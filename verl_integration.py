"""
VERL Integration Interface for Multi-Task Legal Reward System

This module provides the main entry point for VERL training integration, converting
between VERL data formats and the internal legal reward system. Handles both
single-item and batch reward computation with comprehensive error handling,
performance tracking, and fallback mechanisms.

Key Features:
- VERL-compatible reward function interface
- Automatic data format conversion between VERL and internal formats
- Batch processing with concurrent evaluation support
- Comprehensive error handling and graceful degradation
- Performance monitoring and optimization tracking
- Task type inference from data sources
- US jurisdiction automatic detection and routing
- Cost tracking and budget management integration

This is the primary interface that VERL will use during GRPO training to obtain
reward scores for model-generated legal responses.
"""

import time
import asyncio
import logging
import traceback
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Import core components
from core import (
    LegalDataPoint, LegalRewardEvaluation, EvaluationMetadata,
    LegalTaskType, USJurisdiction, LegalDomain,
    LegalRewardSystemError, create_error_context
)

# Import routing system
from routing import (
    MultiTaskLegalRewardRouter, RouterEvaluationResult, EvaluationRequest,
    create_production_router, create_development_router, RouterConfig, RouterMode
)

# Import configuration and utilities
from config import LegalRewardSystemConfig, create_production_config, create_development_config
from utils import get_legal_logger


class VERLDataFormat(Enum):
    """Supported VERL data formats"""
    STANDARD = "standard"           # Standard VERL format (data_source, solution_str, ground_truth)
    EXTENDED = "extended"           # Extended format with extra_info
    BATCH = "batch"                # Batch format with multiple items
    LEGAL_SPECIFIC = "legal_specific"  # Legal-specific format with task_type, jurisdiction


@dataclass
class VERLPerformanceMetrics:
    """Performance metrics for VERL integration"""
    total_calls: int = 0
    total_items_processed: int = 0
    total_processing_time: float = 0.0
    successful_evaluations: int = 0
    failed_evaluations: int = 0
    conversion_errors: int = 0
    router_errors: int = 0
    average_batch_size: float = 0.0
    average_score: float = 0.0
    cost_tracking: Dict[str, float] = field(default_factory=dict)
    
    def update_call_metrics(self, items_count: int, processing_time: float, success: bool):
        """Update call-level metrics"""
        self.total_calls += 1
        self.total_items_processed += items_count
        self.total_processing_time += processing_time
        
        if success:
            self.successful_evaluations += items_count
        else:
            self.failed_evaluations += items_count
        
        self.average_batch_size = self.total_items_processed / self.total_calls
    
    def update_score_metrics(self, scores: List[float]):
        """Update score-related metrics"""
        if scores:
            current_total = self.average_score * max(1, self.successful_evaluations - len(scores))
            new_total = current_total + sum(scores)
            self.average_score = new_total / max(1, self.successful_evaluations)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        success_rate = self.successful_evaluations / max(1, self.total_items_processed)
        avg_time_per_item = self.total_processing_time / max(1, self.total_items_processed)
        
        return {
            "total_calls": self.total_calls,
            "total_items_processed": self.total_items_processed,
            "success_rate": success_rate,
            "average_score": self.average_score,
            "average_batch_size": self.average_batch_size,
            "average_time_per_item_ms": avg_time_per_item * 1000,
            "total_processing_time": self.total_processing_time,
            "conversion_errors": self.conversion_errors,
            "router_errors": self.router_errors,
            "cost_tracking": self.cost_tracking
        }


class VERLDataConverter:
    """Handles conversion between VERL formats and internal legal data structures"""
    
    def __init__(self):
        self.logger = get_legal_logger("verl_converter")
        
        # Task type inference mapping
        self.task_type_keywords = {
            LegalTaskType.JUDICIAL_REASONING: [
                "judge", "ruling", "decision", "court", "opinion", "firac", "analysis",
                "holding", "reasoning", "judicial", "precedent application"
            ],
            LegalTaskType.PRECEDENT_ANALYSIS: [
                "precedent", "case law", "analogous", "distinguish", "cite", "authority",
                "binding", "persuasive", "prior case", "legal precedent"
            ],
            LegalTaskType.OPINION_GENERATION: [
                "argue", "brief", "advocacy", "persuasive", "motion", "memorandum",
                "position", "client", "advocate", "legal argument"
            ],
            LegalTaskType.GENERAL_CHAT: [
                "explain", "help", "question", "general", "advice", "information",
                "understand", "clarify", "what is", "how does"
            ]
        }
        
    def convert_verl_to_legal_data_point(self, 
                                   data_source: str,
                                   solution_str: str,
                                   ground_truth: str = "",
                                   extra_info: Optional[Dict[str, Any]] = None) -> LegalDataPoint:
        """Convert VERL standard format to LegalDataPoint."""
        
        try:
            # Extract or infer task type
            task_type = self._infer_task_type(data_source, solution_str, extra_info)
            
            # Extract or infer jurisdiction
            jurisdiction = self._infer_jurisdiction(solution_str, ground_truth, extra_info)
            
            # Extract legal domains and convert to single domain for LegalDataPoint
            legal_domain = self._infer_legal_domains(solution_str, ground_truth, extra_info)
            
            # Create legal data point with correct parameter name
            legal_data_point = LegalDataPoint(
                query=ground_truth or data_source,  # Use ground_truth as query if available
                response=solution_str,
                task_type=task_type,
                jurisdiction=jurisdiction.value,
                legal_domain=legal_domain,
                metadata={
                    "verl_data_source": data_source,
                    "verl_ground_truth": ground_truth,
                    "verl_extra_info": extra_info or {},
                    "conversion_timestamp": time.time()
                }
            )
            
            self.logger.debug(f"Converted VERL data: task_type={task_type.value}, jurisdiction={jurisdiction.value}")
            return legal_data_point
            
        except Exception as e:
            self.logger.error(f"Error converting VERL data: {e}")
            # Create fallback data point
            return self._create_fallback_data_point(data_source, solution_str, str(e))

    def convert_batch_verl_data(self, batch_data: List[Dict[str, Any]]) -> List[LegalDataPoint]:
        """
        Convert batch of VERL data to LegalDataPoint list.
        
        Args:
            batch_data: List of VERL data dictionaries
            
        Returns:
            List of LegalDataPoint objects
        """
        
        legal_data_points = []
        conversion_errors = 0
        
        for i, data_item in enumerate(batch_data):
            try:
                # Handle different batch data formats
                if isinstance(data_item, dict):
                    data_source = data_item.get("data_source", f"batch_item_{i}")
                    solution_str = data_item.get("solution_str", data_item.get("response", ""))
                    ground_truth = data_item.get("ground_truth", data_item.get("query", ""))
                    extra_info = data_item.get("extra_info", {})
                else:
                    # Handle tuple/list format
                    data_source = f"batch_item_{i}"
                    solution_str = str(data_item)
                    ground_truth = ""
                    extra_info = {}
                
                legal_data_point = self.convert_verl_to_legal_data_point(
                    data_source, solution_str, ground_truth, extra_info
                )
                legal_data_points.append(legal_data_point)
                
            except Exception as e:
                conversion_errors += 1
                self.logger.warning(f"Error converting batch item {i}: {e}")
                # Add fallback data point
                fallback_point = self._create_fallback_data_point(
                    f"batch_item_{i}_error", str(data_item)[:100], str(e)
                )
                legal_data_points.append(fallback_point)
        
        if conversion_errors > 0:
            self.logger.warning(f"Batch conversion completed with {conversion_errors} errors out of {len(batch_data)} items")
        
        return legal_data_points
    
    def _infer_task_type(self, data_source: str, solution_str: str, extra_info: Optional[Dict]) -> LegalTaskType:
        """Infer task type from available data"""
        
        # Check if explicitly provided in extra_info
        if extra_info and "task_type" in extra_info:
            try:
                return LegalTaskType(extra_info["task_type"])
            except ValueError:
                pass
        
        # Infer from data_source
        data_source_lower = data_source.lower()
        solution_lower = solution_str.lower()
        
        # Score each task type based on keyword matches
        task_scores = {}
        for task_type, keywords in self.task_type_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in data_source_lower:
                    score += 2  # Data source matches are weighted higher
                if keyword in solution_lower:
                    score += 1
            task_scores[task_type] = score
        
        # Return task type with highest score, default to GENERAL_CHAT
        if task_scores:
            best_task_type = max(task_scores, key=task_scores.get)
            if task_scores[best_task_type] > 0:
                return best_task_type
        
        return LegalTaskType.GENERAL_CHAT
    
    def _infer_jurisdiction(self, solution_str: str, ground_truth: str, extra_info: Optional[Dict]) -> USJurisdiction:
        """Infer jurisdiction from available data"""
        
        # Check if explicitly provided
        if extra_info and "jurisdiction" in extra_info:
            try:
                return USJurisdiction(extra_info["jurisdiction"])
            except ValueError:
                pass
        
        # Use your existing inference engine
        from jurisdiction.inference_engine import JurisdictionInferenceEngine
        
        if not hasattr(self, '_jurisdiction_engine'):
            self._jurisdiction_engine = JurisdictionInferenceEngine()
        
        # Combine text for analysis
        combined_text = f"{solution_str} {ground_truth}"
        
        # Get inference result
        result = self._jurisdiction_engine.infer_jurisdiction(
            content=combined_text,
            context=extra_info
        )
        
        # Return jurisdiction if confident, otherwise general
        if result.is_confident(threshold=0.6):
            return result.jurisdiction
        else:
            return USJurisdiction.GENERAL
    
    def _infer_legal_domains(self, solution_str: str, ground_truth: str, extra_info: Optional[Dict]) -> List[LegalDomain]:
        """Infer legal domains from content"""
        
        # For now, return general domain; can be enhanced with domain-specific keyword matching
        if extra_info and "legal_domain" in extra_info:
            try:
                return LegalDomain(extra_info["legal_domain"])
            except ValueError:
                pass
            # domains = []
            # for domain_str in extra_info["legal_domains"]:
            #     try:
            #         domains.append(LegalDomain(domain_str))
            #     except ValueError:
            #         continue
            # return domains
        
        return LegalDomain.GENERAL #[LegalDomain.GENERAL]
    
    def _create_fallback_data_point(self, data_source: str, solution_str: str, error_msg: str) -> LegalDataPoint:
        """Create fallback data point when conversion fails"""
        
        return LegalDataPoint(
            query=f"VERL conversion error: {error_msg[:100]}",
            response=solution_str[:500] if solution_str else "No response provided",
            task_type=LegalTaskType.GENERAL_CHAT,
            jurisdiction=USJurisdiction.GENERAL,
            legal_domain=LegalDomain.GENERAL,
            metadata={
                "conversion_error": True,
                "error_message": error_msg,
                "original_data_source": data_source,
                "fallback_timestamp": time.time()
            }
        )

class VERLLegalRewardFunction:
    """
    Main VERL integration class providing comprehensive reward computation capabilities.

    This class serves as the primary interface between VERL and the legal reward system,
    handling data conversion, routing, evaluation, and result formatting.
    """

    def __init__(self, 
                    config: Optional[LegalRewardSystemConfig] = None,
                    router: Optional[MultiTaskLegalRewardRouter] = None,
                    enable_caching: bool = True,
                    enable_cost_tracking: bool = True):
        
        # Initialize configuration
        self.config = config or create_production_config()
        
        if router:
            self.router = router
        else:
            # Create RouterConfig for VERL integration
            router_config = RouterConfig(
                router_mode=RouterMode.PRODUCTION,
                enable_jurisdiction_inference=True,
                enable_hybrid_evaluation=True,
                enable_caching=enable_caching,
                enable_cost_optimization=enable_cost_tracking,
                max_concurrent_evaluations=8,  # Good for VERL batch processing
                evaluation_timeout_seconds=45.0,
                require_jurisdiction_compliance=True,
                fallback_to_general_chat=True,
                max_cost_per_evaluation=0.30,  # Reasonable for training
                aggressive_cost_optimization=True
            )
            
            self.router = MultiTaskLegalRewardRouter(router_config, self.config)
        
        # Initialize components
        self.converter = VERLDataConverter()
        self.logger = get_legal_logger("verl_integration")
        
        # Performance tracking
        self.metrics = VERLPerformanceMetrics()
        self.enable_caching = enable_caching
        self.enable_cost_tracking = enable_cost_tracking
        
        # Configuration
        self.max_batch_size = 100
        self.timeout_seconds = 300.0  # 5 minutes default timeout
        self.fallback_score = 5.0     # Neutral fallback score
        
        self.logger.info("VERLLegalRewardFunction initialized for GRPO training integration")

    
    async def compute_reward(self, 
                           data_source: str,
                           solution_str: str,
                           ground_truth: str = "",
                           extra_info: Optional[Dict[str, Any]] = None) -> float:
        """
        Compute reward for a single data point (VERL standard interface).
        
        Args:
            data_source: Source identifier for task type inference
            solution_str: Model-generated response to evaluate
            ground_truth: Expected response or reference context
            extra_info: Additional metadata (task_type, jurisdiction, etc.)
            
        Returns:
            Reward score (float) for VERL training
        """
        
        start_time = time.time()
        
        try:
            # Convert VERL data to internal format
            legal_data_point = self.converter.convert_verl_to_legal_data_point(
                data_source, solution_str, ground_truth, extra_info
            )

            # Create evaluation request
            request = EvaluationRequest(
                response=legal_data_point.response,
                task_type=LegalTaskType(legal_data_point.task_type),
                prompt=legal_data_point.query,
                jurisdiction=USJurisdiction(legal_data_point.jurisdiction),
                legal_domains=[legal_data_point.legal_domain],
                user_context=legal_data_point.metadata
            )
            
            # Evaluate through router
            result = await self.router.evaluate_response(request)

            # Extract final score
            final_score = result.final_score
            print("Final score: ", final_score)
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics.update_call_metrics(1, processing_time, result.is_successful)
            self.metrics.update_score_metrics([final_score])
            
            if self.enable_cost_tracking and result.total_cost > 0:
                self.metrics.cost_tracking["total_cost"] = self.metrics.cost_tracking.get("total_cost", 0) + result.total_cost
            
            self.logger.debug(f"VERL reward computed: {final_score:.3f} (task: {legal_data_point.task_type.value})")
            
            return final_score
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.metrics.update_call_metrics(1, processing_time, False)
            self.metrics.router_errors += 1
            
            self.logger.error(f"Error computing VERL reward: {e}")
            self.logger.debug(f"Error details: {traceback.format_exc()}")
            
            return self.fallback_score
    
    async def compute_batch_rewards(self, 
                                  batch_data: List[Dict[str, Any]],
                                  max_concurrent: Optional[int] = None) -> List[float]:
        """
        Compute rewards for a batch of data points.
        
        Args:
            batch_data: List of VERL data dictionaries
            max_concurrent: Maximum concurrent evaluations (optional)
            
        Returns:
            List of reward scores for each data point
        """
        
        start_time = time.time()
        batch_size = len(batch_data)
        
        try:
            # Validate batch size
            if batch_size > self.max_batch_size:
                self.logger.warning(f"Batch size {batch_size} exceeds maximum {self.max_batch_size}, processing in chunks")
                return await self._process_large_batch(batch_data, max_concurrent)
            
            # Convert batch data
            legal_data_points = self.converter.convert_batch_verl_data(batch_data)
            
            # Create evaluation requests
            requests = []
            for data_point in legal_data_points:
                request = EvaluationRequest(
                    response=data_point.response,
                    task_type=data_point.task_type,
                    prompt=data_point.query,
                    jurisdiction=data_point.jurisdiction,
                    legal_domains=[data_point.legal_domain],
                    user_context=data_point.metadata
                )
                requests.append(request)
            
            # Process batch through router
            if max_concurrent:
                results = await self._evaluate_batch_concurrent(requests, max_concurrent)
            else:
                results = await self._evaluate_batch_sequential(requests)
            
            # Extract scores
            scores = []
            successful_count = 0
            
            for result in results:
                if result and result.is_successful:
                    scores.append(result.final_score)
                    successful_count += 1
                else:
                    scores.append(self.fallback_score)
            
            # Ensure we return the same number of scores as input
            while len(scores) < batch_size:
                scores.append(self.fallback_score)
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics.update_call_metrics(batch_size, processing_time, successful_count == batch_size)
            self.metrics.update_score_metrics(scores)
            
            # Track costs
            if self.enable_cost_tracking:
                total_batch_cost = sum(r.total_cost for r in results if r)
                self.metrics.cost_tracking["total_cost"] = self.metrics.cost_tracking.get("total_cost", 0) + total_batch_cost
            
            self.logger.info(f"VERL batch processed: {batch_size} items, {successful_count} successful, avg_score: {sum(scores)/len(scores):.3f}")
            
            return scores
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.metrics.update_call_metrics(batch_size, processing_time, False)
            self.metrics.router_errors += 1
            
            self.logger.error(f"Error processing VERL batch: {e}")
            self.logger.debug(f"Error details: {traceback.format_exc()}")
            
            # Return fallback scores for entire batch
            return [self.fallback_score] * batch_size
    
    async def _process_large_batch(self, 
                                 batch_data: List[Dict[str, Any]], 
                                 max_concurrent: Optional[int]) -> List[float]:
        """Process large batches by chunking"""
        
        all_scores = []
        
        for i in range(0, len(batch_data), self.max_batch_size):
            chunk = batch_data[i:i + self.max_batch_size]
            chunk_scores = await self.compute_batch_rewards(chunk, max_concurrent)
            all_scores.extend(chunk_scores)
        
        return all_scores
    
    async def _evaluate_batch_sequential(self, requests: List[EvaluationRequest]) -> List[RouterEvaluationResult]:
        """Evaluate batch sequentially"""
        
        results = []
        
        for request in requests:
            try:
                result = await self.router.evaluate_response(request)
                results.append(result)
            except Exception as e:
                self.logger.warning(f"Sequential evaluation failed for request {request.request_id}: {e}")
                results.append(None)
        
        return results
    
    async def _evaluate_batch_concurrent(self, 
                                       requests: List[EvaluationRequest], 
                                       max_concurrent: int) -> List[RouterEvaluationResult]:
        """Evaluate batch with controlled concurrency"""
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def evaluate_with_semaphore(request: EvaluationRequest) -> RouterEvaluationResult:
            async with semaphore:
                try:
                    return await self.router.evaluate_response(request)
                except Exception as e:
                    self.logger.warning(f"Concurrent evaluation failed for request {request.request_id}: {e}")
                    return None
        
        # Execute all requests concurrently with semaphore control
        tasks = [evaluate_with_semaphore(request) for request in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        clean_results = []
        for result in results:
            if isinstance(result, Exception):
                self.logger.warning(f"Concurrent evaluation exception: {result}")
                clean_results.append(None)
            else:
                clean_results.append(result)
        
        return clean_results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        
        router_stats = self.router.get_statistics() if hasattr(self.router, 'get_statistics') else {}
        
        return {
            "verl_metrics": self.metrics.get_summary(),
            "router_metrics": router_stats,
            "configuration": {
                "max_batch_size": self.max_batch_size,
                "timeout_seconds": self.timeout_seconds,
                "enable_caching": self.enable_caching,
                "enable_cost_tracking": self.enable_cost_tracking,
                "fallback_score": self.fallback_score
            }
        }
    
    def reset_metrics(self):
        """Reset performance metrics"""
        self.metrics = VERLPerformanceMetrics()
        if hasattr(self.router, 'reset_statistics'):
            self.router.reset_statistics()
        self.logger.info("VERL performance metrics reset")


# Global instance for function-based interface
_global_verl_function: Optional[VERLLegalRewardFunction] = None

async def multi_task_legal_reward_function_async(data_source: str,
                                                solution_str: str,
                                                ground_truth: str = "",
                                                extra_info: Optional[Dict[str, Any]] = None) -> float:
    """
    Async version of the VERL reward function.
    
    Args:
        data_source: Source identifier (used for task type inference)
        solution_str: Model-generated response to evaluate
        ground_truth: Expected response or context
        extra_info: Additional metadata (task_type, jurisdiction, etc.)
        
    Returns:
        Final reward score for VERL training (float)
    """
    
    global _global_verl_function
    
    # Initialize global function if needed
    if _global_verl_function is None:
        _global_verl_function = VERLLegalRewardFunction()
    
    return await _global_verl_function.compute_reward(data_source, solution_str, ground_truth, extra_info)


def compute_batch_rewards(batch_data: List[Dict[str, Any]], 
                         max_concurrent: Optional[int] = None) -> List[float]:
    """
    Synchronous batch reward computation for VERL.
    
    Args:
        batch_data: List of VERL data dictionaries
        max_concurrent: Maximum concurrent evaluations
        
    Returns:
        List of reward scores for each data point
    """
    
    global _global_verl_function
    
    if _global_verl_function is None:
        _global_verl_function = VERLLegalRewardFunction()
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(
            _global_verl_function.compute_batch_rewards(batch_data, max_concurrent)
        )
        
        return result
        
    except Exception as e:
        logging.error(f"VERL batch reward computation failed: {e}")
        return [5.0] * len(batch_data)  # Neutral fallback scores
        
    finally:
        try:
            loop.close()
        except:
            pass


async def compute_batch_rewards_async(batch_data: List[Dict[str, Any]], 
                                    max_concurrent: Optional[int] = None) -> List[float]:
    """
    Async batch reward computation for VERL.
    
    Args:
        batch_data: List of VERL data dictionaries
        max_concurrent: Maximum concurrent evaluations
        
    Returns:
        List of reward scores for each data point
    """
    
    global _global_verl_function
    
    if _global_verl_function is None:
        _global_verl_function = VERLLegalRewardFunction()
    
    return await _global_verl_function.compute_batch_rewards(batch_data, max_concurrent)


def get_verl_performance_metrics() -> Dict[str, Any]:
    """Get VERL integration performance metrics"""
    
    global _global_verl_function
    
    if _global_verl_function is None:
        return {"error": "VERL function not initialized"}
    
    return _global_verl_function.get_performance_metrics()


def reset_verl_metrics():
    """Reset VERL performance metrics"""
    
    global _global_verl_function
    
    if _global_verl_function is not None:
        _global_verl_function.reset_metrics()


def configure_verl_function(config: Optional[LegalRewardSystemConfig] = None,
                          max_batch_size: Optional[int] = None,
                          timeout_seconds: Optional[float] = None,
                          fallback_score: Optional[float] = None):
    """
    Configure global VERL function parameters.
    
    Args:
        config: Legal reward system configuration
        max_batch_size: Maximum batch size for processing
        timeout_seconds: Timeout for evaluations
        fallback_score: Fallback score for errors
    """
    
    global _global_verl_function
    
    # Reinitialize with new configuration
    _global_verl_function = VERLLegalRewardFunction(config=config)
    
    if max_batch_size is not None:
        _global_verl_function.max_batch_size = max_batch_size
    
    if timeout_seconds is not None:
        _global_verl_function.timeout_seconds = timeout_seconds
    
    if fallback_score is not None:
        _global_verl_function.fallback_score = fallback_score


# Factory functions for different environments

def create_production_verl_function(config: Optional[LegalRewardSystemConfig] = None) -> VERLLegalRewardFunction:
    """Create production VERL function with optimized settings"""
    
    if config is None:
        config = create_production_config()
    
    router = create_production_router(config)
    
    verl_function = VERLLegalRewardFunction(
        config=config,
        router=router,
        enable_caching=True,
        enable_cost_tracking=True
    )
    
    # Production settings
    verl_function.max_batch_size = 50
    verl_function.timeout_seconds = 180.0
    verl_function.fallback_score = 5.0
    
    return verl_function

def create_training_verl_function(config: Optional[LegalRewardSystemConfig] = None) -> VERLLegalRewardFunction:
    """Create VERL function optimized for training"""
    
    if config is None:
        config = create_production_config()
    
    # Create router specifically for VERL training
    router_config = RouterConfig(
        router_mode=RouterMode.PRODUCTION,
        enable_jurisdiction_inference=True,
        enable_hybrid_evaluation=True,
        enable_caching=True,
        enable_cost_optimization=True,
        max_concurrent_evaluations=12,  # Higher for training
        evaluation_timeout_seconds=60.0,
        require_jurisdiction_compliance=True,
        fallback_to_general_chat=True,
        max_cost_per_evaluation=0.40,
        aggressive_cost_optimization=True
    )
    
    router = MultiTaskLegalRewardRouter(router_config, config)
    
    # Create VERL function with the router
    return VERLLegalRewardFunction(
        config=config,
        router=router,
        enable_caching=True,
        enable_cost_tracking=True
    )

def create_development_verl_function(config: Optional[LegalRewardSystemConfig] = None) -> VERLLegalRewardFunction:
    """Create VERL function for development"""
    
    if config is None:
        config = create_development_config()
    
    # Create router for development
    router_config = RouterConfig(
        router_mode=RouterMode.DEVELOPMENT,
        enable_jurisdiction_inference=True,
        enable_hybrid_evaluation=True,
        enable_caching=False,  # Disable for development
        enable_cost_optimization=False,
        max_concurrent_evaluations=3,
        evaluation_timeout_seconds=30.0,
        require_jurisdiction_compliance=False,
        fallback_to_general_chat=True,
        max_cost_per_evaluation=0.20
    )
    
    # FIX: Use correct constructor interface
    router = MultiTaskLegalRewardRouter(router_config, config)
    
    # Create VERL function with the router
    return VERLLegalRewardFunction(
        config=config,
        router=router,
        enable_caching=False,
        enable_cost_tracking=False
    )

def multi_task_legal_reward_function(data_source: str, 
                                   solution_str: str, 
                                   ground_truth: str, 
                                   extra_info: Optional[Dict] = None) -> float:
    """
    VERL-compatible reward function for multi-task legal AI.
    
    This is the main entry point that VERL will call during training.
    """
    logger = get_legal_logger("multi_task_legal_reward_function")
    print("\n\nData source:  ",data_source)
    print("\n\nSolution_str:  ", solution_str)
    print("\n\nGround truth:  ", ground_truth)
    print("\n\nExtra Info:  ", extra_info)
    try:
        # Create VERL function instance with proper configuration
        if not hasattr(multi_task_legal_reward_function, '_verl_function'):
            # Create and cache the VERL function instance
            config = create_production_config()
            multi_task_legal_reward_function._verl_function = create_training_verl_function(config)
            
        verl_function = multi_task_legal_reward_function._verl_function
        
        # Use the instance to compute reward
        return asyncio.run(verl_function.compute_reward(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info
        ))
        
    except Exception as e:
        # Fallback to neutral score on any error
        logger.error(f"VERL reward function error: {e}")
        return 5.0

# Example VERL configuration for integration
VERL_CONFIG_EXAMPLE = {
    "custom_reward_function": {
        "path": "legal_reward_system.verl_integration",
        "name": "multi_task_legal_reward_function"
    },
    "batch_reward_function": {
        "path": "legal_reward_system.verl_integration", 
        "name": "compute_batch_rewards"
    }
}