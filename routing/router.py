"""
Multi-Task Legal Reward Router for Enhanced Legal Reward System

This module implements the main orchestration component that unifies all system
components into a cohesive legal reward function. It provides the primary interface
for VERL integration and handles task routing, evaluation orchestration, and
comprehensive result aggregation.

Key Features:
- Unified interface for all legal task types
- Intelligent task routing and evaluation orchestration
- Hybrid evaluation system integration (70% specialized + 30% general chat)
- Complete US jurisdiction support with automatic inference
- Cost optimization and performance monitoring
- Production-ready error handling and fallback mechanisms
- Seamless VERL integration with proper data formatting

The router acts as the central hub that coordinates jurisdiction inference,
compliance checking, hybrid evaluation, and result aggregation to provide
a single, reliable interface for the enhanced legal reward system.
"""

import asyncio
from collections import defaultdict
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

# Import core components
from core import (
    LegalRewardEvaluation, JudgeEvaluation, EvaluationMetadata,
    LegalTaskType, USJurisdiction, LegalDomain, APIProvider,
    LegalRewardSystemError, create_error_context
)

# Import system components
from config import LegalRewardSystemConfig, create_production_config
from jurisdiction import (
    JurisdictionInferenceEngine, create_production_inference_engine,
    JurisdictionComplianceJudge, create_production_compliance_judge
)
from judges.general_chat import (
    EnhancedGeneralChatEnsemble, create_production_general_chat_ensemble
)
from judges.api_client import (
    CostOptimizedAPIClient, create_production_api_client
)
from routing.hybrid_evaluation import (
    HybridEvaluationEngine, HybridEvaluationResult, 
    create_production_hybrid_engine, EvaluationMode
)
from utils import (
    LegalRewardLogger, get_legal_logger,
    MultiStrategyLegalRewardCache, create_aggressive_cache,
    MultiProviderRateLimiter, create_production_rate_limiter
)

from judges.base import BaseJudgeEnsemble, create_evaluation_context
from routing.task_weights import TaskDifficultyWeightManager

class RouterMode(Enum):
    """Operating modes for the legal reward router"""
    PRODUCTION = "production"           # Full production mode with all features
    DEVELOPMENT = "development"         # Development mode with relaxed settings
    EVALUATION_ONLY = "evaluation_only" # Evaluation only, no training optimizations
    COST_OPTIMIZED = "cost_optimized"   # Maximum cost optimization
    HIGH_ACCURACY = "high_accuracy"     # Maximum accuracy, higher cost


@dataclass
class RouterConfig:
    """
    Configuration for the multi-task legal reward router.
    
    Contains all configuration parameters for router operation including
    mode settings, performance tuning, and integration parameters.
    """
    
    # Operating mode
    router_mode: RouterMode = RouterMode.PRODUCTION
    
    # Feature flags
    enable_jurisdiction_inference: bool = True
    enable_hybrid_evaluation: bool = True
    enable_caching: bool = True
    enable_cost_optimization: bool = True
    cache_evaluation_results: bool = True
    
    # Performance settings
    max_concurrent_evaluations: int = 10
    evaluation_timeout_seconds: float = 60.0
    
    # Compliance settings
    require_jurisdiction_compliance: bool = True
    fallback_to_general_chat: bool = True
    
    # Cost settings
    max_cost_per_evaluation: float = 0.50
    aggressive_cost_optimization: bool = True
    prefer_cached_results: bool = True
    
    # Hybrid evaluation settings
    specialized_weight: float = 0.7
    general_chat_weight: float = 0.3
    jurisdiction_failure_penalty: float = 0.2

    # Quality control settings
    min_confidence_threshold: float = 0.5  # ← ADD THIS FIELD
    
    def validate_config(self) -> List[str]:
        """Validate router configuration and return any issues"""
        
        issues = []
        
        # Validate numeric ranges
        if self.max_concurrent_evaluations <= 0:
            issues.append("max_concurrent_evaluations must be positive")
        
        if self.evaluation_timeout_seconds <= 0:
            issues.append("evaluation_timeout_seconds must be positive")
        
        if self.max_cost_per_evaluation <= 0:
            issues.append("max_cost_per_evaluation must be positive")
        
        # Validate weights
        if not (0.0 <= self.specialized_weight <= 1.0):
            issues.append("specialized_weight must be between 0.0 and 1.0")
        
        if not (0.0 <= self.general_chat_weight <= 1.0):
            issues.append("general_chat_weight must be between 0.0 and 1.0")
        
        if abs(self.specialized_weight + self.general_chat_weight - 1.0) > 0.01:
            issues.append("specialized_weight + general_chat_weight must equal 1.0")
        
        # Validate enum values
        if self.router_mode not in RouterMode:
            issues.append(f"Invalid router_mode: {self.router_mode}")
        
        # Validate confidence threshold
        if not (0.0 <= self.min_confidence_threshold <= 1.0):
            issues.append("min_confidence_threshold must be between 0.0 and 1.0")

        return issues
    
    def get(self, key: str, default=None):
        """Get configuration value by key name for backwards compatibility"""
        return getattr(self, key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "router_mode": self.router_mode.value,
            "enable_jurisdiction_inference": self.enable_jurisdiction_inference,
            "enable_hybrid_evaluation": self.enable_hybrid_evaluation,
            "enable_caching": self.enable_caching,
            "enable_cost_optimization": self.enable_cost_optimization,
            "max_concurrent_evaluations": self.max_concurrent_evaluations,
            "evaluation_timeout_seconds": self.evaluation_timeout_seconds,
            "require_jurisdiction_compliance": self.require_jurisdiction_compliance,
            "fallback_to_general_chat": self.fallback_to_general_chat,
            "max_cost_per_evaluation": self.max_cost_per_evaluation,
            "aggressive_cost_optimization": self.aggressive_cost_optimization,
            "prefer_cached_results": self.prefer_cached_results,
            "specialized_weight": self.specialized_weight,
            "general_chat_weight": self.general_chat_weight,
            "jurisdiction_failure_penalty": self.jurisdiction_failure_penalty,
            "min_confidence_threshold": self.min_confidence_threshold 
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RouterConfig':
        """Create RouterConfig from dictionary"""
        
        # Convert router_mode string back to enum
        if 'router_mode' in config_dict and isinstance(config_dict['router_mode'], str):
            config_dict['router_mode'] = RouterMode(config_dict['router_mode'])
        
        return cls(**config_dict)


@dataclass
class EvaluationRequest:
    """
    Structured evaluation request for the router.
    
    Contains all information needed for comprehensive legal evaluation
    including task context, jurisdiction requirements, and optimization hints.
    """
    
    # Core evaluation data
    response: str
    task_type: LegalTaskType
    prompt: str = ""
    
    # Jurisdiction context
    jurisdiction: Optional[USJurisdiction] = None
    infer_jurisdiction: bool = True
    legal_domains: List[LegalDomain] = field(default_factory=list)
    
    # Evaluation preferences
    evaluation_mode: Optional[EvaluationMode] = None
    preferred_providers: List[APIProvider] = field(default_factory=list)
    max_cost: Optional[float] = None
    
    # Request metadata
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_context: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1  # 1 = high, 2 = medium, 3 = low
    
    # Performance hints
    cache_key_hint: Optional[str] = None
    expected_complexity: Optional[str] = None  # "simple", "medium", "complex"
    
    def validate_request(self) -> List[str]:
        """Validate evaluation request"""
        
        issues = []
        
        if not self.response.strip():
            issues.append("Response text cannot be empty")
        
        if self.task_type not in LegalTaskType:
            issues.append(f"Invalid task type: {self.task_type}")
        
        if self.jurisdiction and self.jurisdiction not in USJurisdiction:
            issues.append(f"Invalid jurisdiction: {self.jurisdiction}")
        
        if self.max_cost and self.max_cost <= 0:
            issues.append("Max cost must be positive if specified")
        
        if not (1 <= self.priority <= 3):
            issues.append("Priority must be 1, 2, or 3")
        
        return issues
    
    def get_cache_key(self) -> str:
        """Generate cache key for this request"""
        
        if self.cache_key_hint:
            return self.cache_key_hint
        
        import hashlib
        import json
        
        # Create deterministic key from core request data
        key_data = {
            "response_hash": hashlib.md5(self.response.encode()).hexdigest(),
            "task_type": self.task_type.value,
            "jurisdiction": self.jurisdiction.value if self.jurisdiction else None,
            "domains": sorted([d.value for d in self.legal_domains]),
            "prompt_hash": hashlib.md5(self.prompt.encode()).hexdigest() if self.prompt else None
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return f"eval_request_{hashlib.sha256(key_string.encode()).hexdigest()[:16]}"


@dataclass
class RouterEvaluationResult:
    """
    Comprehensive evaluation result from the router.
    
    Contains the final evaluation score along with detailed component
    breakdowns, metadata, and performance information.
    """
    
    # Core results
    final_score: float  # 0.0 to 10.0
    confidence: float   # 0.0 to 1.0
    is_successful: bool = True
    
    # Component results
    hybrid_result: Optional[HybridEvaluationResult] = None
    jurisdiction_inference_result: Optional[Any] = None  # JurisdictionInferenceResult
    compliance_result: Optional[Any] = None  # JurisdictionComplianceResult
    
    # Evaluation path
    evaluation_mode: EvaluationMode = EvaluationMode.AUTO
    jurisdiction_used: USJurisdiction = USJurisdiction.GENERAL
    specialized_evaluation_attempted: bool = False
    gating_passed: bool = True
    
    # Performance and cost
    total_cost: float = 0.0
    processing_time_ms: float = 0.0
    cache_hit: bool = False
    provider_usage: Dict[str, int] = field(default_factory=dict)
    
    # Quality assessment
    reasoning: str = ""
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Detailed metadata
    component_breakdown: Dict[str, Any] = field(default_factory=dict)
    evaluation_metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Request context
    request_id: str = ""
    task_type: LegalTaskType = LegalTaskType.GENERAL_CHAT
    
    def to_legal_reward_evaluation(self) -> LegalRewardEvaluation:
        """Convert to standard LegalRewardEvaluation format"""
        
        # Extract judge evaluations from hybrid result
        judge_evaluations = []
        if self.hybrid_result and self.hybrid_result.general_chat_evaluation:
            general_eval = self.hybrid_result.general_chat_evaluation
            for component_name, evaluation in general_eval.component_evaluations.items():
                judge_evaluations.append(evaluation)
        
        # Create evaluation metadata
        metadata = EvaluationMetadata(
            task_type=self.task_type,
            jurisdiction=self.jurisdiction_used,
            legal_domains=[],  # Will be populated if available
            evaluation_method="multi_task_router",
            processing_time_ms=self.processing_time_ms,
            total_cost=self.total_cost,
            cache_hit=self.cache_hit,
            additional_info={
                "evaluation_mode": self.evaluation_mode.value,
                "gating_passed": self.gating_passed,
                "specialized_attempted": self.specialized_evaluation_attempted,
                "component_breakdown": self.component_breakdown,
                "provider_usage": self.provider_usage,
                "request_id": self.request_id
            }
        )
        
        return LegalRewardEvaluation(
            overall_score=self.final_score,
            confidence=self.confidence,
            reasoning=self.reasoning,
            judge_evaluations=judge_evaluations,
            evaluation_metadata=metadata
        )
    
    def get_summary(self) -> str:
        """Get human-readable summary of the evaluation"""
        
        status = "SUCCESS" if self.is_successful else "FAILED"
        gating = "GATED" if not self.gating_passed else "PASS"
        mode = self.evaluation_mode.value.upper()
        
        summary = f"{status} | Score: {self.final_score:.1f}/10.0 | Confidence: {self.confidence:.2f} | Mode: {mode} | Gating: {gating}"
        
        if self.cache_hit:
            summary += " | CACHED"
        
        if self.total_cost > 0:
            summary += f" | Cost: ${self.total_cost:.4f}"
        
        return summary


class MultiTaskLegalRewardRouter:
    """
    Main orchestration component for the enhanced legal reward system.
    
    Provides unified interface for legal task evaluation with intelligent routing,
    hybrid evaluation, jurisdiction compliance, and comprehensive optimization.
    This is the primary entry point for VERL integration and training.
    """
    
    def __init__(self, 
             config: Optional[RouterConfig] = None,
             system_config: Optional[LegalRewardSystemConfig] = None):
        """Initialize multi-task legal reward router with correct interface"""
        
        # Set up configurations with defaults
        self.config = config or RouterConfig()
        self.system_config = system_config or create_production_config()
        
        # Initialize logger first
        self.logger = get_legal_logger("multi_task_router")
        
        # Validate RouterConfig if it has validation method
        if hasattr(self.config, 'validate_config'):
            config_issues = self.config.validate_config()
            if config_issues:
                raise LegalRewardSystemError(
                    f"Invalid router configuration: {'; '.join(config_issues)}",
                    error_context=create_error_context("router", "init")
                )
        
        try:
            # Initialize core evaluation components
            self.hybrid_evaluation_system = create_production_hybrid_engine() #HybridEvaluationEngine(self.system_config)
            self.weight_manager = TaskDifficultyWeightManager(self.system_config)
            # self.api_client = CostOptimizedAPIClient(self.system_config)
            self._initialize_system_components()
            
            # Initialize jurisdiction components
            # self.jurisdiction_inference_engine = JurisdictionInferenceEngine(self.system_config)
            # self.compliance_judge = JurisdictionComplianceJudge(self.system_config)
            self.jurisdiction_inference_engine = create_production_inference_engine()
            self.compliance_judge = create_production_compliance_judge()
            
            # Initialize judge ensemble registry
            self.judge_ensembles = {}
            self.general_chat_ensemble = None
            
            # Performance tracking
            self.total_evaluations = 0
            self.total_routing_time = 0.0
            self.task_type_counts = {task_type: 0 for task_type in LegalTaskType}
            self.error_count = 0
            
            # FIX: Get concurrency settings from RouterConfig
            if hasattr(self.config, 'max_concurrent_evaluations'):
                self.max_concurrent_evaluations = self.config.max_concurrent_evaluations
            else:
                self.max_concurrent_evaluations = 10  # Default
                
            self.evaluation_semaphore = asyncio.Semaphore(self.max_concurrent_evaluations)
            
            # Router stats (expected by validation)
            self.router_stats = {
                "total_evaluations": 0,
                "successful_evaluations": 0,
                "failed_evaluations": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "total_cost": 0.0,
                "avg_evaluation_time": 0.0,
                "task_type_distribution": {task_type: 0 for task_type in LegalTaskType},
                "jurisdiction_distribution": {},
                "evaluation_mode_distribution": defaultdict(int),
                "gating_failures": 0
            }
            
            self.logger.info("All system components initialized successfully")
            
            # Determine environment for logging
            environment = getattr(self.system_config, 'environment', 'production')
            self.logger.info(f"Multi-task legal reward router initialized in {environment} mode")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize router components: {e}")
            raise LegalRewardSystemError(
                f"Router initialization failed: {e}",
                error_context=create_error_context("router", "init")
            )


    def _get_config_value(self, key: str, default=None):
        """Safely get configuration value from either RouterConfig or system config"""
        
        # Try RouterConfig first
        if hasattr(self.config, key):
            return getattr(self.config, key)
        
        # Try system config
        if hasattr(self.system_config, key):
            return getattr(self.system_config, key)
        
        # Try as dict access for RouterConfig
        if hasattr(self.config, 'get'):
            return self.config.get(key, default)
        
        return default

    def register_judge_ensemble(self, task_type: LegalTaskType, ensemble):
        """Register a judge ensemble for a specific task type"""
        
        if not hasattr(ensemble, 'evaluate_response') and not hasattr(ensemble, 'evaluate_async'):
            self.logger.warning(f"Ensemble {ensemble} may not have proper evaluation interface")
        
        # Register the ensemble
        self.judge_ensembles[task_type] = ensemble
        
        # Keep reference to general chat ensemble for hybrid evaluation
        if task_type == LegalTaskType.GENERAL_CHAT:
            self.general_chat_ensemble = ensemble
        
        # Log registration with ensemble type detection
        ensemble_type = "REAL" if 'Mock' not in ensemble.__class__.__name__ else "MOCK"
        ensemble_name = getattr(ensemble, 'ensemble_name', ensemble.__class__.__name__)
        self.logger.info(f"Registered {ensemble_type} ensemble {ensemble_name} for {task_type.value}")


    def unregister_judge_ensemble(self, task_type: LegalTaskType):
        """Unregister a judge ensemble"""
        if task_type in self.judge_ensembles:
            ensemble = self.judge_ensembles[task_type]
            ensemble_name = getattr(ensemble, 'ensemble_name', ensemble.__class__.__name__)
            del self.judge_ensembles[task_type]
            
            # Clear general chat reference if unregistering it
            if task_type == LegalTaskType.GENERAL_CHAT:
                self.general_chat_ensemble = None
                
            self.logger.info(f"Unregistered {ensemble_name} for {task_type.value}")


    async def route_and_evaluate_single(self, data_point) -> float:
        """Route single data point through real evaluation pipeline"""
        
        start_time = time.time()
        
        try:
            # Determine task type
            if hasattr(data_point, 'task_type'):
                task_type = data_point.task_type
                if isinstance(task_type, str):
                    task_type = LegalTaskType(task_type)
            else:
                task_type = LegalTaskType.GENERAL_CHAT
            
            # Check if we have a registered ensemble for this task type
            if task_type in self.judge_ensembles:
                ensemble = self.judge_ensembles[task_type]
                
                # Get jurisdiction (with fallback)
                jurisdiction = getattr(data_point, 'jurisdiction', 'general')
                if isinstance(jurisdiction, str):
                    try:
                        jurisdiction = USJurisdiction(jurisdiction)
                    except ValueError:
                        jurisdiction = USJurisdiction.GENERAL
                
                # Extract response text and prompt
                response = getattr(data_point, 'response', str(data_point))
                prompt = getattr(data_point, 'query', '')
                
                # Use ensemble evaluation with proper interface
                try:
                    if hasattr(ensemble, 'evaluate_response'):
                        # Use new interface
                        evaluation = await ensemble.evaluate_response(
                            response=response,
                            task_type=task_type,
                            jurisdiction=jurisdiction,
                            prompt=prompt
                        )
                        score = evaluation.score
                    elif hasattr(ensemble, 'evaluate_async'):
                        # Use legacy interface
                        evaluation = await ensemble.evaluate_async(response)
                        score = evaluation.score
                    else:
                        # Fallback for simple ensembles
                        score = 7.0  # Default good score
                    
                    # Apply task difficulty weighting if available
                    if hasattr(self.weight_manager, 'get_weight'):
                        weight = self.weight_manager.get_weight(task_type)
                        final_score = score * weight
                    else:
                        # Default weights if weight manager doesn't have get_weight
                        weights = {
                            LegalTaskType.JUDICIAL_REASONING: 1.5,
                            LegalTaskType.PRECEDENT_ANALYSIS: 1.3,
                            LegalTaskType.OPINION_GENERATION: 1.1,
                            LegalTaskType.GENERAL_CHAT: 1.0
                        }
                        weight = weights.get(task_type, 1.0)
                        final_score = score * weight
                    
                    # Track performance
                    self.total_evaluations += 1
                    self.total_routing_time += time.time() - start_time
                    self.task_type_counts[task_type] += 1
                    
                    # Update router stats
                    self.router_stats["total_evaluations"] += 1
                    self.router_stats["successful_evaluations"] += 1
                    self.router_stats["task_type_distribution"][task_type] += 1
                    
                    return max(0.0, min(15.0, final_score))  # Bound to valid range
                    
                except Exception as e:
                    self.logger.error(f"Ensemble evaluation failed: {e}")
                    self.error_count += 1
                    self.router_stats["failed_evaluations"] += 1
                    return 5.0  # Neutral fallback
                
            else:
                # No ensemble registered for this task type
                self.logger.warning(f"No ensemble registered for task type {task_type.value}")
                self.error_count += 1
                self.router_stats["failed_evaluations"] += 1
                return 5.0  # Neutral fallback score
                
        except Exception as e:
            self.logger.error(f"Error in route_and_evaluate_single: {e}")
            self.error_count += 1
            self.router_stats["failed_evaluations"] += 1
            return 5.0  # Neutral fallback on error


    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        return {
            "router_initialized": True,
            "registered_ensembles": len(self.judge_ensembles),
            "ensemble_types": {
                task.value: ensemble.__class__.__name__ 
                for task, ensemble in self.judge_ensembles.items()
            },
            "total_evaluations": self.total_evaluations,
            "total_routing_time": self.total_routing_time,
            "error_count": self.error_count,
            "task_type_counts": {task.value: count for task, count in self.task_type_counts.items()},
            "general_chat_available": self.general_chat_ensemble is not None,
            "hybrid_evaluation_ready": self.general_chat_ensemble is not None and len(self.judge_ensembles) > 1,
            "router_stats": self.router_stats
        }

    def validate_router_ready(self) -> bool:
        """Validate that router is ready for evaluation"""
        
        # Check basic requirements
        if not hasattr(self, 'judge_ensembles') or not self.judge_ensembles:
            return False
        
        if not hasattr(self, 'api_client') or self.api_client is None:
            return False
        
        # Check that we have at least one ensemble
        if len(self.judge_ensembles) == 0:
            return False
        
        # Check that all registered ensembles have required interface
        for task_type, ensemble in self.judge_ensembles.items():
            if not hasattr(ensemble, 'evaluate_response') and not hasattr(ensemble, 'evaluate_async'):
                return False
        
        return True


    def _initialize_system_components(self):
        """Initialize all system components based on configuration"""
        
        try:
            # Initialize utilities first
            if self.config.enable_caching:
                cache_config = self.system_config.caching_config
                self.cache = create_aggressive_cache(cache_config)
            else:
                self.cache = None
            
            # Initialize rate limiter
            rate_limit_config = self.system_config.rate_limiting_config
            self.rate_limiter = create_production_rate_limiter(rate_limit_config)
            
            # Initialize API client
            provider_configs = {}
            for provider_name, provider_config in self.system_config.api_providers.items():
                provider_configs[provider_name] = provider_config
            
            self.api_client = create_production_api_client(
                provider_configs, self.cache, self.rate_limiter
            )
            
            # Initialize jurisdiction components
            if self.config.enable_jurisdiction_inference:
                self.jurisdiction_inference_engine = create_production_inference_engine()
                self.compliance_judge = create_production_compliance_judge()
            else:
                self.jurisdiction_inference_engine = None
                self.compliance_judge = None
            
            # Initialize evaluation components
            self.general_chat_ensemble = create_production_general_chat_ensemble(
                self.api_client, self.cache, self.rate_limiter
            )
            
            # Initialize hybrid evaluation engine
            if self.config.enable_hybrid_evaluation:
                # For now, specialized ensembles are placeholders
                specialized_ensembles = {}  # Will be populated when specialized ensembles are implemented
                
                self.hybrid_engine = create_production_hybrid_engine(
                    self.general_chat_ensemble, specialized_ensembles
                )
            else:
                self.hybrid_engine = None
            
            self.logger.info("All system components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize system components: {e}")
            raise LegalRewardSystemError(
                f"Router initialization failed: {e}",
                error_context=create_error_context("router", "initialize_components"),
                original_exception=e
            )
    
    async def evaluate_response(self, request: EvaluationRequest) -> RouterEvaluationResult:
        """
        Main evaluation method that orchestrates the complete evaluation process.
        
        This is the primary entry point for all legal response evaluations,
        providing the unified interface for VERL integration.
        
        Args:
            request: Structured evaluation request with all necessary context
            
        Returns:
            RouterEvaluationResult with comprehensive evaluation details
        """
        
        start_time = time.time()
        self.router_stats["total_evaluations"] += 1
        
        try:
            # Validate request
            request_issues = request.validate_request()
            if request_issues:
                raise LegalRewardSystemError(
                    f"Invalid evaluation request: {'; '.join(request_issues)}",
                    error_context=create_error_context("router", "validate_request")
                )
            
            # Check cache first
            if self.config.cache_evaluation_results:
                cached_result = await self._check_evaluation_cache(request)
                if cached_result:
                    self.router_stats["cache_hits"] += 1
                    return cached_result
            
            self.router_stats["cache_misses"] += 1
            
            # Step 1: Jurisdiction inference and validation
            jurisdiction_result = await self._handle_jurisdiction_inference(request)
            
            # Step 2: Route evaluation based on task type and jurisdiction
            evaluation_result = await self._route_evaluation(request, jurisdiction_result)
            
            # Step 3: Create comprehensive result
            router_result = await self._create_router_result(
                request, jurisdiction_result, evaluation_result, start_time
            )
            
            # Step 4: Cache successful result
            if self.config.cache_evaluation_results and router_result.is_successful:
                await self._cache_evaluation_result(request, router_result)
            
            # Step 5: Update statistics
            self._update_router_stats(request, router_result, True)
            
            return router_result
            
        except Exception as e:
            self.logger.error(f"Evaluation failed for request {request.request_id}: {e}")
            
            # Create error result
            error_result = self._create_error_result(request, str(e), time.time() - start_time)
            self._update_router_stats(request, error_result, False)
            
            return error_result
    
    # async def _check_evaluation_cache(self, request: EvaluationRequest) -> Optional[RouterEvaluationResult]:
    #     """Check cache for existing evaluation result"""
        
    #     if not self.cache:
    #         return None
        
    #     try:
    #         cache_key = request.get_cache_key()
    #         # Cache integration would be implemented here
    #         # This is a placeholder for the actual cache lookup
    #         return None
            
    #     except Exception as e:
    #         self.logger.warning(f"Cache check failed: {e}")
    #         return None

    async def _check_evaluation_cache(self, request: EvaluationRequest) -> Optional[RouterEvaluationResult]:
        """Check cache for existing evaluation result"""
        
        if not self.cache:
            return None
        
        try:
            cache_key = request.get_cache_key()
            cached_data = self.cache.get_cached_response(cache_key)
            
            if cached_data:
                self.router_stats["cache_hits"] += 1
                return RouterEvaluationResult.from_cache(cached_data)
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Cache check failed: {e}")
            return None
    
    async def _handle_jurisdiction_inference(self, request: EvaluationRequest) -> Optional[Any]:
        """Handle jurisdiction inference and validation"""
        
        # Use provided jurisdiction if available
        if request.jurisdiction:
            return None  # No inference needed
        
        # Skip inference if disabled
        if not self.config.enable_jurisdiction_inference or not self.jurisdiction_inference_engine:
            return None
        
        try:
            # Infer jurisdiction from response content
            inference_result = self.jurisdiction_inference_engine.infer_jurisdiction(
                content=request.response,
                task_type=request.task_type,
                context=request.user_context
            )
            
            # Update request with inferred jurisdiction
            if inference_result.is_confident():
                request.jurisdiction = inference_result.jurisdiction
                self.logger.debug(f"Inferred jurisdiction: {inference_result.jurisdiction.value} (confidence: {inference_result.confidence:.2f})")
            else:
                request.jurisdiction = USJurisdiction.GENERAL
                self.logger.debug("Low confidence jurisdiction inference, using GENERAL")
            
            return inference_result
            
        except Exception as e:
            self.logger.warning(f"Jurisdiction inference failed: {e}")
            request.jurisdiction = USJurisdiction.GENERAL
            return None
    
    async def _route_evaluation(self, 
                              request: EvaluationRequest, 
                              jurisdiction_result: Optional[Any]) -> HybridEvaluationResult:
        """Route evaluation to appropriate evaluation system"""
        
        # Determine evaluation mode
        evaluation_mode = request.evaluation_mode or self._determine_evaluation_mode(request)
        
        # Use hybrid evaluation if available
        if self.config.enable_hybrid_evaluation and self.hybrid_engine:
            return await self.hybrid_engine.evaluate_hybrid(
                response=request.response,
                task_type=request.task_type,
                jurisdiction=request.jurisdiction or USJurisdiction.GENERAL,
                prompt=request.prompt,
                legal_domains=request.legal_domains,
                evaluation_mode=evaluation_mode
            )
        
        # Fallback to general chat evaluation only
        else:
            from judges.base import create_evaluation_context
            
            context = create_evaluation_context(
                task_type=request.task_type,
                jurisdiction=request.jurisdiction or USJurisdiction.GENERAL,
                prompt=request.prompt,
                legal_domains=request.legal_domains
            )
            
            general_result = await self.general_chat_ensemble.evaluate_with_gating(
                request.response, context
            )
            
            # Convert to hybrid result format
            from .hybrid_evaluation import HybridEvaluationResult, SpecializedEvaluationStatus
            
            return HybridEvaluationResult(
                hybrid_score=general_result.overall_score,
                confidence=general_result.confidence,
                is_successful=True,
                general_chat_score=general_result.overall_score,
                general_chat_confidence=general_result.confidence,
                specialized_status=SpecializedEvaluationStatus.NOT_ATTEMPTED,
                general_chat_gating_passed=general_result.is_gated,
                jurisdiction_compliance_score=general_result.jurisdiction_compliance_score,
                applied_specialized_weight=0.0,
                applied_general_chat_weight=1.0,
                general_chat_evaluation=general_result,
                evaluation_mode=EvaluationMode.GENERAL_CHAT_ONLY,
                reasoning=f"General chat only: {general_result.reasoning}",
                recommendations=general_result.recommendations
            )
    
    def _determine_evaluation_mode(self, request: EvaluationRequest) -> EvaluationMode:
        """Determine appropriate evaluation mode for the request"""
        
        # General chat tasks use general chat only
        if request.task_type == LegalTaskType.GENERAL_CHAT:
            return EvaluationMode.GENERAL_CHAT_ONLY
        
        # For specialized tasks, check if hybrid evaluation is available
        if self.config.enable_hybrid_evaluation and self.hybrid_engine:
            return EvaluationMode.HYBRID
        
        # Fallback to general chat
        return EvaluationMode.GENERAL_CHAT_ONLY
    
    async def _create_router_result(self, 
                                  request: EvaluationRequest,
                                  jurisdiction_result: Optional[Any],
                                  evaluation_result: HybridEvaluationResult,
                                  start_time: float) -> RouterEvaluationResult:
        """Create comprehensive router evaluation result"""
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Extract cost information
        total_cost = 0.0
        provider_usage = {}
        if hasattr(evaluation_result, 'total_cost'):
            total_cost = getattr(evaluation_result, 'total_cost', 0.0)
        
        # Create component breakdown
        component_breakdown = {}
        if evaluation_result:
            component_breakdown = evaluation_result.get_component_breakdown()
        
        # Generate recommendations
        recommendations = evaluation_result.recommendations if evaluation_result else []
        warnings = []
        
        # Add cost warning if needed
        if total_cost > self.config.max_cost_per_evaluation:
            warnings.append(f"Evaluation cost (${total_cost:.4f}) exceeded maximum (${self.config.max_cost_per_evaluation:.4f})")
        
        # Add confidence warning if needed
        if evaluation_result and evaluation_result.confidence < self.config.min_confidence_threshold:
            warnings.append(f"Low confidence evaluation ({evaluation_result.confidence:.2f})")
        
        # Create router result
        return RouterEvaluationResult(
            final_score=evaluation_result.hybrid_score if evaluation_result else 5.0,
            confidence=evaluation_result.confidence if evaluation_result else 0.5,
            is_successful=evaluation_result.is_successful if evaluation_result else False,
            hybrid_result=evaluation_result,
            jurisdiction_inference_result=jurisdiction_result,
            evaluation_mode=evaluation_result.evaluation_mode if evaluation_result else EvaluationMode.GENERAL_CHAT_ONLY,
            jurisdiction_used=request.jurisdiction or USJurisdiction.GENERAL,
            specialized_evaluation_attempted=evaluation_result.specialized_status.value != "not_attempted" if evaluation_result else False,
            gating_passed=evaluation_result.general_chat_gating_passed if evaluation_result else True,
            total_cost=total_cost,
            processing_time_ms=processing_time,
            cache_hit=False,
            provider_usage=provider_usage,
            reasoning=evaluation_result.reasoning if evaluation_result else "Evaluation failed",
            recommendations=recommendations,
            warnings=warnings,
            component_breakdown=component_breakdown,
            evaluation_metadata={
                "router_mode": self.config.router_mode.value,
                "jurisdiction_inference_enabled": self.config.enable_jurisdiction_inference,
                "hybrid_evaluation_enabled": self.config.enable_hybrid_evaluation,
                "cost_optimization_enabled": self.config.enable_cost_optimization
            },
            request_id=request.request_id,
            task_type=request.task_type
        )
    
    async def _cache_evaluation_result(self, request: EvaluationRequest, result: RouterEvaluationResult):
        """Cache evaluation result for future use"""
        
        if not self.cache:
            return
        
        try:
            cache_key = request.get_cache_key()
            # Cache implementation would go here
            # This is a placeholder for the actual cache storage
            pass
            
        except Exception as e:
            self.logger.warning(f"Failed to cache evaluation result: {e}")
    
    def _create_error_result(self, 
                           request: EvaluationRequest, 
                           error_msg: str, 
                           processing_time: float) -> RouterEvaluationResult:
        """Create error result for failed evaluations"""
        
        return RouterEvaluationResult(
            final_score=3.0,  # Below average score for errors
            confidence=0.1,
            is_successful=False,
            evaluation_mode=EvaluationMode.GENERAL_CHAT_ONLY,
            jurisdiction_used=request.jurisdiction or USJurisdiction.GENERAL,
            specialized_evaluation_attempted=False,
            gating_passed=False,
            total_cost=0.0,
            processing_time_ms=processing_time * 1000,
            cache_hit=False,
            reasoning=f"Evaluation failed: {error_msg[:200]}",
            recommendations=["Manual review required due to evaluation error"],
            warnings=[f"System error: {error_msg[:100]}"],
            request_id=request.request_id,
            task_type=request.task_type
        )
    
    def _update_router_stats(self, 
                           request: EvaluationRequest, 
                           result: RouterEvaluationResult, 
                           success: bool):
        """Update router performance statistics"""
        
        if success:
            self.router_stats["successful_evaluations"] += 1
        else:
            self.router_stats["failed_evaluations"] += 1
        
        # Update cost tracking
        self.router_stats["total_cost"] += result.total_cost
        
        # Update timing
        total_evals = self.router_stats["total_evaluations"]
        current_avg = self.router_stats["avg_evaluation_time"]
        self.router_stats["avg_evaluation_time"] = (
            (current_avg * (total_evals - 1) + result.processing_time_ms) / total_evals
        )
        
        # Update distributions
        self.router_stats["task_type_distribution"][request.task_type] += 1
        self.router_stats["evaluation_mode_distribution"][result.evaluation_mode] += 1
        
        jurisdiction_key = result.jurisdiction_used.value
        self.router_stats["jurisdiction_distribution"][jurisdiction_key] = (
            self.router_stats["jurisdiction_distribution"].get(jurisdiction_key, 0) + 1
        )
        
        if not result.gating_passed:
            self.router_stats["gating_failures"] += 1
        
        if result.cache_hit:
            self.router_stats["cache_hits"] += 1
        else:
            self.router_stats["cache_misses"] += 1
    
    # Convenience methods for common use cases
    
    async def evaluate_legal_response(self, 
                                    response: str,
                                    task_type: LegalTaskType,
                                    prompt: str = "",
                                    jurisdiction: Optional[USJurisdiction] = None) -> RouterEvaluationResult:
        """
        Convenience method for evaluating legal responses.
        
        Args:
            response: Legal response text to evaluate
            task_type: Type of legal task
            prompt: Original prompt/question
            jurisdiction: Jurisdiction context (optional)
            
        Returns:
            RouterEvaluationResult with comprehensive evaluation
        """
        
        request = EvaluationRequest(
            response=response,
            task_type=task_type,
            prompt=prompt,
            jurisdiction=jurisdiction,
            infer_jurisdiction=jurisdiction is None
        )
        
        return await self.evaluate_response(request)
    
    async def evaluate_general_chat(self, response: str, prompt: str = "") -> RouterEvaluationResult:
        """
        Convenience method for general chat evaluation.
        
        Args:
            response: Chat response to evaluate
            prompt: Original prompt/question
            
        Returns:
            RouterEvaluationResult with general chat evaluation
        """
        
        request = EvaluationRequest(
            response=response,
            task_type=LegalTaskType.GENERAL_CHAT,
            prompt=prompt,
            jurisdiction=USJurisdiction.GENERAL,
            infer_jurisdiction=False
        )
        
        return await self.evaluate_response(request)
    
    async def evaluate_specialized_task(self, 
                                      response: str,
                                      task_type: LegalTaskType,
                                      jurisdiction: USJurisdiction,
                                      prompt: str = "") -> RouterEvaluationResult:
        """
        Convenience method for specialized legal task evaluation.
        
        Args:
            response: Legal response to evaluate
            task_type: Specialized legal task type
            jurisdiction: Jurisdiction context
            prompt: Original prompt/question
            
        Returns:
            RouterEvaluationResult with specialized evaluation
        """
        
        request = EvaluationRequest(
            response=response,
            task_type=task_type,
            prompt=prompt,
            jurisdiction=jurisdiction,
            infer_jurisdiction=False,
            evaluation_mode=EvaluationMode.HYBRID
        )
        
        return await self.evaluate_response(request)
    
    def evaluate_response_sync(self, request: EvaluationRequest) -> RouterEvaluationResult:
        """
        Synchronous wrapper for evaluate_response.
        
        Args:
            request: Evaluation request
            
        Returns:
            RouterEvaluationResult
        """
        
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.evaluate_response(request))
        except RuntimeError:
            # Create new event loop if none exists
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.evaluate_response(request))
            finally:
                loop.close()
    
    # VERL integration methods
    
    def verl_evaluate_batch(self, batch_data: List[Dict[str, Any]]) -> List[RouterEvaluationResult]:
        """
        Batch evaluation interface for VERL integration.
        
        Args:
            batch_data: List of evaluation data dictionaries
            
        Returns:
            List of RouterEvaluationResult objects
        """
        
        # Convert batch data to evaluation requests
        requests = []
        for item in batch_data:
            try:
                request = self._convert_verl_data_to_request(item)
                requests.append(request)
            except Exception as e:
                self.logger.error(f"Failed to convert VERL data to request: {e}")
                # Create error request
                error_request = EvaluationRequest(
                    response=item.get("response", ""),
                    task_type=LegalTaskType.GENERAL_CHAT,
                    prompt=item.get("prompt", "")
                )
                requests.append(error_request)
        
        # Evaluate requests in parallel with concurrency control
        return asyncio.run(self._evaluate_batch_parallel(requests))
    
    def _convert_verl_data_to_request(self, verl_data: Dict[str, Any]) -> EvaluationRequest:
        """Convert VERL data format to EvaluationRequest"""
        
        # Extract task type
        task_type_str = verl_data.get("task_type", "general_chat")
        try:
            task_type = LegalTaskType(task_type_str)
        except ValueError:
            task_type = LegalTaskType.GENERAL_CHAT
        
        # Extract jurisdiction
        jurisdiction = None
        jurisdiction_str = verl_data.get("jurisdiction")
        if jurisdiction_str:
            try:
                jurisdiction = USJurisdiction(jurisdiction_str)
            except ValueError:
                jurisdiction = None
        
        return EvaluationRequest(
            response=verl_data.get("response", ""),
            task_type=task_type,
            prompt=verl_data.get("prompt", ""),
            jurisdiction=jurisdiction,
            infer_jurisdiction=jurisdiction is None,
            user_context=verl_data.get("context", {})
        )
    
    async def _evaluate_batch_parallel(self, requests: List[EvaluationRequest]) -> List[RouterEvaluationResult]:
        """Evaluate batch of requests in parallel with concurrency control"""
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.config.max_concurrent_evaluations)
        
        async def evaluate_with_semaphore(request):
            async with semaphore:
                return await self.evaluate_response(request)
        
        # Execute all evaluations in parallel
        tasks = [evaluate_with_semaphore(request) for request in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Batch evaluation failed for request {i}: {result}")
                error_result = self._create_error_result(requests[i], str(result), 0.0)
                final_results.append(error_result)
            else:
                final_results.append(result)
        
        return final_results
    
    # Monitoring and performance methods
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        
        total = self.router_stats["total_evaluations"]
        if total == 0:
            return {"status": "No evaluations performed"}
        
        # Calculate rates and averages
        success_rate = self.router_stats["successful_evaluations"] / total
        cache_hit_rate = self.router_stats["cache_hits"] / total
        avg_cost = self.router_stats["total_cost"] / total
        gating_failure_rate = self.router_stats["gating_failures"] / total
        
        # Component performance
        component_performance = {}
        if hasattr(self, 'hybrid_engine') and self.hybrid_engine:
            component_performance["hybrid_engine"] = self.hybrid_engine.get_performance_summary()
        
        if hasattr(self, 'general_chat_ensemble') and self.general_chat_ensemble:
            component_performance["general_chat_ensemble"] = self.general_chat_ensemble.get_gating_performance_summary()
        
        if hasattr(self, 'api_client') and self.api_client:
            component_performance["api_client"] = self.api_client.get_performance_summary()
        
        return {
            "router_performance": {
                "total_evaluations": total,
                "success_rate": success_rate,
                "cache_hit_rate": cache_hit_rate,
                "avg_evaluation_time_ms": self.router_stats["avg_evaluation_time"],
                "avg_cost_per_evaluation": avg_cost,
                "total_cost": self.router_stats["total_cost"],
                "gating_failure_rate": gating_failure_rate
            },
            "distribution_analysis": {
                "task_types": {task.value: count for task, count in self.router_stats["task_type_distribution"].items()},
                "jurisdictions": self.router_stats["jurisdiction_distribution"],
                "evaluation_modes": {mode.value: count for mode, count in self.router_stats["evaluation_mode_distribution"].items()}
            },
            "component_performance": component_performance,
            "configuration": {
                "router_mode": self.config.router_mode.value,
                "max_concurrent_evaluations": self.config.max_concurrent_evaluations,
                "enable_hybrid_evaluation": self.config.enable_hybrid_evaluation,
                "enable_jurisdiction_inference": self.config.enable_jurisdiction_inference,
                "enable_cost_optimization": self.config.enable_cost_optimization
            }
        }
    
    def get_cost_analysis(self) -> Dict[str, Any]:
        """Get detailed cost analysis"""
        
        total_evals = self.router_stats["total_evaluations"]
        total_cost = self.router_stats["total_cost"]
        
        if total_evals == 0:
            return {"status": "No evaluations for cost analysis"}
        
        # Cost projections
        daily_evals = 1000  # Assume 1000 evaluations per day for projection
        monthly_cost_projection = (total_cost / total_evals) * daily_evals * 30
        
        # Cache savings estimation
        cache_hit_rate = self.router_stats["cache_hits"] / total_evals
        estimated_cost_without_cache = total_cost / (1.0 - cache_hit_rate * 0.9) if cache_hit_rate > 0 else total_cost
        estimated_savings = estimated_cost_without_cache - total_cost
        
        return {
            "cost_summary": {
                "total_cost": total_cost,
                "avg_cost_per_evaluation": total_cost / total_evals,
                "max_cost_per_evaluation": self.config.max_cost_per_evaluation,
                "cost_efficiency": min(1.0, self.config.max_cost_per_evaluation / (total_cost / total_evals)) if total_cost > 0 else 1.0
            },
            "cost_projections": {
                "daily_cost_projection": (total_cost / total_evals) * daily_evals,
                "monthly_cost_projection": monthly_cost_projection,
                "annual_cost_projection": monthly_cost_projection * 12
            },
            "optimization_impact": {
                "cache_hit_rate": cache_hit_rate,
                "estimated_cost_savings": estimated_savings,
                "savings_percentage": (estimated_savings / estimated_cost_without_cache) * 100 if estimated_cost_without_cache > 0 else 0
            },
            "component_costs": {
                "api_client": self.api_client.get_cost_breakdown() if hasattr(self, 'api_client') else "Not available"
            }
        }


# Factory functions for different use cases

def create_production_router(system_config: Optional[LegalRewardSystemConfig] = None) -> MultiTaskLegalRewardRouter:
    """
    Create production-ready multi-task legal reward router.
    
    Args:
        system_config: System configuration (optional)
        
    Returns:
        Configured MultiTaskLegalRewardRouter for production use
    """
    
    router_config = RouterConfig(
        router_mode=RouterMode.PRODUCTION,
        enable_jurisdiction_inference=True,
        enable_hybrid_evaluation=True,
        enable_caching=True,
        enable_cost_optimization=True,
        max_concurrent_evaluations=10,
        evaluation_timeout_seconds=60.0,
        require_jurisdiction_compliance=True,
        fallback_to_general_chat=True,
        max_cost_per_evaluation=0.75,
        aggressive_cost_optimization=True
    )
    
    return MultiTaskLegalRewardRouter(router_config, system_config)


def create_development_router() -> MultiTaskLegalRewardRouter:
    """
    Create development-friendly router with relaxed settings.
    
    Returns:
        Configured MultiTaskLegalRewardRouter for development use
    """
    
    router_config = RouterConfig(
        router_mode=RouterMode.DEVELOPMENT,
        enable_jurisdiction_inference=True,
        enable_hybrid_evaluation=True,
        enable_caching=False,  # Disable caching for development
        enable_cost_optimization=False,
        max_concurrent_evaluations=3,
        evaluation_timeout_seconds=30.0,
        require_jurisdiction_compliance=False,
        fallback_to_general_chat=True,
        max_cost_per_evaluation=0.25,
        aggressive_cost_optimization=False
    )
    
    return MultiTaskLegalRewardRouter(router_config)


def create_cost_optimized_router(system_config: Optional[LegalRewardSystemConfig] = None) -> MultiTaskLegalRewardRouter:
    """
    Create cost-optimized router for minimal API usage.
    
    Args:
        system_config: System configuration (optional)
        
    Returns:
        Configured MultiTaskLegalRewardRouter for cost-conscious use
    """
    
    router_config = RouterConfig(
        router_mode=RouterMode.COST_OPTIMIZED,
        enable_jurisdiction_inference=True,
        enable_hybrid_evaluation=True,
        enable_caching=True,
        enable_cost_optimization=True,
        max_concurrent_evaluations=5,
        evaluation_timeout_seconds=45.0,
        require_jurisdiction_compliance=True,
        fallback_to_general_chat=True,
        max_cost_per_evaluation=0.25,  # Lower cost limit
        aggressive_cost_optimization=True,
        prefer_cached_results=True
    )
    
    return MultiTaskLegalRewardRouter(router_config, system_config)


def create_high_accuracy_router(system_config: Optional[LegalRewardSystemConfig] = None) -> MultiTaskLegalRewardRouter:
    """
    Create high-accuracy router prioritizing quality over cost.
    
    Args:
        system_config: System configuration (optional)
        
    Returns:
        Configured MultiTaskLegalRewardRouter for high-accuracy use
    """
    
    router_config = RouterConfig(
        router_mode=RouterMode.HIGH_ACCURACY,
        enable_jurisdiction_inference=True,
        enable_hybrid_evaluation=True,
        enable_caching=True,
        enable_cost_optimization=False,  # Prioritize accuracy over cost
        max_concurrent_evaluations=15,
        evaluation_timeout_seconds=90.0,
        require_jurisdiction_compliance=True,
        fallback_to_general_chat=False,  # Don't fallback - require full evaluation
        max_cost_per_evaluation=2.00,   # Higher cost limit for accuracy
        aggressive_cost_optimization=False,
        min_confidence_threshold=0.7    # Higher confidence requirement
    )
    
    return MultiTaskLegalRewardRouter(router_config, system_config)

def create_production_router_config() -> RouterConfig:
    """Create production router configuration"""
    return RouterConfig(
        router_mode=RouterMode.PRODUCTION,
        enable_jurisdiction_inference=True,
        enable_hybrid_evaluation=True,
        enable_caching=True,
        enable_cost_optimization=True,
        max_concurrent_evaluations=10,
        evaluation_timeout_seconds=60.0,
        require_jurisdiction_compliance=True,
        fallback_to_general_chat=True,
        max_cost_per_evaluation=0.50,
        aggressive_cost_optimization=True,
        prefer_cached_results=True
    )


def create_development_router_config() -> RouterConfig:
    """Create development router configuration"""
    return RouterConfig(
        router_mode=RouterMode.DEVELOPMENT,
        enable_jurisdiction_inference=True,
        enable_hybrid_evaluation=True,
        enable_caching=False,
        enable_cost_optimization=False,
        max_concurrent_evaluations=3,
        evaluation_timeout_seconds=30.0,
        require_jurisdiction_compliance=False,
        fallback_to_general_chat=True,
        max_cost_per_evaluation=0.25,
        aggressive_cost_optimization=False,
        prefer_cached_results=False
    )


def create_cost_optimized_router_config() -> RouterConfig:
    """Create cost-optimized router configuration"""
    return RouterConfig(
        router_mode=RouterMode.COST_OPTIMIZED,
        enable_jurisdiction_inference=True,
        enable_hybrid_evaluation=True,
        enable_caching=True,
        enable_cost_optimization=True,
        max_concurrent_evaluations=5,
        evaluation_timeout_seconds=45.0,
        require_jurisdiction_compliance=True,
        fallback_to_general_chat=True,
        max_cost_per_evaluation=0.25,
        aggressive_cost_optimization=True,
        prefer_cached_results=True
    )


def create_high_accuracy_router_config() -> RouterConfig:
    """Create high-accuracy router configuration"""
    return RouterConfig(
        router_mode=RouterMode.HIGH_ACCURACY,
        enable_jurisdiction_inference=True,
        enable_hybrid_evaluation=True,
        enable_caching=True,
        enable_cost_optimization=False,
        max_concurrent_evaluations=15,
        evaluation_timeout_seconds=120.0,
        require_jurisdiction_compliance=True,
        fallback_to_general_chat=True,
        max_cost_per_evaluation=1.00,  # Higher cost for accuracy
        aggressive_cost_optimization=False,
        prefer_cached_results=False
    )


# Convenience functions for integration

def evaluate_legal_response(response: str,
                          task_type: LegalTaskType,
                          jurisdiction: Optional[USJurisdiction] = None,
                          prompt: str = "") -> RouterEvaluationResult:
    """
    Convenience function for evaluating legal responses.
    
    Args:
        response: Legal response text to evaluate
        task_type: Type of legal task
        jurisdiction: Jurisdiction context (optional)
        prompt: Original prompt/question
        
    Returns:
        RouterEvaluationResult with comprehensive evaluation
    """
    
    router = create_production_router()
    
    request = EvaluationRequest(
        response=response,
        task_type=task_type,
        prompt=prompt,
        jurisdiction=jurisdiction,
        infer_jurisdiction=jurisdiction is None
    )
    
    return router.evaluate_response_sync(request)