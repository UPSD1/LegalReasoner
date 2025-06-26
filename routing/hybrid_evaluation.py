"""
Hybrid Evaluation System for Multi-Task Legal Reward System

This module implements the core innovation of the legal reward system: a hybrid
evaluation approach that combines specialized legal evaluation (70% weight) with
general chat quality assessment (30% weight), with jurisdiction compliance gating.

Key Features:
- Hybrid scoring: 70% specialized + 30% general chat
- Jurisdiction compliance gating (blocks evaluation if non-compliant)
- Dynamic task routing between specialized and general evaluations
- Task difficulty weighting for fair comparison across legal tasks
- Comprehensive error handling and fallback mechanisms
- Performance optimization for training efficiency

The hybrid system ensures that legal responses are evaluated both for their
specialized legal accuracy and their general chat quality, providing a more
comprehensive and robust evaluation framework.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

# Import core components
from ..core import (
    LegalRewardEvaluation, JudgeEvaluation, EvaluationMetadata,
    LegalTaskType, USJurisdiction, LegalDomain,
    LegalRewardSystemError, create_error_context
)
from ..judges.general_chat import (
    EnhancedGeneralChatEnsemble, GeneralChatEvaluationResult,
    create_production_general_chat_ensemble
)
from ..judges.base import EvaluationContext, create_evaluation_context


class EvaluationMode(Enum):
    """Evaluation modes for the hybrid system"""
    GENERAL_CHAT_ONLY = "general_chat_only"       # Only general chat evaluation
    SPECIALIZED_ONLY = "specialized_only"         # Only specialized evaluation  
    HYBRID = "hybrid"                             # Both evaluations combined
    AUTO = "auto"                                 # Automatic mode selection


class SpecializedEvaluationStatus(Enum):
    """Status of specialized evaluation"""
    NOT_ATTEMPTED = "not_attempted"
    SUCCESS = "success"
    FAILED = "failed"
    NOT_AVAILABLE = "not_available"
    GATED_OUT = "gated_out"


@dataclass
class HybridEvaluationConfig:
    """
    Configuration for hybrid evaluation system.
    
    Contains weights, thresholds, and behavior settings for the
    hybrid evaluation approach.
    """
    
    # Core hybrid weights
    specialized_weight: float = 0.7  # 70% weight for specialized evaluation
    general_chat_weight: float = 0.3  # 30% weight for general chat
    
    # Gating configuration
    enable_gating: bool = True
    jurisdiction_failure_penalty: float = 0.2  # Penalty multiplier for jurisdiction failures
    require_jurisdiction_compliance: bool = True
    
    # Evaluation behavior
    evaluation_mode: EvaluationMode = EvaluationMode.AUTO
    fallback_to_general_chat: bool = True  # Fallback if specialized fails
    minimum_ensemble_confidence: float = 0.5
    
    # Task routing
    force_specialized_for_complex: bool = True
    general_chat_threshold: float = 6.0  # Minimum general chat score for specialized routing
    
    # Performance optimization
    enable_parallel_evaluation: bool = True
    max_evaluation_time_seconds: float = 45.0
    cache_hybrid_results: bool = True
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return any issues"""
        
        issues = []
        
        # Check weights sum to 1.0
        total_weight = self.specialized_weight + self.general_chat_weight
        if abs(total_weight - 1.0) > 0.01:
            issues.append(f"Weights must sum to 1.0, got {total_weight}")
        
        # Check weight ranges
        if not (0.0 <= self.specialized_weight <= 1.0):
            issues.append("Specialized weight must be between 0.0 and 1.0")
        
        if not (0.0 <= self.general_chat_weight <= 1.0):
            issues.append("General chat weight must be between 0.0 and 1.0")
        
        # Check penalty range
        if not (0.0 <= self.jurisdiction_failure_penalty <= 1.0):
            issues.append("Jurisdiction failure penalty must be between 0.0 and 1.0")
        
        # Check confidence threshold
        if not (0.0 <= self.minimum_ensemble_confidence <= 1.0):
            issues.append("Minimum ensemble confidence must be between 0.0 and 1.0")
        
        return issues


@dataclass
class HybridEvaluationResult:
    """
    Result of hybrid evaluation combining specialized and general chat scores.
    
    Contains comprehensive evaluation results with detailed breakdown
    of specialized and general chat components.
    """
    
    # Final hybrid results
    hybrid_score: float  # 0.0 to 10.0
    confidence: float    # 0.0 to 1.0
    is_successful: bool = True
    
    # Component results
    specialized_score: Optional[float] = None
    general_chat_score: Optional[float] = None
    specialized_confidence: Optional[float] = None
    general_chat_confidence: Optional[float] = None
    
    # Evaluation details
    specialized_status: SpecializedEvaluationStatus = SpecializedEvaluationStatus.NOT_ATTEMPTED
    general_chat_gating_passed: bool = True
    jurisdiction_compliance_score: float = 10.0
    
    # Weight application
    applied_specialized_weight: float = 0.7
    applied_general_chat_weight: float = 0.3
    
    # Component evaluations
    specialized_evaluation: Optional[LegalRewardEvaluation] = None
    general_chat_evaluation: Optional[GeneralChatEvaluationResult] = None
    
    # Metadata
    evaluation_mode: EvaluationMode = EvaluationMode.HYBRID
    task_difficulty_weight: float = 1.0
    processing_time_ms: float = 0.0
    
    # Quality assessment
    reasoning: str = ""
    recommendations: List[str] = field(default_factory=list)
    
    def get_component_breakdown(self) -> Dict[str, Any]:
        """Get detailed breakdown of evaluation components"""
        
        return {
            "hybrid_score": self.hybrid_score,
            "components": {
                "specialized": {
                    "score": self.specialized_score,
                    "confidence": self.specialized_confidence,
                    "weight": self.applied_specialized_weight,
                    "status": self.specialized_status.value,
                    "contribution": (self.specialized_score or 0.0) * self.applied_specialized_weight
                },
                "general_chat": {
                    "score": self.general_chat_score,
                    "confidence": self.general_chat_confidence,
                    "weight": self.applied_general_chat_weight,
                    "gating_passed": self.general_chat_gating_passed,
                    "contribution": (self.general_chat_score or 0.0) * self.applied_general_chat_weight
                }
            },
            "gating": {
                "jurisdiction_compliance_score": self.jurisdiction_compliance_score,
                "gating_passed": self.general_chat_gating_passed,
                "require_compliance": True
            },
            "metadata": {
                "evaluation_mode": self.evaluation_mode.value,
                "task_difficulty_weight": self.task_difficulty_weight,
                "processing_time_ms": self.processing_time_ms
            }
        }
    
    def has_strong_performance(self, threshold: float = 7.0) -> bool:
        """Check if evaluation shows strong performance"""
        return (self.hybrid_score >= threshold and 
                self.is_successful and 
                self.general_chat_gating_passed)
    
    def get_weakest_component(self) -> Tuple[str, float]:
        """Get the weakest performing component"""
        components = {}
        
        if self.specialized_score is not None:
            components["specialized"] = self.specialized_score
        
        if self.general_chat_score is not None:
            components["general_chat"] = self.general_chat_score
        
        if not components:
            return "none", 0.0
        
        return min(components.items(), key=lambda x: x[1])


class SpecializedEvaluationPlaceholder:
    """
    Placeholder for specialized evaluation ensembles.
    
    This will be replaced with actual specialized ensembles in later steps.
    For now, it provides a consistent interface for the hybrid system.
    """
    
    def __init__(self, task_type: LegalTaskType):
        self.task_type = task_type
        self.logger = logging.getLogger(f"specialized_placeholder.{task_type.value}")
    
    async def evaluate_specialized(self, 
                                 response: str, 
                                 context: EvaluationContext) -> Optional[LegalRewardEvaluation]:
        """
        Placeholder for specialized evaluation.
        
        Returns None to indicate specialized evaluation is not yet available.
        This will be replaced with actual specialized ensemble calls.
        """
        
        self.logger.info(f"Specialized evaluation for {self.task_type.value} not yet implemented")
        return None
    
    def is_available(self) -> bool:
        """Check if specialized evaluation is available"""
        return False


class HybridEvaluationEngine:
    """
    Main hybrid evaluation engine that orchestrates specialized and general chat evaluations.
    
    Implements the core 70/30 hybrid approach with jurisdiction compliance gating
    and intelligent task routing for optimal evaluation quality.
    """
    
    def __init__(self, 
                 config: HybridEvaluationConfig,
                 general_chat_ensemble: Optional[EnhancedGeneralChatEnsemble] = None,
                 specialized_ensembles: Optional[Dict[LegalTaskType, Any]] = None):
        
        self.config = config
        self.logger = logging.getLogger("hybrid_evaluation_engine")
        
        # Validate configuration
        config_issues = self.config.validate_config()
        if config_issues:
            raise LegalRewardSystemError(
                f"Invalid hybrid evaluation configuration: {'; '.join(config_issues)}",
                error_context=create_error_context("hybrid_evaluation", "init")
            )
        
        # Initialize ensembles
        self.general_chat_ensemble = general_chat_ensemble or create_production_general_chat_ensemble()
        self.specialized_ensembles = specialized_ensembles or {}
        
        # Create placeholders for specialized ensembles not yet available
        for task_type in LegalTaskType:
            if task_type not in self.specialized_ensembles:
                self.specialized_ensembles[task_type] = SpecializedEvaluationPlaceholder(task_type)
        
        # Performance tracking
        self.evaluation_stats = {
            "total_hybrid_evaluations": 0,
            "general_chat_only": 0,
            "specialized_only": 0,
            "full_hybrid": 0,
            "gating_failures": 0,
            "specialized_failures": 0,
            "avg_hybrid_score": 0.0,
            "avg_processing_time": 0.0
        }
    
    async def evaluate_hybrid(self, 
                            response: str,
                            task_type: LegalTaskType,
                            jurisdiction: USJurisdiction,
                            prompt: str = "",
                            legal_domains: Optional[List[LegalDomain]] = None,
                            evaluation_mode: Optional[EvaluationMode] = None) -> HybridEvaluationResult:
        """
        Perform hybrid evaluation combining specialized and general chat assessment.
        
        This is the main entry point for the hybrid evaluation system that implements
        the core 70/30 specialized/general chat weighting with jurisdiction gating.
        
        Args:
            response: Legal response text to evaluate
            task_type: Type of legal task being evaluated
            jurisdiction: Jurisdiction context for the evaluation
            prompt: Original prompt/question (optional)
            legal_domains: Detected legal domains (optional)
            evaluation_mode: Override evaluation mode (optional)
            
        Returns:
            HybridEvaluationResult with comprehensive evaluation details
        """
        
        start_time = time.time()
        self.evaluation_stats["total_hybrid_evaluations"] += 1
        
        try:
            # Create evaluation context
            context = create_evaluation_context(task_type, jurisdiction, prompt, legal_domains)
            
            # Determine evaluation mode
            mode = evaluation_mode or self._determine_evaluation_mode(task_type, context)
            
            # Execute evaluation based on mode
            if mode == EvaluationMode.GENERAL_CHAT_ONLY:
                result = await self._evaluate_general_chat_only(response, context)
                self.evaluation_stats["general_chat_only"] += 1
                
            elif mode == EvaluationMode.SPECIALIZED_ONLY:
                result = await self._evaluate_specialized_only(response, context)
                self.evaluation_stats["specialized_only"] += 1
                
            else:  # HYBRID or AUTO
                result = await self._evaluate_full_hybrid(response, context)
                self.evaluation_stats["full_hybrid"] += 1
            
            # Apply task difficulty weighting
            result = self._apply_task_difficulty_weighting(result, task_type)
            
            # Set metadata
            result.evaluation_mode = mode
            result.processing_time_ms = (time.time() - start_time) * 1000
            
            # Update statistics
            self._update_evaluation_stats(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Hybrid evaluation failed: {e}")
            return self._create_error_result(str(e), task_type, jurisdiction)
    
    def _determine_evaluation_mode(self, 
                                 task_type: LegalTaskType, 
                                 context: EvaluationContext) -> EvaluationMode:
        """Determine appropriate evaluation mode based on task and context"""
        
        # Use configured mode if not AUTO
        if self.config.evaluation_mode != EvaluationMode.AUTO:
            return self.config.evaluation_mode
        
        # For general chat tasks, use general chat only
        if task_type == LegalTaskType.GENERAL_CHAT:
            return EvaluationMode.GENERAL_CHAT_ONLY
        
        # For specialized tasks, check if specialized evaluation is available
        specialized_ensemble = self.specialized_ensembles.get(task_type)
        if specialized_ensemble and hasattr(specialized_ensemble, 'is_available') and specialized_ensemble.is_available():
            return EvaluationMode.HYBRID
        else:
            # Fallback to general chat if specialized not available
            return EvaluationMode.GENERAL_CHAT_ONLY
    
    async def _evaluate_general_chat_only(self, 
                                        response: str, 
                                        context: EvaluationContext) -> HybridEvaluationResult:
        """Evaluate using general chat ensemble only"""
        
        try:
            # Evaluate with general chat ensemble (includes gating)
            general_result = await self.general_chat_ensemble.evaluate_with_gating(response, context)
            
            # Create hybrid result from general chat evaluation
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
                reasoning=f"General chat only evaluation: {general_result.reasoning}",
                recommendations=general_result.recommendations
            )
            
        except Exception as e:
            self.logger.error(f"General chat evaluation failed: {e}")
            return self._create_error_result(str(e), context.task_type, context.jurisdiction)
    
    async def _evaluate_specialized_only(self, 
                                       response: str, 
                                       context: EvaluationContext) -> HybridEvaluationResult:
        """Evaluate using specialized ensemble only"""
        
        try:
            # Get specialized ensemble
            specialized_ensemble = self.specialized_ensembles.get(context.task_type)
            
            if not specialized_ensemble or not hasattr(specialized_ensemble, 'is_available') or not specialized_ensemble.is_available():
                # Fallback to general chat if specialized not available
                self.logger.warning(f"Specialized evaluation not available for {context.task_type.value}, falling back to general chat")
                return await self._evaluate_general_chat_only(response, context)
            
            # Evaluate with specialized ensemble
            specialized_result = await specialized_ensemble.evaluate_specialized(response, context)
            
            if specialized_result:
                return HybridEvaluationResult(
                    hybrid_score=specialized_result.overall_score,
                    confidence=specialized_result.confidence,
                    is_successful=True,
                    specialized_score=specialized_result.overall_score,
                    specialized_confidence=specialized_result.confidence,
                    specialized_status=SpecializedEvaluationStatus.SUCCESS,
                    general_chat_gating_passed=True,  # Assume passed for specialized-only
                    applied_specialized_weight=1.0,
                    applied_general_chat_weight=0.0,
                    specialized_evaluation=specialized_result,
                    reasoning=f"Specialized only evaluation: {specialized_result.reasoning}"
                )
            else:
                # Specialized evaluation failed, fallback to general chat
                return await self._evaluate_general_chat_only(response, context)
            
        except Exception as e:
            self.logger.error(f"Specialized evaluation failed: {e}")
            
            # Fallback to general chat on error
            if self.config.fallback_to_general_chat:
                return await self._evaluate_general_chat_only(response, context)
            else:
                return self._create_error_result(str(e), context.task_type, context.jurisdiction)
    
    async def _evaluate_full_hybrid(self, 
                                  response: str, 
                                  context: EvaluationContext) -> HybridEvaluationResult:
        """Perform full hybrid evaluation with both specialized and general chat"""
        
        try:
            # Run evaluations in parallel if enabled
            if self.config.enable_parallel_evaluation:
                general_task = asyncio.create_task(
                    self.general_chat_ensemble.evaluate_with_gating(response, context)
                )
                specialized_task = asyncio.create_task(
                    self._get_specialized_evaluation(response, context)
                )
                
                # Wait for both with timeout
                try:
                    general_result, specialized_result = await asyncio.wait_for(
                        asyncio.gather(general_task, specialized_task, return_exceptions=True),
                        timeout=self.config.max_evaluation_time_seconds
                    )
                except asyncio.TimeoutError:
                    self.logger.warning("Hybrid evaluation timed out, using available results")
                    general_result = general_task.result() if general_task.done() else None
                    specialized_result = specialized_task.result() if specialized_task.done() else None
            
            else:
                # Sequential evaluation
                general_result = await self.general_chat_ensemble.evaluate_with_gating(response, context)
                specialized_result = await self._get_specialized_evaluation(response, context)
            
            # Handle exceptions from parallel execution
            if isinstance(general_result, Exception):
                self.logger.error(f"General chat evaluation failed: {general_result}")
                general_result = None
            
            if isinstance(specialized_result, Exception):
                self.logger.error(f"Specialized evaluation failed: {specialized_result}")
                specialized_result = None
            
            # Check gating decision
            if general_result and not general_result.is_gated:
                self.evaluation_stats["gating_failures"] += 1
                
                # Return gated result if gating is required
                if self.config.require_jurisdiction_compliance:
                    return self._create_gated_result(general_result, context)
            
            # Combine results
            return self._combine_evaluation_results(general_result, specialized_result, context)
            
        except Exception as e:
            self.logger.error(f"Full hybrid evaluation failed: {e}")
            
            # Fallback to general chat only
            if self.config.fallback_to_general_chat:
                return await self._evaluate_general_chat_only(response, context)
            else:
                return self._create_error_result(str(e), context.task_type, context.jurisdiction)
    
    async def _get_specialized_evaluation(self, 
                                        response: str, 
                                        context: EvaluationContext) -> Optional[LegalRewardEvaluation]:
        """Get specialized evaluation for the given task type"""
        
        try:
            specialized_ensemble = self.specialized_ensembles.get(context.task_type)
            
            if not specialized_ensemble:
                return None
            
            # Check if specialized evaluation is available
            if hasattr(specialized_ensemble, 'is_available') and not specialized_ensemble.is_available():
                return None
            
            # Perform specialized evaluation
            return await specialized_ensemble.evaluate_specialized(response, context)
            
        except Exception as e:
            self.logger.warning(f"Specialized evaluation failed for {context.task_type.value}: {e}")
            return None
    
    def _combine_evaluation_results(self, 
                                  general_result: Optional[GeneralChatEvaluationResult],
                                  specialized_result: Optional[LegalRewardEvaluation],
                                  context: EvaluationContext) -> HybridEvaluationResult:
        """Combine general chat and specialized evaluation results using hybrid weights"""
        
        # Determine specialized status
        if specialized_result:
            specialized_status = SpecializedEvaluationStatus.SUCCESS
            self.logger.info(f"Specialized evaluation succeeded for {context.task_type.value}")
        else:
            specialized_status = SpecializedEvaluationStatus.NOT_AVAILABLE
            self.evaluation_stats["specialized_failures"] += 1
        
        # Extract component scores
        general_chat_score = general_result.overall_score if general_result else 5.0
        general_chat_confidence = general_result.confidence if general_result else 0.5
        gating_passed = general_result.is_gated if general_result else True
        jurisdiction_score = general_result.jurisdiction_compliance_score if general_result else 10.0
        
        specialized_score = specialized_result.overall_score if specialized_result else None
        specialized_confidence = specialized_result.confidence if specialized_result else None
        
        # Calculate weights based on availability
        if specialized_result:
            # Full hybrid: use configured weights
            specialized_weight = self.config.specialized_weight
            general_weight = self.config.general_chat_weight
        else:
            # No specialized: use general chat only
            specialized_weight = 0.0
            general_weight = 1.0
        
        # Calculate hybrid score
        hybrid_score = (
            (specialized_score or 0.0) * specialized_weight +
            general_chat_score * general_weight
        )
        
        # Calculate combined confidence
        confidences = [general_chat_confidence]
        if specialized_confidence:
            confidences.append(specialized_confidence)
        
        combined_confidence = sum(confidences) / len(confidences)
        
        # Apply jurisdiction compliance penalty if needed
        if not gating_passed and self.config.require_jurisdiction_compliance:
            penalty_multiplier = 1.0 - self.config.jurisdiction_failure_penalty
            hybrid_score *= penalty_multiplier
            combined_confidence *= 0.7  # Reduce confidence for compliance failures
        
        # Create comprehensive reasoning
        reasoning = self._create_hybrid_reasoning(
            general_result, specialized_result, specialized_weight, general_weight, gating_passed
        )
        
        # Generate recommendations
        recommendations = self._generate_hybrid_recommendations(general_result, specialized_result)
        
        return HybridEvaluationResult(
            hybrid_score=hybrid_score,
            confidence=combined_confidence,
            is_successful=True,
            specialized_score=specialized_score,
            general_chat_score=general_chat_score,
            specialized_confidence=specialized_confidence,
            general_chat_confidence=general_chat_confidence,
            specialized_status=specialized_status,
            general_chat_gating_passed=gating_passed,
            jurisdiction_compliance_score=jurisdiction_score,
            applied_specialized_weight=specialized_weight,
            applied_general_chat_weight=general_weight,
            specialized_evaluation=specialized_result,
            general_chat_evaluation=general_result,
            reasoning=reasoning,
            recommendations=recommendations
        )
    
    def _create_hybrid_reasoning(self, 
                               general_result: Optional[GeneralChatEvaluationResult],
                               specialized_result: Optional[LegalRewardEvaluation],
                               specialized_weight: float,
                               general_weight: float,
                               gating_passed: bool) -> str:
        """Create comprehensive reasoning for hybrid evaluation"""
        
        reasoning_parts = []
        
        # Add hybrid weighting info
        if specialized_result:
            reasoning_parts.append(
                f"Hybrid Evaluation: {specialized_weight:.1%} specialized + {general_weight:.1%} general chat"
            )
        else:
            reasoning_parts.append("General Chat Only: Specialized evaluation not available")
        
        # Add gating status
        gating_status = "PASS" if gating_passed else "FAIL"
        reasoning_parts.append(f"Jurisdiction Gating: {gating_status}")
        
        # Add component summaries
        if specialized_result:
            reasoning_parts.append(
                f"Specialized: {specialized_result.overall_score:.1f}/10.0 (confidence: {specialized_result.confidence:.2f})"
            )
        
        if general_result:
            reasoning_parts.append(
                f"General Chat: {general_result.overall_score:.1f}/10.0 (compliance: {general_result.jurisdiction_compliance_score:.1f})"
            )
        
        # Add brief quality assessment
        if specialized_result and specialized_result.reasoning:
            specialist_insight = specialized_result.reasoning[:60] + "..." if len(specialized_result.reasoning) > 60 else specialized_result.reasoning
            reasoning_parts.append(f"Specialized insight: {specialist_insight}")
        
        return " | ".join(reasoning_parts)
    
    def _generate_hybrid_recommendations(self, 
                                       general_result: Optional[GeneralChatEvaluationResult],
                                       specialized_result: Optional[LegalRewardEvaluation]) -> List[str]:
        """Generate actionable recommendations from hybrid evaluation"""
        
        recommendations = []
        
        # Add general chat recommendations
        if general_result and general_result.recommendations:
            recommendations.extend(general_result.recommendations[:2])  # Top 2
        
        # Add specialized recommendations if available
        if specialized_result and hasattr(specialized_result, 'recommendations'):
            specialized_recs = getattr(specialized_result, 'recommendations', [])
            recommendations.extend(specialized_recs[:2])  # Top 2
        
        # Add hybrid-specific recommendations
        if specialized_result and general_result:
            # Compare component scores for insights
            specialist_score = specialized_result.overall_score
            general_score = general_result.overall_score
            
            if abs(specialist_score - general_score) > 2.0:
                if specialist_score > general_score:
                    recommendations.append("Strong specialized knowledge but consider improving general helpfulness")
                else:
                    recommendations.append("Good general chat quality but specialized legal accuracy needs improvement")
        
        # Add gating recommendations
        if general_result and not general_result.is_gated:
            recommendations.append("Address jurisdiction compliance issues before proceeding")
        
        return recommendations[:4]  # Limit to top 4
    
    def _create_gated_result(self, 
                           general_result: GeneralChatEvaluationResult,
                           context: EvaluationContext) -> HybridEvaluationResult:
        """Create result for gated (failed jurisdiction compliance) evaluation"""
        
        # Apply jurisdiction failure penalty
        penalty_multiplier = 1.0 - self.config.jurisdiction_failure_penalty
        penalized_score = general_result.overall_score * penalty_multiplier
        
        return HybridEvaluationResult(
            hybrid_score=penalized_score,
            confidence=0.3,  # Low confidence for gated results
            is_successful=False,
            general_chat_score=general_result.overall_score,
            general_chat_confidence=general_result.confidence,
            specialized_status=SpecializedEvaluationStatus.GATED_OUT,
            general_chat_gating_passed=False,
            jurisdiction_compliance_score=general_result.jurisdiction_compliance_score,
            applied_specialized_weight=0.0,
            applied_general_chat_weight=1.0,
            general_chat_evaluation=general_result,
            reasoning=f"GATED: Jurisdiction compliance failed (score: {general_result.jurisdiction_compliance_score:.1f})",
            recommendations=["Address jurisdiction compliance violations before re-evaluation"] + general_result.recommendations[:2]
        )
    
    def _apply_task_difficulty_weighting(self, 
                                       result: HybridEvaluationResult,
                                       task_type: LegalTaskType) -> HybridEvaluationResult:
        """Apply task difficulty weighting to normalize scores across task types"""
        
        # Get task difficulty weight (this will be implemented in task_weights.py)
        task_weight = self._get_task_difficulty_weight(task_type)
        
        # Apply weight to hybrid score
        result.hybrid_score *= task_weight
        result.task_difficulty_weight = task_weight
        
        # Ensure score stays within valid range
        result.hybrid_score = max(0.0, min(10.0, result.hybrid_score))
        
        return result
    
    def _get_task_difficulty_weight(self, task_type: LegalTaskType) -> float:
        """Get task difficulty weight (placeholder - will be implemented in task_weights.py)"""
        
        # Placeholder weights based on task complexity
        weights = {
            LegalTaskType.JUDICIAL_REASONING: 1.5,      # Hardest
            LegalTaskType.PRECEDENT_ANALYSIS: 1.3,     # Hard
            LegalTaskType.OPINION_GENERATION: 1.1,     # Medium-hard
            LegalTaskType.GENERAL_CHAT: 1.0            # Baseline
        }
        
        return weights.get(task_type, 1.0)
    
    def _create_error_result(self, 
                           error_msg: str,
                           task_type: LegalTaskType,
                           jurisdiction: USJurisdiction) -> HybridEvaluationResult:
        """Create error fallback result"""
        
        return HybridEvaluationResult(
            hybrid_score=3.0,  # Below average
            confidence=0.1,
            is_successful=False,
            specialized_status=SpecializedEvaluationStatus.FAILED,
            general_chat_gating_passed=False,
            reasoning=f"Hybrid evaluation error: {error_msg[:100]}",
            recommendations=["Manual review required due to evaluation error"]
        )
    
    def _update_evaluation_stats(self, result: HybridEvaluationResult):
        """Update evaluation performance statistics"""
        
        # Update average hybrid score
        total = self.evaluation_stats["total_hybrid_evaluations"]
        current_avg = self.evaluation_stats["avg_hybrid_score"]
        self.evaluation_stats["avg_hybrid_score"] = (
            (current_avg * (total - 1) + result.hybrid_score) / total if total > 0 else result.hybrid_score
        )
        
        # Update average processing time
        current_time_avg = self.evaluation_stats["avg_processing_time"]
        self.evaluation_stats["avg_processing_time"] = (
            (current_time_avg * (total - 1) + result.processing_time_ms) / total if total > 0 else result.processing_time_ms
        )
    
    # Configuration and monitoring methods
    
    def update_config(self, new_config: HybridEvaluationConfig):
        """Update hybrid evaluation configuration"""
        
        # Validate new configuration
        config_issues = new_config.validate_config()
        if config_issues:
            raise LegalRewardSystemError(
                f"Invalid hybrid evaluation configuration: {'; '.join(config_issues)}"
            )
        
        self.config = new_config
        self.logger.info("Updated hybrid evaluation configuration")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        
        total = self.evaluation_stats["total_hybrid_evaluations"]
        if total == 0:
            return {"status": "No hybrid evaluations performed"}
        
        return {
            "hybrid_evaluation_performance": {
                "total_evaluations": total,
                "avg_hybrid_score": self.evaluation_stats["avg_hybrid_score"],
                "avg_processing_time_ms": self.evaluation_stats["avg_processing_time"],
                "evaluation_distribution": {
                    "general_chat_only": self.evaluation_stats["general_chat_only"] / total,
                    "specialized_only": self.evaluation_stats["specialized_only"] / total,
                    "full_hybrid": self.evaluation_stats["full_hybrid"] / total
                },
                "quality_metrics": {
                    "gating_failure_rate": self.evaluation_stats["gating_failures"] / total,
                    "specialized_failure_rate": self.evaluation_stats["specialized_failures"] / total
                }
            },
            "configuration": {
                "specialized_weight": self.config.specialized_weight,
                "general_chat_weight": self.config.general_chat_weight,
                "jurisdiction_failure_penalty": self.config.jurisdiction_failure_penalty,
                "evaluation_mode": self.config.evaluation_mode.value,
                "enable_gating": self.config.enable_gating,
                "parallel_evaluation": self.config.enable_parallel_evaluation
            },
            "ensemble_performance": {
                "general_chat": self.general_chat_ensemble.get_gating_performance_summary() if self.general_chat_ensemble else "Not available",
                "specialized_ensembles": {
                    task_type.value: "Placeholder - not yet implemented"
                    for task_type in self.specialized_ensembles.keys()
                }
            }
        }
    
    def get_evaluation_breakdown(self, result: HybridEvaluationResult) -> Dict[str, Any]:
        """Get detailed breakdown of a specific evaluation result"""
        
        return {
            "summary": {
                "hybrid_score": result.hybrid_score,
                "confidence": result.confidence,
                "success": result.is_successful,
                "gating_passed": result.general_chat_gating_passed
            },
            "component_breakdown": result.get_component_breakdown(),
            "evaluation_path": {
                "mode": result.evaluation_mode.value,
                "specialized_status": result.specialized_status.value,
                "gating_decision": "PASS" if result.general_chat_gating_passed else "FAIL"
            },
            "quality_assessment": {
                "reasoning": result.reasoning,
                "recommendations": result.recommendations,
                "weakest_component": result.get_weakest_component(),
                "strong_performance": result.has_strong_performance()
            },
            "metadata": {
                "processing_time_ms": result.processing_time_ms,
                "task_difficulty_weight": result.task_difficulty_weight,
                "applied_weights": {
                    "specialized": result.applied_specialized_weight,
                    "general_chat": result.applied_general_chat_weight
                }
            }
        }


# Factory functions for different use cases

def create_production_hybrid_engine(general_chat_ensemble: Optional[EnhancedGeneralChatEnsemble] = None,
                                   specialized_ensembles: Optional[Dict[LegalTaskType, Any]] = None) -> HybridEvaluationEngine:
    """
    Create production-ready hybrid evaluation engine.
    
    Args:
        general_chat_ensemble: Enhanced general chat ensemble
        specialized_ensembles: Dictionary of specialized ensembles by task type
        
    Returns:
        Configured HybridEvaluationEngine for production use
    """
    
    config = HybridEvaluationConfig(
        specialized_weight=0.7,
        general_chat_weight=0.3,
        enable_gating=True,
        jurisdiction_failure_penalty=0.2,
        require_jurisdiction_compliance=True,
        evaluation_mode=EvaluationMode.AUTO,
        fallback_to_general_chat=True,
        enable_parallel_evaluation=True,
        max_evaluation_time_seconds=45.0
    )
    
    return HybridEvaluationEngine(config, general_chat_ensemble, specialized_ensembles)


def create_development_hybrid_engine(general_chat_ensemble: Optional[EnhancedGeneralChatEnsemble] = None) -> HybridEvaluationEngine:
    """
    Create development-friendly hybrid evaluation engine.
    
    Args:
        general_chat_ensemble: Enhanced general chat ensemble
        
    Returns:
        Configured HybridEvaluationEngine for development use
    """
    
    config = HybridEvaluationConfig(
        specialized_weight=0.6,  # Slightly lower for development
        general_chat_weight=0.4,
        enable_gating=True,
        jurisdiction_failure_penalty=0.1,  # Lower penalty for development
        require_jurisdiction_compliance=False,  # More forgiving
        evaluation_mode=EvaluationMode.AUTO,
        fallback_to_general_chat=True,
        enable_parallel_evaluation=False,  # Simpler for debugging
        max_evaluation_time_seconds=30.0
    )
    
    return HybridEvaluationEngine(config, general_chat_ensemble)


def create_strict_hybrid_engine(general_chat_ensemble: Optional[EnhancedGeneralChatEnsemble] = None,
                               specialized_ensembles: Optional[Dict[LegalTaskType, Any]] = None) -> HybridEvaluationEngine:
    """
    Create strict hybrid evaluation engine with high compliance standards.
    
    Args:
        general_chat_ensemble: Enhanced general chat ensemble
        specialized_ensembles: Dictionary of specialized ensembles by task type
        
    Returns:
        Configured HybridEvaluationEngine for strict evaluation
    """
    
    config = HybridEvaluationConfig(
        specialized_weight=0.8,  # Higher specialized weight
        general_chat_weight=0.2,
        enable_gating=True,
        jurisdiction_failure_penalty=0.5,  # High penalty for compliance failures
        require_jurisdiction_compliance=True,
        evaluation_mode=EvaluationMode.HYBRID,  # Force hybrid when available
        fallback_to_general_chat=False,  # Strict - no fallback
        enable_parallel_evaluation=True,
        max_evaluation_time_seconds=60.0
    )
    
    return HybridEvaluationEngine(config, general_chat_ensemble, specialized_ensembles)


# Convenience functions for integration

def evaluate_hybrid_response(response: str,
                            task_type: LegalTaskType,
                            jurisdiction: USJurisdiction,
                            prompt: str = "",
                            legal_domains: Optional[List[LegalDomain]] = None) -> HybridEvaluationResult:
    """
    Convenience function for hybrid evaluation.
    
    Args:
        response: Legal response to evaluate
        task_type: Type of legal task
        jurisdiction: Jurisdiction context
        prompt: Original prompt/question
        legal_domains: Legal domains if known
        
    Returns:
        HybridEvaluationResult with comprehensive evaluation
    """
    
    # Create hybrid engine
    engine = create_production_hybrid_engine()
    
    # Run hybrid evaluation
    # Note: This will need full integration to work completely
    return asyncio.run(engine.evaluate_hybrid(response, task_type, jurisdiction, prompt, legal_domains))


def get_hybrid_weights_for_task(task_type: LegalTaskType) -> Tuple[float, float]:
    """
    Get appropriate hybrid weights for a specific task type.
    
    Args:
        task_type: Legal task type
        
    Returns:
        Tuple of (specialized_weight, general_chat_weight)
    """
    
    # Task-specific weight adjustments
    task_weights = {
        LegalTaskType.JUDICIAL_REASONING: (0.8, 0.2),    # Higher specialized weight
        LegalTaskType.PRECEDENT_ANALYSIS: (0.8, 0.2),   # Higher specialized weight
        LegalTaskType.OPINION_GENERATION: (0.7, 0.3),   # Standard weights
        LegalTaskType.GENERAL_CHAT: (0.0, 1.0)          # General chat only
    }
    
    return task_weights.get(task_type, (0.7, 0.3))  # Default 70/30 split
