"""
Base Judge Classes for Multi-Task Legal Reward System

This module provides the foundational abstract base classes and interfaces
for all judge ensembles in the legal reward system. It defines the common
architecture, evaluation patterns, and integration points for specialized
and general chat judge ensembles.

Key Features:
- Abstract base classes for consistent judge implementation
- Common evaluation patterns and scoring interfaces
- Integration with API client and caching systems
- Error handling and recovery mechanisms
- Performance monitoring and cost tracking
- Support for both individual and ensemble judgments

All specialized judge ensembles inherit from these base classes to ensure
consistent behavior and integration with the hybrid evaluation system.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import json

# Import core components
from core import (
    LegalRewardEvaluation, JudgeEvaluation, EvaluationMetadata,
    APIProvider, LegalTaskType, USJurisdiction, LegalDomain,
    LegalRewardSystemError, create_error_context
)
from utils import (
    LegalRewardLogger, get_legal_logger,
    MultiStrategyLegalRewardCache, ManagedCache,
    MultiProviderRateLimiter, ManagedRateLimiter
)


class JudgeType(Enum):
    """Types of judges in the legal reward system"""
    GENERAL_CHAT = "general_chat"
    SPECIALIZED_JUDICIAL = "specialized_judicial"
    SPECIALIZED_PRECEDENT = "specialized_precedent"
    SPECIALIZED_OPINION = "specialized_opinion"
    JURISDICTION_COMPLIANCE = "jurisdiction_compliance"
    INDIVIDUAL_HELPFULNESS = "individual_helpfulness"
    INDIVIDUAL_ETHICS = "individual_ethics"
    INDIVIDUAL_CLARITY = "individual_clarity"


class EvaluationStrategy(Enum):
    """Evaluation strategies for judge ensembles"""
    CONSENSUS = "consensus"              # Require consensus among judges
    MAJORITY_VOTE = "majority_vote"     # Use majority voting
    WEIGHTED_AVERAGE = "weighted_average"  # Weighted average of scores
    BEST_OF_N = "best_of_n"            # Take best score
    HYBRID = "hybrid"                   # Custom hybrid approach


@dataclass
class JudgeConfig:
    """
    Configuration for individual judges and ensembles.
    
    Contains all necessary configuration parameters for judge initialization,
    evaluation behavior, and performance optimization.
    """
    
    # Judge identification
    judge_type: JudgeType
    judge_name: str
    
    # API configuration
    preferred_providers: List[APIProvider] = field(default_factory=lambda: [APIProvider.ANTHROPIC])
    fallback_providers: List[APIProvider] = field(default_factory=list)
    max_retries: int = 3
    timeout_seconds: int = 30
    
    # Evaluation configuration
    evaluation_strategy: EvaluationStrategy = EvaluationStrategy.WEIGHTED_AVERAGE
    confidence_threshold: float = 0.7
    score_range: Tuple[float, float] = (0.0, 10.0)
    
    # Performance configuration
    enable_caching: bool = True
    cache_ttl_hours: int = 24
    enable_async: bool = True
    max_concurrent: int = 3
    
    # Cost optimization
    enable_cost_tracking: bool = True
    max_cost_per_evaluation: float = 0.50  # USD
    prefer_cheaper_providers: bool = False
    
    # Specialized configuration
    specialized_prompts: Dict[str, str] = field(default_factory=dict)
    evaluation_weights: Dict[str, float] = field(default_factory=dict)
    additional_config: Dict[str, Any] = field(default_factory=dict)
    
    def get_primary_provider(self) -> APIProvider:
        """Get primary API provider"""
        return self.preferred_providers[0] if self.preferred_providers else APIProvider.ANTHROPIC
    
    def get_fallback_chain(self) -> List[APIProvider]:
        """Get complete fallback chain"""
        return self.preferred_providers + self.fallback_providers
    
    def is_specialized_judge(self) -> bool:
        """Check if this is a specialized judge configuration"""
        return self.judge_type.value.startswith("specialized_")


@dataclass
class EvaluationContext:
    """
    Context information for judge evaluations.
    
    Contains all contextual information needed for judges to perform
    accurate and appropriate evaluations.
    """
    
    # Core context
    task_type: LegalTaskType
    jurisdiction: USJurisdiction
    legal_domains: List[LegalDomain] = field(default_factory=list)
    
    # Input information
    prompt: str = ""
    expected_response_type: str = "general"
    
    # Evaluation metadata
    evaluation_id: str = ""
    timestamp: float = 0.0
    user_context: Dict[str, Any] = field(default_factory=dict)
    
    # Performance context
    max_evaluation_time: float = 30.0
    target_cost_budget: float = 0.25
    
    # Integration context
    previous_evaluations: List[JudgeEvaluation] = field(default_factory=list)
    system_state: Dict[str, Any] = field(default_factory=dict)
    
    def get_prompt_length(self) -> int:
        """Get length of evaluation prompt"""
        return len(self.prompt)
    
    def has_legal_domains(self) -> bool:
        """Check if legal domains are specified"""
        return len(self.legal_domains) > 0
    
    def is_federal_jurisdiction(self) -> bool:
        """Check if jurisdiction is federal"""
        return self.jurisdiction == USJurisdiction.FEDERAL


class BaseJudge(ABC):
    """
    Abstract base class for all individual judges.
    
    Defines the common interface and behavior that all judges must implement,
    including evaluation methods, error handling, and performance tracking.
    """
    
    def __init__(self, config: JudgeConfig, api_client=None, cache=None, rate_limiter=None):
        self.config = config
        self.logger = get_legal_logger(f"judge.{config.judge_name}")
        
        # External dependencies (will be injected or created)
        self.api_client = api_client
        self.cache = cache
        self.rate_limiter = rate_limiter
        
        # Performance tracking
        self.evaluation_stats = {
            "total_evaluations": 0,
            "successful_evaluations": 0,
            "failed_evaluations": 0,
            "avg_evaluation_time": 0.0,
            "total_cost": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        # Initialize judge-specific components
        self._initialize_judge()
    
    @abstractmethod
    def _initialize_judge(self):
        """Initialize judge-specific components (must be implemented by subclasses)"""
        pass
    
    @abstractmethod
    async def evaluate_async(self, 
                           response: str, 
                           context: EvaluationContext) -> JudgeEvaluation:
        """
        Asynchronously evaluate a legal response (must be implemented by subclasses).
        
        Args:
            response: Legal response text to evaluate
            context: Evaluation context and metadata
            
        Returns:
            JudgeEvaluation with score, reasoning, and metadata
        """
        pass
    
    def evaluate(self, response: str, context: EvaluationContext) -> JudgeEvaluation:
        """
        Synchronous wrapper for evaluate_async.
        
        Args:
            response: Legal response text to evaluate
            context: Evaluation context and metadata
            
        Returns:
            JudgeEvaluation with score, reasoning, and metadata
        """
        if self.config.enable_async:
            # Run async evaluation in event loop
            try:
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(self.evaluate_async(response, context))
            except RuntimeError:
                # Create new event loop if none exists
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(self.evaluate_async(response, context))
                finally:
                    loop.close()
        else:
            # Fallback to sync implementation (subclasses should override if needed)
            return asyncio.run(self.evaluate_async(response, context))
    
    def _create_cache_key(self, response: str, context: EvaluationContext) -> str:
        """Create cache key for evaluation result"""
        import hashlib
        
        # Create deterministic key from response and context
        key_data = {
            "judge": self.config.judge_name,
            "response_hash": hashlib.md5(response.encode()).hexdigest(),
            "task_type": context.task_type.value,
            "jurisdiction": context.jurisdiction.value,
            "domains": sorted([d.value for d in context.legal_domains])
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return f"judge_eval_{hashlib.sha256(key_string.encode()).hexdigest()[:16]}"
    
    def _update_performance_stats(self, 
                                 evaluation_time: float, 
                                 cost: float = 0.0, 
                                 success: bool = True,
                                 cache_hit: bool = False):
        """Update performance statistics"""
        
        self.evaluation_stats["total_evaluations"] += 1
        
        if success:
            self.evaluation_stats["successful_evaluations"] += 1
        else:
            self.evaluation_stats["failed_evaluations"] += 1
        
        if cache_hit:
            self.evaluation_stats["cache_hits"] += 1
        else:
            self.evaluation_stats["cache_misses"] += 1
        
        # Update running averages
        total = self.evaluation_stats["total_evaluations"]
        current_avg = self.evaluation_stats["avg_evaluation_time"]
        self.evaluation_stats["avg_evaluation_time"] = (
            (current_avg * (total - 1) + evaluation_time) / total
        )
        
        self.evaluation_stats["total_cost"] += cost
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for monitoring"""
        
        total = self.evaluation_stats["total_evaluations"]
        if total == 0:
            return {"status": "No evaluations performed"}
        
        return {
            "judge_name": self.config.judge_name,
            "judge_type": self.config.judge_type.value,
            "total_evaluations": total,
            "success_rate": self.evaluation_stats["successful_evaluations"] / total,
            "avg_evaluation_time": self.evaluation_stats["avg_evaluation_time"],
            "total_cost": self.evaluation_stats["total_cost"],
            "avg_cost_per_evaluation": self.evaluation_stats["total_cost"] / total,
            "cache_hit_rate": self.evaluation_stats["cache_hits"] / total if total > 0 else 0.0,
            "configuration": {
                "preferred_providers": [p.value for p in self.config.preferred_providers],
                "evaluation_strategy": self.config.evaluation_strategy.value,
                "caching_enabled": self.config.enable_caching
            }
        }


class BaseJudgeEnsemble(ABC):
    """
    Abstract base class for judge ensembles.
    
    Manages multiple individual judges and provides ensemble evaluation
    capabilities with consensus, voting, and aggregation mechanisms.
    """
    
    def __init__(self, 
                 ensemble_config: Dict[str, Any],
                 api_client=None,
                 cache=None,
                 rate_limiter=None):
        
        self.ensemble_config = ensemble_config
        self.ensemble_name = ensemble_config.get("name", "unknown_ensemble")
        self.logger = get_legal_logger(f"ensemble.{self.ensemble_name}")
        
        # External dependencies
        self.api_client = api_client
        self.cache = cache
        self.rate_limiter = rate_limiter
        
        # Ensemble configuration
        self.evaluation_strategy = EvaluationStrategy(
            ensemble_config.get("evaluation_strategy", "weighted_average")
        )
        self.judge_weights = ensemble_config.get("judge_weights", {})
        self.consensus_threshold = ensemble_config.get("consensus_threshold", 0.8)
        self.min_judges_required = ensemble_config.get("min_judges_required", 1)
        
        # Individual judges (will be initialized by subclasses)
        self.judges: Dict[str, BaseJudge] = {}
        
        # Ensemble performance tracking
        self.ensemble_stats = {
            "total_ensemble_evaluations": 0,
            "successful_ensemble_evaluations": 0,
            "consensus_achieved": 0,
            "avg_ensemble_time": 0.0,
            "total_ensemble_cost": 0.0,
            "judge_disagreement_rate": 0.0
        }
        
        # Initialize ensemble-specific components
        self._initialize_ensemble()
    
    @abstractmethod
    def _initialize_ensemble(self):
        """Initialize ensemble-specific components (must be implemented by subclasses)"""
        pass
    
    @abstractmethod
    def _create_individual_judges(self) -> Dict[str, BaseJudge]:
        """Create individual judges for this ensemble (must be implemented by subclasses)"""
        pass
    
    async def evaluate_ensemble_async(self, 
                                    response: str, 
                                    context: EvaluationContext) -> LegalRewardEvaluation:
        """
        Asynchronously evaluate response using the full judge ensemble.
        
        Args:
            response: Legal response text to evaluate
            context: Evaluation context and metadata
            
        Returns:
            LegalRewardEvaluation with ensemble results
        """
        
        start_time = time.time()
        self.ensemble_stats["total_ensemble_evaluations"] += 1
        
        try:
            # Validate input
            if not response.strip():
                raise LegalRewardSystemError(
                    "Empty response provided for evaluation",
                    error_context=create_error_context("judge_ensemble", "evaluate")
                )
            
            # Check cache first
            ensemble_cache_key = self._create_ensemble_cache_key(response, context)
            cached_result = await self._check_ensemble_cache(ensemble_cache_key)
            if cached_result:
                return cached_result
            
            # Evaluate with individual judges
            judge_evaluations = await self._evaluate_with_judges(response, context)
            
            # Validate judge results
            valid_evaluations = self._validate_judge_evaluations(judge_evaluations)
            
            if len(valid_evaluations) < self.min_judges_required:
                raise LegalRewardSystemError(
                    f"Insufficient valid judge evaluations: {len(valid_evaluations)}/{self.min_judges_required}",
                    error_context=create_error_context("judge_ensemble", "insufficient_judges")
                )
            
            # Apply ensemble strategy
            ensemble_result = self._apply_ensemble_strategy(valid_evaluations, context)
            
            # Add ensemble metadata
            ensemble_result.evaluation_metadata.processing_time_ms = (time.time() - start_time) * 1000
            ensemble_result.evaluation_metadata.judge_count = len(valid_evaluations)
            ensemble_result.evaluation_metadata.ensemble_name = self.ensemble_name
            
            # Cache result
            await self._cache_ensemble_result(ensemble_cache_key, ensemble_result)
            
            # Update stats
            self._update_ensemble_stats(time.time() - start_time, ensemble_result, True)
            
            self.ensemble_stats["successful_ensemble_evaluations"] += 1
            
            return ensemble_result
            
        except Exception as e:
            self.logger.error(f"Ensemble evaluation failed: {e}")
            self._update_ensemble_stats(time.time() - start_time, None, False)
            
            # Return error fallback
            return self._create_error_fallback(str(e), context)
    
    def evaluate_ensemble(self, response: str, context: EvaluationContext) -> LegalRewardEvaluation:
        """
        Synchronous wrapper for evaluate_ensemble_async.
        
        Args:
            response: Legal response text to evaluate
            context: Evaluation context and metadata
            
        Returns:
            LegalRewardEvaluation with ensemble results
        """
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.evaluate_ensemble_async(response, context))
        except RuntimeError:
            # Create new event loop if none exists
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self.evaluate_ensemble_async(response, context))
            finally:
                loop.close()
    
    async def _evaluate_with_judges(self, 
                                   response: str, 
                                   context: EvaluationContext) -> Dict[str, JudgeEvaluation]:
        """Evaluate response with all individual judges"""
        
        if not self.judges:
            self.judges = self._create_individual_judges()
        
        # Create evaluation tasks
        evaluation_tasks = []
        for judge_name, judge in self.judges.items():
            task = asyncio.create_task(
                self._safe_judge_evaluation(judge, response, context, judge_name)
            )
            evaluation_tasks.append((judge_name, task))
        
        # Execute evaluations concurrently
        judge_evaluations = {}
        completed_tasks = await asyncio.gather(
            *[task for _, task in evaluation_tasks], 
            return_exceptions=True
        )
        
        # Process results
        for (judge_name, _), result in zip(evaluation_tasks, completed_tasks):
            if isinstance(result, Exception):
                self.logger.warning(f"Judge {judge_name} evaluation failed: {result}")
                continue
            
            if isinstance(result, JudgeEvaluation):
                judge_evaluations[judge_name] = result
        
        return judge_evaluations
    
    async def _safe_judge_evaluation(self, 
                                   judge: BaseJudge, 
                                   response: str, 
                                   context: EvaluationContext,
                                   judge_name: str) -> JudgeEvaluation:
        """Safely evaluate with individual judge with error handling"""
        
        try:
            return await judge.evaluate_async(response, context)
        except Exception as e:
            self.logger.warning(f"Judge {judge_name} evaluation failed: {e}")
            
            # Return fallback evaluation
            return JudgeEvaluation(
                score=5.0,  # Neutral score
                reasoning=f"Judge evaluation failed: {str(e)[:100]}",
                confidence=0.1,
                judge_type=judge.config.judge_type.value,
                evaluation_metadata=EvaluationMetadata(
                    judge_name=judge_name,
                    evaluation_method="error_fallback",
                    additional_info={"error": str(e)}
                )
            )
    
    def _validate_judge_evaluations(self, 
                                   evaluations: Dict[str, JudgeEvaluation]) -> Dict[str, JudgeEvaluation]:
        """Validate and filter judge evaluations"""
        
        valid_evaluations = {}
        
        for judge_name, evaluation in evaluations.items():
            # Basic validation
            if not isinstance(evaluation, JudgeEvaluation):
                self.logger.warning(f"Invalid evaluation type from {judge_name}")
                continue
            
            # Score validation
            if not (0.0 <= evaluation.score <= 10.0):
                self.logger.warning(f"Invalid score from {judge_name}: {evaluation.score}")
                continue
            
            # Confidence validation
            if not (0.0 <= evaluation.confidence <= 1.0):
                self.logger.warning(f"Invalid confidence from {judge_name}: {evaluation.confidence}")
                continue
            
            valid_evaluations[judge_name] = evaluation
        
        return valid_evaluations
    
    def _apply_ensemble_strategy(self, 
                               evaluations: Dict[str, JudgeEvaluation], 
                               context: EvaluationContext) -> LegalRewardEvaluation:
        """Apply ensemble strategy to combine judge evaluations"""
        
        if self.evaluation_strategy == EvaluationStrategy.WEIGHTED_AVERAGE:
            return self._weighted_average_strategy(evaluations, context)
        elif self.evaluation_strategy == EvaluationStrategy.CONSENSUS:
            return self._consensus_strategy(evaluations, context)
        elif self.evaluation_strategy == EvaluationStrategy.MAJORITY_VOTE:
            return self._majority_vote_strategy(evaluations, context)
        elif self.evaluation_strategy == EvaluationStrategy.BEST_OF_N:
            return self._best_of_n_strategy(evaluations, context)
        else:
            # Default to weighted average
            return self._weighted_average_strategy(evaluations, context)
    
    def _weighted_average_strategy(self, 
                                 evaluations: Dict[str, JudgeEvaluation], 
                                 context: EvaluationContext) -> LegalRewardEvaluation:
        """Apply weighted average ensemble strategy"""
        
        total_weight = 0.0
        weighted_score = 0.0
        weighted_confidence = 0.0
        
        all_reasoning = []
        all_judges = []
        
        for judge_name, evaluation in evaluations.items():
            # Get judge weight (default to 1.0)
            weight = self.judge_weights.get(judge_name, 1.0)
            
            # Apply confidence weighting
            effective_weight = weight * evaluation.confidence
            
            weighted_score += evaluation.score * effective_weight
            weighted_confidence += evaluation.confidence * effective_weight
            total_weight += effective_weight
            
            all_reasoning.append(f"{judge_name}: {evaluation.reasoning}")
            all_judges.append(evaluation)
        
        if total_weight == 0.0:
            # Fallback if all weights are zero
            final_score = sum(e.score for e in evaluations.values()) / len(evaluations)
            final_confidence = sum(e.confidence for e in evaluations.values()) / len(evaluations)
        else:
            final_score = weighted_score / total_weight
            final_confidence = weighted_confidence / total_weight
        
        # Create ensemble reasoning
        ensemble_reasoning = self._create_ensemble_reasoning(evaluations, "weighted_average")
        
        return LegalRewardEvaluation(
            overall_score=final_score,
            confidence=final_confidence,
            reasoning=ensemble_reasoning,
            judge_evaluations=all_judges,
            evaluation_metadata=EvaluationMetadata(
                task_type=context.task_type,
                jurisdiction=context.jurisdiction,
                legal_domains=context.legal_domains,
                evaluation_method=f"ensemble_{self.evaluation_strategy.value}",
                ensemble_name=self.ensemble_name,
                additional_info={
                    "strategy": "weighted_average",
                    "total_weight": total_weight,
                    "judge_weights": self.judge_weights,
                    "judge_count": len(evaluations)
                }
            )
        )
    
    def _consensus_strategy(self, 
                          evaluations: Dict[str, JudgeEvaluation], 
                          context: EvaluationContext) -> LegalRewardEvaluation:
        """Apply consensus ensemble strategy"""
        
        scores = [e.score for e in evaluations.values()]
        
        # Calculate score variance
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        std_dev = variance ** 0.5
        
        # Check if consensus is achieved
        consensus_achieved = std_dev <= (10.0 * (1.0 - self.consensus_threshold))
        
        if consensus_achieved:
            self.ensemble_stats["consensus_achieved"] += 1
            confidence = min(0.9, self.consensus_threshold + (1.0 - self.consensus_threshold) * (1.0 - std_dev / 5.0))
        else:
            confidence = 0.5  # Lower confidence without consensus
        
        ensemble_reasoning = self._create_ensemble_reasoning(
            evaluations, 
            f"consensus ({'achieved' if consensus_achieved else 'not achieved'})"
        )
        
        return LegalRewardEvaluation(
            overall_score=mean_score,
            confidence=confidence,
            reasoning=ensemble_reasoning,
            judge_evaluations=list(evaluations.values()),
            evaluation_metadata=EvaluationMetadata(
                task_type=context.task_type,
                jurisdiction=context.jurisdiction,
                legal_domains=context.legal_domains,
                evaluation_method=f"ensemble_consensus",
                ensemble_name=self.ensemble_name,
                additional_info={
                    "consensus_achieved": consensus_achieved,
                    "score_variance": variance,
                    "consensus_threshold": self.consensus_threshold
                }
            )
        )
    
    def _majority_vote_strategy(self, 
                              evaluations: Dict[str, JudgeEvaluation], 
                              context: EvaluationContext) -> LegalRewardEvaluation:
        """Apply majority vote ensemble strategy (for categorical decisions)"""
        
        # Convert scores to categories (pass/fail based on threshold)
        threshold = 6.0
        votes = {"pass": [], "fail": []}
        
        for judge_name, evaluation in evaluations.items():
            category = "pass" if evaluation.score >= threshold else "fail"
            votes[category].append((judge_name, evaluation))
        
        # Determine majority
        majority_category = "pass" if len(votes["pass"]) > len(votes["fail"]) else "fail"
        majority_evaluations = votes[majority_category]
        
        # Calculate final score from majority
        if majority_evaluations:
            final_score = sum(e.score for _, e in majority_evaluations) / len(majority_evaluations)
            final_confidence = sum(e.confidence for _, e in majority_evaluations) / len(majority_evaluations)
        else:
            final_score = 5.0  # Neutral if no majority
            final_confidence = 0.5
        
        ensemble_reasoning = self._create_ensemble_reasoning(
            evaluations, 
            f"majority_vote ({majority_category})"
        )
        
        return LegalRewardEvaluation(
            overall_score=final_score,
            confidence=final_confidence,
            reasoning=ensemble_reasoning,
            judge_evaluations=list(evaluations.values()),
            evaluation_metadata=EvaluationMetadata(
                task_type=context.task_type,
                jurisdiction=context.jurisdiction,
                legal_domains=context.legal_domains,
                evaluation_method="ensemble_majority_vote",
                ensemble_name=self.ensemble_name,
                additional_info={
                    "majority_category": majority_category,
                    "vote_distribution": {cat: len(evals) for cat, evals in votes.items()},
                    "voting_threshold": threshold
                }
            )
        )
    
    def _best_of_n_strategy(self, 
                          evaluations: Dict[str, JudgeEvaluation], 
                          context: EvaluationContext) -> LegalRewardEvaluation:
        """Apply best-of-N ensemble strategy"""
        
        # Find best evaluation based on score and confidence
        best_evaluation = max(
            evaluations.values(), 
            key=lambda e: e.score * e.confidence
        )
        
        # Use best evaluation but with ensemble metadata
        ensemble_reasoning = self._create_ensemble_reasoning(
            evaluations, 
            f"best_of_n (selected: {best_evaluation.judge_type})"
        )
        
        return LegalRewardEvaluation(
            overall_score=best_evaluation.score,
            confidence=best_evaluation.confidence,
            reasoning=ensemble_reasoning,
            judge_evaluations=list(evaluations.values()),
            evaluation_metadata=EvaluationMetadata(
                task_type=context.task_type,
                jurisdiction=context.jurisdiction,
                legal_domains=context.legal_domains,
                evaluation_method="ensemble_best_of_n",
                ensemble_name=self.ensemble_name,
                additional_info={
                    "selected_judge": best_evaluation.judge_type,
                    "selection_criteria": "score * confidence",
                    "all_scores": {name: e.score for name, e in evaluations.items()}
                }
            )
        )
    
    def _create_ensemble_reasoning(self, 
                                 evaluations: Dict[str, JudgeEvaluation], 
                                 strategy: str) -> str:
        """Create comprehensive ensemble reasoning"""
        
        reasoning_parts = [f"Ensemble evaluation using {strategy} strategy:"]
        
        # Add individual judge summaries
        for judge_name, evaluation in evaluations.items():
            weight = self.judge_weights.get(judge_name, 1.0)
            reasoning_parts.append(
                f"• {judge_name} (weight: {weight:.2f}): {evaluation.score:.1f}/10.0 "
                f"(confidence: {evaluation.confidence:.2f}) - {evaluation.reasoning[:80]}..."
            )
        
        # Add aggregate statistics
        scores = [e.score for e in evaluations.values()]
        reasoning_parts.append(
            f"Score range: {min(scores):.1f}-{max(scores):.1f}, "
            f"std dev: {(sum((s - sum(scores)/len(scores))**2 for s in scores)/len(scores))**0.5:.2f}"
        )
        
        return " | ".join(reasoning_parts)
    
    def _create_ensemble_cache_key(self, response: str, context: EvaluationContext) -> str:
        """Create cache key for ensemble evaluation"""
        import hashlib
        
        key_data = {
            "ensemble": self.ensemble_name,
            "response_hash": hashlib.md5(response.encode()).hexdigest(),
            "context": {
                "task_type": context.task_type.value,
                "jurisdiction": context.jurisdiction.value,
                "domains": sorted([d.value for d in context.legal_domains])
            },
            "config": {
                "strategy": self.evaluation_strategy.value,
                "judges": sorted(self.judges.keys()) if self.judges else []
            }
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        return f"ensemble_eval_{hashlib.sha256(key_string.encode()).hexdigest()[:16]}"
    
    async def _check_ensemble_cache(self, cache_key: str) -> Optional[LegalRewardEvaluation]:
        """Check cache for ensemble evaluation result"""
        
        if not self.cache or not self.cache:
            return None
        
        try:
            # Implementation depends on cache interface
            # This is a placeholder - actual implementation in API client
            return None
        except Exception as e:
            self.logger.warning(f"Cache check failed: {e}")
            return None
    
    async def _cache_ensemble_result(self, cache_key: str, result: LegalRewardEvaluation):
        """Cache ensemble evaluation result"""
        
        if not self.cache:
            return
        
        try:
            # Implementation depends on cache interface
            # This is a placeholder - actual implementation in API client
            pass
        except Exception as e:
            self.logger.warning(f"Cache store failed: {e}")
    
    def _create_error_fallback(self, error_msg: str, context: EvaluationContext) -> LegalRewardEvaluation:
        """Create error fallback evaluation"""
        
        return LegalRewardEvaluation(
            overall_score=5.0,  # Neutral score
            confidence=0.1,
            reasoning=f"Ensemble evaluation failed: {error_msg[:100]}",
            judge_evaluations=[],
            evaluation_metadata=EvaluationMetadata(
                task_type=context.task_type,
                jurisdiction=context.jurisdiction,
                legal_domains=context.legal_domains,
                evaluation_method="error_fallback",
                ensemble_name=self.ensemble_name,
                additional_info={"error": error_msg}
            )
        )
    
    def _update_ensemble_stats(self, 
                             evaluation_time: float, 
                             result: Optional[LegalRewardEvaluation], 
                             success: bool):
        """Update ensemble performance statistics"""
        
        # Update timing
        total = self.ensemble_stats["total_ensemble_evaluations"]
        current_avg = self.ensemble_stats["avg_ensemble_time"]
        self.ensemble_stats["avg_ensemble_time"] = (
            (current_avg * (total - 1) + evaluation_time) / total if total > 0 else evaluation_time
        )
        
        # Update cost (if available in result)
        if result and hasattr(result.evaluation_metadata, 'total_cost'):
            self.ensemble_stats["total_ensemble_cost"] += getattr(result.evaluation_metadata, 'total_cost', 0.0)
    
    def get_ensemble_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive ensemble performance summary"""
        
        total = self.ensemble_stats["total_ensemble_evaluations"]
        if total == 0:
            return {"status": "No ensemble evaluations performed"}
        
        # Individual judge summaries
        judge_summaries = {}
        for judge_name, judge in self.judges.items():
            judge_summaries[judge_name] = judge.get_performance_summary()
        
        return {
            "ensemble_name": self.ensemble_name,
            "evaluation_strategy": self.evaluation_strategy.value,
            "total_ensemble_evaluations": total,
            "success_rate": self.ensemble_stats["successful_ensemble_evaluations"] / total,
            "consensus_rate": self.ensemble_stats["consensus_achieved"] / total,
            "avg_ensemble_time": self.ensemble_stats["avg_ensemble_time"],
            "total_ensemble_cost": self.ensemble_stats["total_ensemble_cost"],
            "avg_cost_per_evaluation": self.ensemble_stats["total_ensemble_cost"] / total,
            "judge_count": len(self.judges),
            "judge_weights": self.judge_weights,
            "individual_judge_performance": judge_summaries,
            "configuration": {
                "consensus_threshold": self.consensus_threshold,
                "min_judges_required": self.min_judges_required,
                "caching_enabled": bool(self.cache)
            }
        }


# Factory functions for creating judge configurations

def create_general_chat_judge_config(judge_name: str) -> JudgeConfig:
    """Create configuration for general chat judges"""
    return JudgeConfig(
        judge_type=JudgeType.GENERAL_CHAT,
        judge_name=judge_name,
        preferred_providers=[APIProvider.ANTHROPIC, APIProvider.OPENAI],
        fallback_providers=[APIProvider.GOOGLE],
        evaluation_strategy=EvaluationStrategy.WEIGHTED_AVERAGE,
        confidence_threshold=0.6,
        enable_caching=True,
        max_cost_per_evaluation=0.25
    )


def create_specialized_judge_config(judge_name: str, judge_type: JudgeType) -> JudgeConfig:
    """Create configuration for specialized judges"""
    return JudgeConfig(
        judge_type=judge_type,
        judge_name=judge_name,
        preferred_providers=[APIProvider.OPENAI, APIProvider.ANTHROPIC],
        fallback_providers=[APIProvider.GOOGLE],
        evaluation_strategy=EvaluationStrategy.CONSENSUS,
        confidence_threshold=0.7,
        enable_caching=True,
        max_cost_per_evaluation=0.50,  # Higher cost for specialized evaluation
        cache_ttl_hours=48  # Longer cache for specialized results
    )


def create_individual_judge_config(judge_name: str, judge_type: JudgeType) -> JudgeConfig:
    """Create configuration for individual component judges"""
    return JudgeConfig(
        judge_type=judge_type,
        judge_name=judge_name,
        preferred_providers=[APIProvider.ANTHROPIC],
        fallback_providers=[APIProvider.OPENAI, APIProvider.GOOGLE],
        evaluation_strategy=EvaluationStrategy.WEIGHTED_AVERAGE,
        confidence_threshold=0.6,
        enable_caching=True,
        max_cost_per_evaluation=0.15  # Lower cost for individual components
    )


# Utility functions for ensemble management

def create_evaluation_context(task_type: LegalTaskType,
                            jurisdiction: USJurisdiction,
                            prompt: str = "",
                            legal_domains: Optional[List[LegalDomain]] = None) -> EvaluationContext:
    """Create evaluation context for judge evaluations"""
    
    import time
    import uuid
    
    return EvaluationContext(
        task_type=task_type,
        jurisdiction=jurisdiction,
        legal_domains=legal_domains or [],
        prompt=prompt,
        evaluation_id=str(uuid.uuid4()),
        timestamp=time.time()
    )


def validate_judge_configuration(config: JudgeConfig) -> List[str]:
    """Validate judge configuration and return any issues"""
    
    issues = []
    
    # Check required fields
    if not config.judge_name:
        issues.append("Judge name is required")
    
    if not config.preferred_providers:
        issues.append("At least one preferred provider is required")
    
    # Check numeric ranges
    if not (0.0 <= config.confidence_threshold <= 1.0):
        issues.append("Confidence threshold must be between 0.0 and 1.0")
    
    if config.max_cost_per_evaluation <= 0:
        issues.append("Max cost per evaluation must be positive")
    
    if config.timeout_seconds <= 0:
        issues.append("Timeout seconds must be positive")
    
    # Check provider validity
    for provider in config.preferred_providers + config.fallback_providers:
        if provider not in APIProvider:
            issues.append(f"Invalid API provider: {provider}")
    
    return issues
