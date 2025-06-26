"""
Task Difficulty Weight Manager for Multi-Task Legal Reward System

This module manages task difficulty weights for different legal task types,
allowing for dynamic adjustment and optimization during training. Provides
comprehensive weight tracking, performance-based optimization, and analytics
for training optimization.

Key Features:
- Default weights based on comprehensive task complexity analysis
- Dynamic weight adjustment with tracking and history
- Performance-based weight optimization
- Statistical analysis and optimization suggestions
- Configuration import/export for persistence
- Comprehensive analytics and monitoring

The weight manager is critical for training optimization, ensuring that
different task types receive appropriate difficulty weighting to produce
optimal training signals for VERL integration.
"""

import time
import statistics
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Import core components
from ..core import (
    LegalTaskType, LegalRewardSystemError, create_error_context
)
from ..config import LegalRewardSystemConfig
from ..utils import get_legal_logger


class WeightAdjustmentReason(Enum):
    """Reasons for weight adjustments"""
    MANUAL_ADJUSTMENT = "manual_adjustment"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    CONFIG_IMPORT = "config_import"
    AUTOMATIC_ADJUSTMENT = "automatic_adjustment"
    SYSTEM_INITIALIZATION = "system_initialization"
    TRAINING_FEEDBACK = "training_feedback" 
    USER_OVERRIDE = "user_override"
    AUTOMATIC_REBALANCING = "automatic_rebalancing"
    CONFIGURATION_UPDATE = "configuration_update"


@dataclass
class WeightChangeRecord:
    """Record of a weight change for audit trail"""
    timestamp: float
    task_type: LegalTaskType
    old_weight: float
    new_weight: float
    reason: WeightAdjustmentReason
    source: str = "system"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_change_magnitude(self) -> float:
        """Get magnitude of weight change"""
        return abs(self.new_weight - self.old_weight)
    
    def get_change_percentage(self) -> float:
        """Get percentage change in weight"""
        if self.old_weight == 0:
            return 100.0 if self.new_weight > 0 else 0.0
        return ((self.new_weight - self.old_weight) / self.old_weight) * 100


@dataclass
class TaskPerformanceData:
    """Performance data for a specific task type"""
    task_type: LegalTaskType
    evaluation_count: int = 0
    total_score: float = 0.0
    score_history: List[float] = field(default_factory=list)
    confidence_history: List[float] = field(default_factory=list)
    cost_history: List[float] = field(default_factory=list)
    processing_time_history: List[float] = field(default_factory=list)
    last_updated: float = field(default_factory=time.time)
    
    def add_evaluation(self, score: float, confidence: float = 1.0, 
                      cost: float = 0.0, processing_time: float = 0.0):
        """Add evaluation data"""
        self.evaluation_count += 1
        self.total_score += score
        self.score_history.append(score)
        self.confidence_history.append(confidence)
        self.cost_history.append(cost)
        self.processing_time_history.append(processing_time)
        self.last_updated = time.time()
        
        # Keep only recent history to prevent memory bloat
        max_history = 1000
        if len(self.score_history) > max_history:
            self.score_history = self.score_history[-max_history:]
            self.confidence_history = self.confidence_history[-max_history:]
            self.cost_history = self.cost_history[-max_history:]
            self.processing_time_history = self.processing_time_history[-max_history:]
    
    def get_average_score(self) -> float:
        """Get average score"""
        if self.evaluation_count == 0:
            return 0.0
        return self.total_score / self.evaluation_count
    
    def get_score_variance(self) -> float:
        """Get score variance"""
        if len(self.score_history) < 2:
            return 0.0
        return statistics.variance(self.score_history)
    
    def get_score_std_dev(self) -> float:
        """Get score standard deviation"""
        if len(self.score_history) < 2:
            return 0.0
        return statistics.stdev(self.score_history)
    
    def get_recent_performance(self, recent_count: int = 100) -> Dict[str, float]:
        """Get recent performance statistics"""
        if not self.score_history:
            return {"avg_score": 0.0, "variance": 0.0, "count": 0}
        
        recent_scores = self.score_history[-recent_count:]
        
        return {
            "avg_score": statistics.mean(recent_scores),
            "variance": statistics.variance(recent_scores) if len(recent_scores) > 1 else 0.0,
            "std_dev": statistics.stdev(recent_scores) if len(recent_scores) > 1 else 0.0,
            "count": len(recent_scores),
            "min_score": min(recent_scores),
            "max_score": max(recent_scores)
        }


class TaskDifficultyWeightManager:
    """
    Manages task difficulty weights for different legal task types.
    
    Provides comprehensive weight management including:
    - Default weights based on task complexity analysis
    - Dynamic weight adjustment with full audit trail
    - Performance-based weight optimization
    - Statistical analysis and optimization suggestions
    - Configuration persistence and import/export
    """
    
    def __init__(self, config: Optional[LegalRewardSystemConfig] = None):
        """Initialize TaskDifficultyWeightManager with proper defaults"""
        
        self.config = config
        self.logger = get_legal_logger("task_weights")
        
        # Initialize current weights with defaults
        self.current_weights = {
            LegalTaskType.JUDICIAL_REASONING: 1.5,
            LegalTaskType.PRECEDENT_ANALYSIS: 1.3,
            LegalTaskType.OPINION_GENERATION: 1.1,
            LegalTaskType.GENERAL_CHAT: 1.0
        }
        
        # Performance tracking
        self.weight_adjustments = []
        self.last_optimization = None
        
        # Configuration
        self.enable_auto_adjustment = getattr(config, 'enable_auto_weight_adjustment', True) if config else True
        self.adjustment_frequency = getattr(config, 'weight_adjustment_frequency', 100) if config else 100
        
        self.logger.info(f"TaskDifficultyWeightManager initialized with weights: {self.current_weights}")


    def get_weight(self, task_type: LegalTaskType) -> float:
        """
        Get current weight for a specific task type.
        
        This method is called by the router during evaluation.
        
        Args:
            task_type: Legal task type to get weight for
            
        Returns:
            Current weight for the task type
        """
        
        # Get current weights
        current_weights = self.get_current_weights()
        
        # Return weight for the specific task type
        if task_type in current_weights:
            return current_weights[task_type]
        else:
            # Fallback to default weights
            default_weights = {
                LegalTaskType.JUDICIAL_REASONING: 1.5,
                LegalTaskType.PRECEDENT_ANALYSIS: 1.3,
                LegalTaskType.OPINION_GENERATION: 1.1,
                LegalTaskType.GENERAL_CHAT: 1.0
            }
            return default_weights.get(task_type, 1.0)
    
    def get_current_weights(self) -> Dict[LegalTaskType, float]:
        """Get current weights for all task types"""
        
        # If the manager has been properly initialized, return current weights
        if hasattr(self, 'current_weights') and self.current_weights:
            return self.current_weights.copy()
        
        # Otherwise return default weights
        return {
            LegalTaskType.JUDICIAL_REASONING: 1.5,
            LegalTaskType.PRECEDENT_ANALYSIS: 1.3,
            LegalTaskType.OPINION_GENERATION: 1.1,
            LegalTaskType.GENERAL_CHAT: 1.0
        }


    def set_weight(self, task_type: LegalTaskType, weight: float) -> None:
        """Set weight for a specific task type"""
        
        if not hasattr(self, 'current_weights'):
            self.current_weights = self.get_current_weights()
        
        # Validate weight
        if weight < 0.1 or weight > 5.0:
            self.logger.warning(f"Weight {weight} for {task_type.value} is outside recommended range [0.1, 5.0]")
        
        # Update weight
        old_weight = self.current_weights.get(task_type, 1.0)
        self.current_weights[task_type] = weight
        
        # Log the change
        self.logger.info(f"Updated weight for {task_type.value}: {old_weight} → {weight}")

    
    def update_weight(self, 
                     task_type: LegalTaskType, 
                     weight: float,
                     reason: WeightAdjustmentReason = WeightAdjustmentReason.MANUAL_ADJUSTMENT,
                     source: str = "manual",
                     metadata: Optional[Dict[str, Any]] = None):
        """
        Update weight for a specific task type with full tracking.
        
        Args:
            task_type: Legal task type to update
            weight: New weight value
            reason: Reason for the adjustment
            source: Source of the adjustment
            metadata: Additional metadata about the change
            
        Raises:
            LegalRewardSystemError: If weight is invalid
        """
        
        # Validate weight
        if not isinstance(weight, (int, float)):
            raise LegalRewardSystemError(
                f"Weight must be numeric, got {type(weight).__name__}",
                error_context=create_error_context("task_weights", "update_weight")
            )
        
        if weight < self.min_weight or weight > self.max_weight:
            raise LegalRewardSystemError(
                f"Weight {weight} outside valid range [{self.min_weight}, {self.max_weight}]",
                error_context=create_error_context("task_weights", "update_weight")
            )
        
        # Record the change
        old_weight = self.weights.get(task_type, 1.0)
        self.weights[task_type] = weight
        
        self._record_weight_change(task_type, old_weight, weight, reason, source, metadata)
        
        self.logger.info(f"Updated weight for {task_type.value}: {old_weight:.3f} -> {weight:.3f} (reason: {reason.value})")
    
    def get_all_weights(self) -> Dict[LegalTaskType, float]:
        """Get copy of all current weights"""
        return self.weights.copy()
    
    def reset_weight(self, task_type: LegalTaskType, 
                    reason: WeightAdjustmentReason = WeightAdjustmentReason.MANUAL_ADJUSTMENT):
        """Reset weight to default value"""
        default_weight = self.default_weights.get(task_type, 1.0)
        self.update_weight(task_type, default_weight, reason, "reset")
    
    def reset_all_weights(self, 
                         reason: WeightAdjustmentReason = WeightAdjustmentReason.MANUAL_ADJUSTMENT):
        """Reset all weights to default values"""
        for task_type in LegalTaskType:
            self.reset_weight(task_type, reason)
    
    def add_performance_data(self, 
                           task_type: LegalTaskType,
                           score: float,
                           confidence: float = 1.0,
                           cost: float = 0.0,
                           processing_time: float = 0.0):
        """
        Add performance data for a task type.
        
        Args:
            task_type: Task type that was evaluated
            score: Evaluation score (0-10)
            confidence: Confidence in the evaluation (0-1)
            cost: Cost of the evaluation
            processing_time: Processing time in milliseconds
        """
        
        if task_type not in self.performance_data:
            self.performance_data[task_type] = TaskPerformanceData(task_type)
        
        self.performance_data[task_type].add_evaluation(score, confidence, cost, processing_time)
        
        # Check if auto-adjustment should be triggered
        if (self.optimization_enabled and 
            self.performance_data[task_type].evaluation_count >= self.auto_adjustment_threshold):
            self._consider_automatic_adjustment(task_type)
    
    def suggest_weight_adjustment(self, task_type: LegalTaskType) -> Optional[Tuple[float, str]]:
        """
        Suggest weight adjustment based on performance data.
        
        Args:
            task_type: Task type to analyze
            
        Returns:
            Tuple of (suggested_weight, reasoning) or None if no adjustment suggested
        """
        
        if task_type not in self.performance_data:
            return None
        
        perf_data = self.performance_data[task_type]
        
        if perf_data.evaluation_count < 10:
            return None  # Not enough data
        
        current_weight = self.weights.get(task_type, 1.0)
        avg_score = perf_data.get_average_score()
        score_variance = perf_data.get_score_variance()
        
        # Target score is around 7.0 (good performance)
        target_score = 7.0
        score_deviation = avg_score - target_score
        
        # Calculate adjustment factor
        confidence_factor = min(perf_data.evaluation_count / 100, 1.0)  # More data = more confidence
        variance_factor = max(0.1, 1.0 - (score_variance / 5.0))  # Lower variance = more confidence
        
        adjustment_magnitude = score_deviation * 0.1 * confidence_factor * variance_factor
        
        # Apply bounds to prevent extreme adjustments
        suggested_weight = current_weight - adjustment_magnitude  # Inverse relationship
        suggested_weight = max(self.min_weight, min(self.max_weight, suggested_weight))
        
        # Only suggest if change is meaningful
        if abs(suggested_weight - current_weight) < 0.05:
            return None
        
        reasoning = f"Based on {perf_data.evaluation_count} evaluations: avg_score={avg_score:.2f}, target={target_score:.2f}, variance={score_variance:.2f}"
        
        return suggested_weight, reasoning
    
    def optimize_weights(self, apply_suggestions: bool = False) -> Dict[LegalTaskType, Tuple[float, str]]:
        """
        Optimize all weights based on performance data.
        
        Args:
            apply_suggestions: Whether to automatically apply suggested changes
            
        Returns:
            Dictionary of suggested weight changes
        """
        
        suggestions = {}
        
        for task_type in LegalTaskType:
            suggestion = self.suggest_weight_adjustment(task_type)
            if suggestion:
                suggested_weight, reasoning = suggestion
                suggestions[task_type] = (suggested_weight, reasoning)
                
                if apply_suggestions:
                    self.update_weight(
                        task_type, suggested_weight,
                        WeightAdjustmentReason.PERFORMANCE_OPTIMIZATION,
                        "auto_optimizer",
                        {"reasoning": reasoning}
                    )
        
        return suggestions
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        
        summary = {
            "total_evaluations": sum(data.evaluation_count for data in self.performance_data.values()),
            "task_performance": {},
            "weight_stability": self._analyze_weight_stability(),
            "optimization_opportunities": len(self.optimize_weights(apply_suggestions=False))
        }
        
        for task_type, perf_data in self.performance_data.items():
            if perf_data.evaluation_count > 0:
                recent_perf = perf_data.get_recent_performance(50)
                summary["task_performance"][task_type.value] = {
                    "evaluation_count": perf_data.evaluation_count,
                    "average_score": perf_data.get_average_score(),
                    "score_variance": perf_data.get_score_variance(),
                    "recent_performance": recent_perf,
                    "current_weight": self.weights.get(task_type, 1.0)
                }
        
        return summary
    
    def get_weight_statistics(self) -> Dict[str, Any]:
        """Get comprehensive weight statistics and history"""
        
        return {
            "current_weights": {k.value: v for k, v in self.weights.items()},
            "default_weights": {k.value: v for k, v in self.default_weights.items()},
            "weight_ranges": {"min": self.min_weight, "max": self.max_weight},
            "weight_changes": len(self.weight_history),
            "recent_changes": [self._format_weight_change(change) for change in self.weight_history[-5:]],
            "performance_data_available": {k.value: v.evaluation_count for k, v in self.performance_data.items()},
            "total_weight_sum": sum(self.weights.values()),
            "weight_distribution": self._analyze_weight_distribution(),
            "optimization_enabled": self.optimization_enabled
        }
    
    def export_weights_config(self) -> Dict[str, float]:
        """Export current weights in configuration format"""
        return {task_type.value: weight for task_type, weight in self.weights.items()}
    
    def import_weights_config(self, weights_config: Dict[str, float], 
                            source: str = "config_import"):
        """
        Import weights from configuration format.
        
        Args:
            weights_config: Dictionary mapping task type strings to weights
            source: Source identifier for the import
        """
        
        imported_count = 0
        errors = []
        
        for task_type_str, weight in weights_config.items():
            try:
                task_type = LegalTaskType(task_type_str)
                self.update_weight(
                    task_type, weight,
                    WeightAdjustmentReason.CONFIG_IMPORT,
                    source,
                    {"config_data": weights_config}
                )
                imported_count += 1
            except ValueError:
                error_msg = f"Invalid task type in config: {task_type_str}"
                errors.append(error_msg)
                self.logger.warning(error_msg)
            except LegalRewardSystemError as e:
                error_msg = f"Invalid weight for {task_type_str}: {weight} - {e}"
                errors.append(error_msg)
                self.logger.warning(error_msg)
        
        self.logger.info(f"Imported {imported_count} weights from config, {len(errors)} errors")
        
        if errors:
            raise LegalRewardSystemError(
                f"Weight import completed with errors: {'; '.join(errors)}",
                error_context=create_error_context("task_weights", "import_config")
            )
    
    def get_weight_history(self, task_type: Optional[LegalTaskType] = None, 
                          limit: Optional[int] = None) -> List[WeightChangeRecord]:
        """
        Get weight change history.
        
        Args:
            task_type: Filter by specific task type (optional)
            limit: Maximum number of records to return (optional)
            
        Returns:
            List of weight change records
        """
        
        history = self.weight_history
        
        if task_type:
            history = [record for record in history if record.task_type == task_type]
        
        if limit:
            history = history[-limit:]
        
        return history
    
    def enable_optimization(self, enabled: bool = True):
        """Enable or disable automatic weight optimization"""
        self.optimization_enabled = enabled
        self.logger.info(f"Automatic weight optimization {'enabled' if enabled else 'disabled'}")
    
    def set_weight_bounds(self, min_weight: float, max_weight: float):
        """
        Set weight bounds for validation.
        
        Args:
            min_weight: Minimum allowed weight
            max_weight: Maximum allowed weight
        """
        
        if min_weight <= 0 or max_weight <= min_weight:
            raise LegalRewardSystemError(
                f"Invalid weight bounds: min={min_weight}, max={max_weight}",
                error_context=create_error_context("task_weights", "set_bounds")
            )
        
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        self.logger.info(f"Weight bounds updated: [{min_weight}, {max_weight}]")
    
    def _record_weight_change(self, 
                            task_type: LegalTaskType,
                            old_weight: float,
                            new_weight: float,
                            reason: WeightAdjustmentReason,
                            source: str,
                            metadata: Optional[Dict[str, Any]] = None):
        """Record a weight change in the audit trail"""
        
        record = WeightChangeRecord(
            timestamp=time.time(),
            task_type=task_type,
            old_weight=old_weight,
            new_weight=new_weight,
            reason=reason,
            source=source,
            metadata=metadata or {}
        )
        
        self.weight_history.append(record)
        
        # Keep history manageable
        max_history = 1000
        if len(self.weight_history) > max_history:
            self.weight_history = self.weight_history[-max_history:]
    
    def _consider_automatic_adjustment(self, task_type: LegalTaskType):
        """Consider automatic weight adjustment for a task type"""
        
        if not self.optimization_enabled:
            return
        
        suggestion = self.suggest_weight_adjustment(task_type)
        if suggestion:
            suggested_weight, reasoning = suggestion
            
            # Apply conservative automatic adjustments
            current_weight = self.weights.get(task_type, 1.0)
            max_auto_change = 0.1  # Maximum 10% automatic change
            
            if abs(suggested_weight - current_weight) <= max_auto_change:
                self.update_weight(
                    task_type, suggested_weight,
                    WeightAdjustmentReason.AUTOMATIC_ADJUSTMENT,
                    "auto_system",
                    {"reasoning": reasoning, "conservative_adjustment": True}
                )
    
    def _analyze_weight_stability(self) -> Dict[str, Any]:
        """Analyze weight stability over time"""
        
        if len(self.weight_history) < 2:
            return {"status": "insufficient_data"}
        
        recent_changes = self.weight_history[-20:]  # Last 20 changes
        total_changes = len(recent_changes)
        
        # Calculate change frequency by task type
        task_change_counts = {}
        total_magnitude = 0.0
        
        for record in recent_changes:
            task_type = record.task_type
            task_change_counts[task_type] = task_change_counts.get(task_type, 0) + 1
            total_magnitude += record.get_change_magnitude()
        
        avg_magnitude = total_magnitude / total_changes if total_changes > 0 else 0.0
        
        return {
            "status": "analyzed",
            "recent_changes": total_changes,
            "average_change_magnitude": avg_magnitude,
            "most_adjusted_task": max(task_change_counts, key=task_change_counts.get) if task_change_counts else None,
            "stability_score": max(0.0, 1.0 - (avg_magnitude / 0.5))  # Higher is more stable
        }
    
    def _analyze_weight_distribution(self) -> Dict[str, Any]:
        """Analyze current weight distribution"""
        
        weights_list = list(self.weights.values())
        
        if not weights_list:
            return {"status": "no_weights"}
        
        return {
            "mean": statistics.mean(weights_list),
            "median": statistics.median(weights_list),
            "std_dev": statistics.stdev(weights_list) if len(weights_list) > 1 else 0.0,
            "min": min(weights_list),
            "max": max(weights_list),
            "range": max(weights_list) - min(weights_list),
            "coefficient_of_variation": statistics.stdev(weights_list) / statistics.mean(weights_list) if len(weights_list) > 1 and statistics.mean(weights_list) > 0 else 0.0
        }
    
    def _format_weight_change(self, record: WeightChangeRecord) -> Dict[str, Any]:
        """Format weight change record for display"""
        
        return {
            "timestamp": record.timestamp,
            "task_type": record.task_type.value,
            "old_weight": record.old_weight,
            "new_weight": record.new_weight,
            "change_magnitude": record.get_change_magnitude(),
            "change_percentage": record.get_change_percentage(),
            "reason": record.reason.value,
            "source": record.source
        }


class WeightOptimizer:
    """Advanced weight optimization using training performance data"""
    
    def __init__(self, weight_manager: TaskDifficultyWeightManager):
        self.weight_manager = weight_manager
        self.optimization_history: List[Dict[str, Any]] = []
        self.logger = get_legal_logger("weight_optimizer")
    
    def optimize_weights_from_training_data(self, 
                                          training_results: List[Dict[str, Any]]) -> Dict[LegalTaskType, float]:
        """
        Optimize weights based on comprehensive training performance data.
        
        Args:
            training_results: List of training result dictionaries
            
        Returns:
            Dictionary of optimized weights
        """
        
        # Analyze training performance patterns
        task_performance = self._analyze_training_performance(training_results)
        
        # Calculate optimal weights
        optimized_weights = {}
        
        for task_type in LegalTaskType:
            if task_type in task_performance:
                perf_data = task_performance[task_type]
                optimized_weight = self._calculate_optimal_weight(task_type, perf_data)
                optimized_weights[task_type] = optimized_weight
            else:
                optimized_weights[task_type] = self.weight_manager.get_weight(task_type)
        
        # Record optimization
        optimization_record = {
            "timestamp": time.time(),
            "training_samples": len(training_results),
            "optimized_weights": optimized_weights,
            "performance_data": task_performance
        }
        self.optimization_history.append(optimization_record)
        
        self.logger.info(f"Optimized weights based on {len(training_results)} training samples")
        
        return optimized_weights
    
    def _analyze_training_performance(self, training_results: List[Dict[str, Any]]) -> Dict[LegalTaskType, Dict[str, float]]:
        """Analyze training performance by task type"""
        
        task_data = {}
        
        for result in training_results:
            task_type_str = result.get("task_type")
            if not task_type_str:
                continue
            
            try:
                task_type = LegalTaskType(task_type_str)
            except ValueError:
                continue
            
            if task_type not in task_data:
                task_data[task_type] = {
                    "scores": [],
                    "training_losses": [],
                    "convergence_rates": []
                }
            
            # Extract performance metrics
            if "score" in result:
                task_data[task_type]["scores"].append(result["score"])
            if "training_loss" in result:
                task_data[task_type]["training_losses"].append(result["training_loss"])
            if "convergence_rate" in result:
                task_data[task_type]["convergence_rates"].append(result["convergence_rate"])
        
        # Calculate aggregated statistics
        performance_summary = {}
        for task_type, data in task_data.items():
            performance_summary[task_type] = {
                "avg_score": statistics.mean(data["scores"]) if data["scores"] else 0.0,
                "score_variance": statistics.variance(data["scores"]) if len(data["scores"]) > 1 else 0.0,
                "avg_training_loss": statistics.mean(data["training_losses"]) if data["training_losses"] else 0.0,
                "avg_convergence_rate": statistics.mean(data["convergence_rates"]) if data["convergence_rates"] else 0.0,
                "sample_count": len(data["scores"])
            }
        
        return performance_summary
    
    def _calculate_optimal_weight(self, task_type: LegalTaskType, performance_data: Dict[str, float]) -> float:
        """Calculate optimal weight based on performance data"""
        
        current_weight = self.weight_manager.get_weight(task_type)
        
        # If insufficient data, keep current weight
        if performance_data["sample_count"] < 5:
            return current_weight
        
        # Weight adjustment based on multiple factors
        avg_score = performance_data["avg_score"]
        score_variance = performance_data["score_variance"]
        convergence_rate = performance_data.get("avg_convergence_rate", 1.0)
        
        # Target metrics
        target_score = 7.0
        target_variance = 1.0
        
        # Calculate adjustment factors
        score_factor = 1.0 - ((avg_score - target_score) / 10.0)  # Normalize score influence
        variance_factor = min(score_variance / target_variance, 2.0)  # High variance increases weight
        convergence_factor = max(0.5, min(1.5, convergence_rate))  # Slow convergence increases weight
        
        # Combine factors
        adjustment_factor = score_factor * variance_factor * convergence_factor
        suggested_weight = current_weight * adjustment_factor
        
        # Apply bounds
        suggested_weight = max(
            self.weight_manager.min_weight,
            min(self.weight_manager.max_weight, suggested_weight)
        )
        
        return suggested_weight


class WeightAnalyzer:
    """Analyze weight impact on training performance"""
    
    def __init__(self, weight_manager: TaskDifficultyWeightManager):
        self.weight_manager = weight_manager
        self.logger = get_legal_logger("weight_analyzer")
    
    def analyze_weight_impact(self, 
                            performance_data: List[Dict[str, Any]], 
                            weight_history: List[WeightChangeRecord]) -> Dict[str, Any]:
        """
        Analyze how weight changes affected training performance.
        
        Args:
            performance_data: Training performance data
            weight_history: Weight change history
            
        Returns:
            Analysis results with insights and recommendations
        """
        
        # Correlate weight changes with performance changes
        correlations = self._calculate_weight_performance_correlations(performance_data, weight_history)
        
        # Identify optimal weight ranges
        optimal_ranges = self._identify_optimal_weight_ranges(performance_data)
        
        # Generate recommendations
        recommendations = self._generate_weight_recommendations(correlations, optimal_ranges)
        
        analysis_result = {
            "correlations": correlations,
            "optimal_ranges": optimal_ranges,
            "recommendations": recommendations,
            "analysis_timestamp": time.time(),
            "data_points_analyzed": len(performance_data),
            "weight_changes_analyzed": len(weight_history)
        }
        
        self.logger.info(f"Weight impact analysis completed: {len(recommendations)} recommendations generated")
        
        return analysis_result
    
    def _calculate_weight_performance_correlations(self, 
                                                  performance_data: List[Dict[str, Any]], 
                                                  weight_history: List[WeightChangeRecord]) -> Dict[str, float]:
        """Calculate correlations between weight changes and performance"""
        
        correlations = {}
        
        for task_type in LegalTaskType:
            # Extract performance data for this task type
            task_performance = [p for p in performance_data if p.get("task_type") == task_type.value]
            task_weight_changes = [w for w in weight_history if w.task_type == task_type]
            
            if len(task_performance) < 5 or len(task_weight_changes) < 2:
                correlations[task_type.value] = 0.0
                continue
            
            # Calculate correlation (simplified implementation)
            # In practice, this would use more sophisticated statistical methods
            performance_scores = [p.get("score", 0.0) for p in task_performance]
            weights = [w.new_weight for w in task_weight_changes]
            
            if len(performance_scores) == len(weights):
                try:
                    # Simple correlation calculation
                    correlation = self._calculate_correlation(weights, performance_scores)
                    correlations[task_type.value] = correlation
                except:
                    correlations[task_type.value] = 0.0
            else:
                correlations[task_type.value] = 0.0
        
        return correlations
    
    def _identify_optimal_weight_ranges(self, performance_data: List[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
        """Identify optimal weight ranges for each task type"""
        
        optimal_ranges = {}
        
        for task_type in LegalTaskType:
            task_data = [p for p in performance_data if p.get("task_type") == task_type.value]
            
            if len(task_data) < 10:
                # Use default range if insufficient data
                current_weight = self.weight_manager.get_weight(task_type)
                optimal_ranges[task_type.value] = (current_weight * 0.8, current_weight * 1.2)
                continue
            
            # Find weight range associated with best performance
            best_performance_threshold = 7.5  # Define what constitutes "good" performance
            good_performance_data = [p for p in task_data if p.get("score", 0) >= best_performance_threshold]
            
            if good_performance_data:
                weights = [p.get("weight", self.weight_manager.get_weight(task_type)) for p in good_performance_data]
                optimal_ranges[task_type.value] = (min(weights), max(weights))
            else:
                # Fallback to current weight range
                current_weight = self.weight_manager.get_weight(task_type)
                optimal_ranges[task_type.value] = (current_weight * 0.9, current_weight * 1.1)
        
        return optimal_ranges
    
    def _generate_weight_recommendations(self, 
                                       correlations: Dict[str, float], 
                                       optimal_ranges: Dict[str, Tuple[float, float]]) -> List[str]:
        """Generate actionable weight recommendations"""
        
        recommendations = []
        
        for task_type in LegalTaskType:
            task_type_str = task_type.value
            current_weight = self.weight_manager.get_weight(task_type)
            
            if task_type_str in correlations:
                correlation = correlations[task_type_str]
                optimal_min, optimal_max = optimal_ranges.get(task_type_str, (current_weight, current_weight))
                
                # Generate specific recommendations
                if current_weight < optimal_min:
                    recommendations.append(f"Increase {task_type_str} weight to at least {optimal_min:.2f} (currently {current_weight:.2f})")
                elif current_weight > optimal_max:
                    recommendations.append(f"Decrease {task_type_str} weight to at most {optimal_max:.2f} (currently {current_weight:.2f})")
                elif abs(correlation) > 0.5:
                    if correlation > 0:
                        recommendations.append(f"Strong positive correlation for {task_type_str}: consider increasing weight for better performance")
                    else:
                        recommendations.append(f"Strong negative correlation for {task_type_str}: consider decreasing weight")
        
        return recommendations
    
    def _calculate_correlation(self, x: List[float], y: List[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        sum_y2 = sum(y[i] ** 2 for i in range(n))
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator


# Factory functions for creating weight managers

def create_production_weight_manager(config: Optional[LegalRewardSystemConfig] = None) -> TaskDifficultyWeightManager:
    """Create production weight manager"""
    return TaskDifficultyWeightManager(config)


def create_development_weight_manager(config: Optional[LegalRewardSystemConfig] = None) -> TaskDifficultyWeightManager:
    """Create development weight manager with relaxed settings"""
    manager = TaskDifficultyWeightManager(config)
    manager.enable_auto_adjustment = False  # Disable auto-adjustment for development
    return manager


def create_training_weight_manager(config: Optional[LegalRewardSystemConfig] = None) -> TaskDifficultyWeightManager:
    """Create weight manager optimized for training"""
    manager = TaskDifficultyWeightManager(config)
    manager.enable_auto_adjustment = True
    manager.adjustment_frequency = 50  # More frequent adjustments during training
    return manager