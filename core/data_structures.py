"""
Core Data Structures for Multi-Task Legal Reward System

This module defines the fundamental data structures used throughout the legal reward system:
- LegalDataPoint: Single legal training data point with task routing information
- EnsembleScore: Score from a judge ensemble with detailed evaluation info  
- HybridEvaluationResult: Result from hybrid evaluation (specialized + general chat)
- RoutedReward: Final reward after routing and weighting

These structures serve as the foundation for all system components and ensure
consistent data handling across judges, routing, and VERL integration.
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod


from .enums import (
    LegalTaskType, LegalDomain, USJurisdiction, USJurisdictionLevel, 
    EvaluationMethod, APIProvider, DEFAULT_LEGAL_TASK_TYPE, 
    DEFAULT_LEGAL_DOMAIN, DEFAULT_US_JURISDICTION
)


@dataclass
class LegalRewardEvaluation:
    """
    Final legal reward evaluation result.
    
    This is the main evaluation result that contains the final reward score
    and comprehensive metadata about the evaluation process.
    """
    reward_score: float
    evaluation_method: str
    task_type: 'LegalTaskType'
    jurisdiction: str
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    evaluation_time: float = field(default_factory=time.time)
    evaluation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def __post_init__(self):
        """Validate reward evaluation after initialization"""
        if not isinstance(self.reward_score, (int, float)):
            raise ValueError("Reward score must be numeric")
        if not 0.0 <= self.reward_score <= 10.0:
            raise ValueError("Reward score must be between 0.0 and 10.0")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")


@dataclass  
class JudgeEvaluation:
    """
    Individual judge evaluation result.
    
    Contains the evaluation from a single judge component with
    detailed scoring and reasoning.
    """
    judge_name: str
    score: float
    reasoning: str = ""
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    evaluation_time: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Validate judge evaluation after initialization"""
        if not isinstance(self.score, (int, float)):
            raise ValueError("Judge score must be numeric")
        if not 0.0 <= self.score <= 10.0:
            raise ValueError("Judge score must be between 0.0 and 10.0")


@dataclass
class EvaluationMetadata:
    """
    Metadata for evaluation tracking and debugging.
    
    Contains comprehensive information about the evaluation process
    for monitoring, debugging, and performance analysis.
    """
    evaluation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    evaluation_type: str = "unknown"
    api_provider: Optional[str] = None
    cache_hit: bool = False
    total_cost: float = 0.0
    execution_time: float = 0.0
    additional_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APIResponse:
    """
    API response wrapper for consistent handling.
    
    Standardizes API responses across different providers
    with cost tracking and metadata.
    """
    content: str
    provider: str
    model: str = "unknown"
    tokens_used: int = 0
    cost: float = 0.0
    response_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


@dataclass
class CostInformation:
    """
    Cost tracking information for API usage.
    
    Tracks costs across providers and operations
    for budget management and optimization.
    """
    provider: str
    operation: str
    tokens_used: int = 0
    cost: float = 0.0
    timestamp: float = field(default_factory=time.time)
    model: str = "unknown"
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class PerformanceMetrics:
    """
    Performance metrics for system monitoring.
    
    Tracks system performance metrics for optimization
    and monitoring purposes.
    """
    operation: str
    execution_time: float
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    success: bool = True
    timestamp: float = field(default_factory=time.time)
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LegalDataPoint:
    """
    Single legal training data point with comprehensive task routing information.
    
    This is the core data structure that flows through the entire system,
    from VERL input to final reward computation. Contains all necessary
    metadata for proper routing, evaluation, and US jurisdiction handling.
    
    Attributes:
        query: User input/legal question
        response: Generated model response (during training) 
        task_type: Legal task type for routing (judicial_reasoning, precedent_analysis, etc.)
        jurisdiction: Jurisdiction context (state name, "federal", or "general")
        legal_domain: Area of law (contract, tort, constitutional, etc.)
        metadata: Additional context and routing information
        data_id: Unique identifier for tracking and logging
        timestamp: Creation timestamp for performance tracking
        
        # US-specific jurisdiction fields
        us_jurisdiction: Parsed US jurisdiction enum (set during inference)
        jurisdiction_level: Federal/state/local level classification
        jurisdiction_inferred: Whether jurisdiction was inferred vs explicit
    """
    
    # Core required fields
    query: str
    response: str
    task_type: LegalTaskType
    jurisdiction: str
    legal_domain: LegalDomain
    
    # Optional metadata and tracking
    metadata: Dict[str, Any] = field(default_factory=dict)
    data_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    
    # US jurisdiction-specific fields (populated during processing)
    us_jurisdiction: Optional[USJurisdiction] = None
    jurisdiction_level: Optional[USJurisdictionLevel] = None
    jurisdiction_inferred: bool = False
    
    def __post_init__(self):
        """Validate and normalize data after initialization"""
        # Basic validation
        if not self.query or not self.response:
            raise ValueError("Query and response are required")
        
        if not self.task_type:
            raise ValueError("Task type is required for routing")
        
        # Normalize string fields
        self.query = self.query.strip()
        self.response = self.response.strip()
        self.jurisdiction = self.jurisdiction.lower().strip()
        
        # Set default values
        if not self.jurisdiction:
            self.jurisdiction = "general"
         
        # Validate enum types
        if isinstance(self.task_type, str):
            try:
                self.task_type = LegalTaskType(self.task_type)
            except ValueError:
                self.task_type = DEFAULT_LEGAL_TASK_TYPE
        
        if isinstance(self.legal_domain, str):
            try:
                self.legal_domain = LegalDomain(self.legal_domain)
            except ValueError:
                self.legal_domain = DEFAULT_LEGAL_DOMAIN
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LegalDataPoint':
        """
        Create LegalDataPoint from dictionary (for VERL integration).
        
        This method handles conversion from various input formats including
        VERL's expected format and internal configuration formats.
        
        Args:
            data: Dictionary containing legal data point information
            
        Returns:
            LegalDataPoint instance
            
        Raises:
            ValueError: If required fields are missing or invalid
        """
        try:
            # Handle different input field names (VERL compatibility)
            query = data.get('query') or data.get('data_source', '')
            response = data.get('response') or data.get('solution_str', '')
            
            # Extract task type with enum conversion
            task_type_str = data.get('task_type', 'general_chat')
            try:
                task_type = LegalTaskType(task_type_str)
            except ValueError:
                task_type = DEFAULT_LEGAL_TASK_TYPE
            
            # Extract jurisdiction with validation
            jurisdiction = data.get('jurisdiction', 'general')
            
            # Extract legal domain with enum conversion
            legal_domain_str = data.get('legal_domain', 'general')
            try:
                legal_domain = LegalDomain(legal_domain_str)
            except ValueError:
                legal_domain = DEFAULT_LEGAL_DOMAIN
            
            # Create instance with core fields
            instance = cls(
                query=query,
                response=response,
                task_type=task_type,
                jurisdiction=jurisdiction,
                legal_domain=legal_domain,
                metadata=data.get('metadata', {}),
                data_id=data.get('data_id', str(uuid.uuid4()))
            )
            
            # Set US jurisdiction fields if provided with enum conversion
            if 'us_jurisdiction' in data:
                if isinstance(data['us_jurisdiction'], str):
                    instance.us_jurisdiction = USJurisdiction.from_string(data['us_jurisdiction'])
                else:
                    instance.us_jurisdiction = data['us_jurisdiction']
                    
            if 'jurisdiction_level' in data:
                if isinstance(data['jurisdiction_level'], str):
                    try:
                        instance.jurisdiction_level = USJurisdictionLevel(data['jurisdiction_level'])
                    except ValueError:
                        instance.jurisdiction_level = None
                else:
                    instance.jurisdiction_level = data['jurisdiction_level']
                    
            if 'jurisdiction_inferred' in data:
                instance.jurisdiction_inferred = data['jurisdiction_inferred']
            
            return instance
            
        except Exception as e:
            raise ValueError(f"Invalid data format for LegalDataPoint: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert LegalDataPoint to dictionary for serialization.
        
        Returns:
            Dictionary representation suitable for JSON serialization
        """
        return {
            'query': self.query,
            'response': self.response,
            'task_type': self.task_type.value,
            'jurisdiction': self.jurisdiction,
            'legal_domain': self.legal_domain.value,
            'metadata': self.metadata,
            'data_id': self.data_id,
            'timestamp': self.timestamp,
            'us_jurisdiction': self.us_jurisdiction.value if self.us_jurisdiction else None,
            'jurisdiction_level': self.jurisdiction_level.value if self.jurisdiction_level else None,
            'jurisdiction_inferred': self.jurisdiction_inferred
        }
    
    def is_jurisdiction_critical(self) -> bool:
        """
        Check if this data point requires specific jurisdiction handling.
        
        Returns:
            True if jurisdiction-specific evaluation is critical for accuracy
        """
        return (
            self.legal_domain.is_jurisdiction_critical() or
            self.jurisdiction != 'general' or
            'jurisdiction_critical' in self.metadata
        )
    
    def get_content_hash(self) -> str:
        """
        Generate content-based hash for caching purposes.
        
        Returns:
            SHA-256 hash of core content fields
        """
        import hashlib
        
        content = f"{self.query}|{self.response}|{self.task_type.value}|{self.jurisdiction}|{self.legal_domain.value}"
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class EnsembleScore:
    """
    Score from a judge ensemble with comprehensive evaluation information.
    
    This structure contains detailed results from any judge ensemble evaluation,
    including specialized legal judges and general chat quality assessment.
    Used for both individual judge results and combined hybrid evaluation.
    
    Attributes:
        task_type: Legal task type that was evaluated
        raw_score: Core evaluation score on 0-10 scale
        confidence: Confidence level in the evaluation (0-1 scale)
        individual_scores: Scores from individual judges within the ensemble
        reasoning: Human-readable explanation of the evaluation
        evaluation_time: Time taken for evaluation (seconds)
        metadata: Additional evaluation context and debug information
        
        # Hybrid evaluation fields
        is_primary_evaluation: True for specialized ensembles, False for chat quality
        ensemble_type: Type of ensemble ("specialized", "general_chat", "jurisdiction")
    """
    
    task_type: LegalTaskType
    raw_score: float  # 0-10 scale from ensemble
    confidence: float  # 0-1 confidence level
    individual_scores: Dict[str, float] = field(default_factory=dict)
    reasoning: str = ""
    evaluation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Hybrid evaluation fields
    is_primary_evaluation: bool = True
    ensemble_type: str = ""  # "specialized", "general_chat", "jurisdiction"
    
    def __post_init__(self):
        """Validate score ranges and normalize data"""
        # Validate score ranges
        if not (0 <= self.raw_score <= 10):
            raise ValueError(f"Raw score must be between 0-10, got {self.raw_score}")
        
        if not (0 <= self.confidence <= 1):
            raise ValueError(f"Confidence must be between 0-1, got {self.confidence}")
        
        # Validate individual scores
        for judge_name, score in self.individual_scores.items():
            if not (0 <= score <= 10):
                raise ValueError(f"Individual score for {judge_name} must be between 0-10, got {score}")
    
    def get_weighted_confidence(self) -> float:
        """
        Calculate confidence weighted by score quality.
        
        Higher scores with high confidence are more reliable than
        low scores with high confidence.
        
        Returns:
            Weighted confidence score (0-1)
        """
        # Normalize score to 0-1 range
        normalized_score = self.raw_score / 10.0
        
        # Weight confidence by score quality
        # High scores with high confidence get full weight
        # Low scores with high confidence get reduced weight
        return self.confidence * (0.5 + 0.5 * normalized_score)
    
    def is_reliable(self) -> bool:
        """
        Check if this evaluation result is reliable enough to use.
        
        Returns:
            True if the score meets reliability thresholds
        """
        return (
            self.confidence >= 0.7 and 
            self.raw_score >= 1.0 and 
            self.evaluation_time < 30.0 and  # Not stuck/timeout
            not self.metadata.get("error", False)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'task_type': self.task_type.value,
            'raw_score': self.raw_score,
            'confidence': self.confidence,
            'individual_scores': self.individual_scores,
            'reasoning': self.reasoning,
            'evaluation_time': self.evaluation_time,
            'metadata': self.metadata,
            'is_primary_evaluation': self.is_primary_evaluation,
            'ensemble_type': self.ensemble_type,
            'weighted_confidence': self.get_weighted_confidence(),
            'is_reliable': self.is_reliable()
        }


@dataclass
class HybridEvaluationResult:
    """
    Result from hybrid evaluation combining specialized and general chat assessment.
    
    This structure represents the core innovation of our system: combining
    specialized legal expertise (70%) with general chat quality (30%) while
    applying jurisdiction compliance as a gating function.
    
    Attributes:
        specialized_score: Score from specialized legal ensemble (if applicable)
        general_chat_score: Score from enhanced general chat ensemble
        jurisdiction_compliance_score: Critical jurisdiction handling score
        hybrid_raw_score: Combined weighted score before task difficulty weighting
        evaluation_method: Method used for evaluation
        weighting_used: Actual weights applied in calculation
        metadata: Additional context and performance information
    """
    
    specialized_score: Optional[EnsembleScore] = None
    general_chat_score: Optional[EnsembleScore] = None
    jurisdiction_compliance_score: float = 10.0  # Default to perfect compliance
    hybrid_raw_score: float = 0.0
    evaluation_method: EvaluationMethod = EvaluationMethod.GENERAL_CHAT_ONLY
    weighting_used: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate hybrid evaluation result"""
        # Ensure at least one score is present
        if self.specialized_score is None and self.general_chat_score is None:
            raise ValueError("At least one evaluation score must be provided")
        
        # Validate jurisdiction compliance score
        if not (0 <= self.jurisdiction_compliance_score <= 10):
            raise ValueError(f"Jurisdiction compliance score must be 0-10, got {self.jurisdiction_compliance_score}")
        
        # Validate hybrid score
        if not (0 <= self.hybrid_raw_score <= 10):
            raise ValueError(f"Hybrid raw score must be 0-10, got {self.hybrid_raw_score}")
    
    def jurisdiction_compliance_passed(self) -> bool:
        """
        Check if jurisdiction compliance requirements are met.
        
        Returns:
            True if jurisdiction compliance score meets minimum threshold
        """
        return self.jurisdiction_compliance_score >= 3.0
    
    def is_hybrid_evaluation(self) -> bool:
        """
        Check if this used true hybrid evaluation (both specialized and chat).
        
        Returns:
            True if both specialized and general chat scores are present
        """
        return self.specialized_score is not None and self.general_chat_score is not None
    
    def get_evaluation_confidence(self) -> float:
        """
        Calculate overall confidence in the hybrid evaluation.
        
        Combines confidence from both evaluation components, weighted by
        their contribution to the final score.
        
        Returns:
            Overall confidence score (0-1)
        """
        if not self.is_hybrid_evaluation():
            # Single evaluation case
            single_score = self.specialized_score or self.general_chat_score
            return single_score.confidence if single_score else 0.0
        
        # Hybrid case - weight by contribution
        specialized_weight = self.weighting_used.get("specialized", 0.7)
        chat_weight = self.weighting_used.get("general_chat", 0.3)
        
        weighted_confidence = (
            self.specialized_score.confidence * specialized_weight +
            self.general_chat_score.confidence * chat_weight
        )
        
        # Apply jurisdiction compliance penalty if failed
        if not self.jurisdiction_compliance_passed():
            weighted_confidence *= 0.5  # Reduce confidence for jurisdiction failures
        
        return weighted_confidence
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary for monitoring and optimization.
        
        Returns:
            Dictionary with key performance metrics
        """
        total_time = 0.0
        if self.specialized_score:
            total_time += self.specialized_score.evaluation_time
        if self.general_chat_score:
            total_time += self.general_chat_score.evaluation_time
        
        return {
            'evaluation_method': self.evaluation_method.value,
            'hybrid_evaluation': self.is_hybrid_evaluation(),
            'jurisdiction_passed': self.jurisdiction_compliance_passed(),
            'total_evaluation_time': total_time,
            'overall_confidence': self.get_evaluation_confidence(),
            'hybrid_raw_score': self.hybrid_raw_score,
            'jurisdiction_score': self.jurisdiction_compliance_score
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'specialized_score': self.specialized_score.to_dict() if self.specialized_score else None,
            'general_chat_score': self.general_chat_score.to_dict() if self.general_chat_score else None,
            'jurisdiction_compliance_score': self.jurisdiction_compliance_score,
            'hybrid_raw_score': self.hybrid_raw_score,
            'evaluation_method': self.evaluation_method.value,
            'weighting_used': self.weighting_used,
            'metadata': self.metadata,
            'performance_summary': self.get_performance_summary()
        }


@dataclass
class RoutedReward:
    """
    Final reward after complete routing, evaluation, and weighting.
    
    This is the ultimate output of our multi-task legal reward system,
    containing the final reward score for VERL training along with
    comprehensive metadata for monitoring and optimization.
    
    Attributes:
        data_id: Unique identifier for tracking
        task_type: Legal task type that was evaluated
        ensemble_score: Primary ensemble score used for reward calculation
        task_weight: Difficulty weight applied for this task type
        final_reward: Ultimate reward score for VERL (ensemble_score * task_weight)
        routing_time: Total time for complete evaluation pipeline
        hybrid_evaluation: Detailed hybrid evaluation results
        jurisdiction_compliance_passed: Whether jurisdiction requirements were met
        metadata: Additional routing and performance information
    """
    
    data_id: str
    task_type: LegalTaskType
    ensemble_score: EnsembleScore
    task_weight: float
    final_reward: float
    routing_time: float
    hybrid_evaluation: Optional[HybridEvaluationResult] = None
    jurisdiction_compliance_passed: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate routed reward data"""
        if self.task_weight <= 0:
            raise ValueError(f"Task weight must be positive, got {self.task_weight}")
        
        if self.routing_time < 0:
            raise ValueError(f"Routing time cannot be negative, got {self.routing_time}")
        
        # Verify final reward calculation
        expected_reward = self.ensemble_score.raw_score * self.task_weight
        if abs(self.final_reward - expected_reward) > 0.01:  # Allow small floating point errors
            raise ValueError(f"Final reward calculation mismatch: expected {expected_reward}, got {self.final_reward}")
    
    def get_cost_effectiveness_score(self) -> float:
        """
        Calculate cost-effectiveness of this evaluation.
        
        Higher scores with shorter routing times are more cost-effective.
        
        Returns:
            Cost-effectiveness score (higher is better)
        """
        if self.routing_time <= 0:
            return float('inf')  # Instant evaluation is infinitely cost-effective
        
        return self.final_reward / self.routing_time
    
    def meets_quality_threshold(self, min_score: float = 5.0, min_confidence: float = 0.5) -> bool:
        """
        Check if this reward meets quality thresholds for training.
        
        Args:
            min_score: Minimum acceptable ensemble score
            min_confidence: Minimum acceptable confidence level
            
        Returns:
            True if quality thresholds are met
        """
        return (
            self.ensemble_score.raw_score >= min_score and
            self.ensemble_score.confidence >= min_confidence and
            self.jurisdiction_compliance_passed and
            not self.metadata.get("error", False)
        )
    
    def get_training_signal_quality(self) -> Dict[str, Any]:
        """
        Assess the quality of this reward as a training signal.
        
        Returns:
            Dictionary with training signal quality metrics
        """
        return {
            'final_reward': self.final_reward,
            'ensemble_confidence': self.ensemble_score.confidence,
            'jurisdiction_compliant': self.jurisdiction_compliance_passed,
            'evaluation_reliable': self.ensemble_score.is_reliable(),
            'cost_effectiveness': self.get_cost_effectiveness_score(),
            'meets_threshold': self.meets_quality_threshold(),
            'task_weight_applied': self.task_weight,
            'routing_efficiency': 1.0 / max(self.routing_time, 0.001)  # Avoid division by zero
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization and logging"""
        return {
            'data_id': self.data_id,
            'task_type': self.task_type.value,
            'ensemble_score': self.ensemble_score.to_dict(),
            'task_weight': self.task_weight,
            'final_reward': self.final_reward,
            'routing_time': self.routing_time,
            'hybrid_evaluation': self.hybrid_evaluation.to_dict() if self.hybrid_evaluation else None,
            'jurisdiction_compliance_passed': self.jurisdiction_compliance_passed,
            'metadata': self.metadata,
            'training_signal_quality': self.get_training_signal_quality()
        }


# Utility functions for data structure operations

def validate_score_range(score: float, min_val: float = 0.0, max_val: float = 10.0, field_name: str = "score") -> float:
    """
    Validate that a score falls within the expected range.
    
    Args:
        score: Score value to validate
        min_val: Minimum acceptable value
        max_val: Maximum acceptable value
        field_name: Name of field for error messages
        
    Returns:
        Validated score
        
    Raises:
        ValueError: If score is outside acceptable range
    """
    if not (min_val <= score <= max_val):
        raise ValueError(f"{field_name} must be between {min_val}-{max_val}, got {score}")
    return score


def create_fallback_ensemble_score(task_type: LegalTaskType, error_context: str = "unknown_error") -> EnsembleScore:
    """
    Create a safe fallback ensemble score for error cases.
    
    Args:
        task_type: Task type being evaluated
        error_context: Context about what went wrong
        
    Returns:
        Safe fallback EnsembleScore with neutral values
    """
    return EnsembleScore(
        task_type=task_type,
        raw_score=5.0,  # Neutral score
        confidence=0.1,  # Low confidence to indicate uncertainty
        individual_scores={},
        reasoning=f"Fallback score due to {error_context}",
        evaluation_time=0.0,
        metadata={
            "error": True,
            "error_context": error_context,
            "fallback_score": True
        },
        is_primary_evaluation=False,
        ensemble_type="fallback"
    )


def aggregate_ensemble_scores(scores: List[EnsembleScore], weights: Optional[Dict[str, float]] = None) -> EnsembleScore:
    """
    Aggregate multiple ensemble scores into a single combined score.
    
    Args:
        scores: List of EnsembleScore objects to combine
        weights: Optional weights for each score (by ensemble_type)
        
    Returns:
        Combined EnsembleScore
        
    Raises:
        ValueError: If no scores provided or weights don't match
    """
    if not scores:
        raise ValueError("At least one score must be provided for aggregation")
    
    if len(scores) == 1:
        return scores[0]  # No aggregation needed
    
    # Use equal weights if none provided
    if weights is None:
        weight_per_score = 1.0 / len(scores)
        weights = {score.ensemble_type: weight_per_score for score in scores}
    
    # Calculate weighted averages
    total_weight = 0.0
    weighted_score = 0.0
    weighted_confidence = 0.0
    total_time = 0.0
    combined_individual_scores = {}
    combined_reasoning_parts = []
    combined_metadata = {}
    
    for score in scores:
        weight = weights.get(score.ensemble_type, 0.0)
        total_weight += weight
        
        weighted_score += score.raw_score * weight
        weighted_confidence += score.confidence * weight
        total_time += score.evaluation_time
        
        # Combine individual scores with prefixes
        for judge_name, judge_score in score.individual_scores.items():
            prefixed_name = f"{score.ensemble_type}_{judge_name}"
            combined_individual_scores[prefixed_name] = judge_score
        
        # Collect reasoning
        if score.reasoning:
            combined_reasoning_parts.append(f"{score.ensemble_type}: {score.reasoning}")
        
        # Merge metadata
        combined_metadata.update(score.metadata)
    
    # Normalize by total weight
    if total_weight > 0:
        weighted_score /= total_weight
        weighted_confidence /= total_weight
    
    return EnsembleScore(
        task_type=scores[0].task_type,  # Assume all scores are for same task
        raw_score=weighted_score,
        confidence=weighted_confidence,
        individual_scores=combined_individual_scores,
        reasoning=" | ".join(combined_reasoning_parts),
        evaluation_time=total_time,
        metadata={
            **combined_metadata,
            "aggregated": True,
            "source_scores": len(scores),
            "weights_used": weights
        },
        is_primary_evaluation=any(score.is_primary_evaluation for score in scores),
        ensemble_type="aggregated"
    )


# Type aliases for better code documentation
ScoreDict = Dict[str, float]  # Individual judge scores
MetadataDict = Dict[str, Any]  # Flexible metadata container
JurisdictionInfo = Dict[str, str]  # Jurisdiction-related information