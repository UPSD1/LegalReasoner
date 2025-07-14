"""
Specialized Judge Ensembles Package - Real Implementations

This package contains real API-based specialized legal evaluation ensembles that provide the 
70% specialized component of the hybrid evaluation system. Each ensemble is 
designed to assess specific aspects of legal reasoning and writing at a 
professional level with US jurisdiction awareness.

Real Ensembles:
- JudicialReasoningEnsemble: Formal judicial analysis, FIRAC structure, legal doctrine
- PrecedentAnalysisEnsemble: Case law analysis, analogical reasoning, precedent hierarchy  
- OpinionGenerationEnsemble: Legal advocacy, persuasive writing, client representation

These ensembles work together with the Enhanced General Chat Ensemble (30%) 
to provide comprehensive legal evaluation through the hybrid system.
"""

# Import all real specialized ensembles
from judges.specialized.judicial_reasoning import (
    JudicialReasoningEnsemble,
    JudicialReasoningScore,
    JudicialAnalysisComponent,
    create_judicial_reasoning_ensemble
)

from judges.specialized.precedent_analysis import (
    PrecedentAnalysisEnsemble,
    PrecedentAnalysisScore,
    PrecedentAnalysisComponent,
    create_precedent_analysis_ensemble
)

from judges.specialized.opinion_generation import (
    OpinionGenerationEnsemble,
    OpinionGenerationScore,
    OpinionGenerationComponent,
    create_opinion_generation_ensemble
)

# Import core components for type hints
from typing import Dict, Optional
from core import LegalTaskType
from config import LegalRewardSystemConfig
from judges.api_client import CostOptimizedAPIClient
from judges.base import BaseJudgeEnsemble

# Package version and metadata
__version__ = "1.0.0"
__author__ = "Legal Reward System Team"
__description__ = "Real specialized legal evaluation ensembles for hybrid reward system"

# Main exports
__all__ = [
    # Real ensemble classes
    "JudicialReasoningEnsemble",
    "PrecedentAnalysisEnsemble", 
    "OpinionGenerationEnsemble",
    
    # Score classes
    "JudicialReasoningScore",
    "PrecedentAnalysisScore",
    "OpinionGenerationScore",
    
    # Component enums
    "JudicialAnalysisComponent",
    "PrecedentAnalysisComponent",
    "OpinionGenerationComponent",
    
    # Factory functions
    "create_judicial_reasoning_ensemble",
    "create_precedent_analysis_ensemble",
    "create_opinion_generation_ensemble",
    
    # Utility functions
    "create_all_specialized_ensembles",
    "get_specialized_ensemble_for_task",
    "validate_specialized_ensembles"
]


def create_all_specialized_ensembles(config: LegalRewardSystemConfig,
                                   api_client: Optional[CostOptimizedAPIClient] = None) -> Dict[LegalTaskType, BaseJudgeEnsemble]:
    """
    Create all specialized ensembles for production use.
    
    Args:
        config: Legal reward system configuration
        api_client: Optional shared API client for cost optimization
        
    Returns:
        Dictionary mapping task types to their specialized ensembles
    """
    
    ensembles = {}
    
    # Create shared API client if not provided
    if api_client is None:
        api_client = CostOptimizedAPIClient(config)
    
    # Create all specialized ensembles
    ensembles[LegalTaskType.JUDICIAL_REASONING] = create_judicial_reasoning_ensemble(config, api_client)
    ensembles[LegalTaskType.PRECEDENT_ANALYSIS] = create_precedent_analysis_ensemble(config, api_client)
    ensembles[LegalTaskType.OPINION_GENERATION] = create_opinion_generation_ensemble(config, api_client)
    
    return ensembles


def get_specialized_ensemble_for_task(task_type: LegalTaskType,
                                    config: LegalRewardSystemConfig,
                                    api_client: Optional[CostOptimizedAPIClient] = None) -> BaseJudgeEnsemble:
    """
    Get specialized ensemble for a specific task type.
    
    Args:
        task_type: Legal task type
        config: Legal reward system configuration
        api_client: Optional API client
        
    Returns:
        Specialized ensemble for the task type
        
    Raises:
        ValueError: If task type doesn't have a specialized ensemble
    """
    
    if api_client is None:
        api_client = CostOptimizedAPIClient(config)
    
    ensemble_factories = {
        LegalTaskType.JUDICIAL_REASONING: create_judicial_reasoning_ensemble,
        LegalTaskType.PRECEDENT_ANALYSIS: create_precedent_analysis_ensemble,
        LegalTaskType.OPINION_GENERATION: create_opinion_generation_ensemble
    }
    
    if task_type not in ensemble_factories:
        raise ValueError(f"No specialized ensemble available for task type: {task_type.value}")
    
    return ensemble_factories[task_type](config, api_client)


def validate_specialized_ensembles(ensembles: Dict[LegalTaskType, BaseJudgeEnsemble]) -> Dict[str, bool]:
    """
    Validate specialized ensembles.
    
    Args:
        ensembles: Dictionary of ensembles to validate
        
    Returns:
        Dictionary with validation results for each ensemble
    """
    
    validation_results = {}
    
    required_ensembles = {
        LegalTaskType.JUDICIAL_REASONING: "JudicialReasoningEnsemble",
        LegalTaskType.PRECEDENT_ANALYSIS: "PrecedentAnalysisEnsemble", 
        LegalTaskType.OPINION_GENERATION: "OpinionGenerationEnsemble"
    }
    
    for task_type, expected_class_name in required_ensembles.items():
        if task_type in ensembles:
            ensemble = ensembles[task_type]
            
            # Check if it's the correct type
            is_correct_type = expected_class_name in str(type(ensemble).__name__)
            
            # Check if it has required methods
            has_evaluate_method = hasattr(ensemble, 'evaluate_response')
            has_performance_method = hasattr(ensemble, 'get_performance_analytics')
            
            # Check if it's NOT a mock (for real implementations)
            is_real_implementation = 'Mock' not in str(type(ensemble).__name__)
            
            validation_results[task_type.value] = (
                is_correct_type and 
                has_evaluate_method and 
                has_performance_method and
                is_real_implementation
            )
        else:
            validation_results[task_type.value] = False
    
    return validation_results


def get_ensemble_performance_summary(ensembles: Dict[LegalTaskType, BaseJudgeEnsemble]) -> Dict[str, Dict[str, any]]:
    """
    Get performance summary for all specialized ensembles.
    
    Args:
        ensembles: Dictionary of ensembles
        
    Returns:
        Performance summary for each ensemble
    """
    
    performance_summary = {}
    
    for task_type, ensemble in ensembles.items():
        if hasattr(ensemble, 'get_performance_analytics'):
            performance_summary[task_type.value] = ensemble.get_performance_analytics()
        else:
            performance_summary[task_type.value] = {
                "error": "Ensemble does not support performance analytics"
            }
    
    return performance_summary