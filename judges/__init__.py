"""
Judge Framework for Multi-Task Legal Reward System

This package contains all judge implementations including API-based judges
and real specialized ensembles for production legal evaluation.
"""

# Base judge framework
from .base import (
    BaseJudge,
    BaseJudgeEnsemble,
    JudgeConfig,
    EvaluationContext,
    JudgeType,
    EvaluationStrategy,
    # ComponentStatus,
    # create_production_judge_config,
    # create_development_judge_config,
    create_evaluation_context,
    validate_judge_configuration
)

# API client for all judges
from .api_client import (
    CostOptimizedAPIClient,
    # APIClientConfig,
    # ProviderConfig,
    APIResponse,
    # CostTracking,
    create_production_api_client,
    create_development_api_client,
    # create_cost_optimized_client
)

# General chat ensemble
from .general_chat import (
    EnhancedGeneralChatEnsemble,
    # GeneralChatConfig,
    # GeneralChatJudge,
    create_production_general_chat_ensemble,
    create_development_general_chat_ensemble
)

# REAL SPECIALIZED ENSEMBLES - ADD THESE IMPORTS
from .specialized import (
    # Real ensemble classes
    JudicialReasoningEnsemble,
    PrecedentAnalysisEnsemble,
    OpinionGenerationEnsemble,
    
    # Score classes  
    JudicialReasoningScore,
    PrecedentAnalysisScore,
    OpinionGenerationScore,
    
    # Component enums
    JudicialAnalysisComponent,
    PrecedentAnalysisComponent,
    OpinionGenerationComponent,
    
    # Factory functions
    create_judicial_reasoning_ensemble,
    create_precedent_analysis_ensemble,
    create_opinion_generation_ensemble,
    
    # Utility functions
    create_all_specialized_ensembles,
    get_specialized_ensemble_for_task,
    validate_specialized_ensembles
)

# Export all main components
__all__ = [
    # Base framework
    "BaseJudge",
    "BaseJudgeEnsemble", 
    "JudgeConfig",
    "EvaluationContext",
    "JudgeType",
    "EvaluationStrategy",
    # "ComponentStatus",
    # "create_production_judge_config",
    # "create_development_judge_config",
    "create_evaluation_context",
    "validate_judge_configuration",
    
    # API client
    "CostOptimizedAPIClient",
    # "APIClientConfig",
    # "ProviderConfig",
    "APIResponse", 
    # "CostTracking",
    "create_production_api_client",
    "create_development_api_client",
    # "create_cost_optimized_client",
    
    # General chat
    "EnhancedGeneralChatEnsemble",
    # "GeneralChatConfig",
    # "GeneralChatJudge",
    "create_production_general_chat_ensemble", 
    "create_development_general_chat_ensemble",
    
    # REAL SPECIALIZED ENSEMBLES
    "JudicialReasoningEnsemble",
    "PrecedentAnalysisEnsemble",
    "OpinionGenerationEnsemble",
    
    # Specialized scores
    "JudicialReasoningScore",
    "PrecedentAnalysisScore",
    "OpinionGenerationScore",
    
    # Component enums
    "JudicialAnalysisComponent",
    "PrecedentAnalysisComponent", 
    "OpinionGenerationComponent",
    
    # Specialized factory functions
    "create_judicial_reasoning_ensemble",
    "create_precedent_analysis_ensemble",
    "create_opinion_generation_ensemble",
    
    # Specialized utilities
    "create_all_specialized_ensembles",
    "get_specialized_ensemble_for_task",
    "validate_specialized_ensembles"
]