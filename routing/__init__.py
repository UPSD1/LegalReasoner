"""
Routing Package for Multi-Task Legal Reward System

This package provides the core routing and evaluation logic including:
- Hybrid evaluation system (70% specialized + 30% general chat)
- Task difficulty weight management
- Multi-task legal reward router (main orchestration component)
"""

# Import router components - Make sure RouterConfig and RouterMode are imported first
from routing.router import (
    # Configuration classes
    RouterConfig,
    RouterMode,
    create_production_router_config,
    create_development_router_config,
    create_cost_optimized_router_config,
    create_high_accuracy_router_config,
    
    # Main router class
    MultiTaskLegalRewardRouter,
    
    # Evaluation classes
    RouterEvaluationResult,
    EvaluationRequest,
    
    # Factory functions
    create_production_router,
    create_development_router,
    create_cost_optimized_router,
    create_high_accuracy_router
)

# Hybrid Evaluation System
from routing.hybrid_evaluation import (
    HybridEvaluationEngine,
    HybridEvaluationResult,
    EvaluationMode,
    SpecializedEvaluationStatus,
    HybridEvaluationConfig,
    create_production_hybrid_engine
)

# Task Weight Management
from routing.task_weights import (
    TaskDifficultyWeightManager,
    TaskPerformanceData,
    WeightChangeRecord,
    create_production_weight_manager,
    create_development_weight_manager,
    create_training_weight_manager,
)

# Package metadata
__version__ = "1.0.0"
__author__ = "Legal AI Development Team"  
__description__ = "Routing and Evaluation Logic for Multi-Task Legal Reward System"

# Main exports - RouterConfig and RouterMode are now first in the list
__all__ = [
    # Configuration (FIRST - most important for other modules)
    "RouterConfig",
    "RouterMode", 
    "create_production_router_config",
    "create_development_router_config",
    "create_cost_optimized_router_config",
    "create_high_accuracy_router_config",
    
    # Main Router
    "MultiTaskLegalRewardRouter",
    "RouterEvaluationResult", 
    "EvaluationRequest", 
    "create_production_router",
    "create_development_router",
    "create_cost_optimized_router",
    "create_high_accuracy_router",
    
    # Hybrid Evaluation
    "HybridEvaluationEngine",
    "HybridEvaluationResult",
    "EvaluationMode",
    "SpecializedEvaluationStatus", 
    "HybridEvaluationConfig",
    "create_production_hybrid_engine",
    
    # Task Weight Management
    "TaskDifficultyWeightManager",
    "TaskPerformanceData",
    "WeightChangeRecord",
    "create_production_weight_manager",
    "create_development_weight_manager", 
    "create_training_weight_manager"
]

def validate_routing_system() -> dict:
    """
    Validate that the routing system is working correctly.
    
    Returns:
        Dictionary with validation results
    """
    
    validation_results = {
        "router_config_available": False,
        "router_mode_available": False,
        "router_class_available": False,
        "hybrid_evaluation_available": False,
        "weight_manager_available": False,
        "all_components_available": False
    }
    
    try:
        # Test RouterConfig
        config = RouterConfig()
        validation_results["router_config_available"] = True
        
        # Test RouterMode
        mode = RouterMode.PRODUCTION
        validation_results["router_mode_available"] = True
        
        # Test router creation
        router = MultiTaskLegalRewardRouter(config)
        validation_results["router_class_available"] = True
        
        # Test hybrid evaluation
        hybrid_engine = HybridEvaluationEngine()
        validation_results["hybrid_evaluation_available"] = True
        
        # Test weight manager
        weight_manager = TaskDifficultyWeightManager()
        validation_results["weight_manager_available"] = True
        
        # All components working
        validation_results["all_components_available"] = all(validation_results.values())
        
    except Exception as e:
        validation_results["error"] = str(e)
    
    return validation_results