"""
Factory Functions and System Setup for Multi-Task Legal Reward System

This module provides comprehensive factory functions for creating and configuring
the legal reward system in different environments. Handles system initialization,
validation, and environment-specific optimization for production, development,
and testing scenarios.

Key Features:
- Environment-specific router creation (production, development, test)
- Comprehensive system validation and health checks
- Automated component initialization and integration
- Performance optimization and monitoring setup
- Cost tracking and budget management
- Graceful error handling and recovery mechanisms
- Extensive logging and debugging support

This is the primary module for system setup and provides the foundation
for reliable deployment across different environments.
"""

import os
import time
import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

# Import core components
from .core import (
    LegalTaskType, USJurisdiction, LegalDomain,
    LegalRewardSystemError, create_error_context
)

# Import routing system
from .routing import (
    MultiTaskLegalRewardRouter, RouterConfig, RouterMode,
    TaskDifficultyWeightManager, HybridEvaluationEngine,
    create_production_router, create_development_router, 
    create_cost_optimized_router, create_high_accuracy_router,
    create_production_weight_manager, create_development_weight_manager,
    create_training_weight_manager
)

# Import jurisdiction system
from .jurisdiction import (
    USJurisdictionInferenceEngine, JurisdictionComplianceJudge,
    create_production_inference_engine, create_production_compliance_judge
)

# Import judge system
from .judges import (
    EnhancedGeneralChatEnsemble, CostOptimizedAPIClient,
    create_production_general_chat_ensemble, create_production_api_client
)

# Import configuration and utilities
from .config import (
    LegalRewardSystemConfig, create_production_config, 
    create_development_config, create_test_config
)
from .utils import (
    get_legal_logger, MultiStrategyLegalRewardCache, MultiProviderRateLimiter,
    create_aggressive_cache, create_production_rate_limiter
)

# Import VERL integration
from .verl_integration import (
    VERLLegalRewardFunction, create_production_verl_function,
    create_development_verl_function, create_training_verl_function
)


class SystemEnvironment(Enum):
    """System deployment environments"""
    PRODUCTION = "production"
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    TRAINING = "training"


class ComponentStatus(Enum):
    """Status of system components"""
    HEALTHY = "healthy"
    WARNING = "warning"
    ERROR = "error"
    NOT_AVAILABLE = "not_available"


@dataclass
class SystemValidationResult:
    """Results of system validation"""
    is_valid: bool
    environment: SystemEnvironment
    components_status: Dict[str, ComponentStatus] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    setup_timestamp: float = field(default_factory=time.time)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive validation summary"""
        healthy_components = sum(1 for status in self.components_status.values() if status == ComponentStatus.HEALTHY)
        total_components = len(self.components_status)
        
        return {
            "is_valid": self.is_valid,
            "environment": self.environment.value,
            "component_health": f"{healthy_components}/{total_components}",
            "health_percentage": (healthy_components / total_components * 100) if total_components > 0 else 0,
            "warnings_count": len(self.warnings),
            "errors_count": len(self.errors),
            "setup_time": time.time() - self.setup_timestamp
        }


class LegalRewardSystemFactory:
    """
    Comprehensive factory for creating and configuring legal reward systems.
    
    Provides environment-specific system creation with automatic component
    initialization, validation, and optimization.
    """
    
    def __init__(self):
        self.logger = get_legal_logger("system_factory")
        self._creation_cache = {}
        self._validation_cache = {}
    
    def create_system(self, 
                     environment: SystemEnvironment,
                     config: Optional[LegalRewardSystemConfig] = None,
                     enable_caching: bool = True,
                     enable_cost_tracking: bool = True,
                     custom_components: Optional[Dict[str, Any]] = None) -> Tuple[MultiTaskLegalRewardRouter, SystemValidationResult]:
        """
        Create complete legal reward system for specified environment.
        
        Args:
            environment: Target deployment environment
            config: System configuration (optional)
            enable_caching: Enable caching systems
            enable_cost_tracking: Enable cost tracking and optimization
            custom_components: Custom component overrides
            
        Returns:
            Tuple of (configured router, validation result)
        """
        
        start_time = time.time()
        self.logger.info(f"Creating legal reward system for {environment.value} environment")
        
        try:
            # Create environment-specific configuration
            if config is None:
                config = self._create_environment_config(environment)
            
            # Create router based on environment
            router = self._create_environment_router(environment, config, enable_caching, enable_cost_tracking)
            
            # Register components
            self._register_system_components(router, environment, config, custom_components)
            
            # Validate system
            validation_result = self.validate_system(router, environment)
            
            # Log setup completion
            setup_time = time.time() - start_time
            self.logger.info(f"System creation completed in {setup_time:.2f}s: {validation_result.get_summary()}")
            
            return router, validation_result
            
        except Exception as e:
            self.logger.error(f"System creation failed: {e}")
            # Create error validation result
            validation_result = SystemValidationResult(
                is_valid=False,
                environment=environment,
                errors=[f"System creation failed: {str(e)}"]
            )
            
            # Return minimal router for fallback
            fallback_router = self._create_fallback_router()
            return fallback_router, validation_result
    
    def create_production_system(self, 
                               config: Optional[LegalRewardSystemConfig] = None) -> Tuple[MultiTaskLegalRewardRouter, SystemValidationResult]:
        """Create production-ready system with full optimization"""
        
        if config is None:
            config = create_production_config()
        
        return self.create_system(
            SystemEnvironment.PRODUCTION,
            config=config,
            enable_caching=True,
            enable_cost_tracking=True
        )
    
    def create_development_system(self, 
                                config: Optional[LegalRewardSystemConfig] = None) -> Tuple[MultiTaskLegalRewardRouter, SystemValidationResult]:
        """Create development system with debugging features"""
        
        if config is None:
            config = create_development_config()
        
        return self.create_system(
            SystemEnvironment.DEVELOPMENT,
            config=config,
            enable_caching=False,  # Disable caching for development
            enable_cost_tracking=False
        )
    
    def create_training_system(self, 
                             config: LegalRewardSystemConfig) -> Tuple[MultiTaskLegalRewardRouter, SystemValidationResult]:
        """Create system optimized for VERL training"""
        
        return self.create_system(
            SystemEnvironment.TRAINING,
            config=config,
            enable_caching=True,
            enable_cost_tracking=True
        )
    
    def create_testing_system(self, 
                            config: Optional[LegalRewardSystemConfig] = None) -> Tuple[MultiTaskLegalRewardRouter, SystemValidationResult]:
        """Create system for automated testing"""
        
        if config is None:
            config = create_test_config()
        
        return self.create_system(
            SystemEnvironment.TESTING,
            config=config,
            enable_caching=False,
            enable_cost_tracking=False
        )
    
    def validate_system(self, 
                       router: MultiTaskLegalRewardRouter, 
                       environment: SystemEnvironment) -> SystemValidationResult:
        """
        Comprehensive system validation.
        
        Args:
            router: Router to validate
            environment: Target environment
            
        Returns:
            Detailed validation results
        """
        
        validation_result = SystemValidationResult(
            is_valid=True,
            environment=environment
        )
        
        try:
            # Validate core components
            self._validate_core_components(router, validation_result)
            
            # Validate judge ensembles
            self._validate_judge_ensembles(router, validation_result)
            
            # Validate jurisdiction system
            self._validate_jurisdiction_system(router, validation_result)
            
            # Validate configuration
            self._validate_configuration(router, validation_result)
            
            # Validate performance metrics
            self._validate_performance_metrics(router, validation_result)
            
            # Check for critical errors
            critical_errors = [error for error in validation_result.errors if "critical" in error.lower()]
            if critical_errors:
                validation_result.is_valid = False
            
            # Generate recommendations
            self._generate_recommendations(validation_result)
            
        except Exception as e:
            validation_result.is_valid = False
            validation_result.errors.append(f"Validation failed: {str(e)}")
            self.logger.error(f"System validation error: {e}")
        
        return validation_result
    
    def _create_environment_config(self, environment: SystemEnvironment) -> LegalRewardSystemConfig:
        """Create environment-specific configuration"""
        
        if environment == SystemEnvironment.PRODUCTION:
            return create_production_config()
        elif environment == SystemEnvironment.DEVELOPMENT:
            return create_development_config()
        elif environment in [SystemEnvironment.TESTING, SystemEnvironment.STAGING]:
            return create_test_config()
        elif environment == SystemEnvironment.TRAINING:
            return create_production_config()  # Use production config for training
        else:
            self.logger.warning(f"Unknown environment {environment}, using production config")
            return create_production_config()
    
    def _create_environment_router(self, 
                                 environment: SystemEnvironment,
                                 config: LegalRewardSystemConfig,
                                 enable_caching: bool,
                                 enable_cost_tracking: bool) -> MultiTaskLegalRewardRouter:
        """Create environment-specific router"""
        
        if environment == SystemEnvironment.PRODUCTION:
            return create_production_router(config)
        elif environment == SystemEnvironment.DEVELOPMENT:
            return create_development_router(config)
        elif environment == SystemEnvironment.TESTING:
            # Create router with minimal settings for testing
            router_config = RouterConfig(
                router_mode=RouterMode.DEVELOPMENT,
                enable_caching=enable_caching,
                enable_cost_optimization=enable_cost_tracking,
                max_concurrent_evaluations=3,
                evaluation_timeout_seconds=30.0
            )
            return MultiTaskLegalRewardRouter(router_config, config)
        elif environment == SystemEnvironment.TRAINING:
            return create_production_router(config)
        elif environment == SystemEnvironment.STAGING:
            return create_cost_optimized_router(config)
        else:
            self.logger.warning(f"Unknown environment {environment}, using development router")
            return create_development_router(config)
    
    def _register_system_components(self, 
                                  router: MultiTaskLegalRewardRouter,
                                  environment: SystemEnvironment,
                                  config: LegalRewardSystemConfig,
                                  custom_components: Optional[Dict[str, Any]]):
        """Register all system components"""
        
        try:
            # Register weight manager
            if environment == SystemEnvironment.PRODUCTION:
                weight_manager = create_production_weight_manager(config)
            elif environment == SystemEnvironment.TRAINING:
                weight_manager = create_training_weight_manager(config)
            else:
                weight_manager = create_development_weight_manager()
            
            # Set weight manager on router if it has the method
            if hasattr(router, 'set_weight_manager'):
                router.set_weight_manager(weight_manager)
            
            # Register additional components based on custom overrides
            if custom_components:
                for component_name, component in custom_components.items():
                    if hasattr(router, f'set_{component_name}'):
                        getattr(router, f'set_{component_name}')(component)
            
            self.logger.info(f"System components registered for {environment.value}")
            
        except Exception as e:
            self.logger.error(f"Component registration failed: {e}")
            raise LegalRewardSystemError(
                f"Failed to register system components: {e}",
                error_context=create_error_context("factory", "register_components")
            )
    
    def _create_fallback_router(self) -> MultiTaskLegalRewardRouter:
        """Create minimal fallback router for error cases"""
        
        try:
            config = create_development_config()
            router_config = RouterConfig(
                router_mode=RouterMode.DEVELOPMENT,
                enable_caching=False,
                enable_cost_optimization=False,
                max_concurrent_evaluations=1,
                evaluation_timeout_seconds=60.0,
                fallback_to_general_chat=True
            )
            return MultiTaskLegalRewardRouter(router_config, config)
            
        except Exception as e:
            self.logger.critical(f"Fallback router creation failed: {e}")
            raise LegalRewardSystemError(
                f"Critical system failure: cannot create fallback router: {e}",
                error_context=create_error_context("factory", "fallback_router")
            )
    
    def _validate_core_components(self, router: MultiTaskLegalRewardRouter, result: SystemValidationResult):
        """Validate core router components"""
        
        # Check router initialization
        if router is None:
            result.components_status["router"] = ComponentStatus.ERROR
            result.errors.append("Critical: Router is None")
            return
        
        result.components_status["router"] = ComponentStatus.HEALTHY
        
        # Check configuration
        if hasattr(router, 'config') and router.config:
            result.components_status["router_config"] = ComponentStatus.HEALTHY
        else:
            result.components_status["router_config"] = ComponentStatus.WARNING
            result.warnings.append("Router configuration missing or incomplete")
        
        # Check system configuration
        if hasattr(router, 'system_config') and router.system_config:
            result.components_status["system_config"] = ComponentStatus.HEALTHY
        else:
            result.components_status["system_config"] = ComponentStatus.WARNING
            result.warnings.append("System configuration missing")
        
        # Check logger
        if hasattr(router, 'logger') and router.logger:
            result.components_status["logging"] = ComponentStatus.HEALTHY
        else:
            result.components_status["logging"] = ComponentStatus.WARNING
            result.warnings.append("Logging not properly initialized")
    
    def _validate_judge_ensembles(self, router: MultiTaskLegalRewardRouter, result: SystemValidationResult):
        """Validate judge ensemble registration and functionality"""
        
        # Check if router has judge ensembles
        if not hasattr(router, 'judge_ensembles'):
            result.components_status["judge_ensembles"] = ComponentStatus.ERROR
            result.errors.append("Critical: Router missing judge_ensembles attribute")
            return
        
        # Check ensemble registration
        registered_ensembles = getattr(router, 'judge_ensembles', {})
        expected_tasks = list(LegalTaskType)
        
        missing_ensembles = []
        for task_type in expected_tasks:
            if task_type not in registered_ensembles:
                missing_ensembles.append(task_type.value)
        
        if missing_ensembles:
            result.components_status["judge_ensembles"] = ComponentStatus.WARNING
            result.warnings.append(f"Missing judge ensembles for: {', '.join(missing_ensembles)}")
        else:
            result.components_status["judge_ensembles"] = ComponentStatus.HEALTHY
        
        # Check general chat ensemble
        if hasattr(router, 'general_chat_ensemble'):
            if router.general_chat_ensemble:
                result.components_status["general_chat"] = ComponentStatus.HEALTHY
            else:
                result.components_status["general_chat"] = ComponentStatus.WARNING
                result.warnings.append("General chat ensemble not initialized")
        
        # Check API client
        if hasattr(router, 'api_client'):
            if router.api_client:
                result.components_status["api_client"] = ComponentStatus.HEALTHY
            else:
                result.components_status["api_client"] = ComponentStatus.WARNING
                result.warnings.append("API client not initialized")
    
    def _validate_jurisdiction_system(self, router: MultiTaskLegalRewardRouter, result: SystemValidationResult):
        """Validate jurisdiction inference and compliance components"""
        
        # Check jurisdiction inference engine
        if hasattr(router, 'jurisdiction_inference_engine'):
            if router.jurisdiction_inference_engine:
                result.components_status["jurisdiction_inference"] = ComponentStatus.HEALTHY
            else:
                result.components_status["jurisdiction_inference"] = ComponentStatus.WARNING
                result.warnings.append("Jurisdiction inference engine not available")
        
        # Check compliance judge
        if hasattr(router, 'compliance_judge'):
            if router.compliance_judge:
                result.components_status["compliance_judge"] = ComponentStatus.HEALTHY
            else:
                result.components_status["compliance_judge"] = ComponentStatus.WARNING
                result.warnings.append("Jurisdiction compliance judge not available")
        
        # Check hybrid evaluation
        if hasattr(router, 'hybrid_engine'):
            if router.hybrid_engine:
                result.components_status["hybrid_evaluation"] = ComponentStatus.HEALTHY
            else:
                result.components_status["hybrid_evaluation"] = ComponentStatus.WARNING
                result.warnings.append("Hybrid evaluation engine not available")
    
    def _validate_configuration(self, router: MultiTaskLegalRewardRouter, result: SystemValidationResult):
        """Validate system configuration"""
        
        try:
            # Check router configuration
            if hasattr(router, 'config'):
                config_issues = router.config.validate_config()
                if config_issues:
                    result.components_status["configuration"] = ComponentStatus.WARNING
                    result.warnings.extend([f"Config: {issue}" for issue in config_issues])
                else:
                    result.components_status["configuration"] = ComponentStatus.HEALTHY
            
            # Check API keys if system config available
            if hasattr(router, 'system_config') and router.system_config:
                missing_keys = []
                
                # Check for required API keys (these would be environment variables)
                required_keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
                for key in required_keys:
                    if not os.getenv(key):
                        missing_keys.append(key)
                
                if missing_keys:
                    result.components_status["api_keys"] = ComponentStatus.WARNING
                    result.warnings.append(f"Missing API keys: {', '.join(missing_keys)}")
                else:
                    result.components_status["api_keys"] = ComponentStatus.HEALTHY
            
        except Exception as e:
            result.components_status["configuration"] = ComponentStatus.ERROR
            result.errors.append(f"Configuration validation failed: {e}")
    
    def _validate_performance_metrics(self, router: MultiTaskLegalRewardRouter, result: SystemValidationResult):
        """Validate performance metrics and system health"""
        
        try:
            # Get router statistics if available
            if hasattr(router, 'get_statistics'):
                stats = router.get_statistics()
                result.performance_metrics.update(stats)
                
                # Check error rates
                if 'error_rate' in stats:
                    error_rate = stats['error_rate']
                    if error_rate > 0.1:  # More than 10% errors
                        result.components_status["error_rate"] = ComponentStatus.WARNING
                        result.warnings.append(f"High error rate: {error_rate:.1%}")
                    else:
                        result.components_status["error_rate"] = ComponentStatus.HEALTHY
                
                # Check evaluation counts
                if 'total_evaluations' in stats:
                    total_evals = stats['total_evaluations']
                    result.performance_metrics["total_evaluations"] = total_evals
            
            # Check cache performance if available
            if hasattr(router, 'cache') and router.cache:
                result.components_status["caching"] = ComponentStatus.HEALTHY
            else:
                result.components_status["caching"] = ComponentStatus.NOT_AVAILABLE
            
            # Check rate limiting
            if hasattr(router, 'rate_limiter') and router.rate_limiter:
                result.components_status["rate_limiting"] = ComponentStatus.HEALTHY
            else:
                result.components_status["rate_limiting"] = ComponentStatus.NOT_AVAILABLE
            
        except Exception as e:
            result.components_status["performance_metrics"] = ComponentStatus.ERROR
            result.errors.append(f"Performance metrics validation failed: {e}")
    
    def _generate_recommendations(self, result: SystemValidationResult):
        """Generate actionable recommendations based on validation results"""
        
        # Recommendations based on warnings
        if result.warnings:
            if any("API key" in warning for warning in result.warnings):
                result.recommendations.append("Set up missing API keys in environment variables")
            
            if any("ensemble" in warning for warning in result.warnings):
                result.recommendations.append("Ensure all judge ensembles are properly registered")
            
            if any("configuration" in warning for warning in result.warnings):
                result.recommendations.append("Review and fix configuration issues")
        
        # Recommendations based on component status
        error_components = [name for name, status in result.components_status.items() if status == ComponentStatus.ERROR]
        if error_components:
            result.recommendations.append(f"Fix critical component errors: {', '.join(error_components)}")
        
        warning_components = [name for name, status in result.components_status.items() if status == ComponentStatus.WARNING]
        if warning_components:
            result.recommendations.append(f"Address component warnings: {', '.join(warning_components)}")
        
        # Environment-specific recommendations
        if result.environment == SystemEnvironment.PRODUCTION:
            if ComponentStatus.WARNING in result.components_status.values():
                result.recommendations.append("Resolve all warnings before production deployment")
        
        elif result.environment == SystemEnvironment.DEVELOPMENT:
            if ComponentStatus.NOT_AVAILABLE in result.components_status.values():
                result.recommendations.append("Some components unavailable - normal for development")


# Global factory instance
_system_factory = LegalRewardSystemFactory()


# High-level factory functions for easy use
def create_production_legal_reward_router(config: Optional[LegalRewardSystemConfig] = None) -> MultiTaskLegalRewardRouter:
    """Create production-ready legal reward router with real judge ensembles"""
    
    if config is None:
        config = create_production_config()
    
    # FIX: Create RouterConfig for the router constructor
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
        max_cost_per_evaluation=0.50,
        aggressive_cost_optimization=True
    )
    
    # FIX: Use correct constructor interface - RouterConfig first, then LegalRewardSystemConfig
    router = MultiTaskLegalRewardRouter(router_config, config)
    
    # Create shared API client for cost optimization
    api_client = CostOptimizedAPIClient(config)

    logger = get_legal_logger("production_router_factory")
    
    # Register REAL judge ensembles
    try:
        # Import real ensemble implementations
        from judges.general_chat import EnhancedGeneralChatEnsemble
        from judges.specialized.judicial_reasoning import JudicialReasoningEnsemble
        from judges.specialized.precedent_analysis import PrecedentAnalysisEnsemble
        from judges.specialized.opinion_generation import OpinionGenerationEnsemble
        
        # Create and register real ensembles
        ensembles = {}
        
        # General chat ensemble (always real)
        general_chat_ensemble = EnhancedGeneralChatEnsemble(config, api_client)
        ensembles[LegalTaskType.GENERAL_CHAT] = general_chat_ensemble
        router.register_judge_ensemble(LegalTaskType.GENERAL_CHAT, general_chat_ensemble)
        
        # Specialized real ensembles
        judicial_ensemble = JudicialReasoningEnsemble(config, api_client)
        ensembles[LegalTaskType.JUDICIAL_REASONING] = judicial_ensemble
        router.register_judge_ensemble(LegalTaskType.JUDICIAL_REASONING, judicial_ensemble)
        
        precedent_ensemble = PrecedentAnalysisEnsemble(config, api_client)
        ensembles[LegalTaskType.PRECEDENT_ANALYSIS] = precedent_ensemble
        router.register_judge_ensemble(LegalTaskType.PRECEDENT_ANALYSIS, precedent_ensemble)
        
        opinion_ensemble = OpinionGenerationEnsemble(config, api_client)
        ensembles[LegalTaskType.OPINION_GENERATION] = opinion_ensemble
        router.register_judge_ensemble(LegalTaskType.OPINION_GENERATION, opinion_ensemble)
        
        logger.info(f"Successfully registered {len(ensembles)} real judge ensembles for production")
        
    except Exception as e:
        logger.error(f"Error registering real judge ensembles: {e}")
        raise LegalRewardSystemError(
            f"Failed to initialize production legal reward router: {e}",
            error_context=create_error_context("factory", "create_production_router")
        )
    
    return router


def create_development_legal_reward_router(config: Optional[LegalRewardSystemConfig] = None) -> MultiTaskLegalRewardRouter:
    """Create development router with real judge ensembles"""
    
    if config is None:
        config = create_development_config()
    
    # FIX: Create RouterConfig for development
    router_config = RouterConfig(
        router_mode=RouterMode.DEVELOPMENT,
        enable_jurisdiction_inference=True,
        enable_hybrid_evaluation=True,
        enable_caching=False,  # Disable caching for development
        enable_cost_optimization=False,
        max_concurrent_evaluations=3,
        evaluation_timeout_seconds=30.0,
        require_jurisdiction_compliance=False,  # More relaxed for development
        fallback_to_general_chat=True,
        max_cost_per_evaluation=0.25
    )
    
    # FIX: Use correct constructor interface
    router = MultiTaskLegalRewardRouter(router_config, config)
    
    # Create shared API client with development settings
    api_client = CostOptimizedAPIClient(config)

    logger = get_legal_logger("development_router_factory")
    
    try:
        # Import real ensemble implementations
        from judges.general_chat import EnhancedGeneralChatEnsemble
        from judges.specialized.judicial_reasoning import JudicialReasoningEnsemble
        from judges.specialized.precedent_analysis import PrecedentAnalysisEnsemble
        from judges.specialized.opinion_generation import OpinionGenerationEnsemble
        
        # Create and register real ensembles with development settings
        general_chat_ensemble = EnhancedGeneralChatEnsemble(config, api_client)
        router.register_judge_ensemble(LegalTaskType.GENERAL_CHAT, general_chat_ensemble)
        
        judicial_ensemble = JudicialReasoningEnsemble(config, api_client)
        router.register_judge_ensemble(LegalTaskType.JUDICIAL_REASONING, judicial_ensemble)
        
        precedent_ensemble = PrecedentAnalysisEnsemble(config, api_client)
        router.register_judge_ensemble(LegalTaskType.PRECEDENT_ANALYSIS, precedent_ensemble)
        
        opinion_ensemble = OpinionGenerationEnsemble(config, api_client)
        router.register_judge_ensemble(LegalTaskType.OPINION_GENERATION, opinion_ensemble)
        
        logger.info("Successfully registered real judge ensembles for development")
        
    except Exception as e:
        logger.warning(f"Error registering development judge ensembles: {e}")
        # For development, we could fall back to a simpler implementation
        # but since you want real implementations, we'll raise the error
        raise LegalRewardSystemError(
            f"Failed to initialize development legal reward router: {e}",
            error_context=create_error_context("factory", "create_development_router")
        )
        
    return router


def create_test_legal_reward_router(config: Optional[LegalRewardSystemConfig] = None) -> MultiTaskLegalRewardRouter:
    """Create test router with real judge ensembles but optimized for testing"""
    
    if config is None:
        config = create_test_config()
    
    # FIX: Create RouterConfig for testing
    router_config = RouterConfig(
        router_mode=RouterMode.DEVELOPMENT,  # Use development mode for testing
        enable_jurisdiction_inference=True,
        enable_hybrid_evaluation=True,
        enable_caching=False,  # Disable caching for testing
        enable_cost_optimization=False,
        max_concurrent_evaluations=2,  # Lower for testing
        evaluation_timeout_seconds=20.0,  # Shorter timeout for testing
        require_jurisdiction_compliance=False,  # More relaxed for testing
        fallback_to_general_chat=True,
        max_cost_per_evaluation=0.10  # Lower cost for testing
    )
    
    # FIX: Use correct constructor interface
    router = MultiTaskLegalRewardRouter(router_config, config)
    
    # Create API client with test-optimized settings
    api_client = CostOptimizedAPIClient(config)

    logger = get_legal_logger("test_router_factory")
    
    try:
        # Use real ensembles but with test configurations
        from judges.general_chat import EnhancedGeneralChatEnsemble
        from judges.specialized.judicial_reasoning import JudicialReasoningEnsemble
        from judges.specialized.precedent_analysis import PrecedentAnalysisEnsemble
        from judges.specialized.opinion_generation import OpinionGenerationEnsemble
        
        # Register all real ensembles
        ensembles = [
            (LegalTaskType.GENERAL_CHAT, EnhancedGeneralChatEnsemble(config, api_client)),
            (LegalTaskType.JUDICIAL_REASONING, JudicialReasoningEnsemble(config, api_client)),
            (LegalTaskType.PRECEDENT_ANALYSIS, PrecedentAnalysisEnsemble(config, api_client)),
            (LegalTaskType.OPINION_GENERATION, OpinionGenerationEnsemble(config, api_client))
        ]
        
        for task_type, ensemble in ensembles:
            router.register_judge_ensemble(task_type, ensemble)
        
        logger.info("Successfully registered real judge ensembles for testing")
        
    except Exception as e:
        logger.error(f"Error registering test judge ensembles: {e}")
        raise LegalRewardSystemError(
            f"Failed to initialize test legal reward router: {e}",
            error_context=create_error_context("factory", "create_test_router")
        )
        
    return router

def create_training_legal_reward_router(config: LegalRewardSystemConfig) -> MultiTaskLegalRewardRouter:
    """
    Create router optimized for VERL training scenarios.
    
    Args:
        config: System configuration (required for training)
        
    Returns:
        Configured MultiTaskLegalRewardRouter for training
    """
    
    router, validation_result = _system_factory.create_training_system(config)
    
    if not validation_result.is_valid:
        logger = get_legal_logger("factory")
        logger.error(f"Training system created with issues: {validation_result.errors}")
    
    return router


# System setup and validation functions

def setup_legal_reward_system(environment: str = "production",
                            config_path: Optional[str] = None,
                            enable_validation: bool = True) -> Tuple[MultiTaskLegalRewardRouter, Dict[str, Any]]:
    """
    Set up complete legal reward system based on environment.
    
    Args:
        environment: Target environment ("production", "development", "testing", "training")
        config_path: Path to configuration file (optional)
        enable_validation: Whether to run system validation
        
    Returns:
        Tuple of (router, setup_results)
    """
    
    logger = get_legal_logger("setup")
    logger.info(f"Setting up legal reward system for {environment} environment")
    
    try:
        # Parse environment
        env = SystemEnvironment(environment.lower())
        
        # Create configuration
        if config_path:
            config = LegalRewardSystemConfig.from_file(config_path)
        else:
            config = None
        
        # Create system
        router, validation_result = _system_factory.create_system(env, config)
        
        # Prepare results
        setup_results = {
            "success": validation_result.is_valid,
            "environment": environment,
            "validation_summary": validation_result.get_summary(),
            "warnings": validation_result.warnings,
            "errors": validation_result.errors,
            "recommendations": validation_result.recommendations,
            "components_status": {name: status.value for name, status in validation_result.components_status.items()}
        }
        
        logger.info(f"System setup completed: {setup_results['validation_summary']}")
        
        return router, setup_results
        
    except Exception as e:
        logger.error(f"System setup failed: {e}")
        
        # Return fallback system
        fallback_router = _system_factory._create_fallback_router()
        error_results = {
            "success": False,
            "environment": environment,
            "errors": [str(e)],
            "fallback_used": True
        }
        
        return fallback_router, error_results


def validate_system_setup(router: MultiTaskLegalRewardRouter) -> Dict[str, Any]:
    """Validate that the legal reward system is properly configured with real ensembles"""
    
    issues = []
    warnings = []
    component_health = 0
    total_components = 12  # Expected number of system components
    
    # Check judge ensembles attribute exists
    if not hasattr(router, 'judge_ensembles'):
        issues.append("Critical: Router missing judge_ensembles attribute")
        return {
            "is_valid": False,
            "environment": getattr(router.config, 'environment', 'unknown'),
            "component_health": f"{component_health}/{total_components}",
            "health_percentage": 0.0,
            "warnings_count": len(warnings),
            "errors_count": len(issues),
            "setup_time": 0.0
        }
    
    component_health += 1  # judge_ensembles exists
    
    # Check that all task types have registered ensembles
    registered_ensembles = getattr(router, 'judge_ensembles', {})
    expected_tasks = list(LegalTaskType)
    
    for task_type in expected_tasks:
        if task_type in registered_ensembles:
            ensemble = registered_ensembles[task_type]
            # Verify it's a real ensemble, not a mock
            ensemble_class_name = ensemble.__class__.__name__
            if 'Mock' not in ensemble_class_name and 'Test' not in ensemble_class_name:
                component_health += 1
            else:
                warnings.append(f"Task {task_type.value} using mock/test ensemble instead of real implementation")
        else:
            issues.append(f"Missing judge ensemble for {task_type.value}")
    
    # Check general chat ensemble specifically
    if hasattr(router, 'general_chat_ensemble') and router.general_chat_ensemble is not None:
        component_health += 1
    else:
        warnings.append("General chat ensemble not available for hybrid evaluation")
    
    # Check API client
    if hasattr(router, 'api_client') and router.api_client is not None:
        component_health += 1
    else:
        warnings.append("API client not properly initialized")
    
    # Check other core components
    core_components = [
        'hybrid_evaluation_system', 'weight_manager', 'jurisdiction_inference_engine', 
        'compliance_judge', 'config', 'logger'
    ]
    
    for component in core_components:
        if hasattr(router, component) and getattr(router, component) is not None:
            component_health += 1
        else:
            warnings.append(f"Component {component} not properly initialized")
    
    # Calculate health metrics
    health_percentage = (component_health / total_components) * 100
    is_valid = len(issues) == 0 and health_percentage >= 80.0
    
    return {
        "is_valid": is_valid,
        "environment": getattr(router.config, 'environment', 'unknown'),
        "component_health": f"{component_health}/{total_components}",
        "health_percentage": health_percentage,
        "warnings_count": len(warnings),
        "errors_count": len(issues),
        "setup_time": 0.0001,  # Minimal setup time for validation
        "issues": issues,
        "warnings": warnings,
        "registered_ensembles": [task.value for task in registered_ensembles.keys()] if registered_ensembles else [],
        "system_ready": is_valid
    }

# Specialized factory functions

def create_verl_training_system(config: LegalRewardSystemConfig) -> Tuple[VERLLegalRewardFunction, MultiTaskLegalRewardRouter]:
    """
    Create complete VERL training system with router and VERL interface.
    
    Args:
        config: System configuration for training
        
    Returns:
        Tuple of (VERL function, router)
    """
    
    # Create training router
    router = create_training_legal_reward_router(config)
    
    # Create VERL function
    verl_function = create_training_verl_function(config)
    
    return verl_function, router


def create_api_development_system(config: Optional[LegalRewardSystemConfig] = None) -> MultiTaskLegalRewardRouter:
    """
    Create system for API development and integration testing.
    
    Args:
        config: System configuration (optional)
        
    Returns:
        Configured router for API development
    """
    
    if config is None:
        config = create_development_config()
    
    # Create router with API-focused settings
    router_config = RouterConfig(
        router_mode=RouterMode.DEVELOPMENT,
        enable_caching=True,
        enable_cost_optimization=True,
        max_concurrent_evaluations=5,
        evaluation_timeout_seconds=45.0,
        require_jurisdiction_compliance=True,
        fallback_to_general_chat=True
    )
    
    return MultiTaskLegalRewardRouter(router_config, config)


def create_performance_testing_system(config: Optional[LegalRewardSystemConfig] = None) -> MultiTaskLegalRewardRouter:
    """
    Create system optimized for performance testing and benchmarking.
    
    Args:
        config: System configuration (optional)
        
    Returns:
        Configured router for performance testing
    """
    
    if config is None:
        config = create_production_config()
    
    # Create high-performance router
    router_config = RouterConfig(
        router_mode=RouterMode.HIGH_ACCURACY,
        enable_caching=True,
        enable_cost_optimization=False,  # Disable for consistent performance testing
        max_concurrent_evaluations=20,
        evaluation_timeout_seconds=120.0,
        track_performance_metrics=True
    )
    
    return MultiTaskLegalRewardRouter(router_config, config)


# Utility functions for system management

def get_system_health_check() -> Dict[str, Any]:
    """
    Get system health check for monitoring and alerting.
    
    Returns:
        Dictionary with system health information
    """
    
    try:
        # Create minimal system for health check
        router, validation_result = _system_factory.create_development_system()
        
        health_status = {
            "status": "healthy" if validation_result.is_valid else "unhealthy",
            "timestamp": time.time(),
            "components": {name: status.value for name, status in validation_result.components_status.items()},
            "error_count": len(validation_result.errors),
            "warning_count": len(validation_result.warnings)
        }
        
        return health_status
        
    except Exception as e:
        return {
            "status": "error",
            "timestamp": time.time(),
            "error": str(e)
        }


def reset_system_cache():
    """Reset system-wide caches and statistics"""
    
    global _system_factory
    _system_factory._creation_cache.clear()
    _system_factory._validation_cache.clear()
    
    logger = get_legal_logger("factory")
    logger.info("System caches reset")


async def warm_up_system(router: MultiTaskLegalRewardRouter) -> Dict[str, Any]:
    """
    Warm up system by running test evaluations.
    
    Args:
        router: Router to warm up
        
    Returns:
        Warm-up results and performance metrics
    """
    
    logger = get_legal_logger("warmup")
    logger.info("Starting system warm-up")
    
    start_time = time.time()
    
    try:
        from .routing import EvaluationRequest
        
        # Create test requests for each task type
        test_requests = []
        for task_type in LegalTaskType:
            request = EvaluationRequest(
                response=f"Test response for {task_type.value}",
                task_type=task_type,
                prompt=f"Test prompt for {task_type.value}",
                request_id=f"warmup_{task_type.value}"
            )
            test_requests.append(request)
        
        # Execute warm-up evaluations
        successful_warmups = 0
        total_warmups = len(test_requests)
        
        for request in test_requests:
            try:
                result = await router.evaluate_response(request)
                if result.is_successful:
                    successful_warmups += 1
            except Exception as e:
                logger.warning(f"Warm-up failed for {request.task_type.value}: {e}")
        
        warmup_time = time.time() - start_time
        
        warmup_results = {
            "warmup_successful": successful_warmups == total_warmups,
            "successful_warmups": successful_warmups,
            "total_warmups": total_warmups,
            "warmup_time": warmup_time,
            "average_time_per_evaluation": warmup_time / total_warmups,
            "system_ready": successful_warmups > 0
        }
        
        logger.info(f"System warm-up completed: {warmup_results}")
        
        return warmup_results
        
    except Exception as e:
        logger.error(f"System warm-up failed: {e}")
        return {
            "warmup_successful": False,
            "error": str(e),
            "system_ready": False
        }


# Configuration helper functions

def get_recommended_config(environment: str, use_case: str = "general") -> Dict[str, Any]:
    """
    Get recommended configuration for specific environment and use case.
    
    Args:
        environment: Target environment
        use_case: Specific use case ("general", "training", "development", "testing")
        
    Returns:
        Dictionary with recommended configuration
    """
    
    base_configs = {
        "production": {
            "router_mode": "production",
            "enable_caching": True,
            "enable_cost_optimization": True,
            "max_concurrent_evaluations": 10,
            "evaluation_timeout_seconds": 60.0,
            "require_jurisdiction_compliance": True
        },
        "development": {
            "router_mode": "development", 
            "enable_caching": False,
            "enable_cost_optimization": False,
            "max_concurrent_evaluations": 3,
            "evaluation_timeout_seconds": 30.0,
            "require_jurisdiction_compliance": False
        },
        "training": {
            "router_mode": "production",
            "enable_caching": True,
            "enable_cost_optimization": True,
            "max_concurrent_evaluations": 15,
            "evaluation_timeout_seconds": 90.0,
            "require_jurisdiction_compliance": True
        }
    }
    
    config = base_configs.get(environment, base_configs["production"]).copy()
    
    # Use case specific adjustments
    if use_case == "training":
        config.update({
            "max_concurrent_evaluations": 20,
            "evaluation_timeout_seconds": 120.0,
            "track_performance_metrics": True
        })
    elif use_case == "testing":
        config.update({
            "max_concurrent_evaluations": 5,
            "evaluation_timeout_seconds": 45.0,
            "enable_caching": False
        })
    
    return config


def create_custom_system(custom_config: Dict[str, Any]) -> MultiTaskLegalRewardRouter:
    """
    Create system with custom configuration parameters.
    
    Args:
        custom_config: Custom configuration dictionary
        
    Returns:
        Configured router with custom settings
    """
    
    # Create base configuration
    base_config = create_production_config()
    
    # Create custom router config
    router_config = RouterConfig(
        router_mode=RouterMode(custom_config.get("router_mode", "production")),
        enable_caching=custom_config.get("enable_caching", True),
        enable_cost_optimization=custom_config.get("enable_cost_optimization", True),
        max_concurrent_evaluations=custom_config.get("max_concurrent_evaluations", 10),
        evaluation_timeout_seconds=custom_config.get("evaluation_timeout_seconds", 60.0),
        require_jurisdiction_compliance=custom_config.get("require_jurisdiction_compliance", True),
        fallback_to_general_chat=custom_config.get("fallback_to_general_chat", True)
    )
    
    return MultiTaskLegalRewardRouter(router_config, base_config)