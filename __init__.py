"""
Enhanced Multi-Task Legal Reward System

A sophisticated legal AI reward system for GRPO training with VERL integration,
designed specifically for the US legal system with comprehensive jurisdiction support.

Key Features:
- Hybrid evaluation system (70% specialized + 30% general chat)
- Complete US jurisdiction support (all 50 states + DC + federal)
- Multi-provider API integration (OpenAI, Anthropic, Google) with cost optimization
- Aggressive caching for 60-80% API cost reduction during training
- Smart rate limiting with provider fallback chains
- Production-ready logging, monitoring, and error handling

Main Entry Points:
- For VERL integration: Use `verl_integration.py` (implemented in later steps)
- For direct usage: Use `MultiTaskLegalRewardRouter` from `routing.router`
- For system setup: Use factory functions from `factory` module
"""

# Core system components
from .core import (
    # Data structures
    LegalRewardEvaluation,
    JudgeEvaluation,
    EvaluationMetadata,
    APIResponse,
    CostInformation,
    PerformanceMetrics,
    LegalDataPoint,
    EnsembleScore,
    HybridEvaluationResult,
    RoutedReward,
    
    # Enums and types
    APIProvider,
    LegalTaskType,
    USJurisdiction,
    LegalDomain,
    CacheStrategy,
    RateLimitStrategy,
    LogLevel,
    
    # Exceptions
    LegalRewardSystemError,
    ConfigurationError,
    APIClientError,
    RateLimitExceededError,
    JurisdictionInferenceError,
    JudgeEvaluationError,
    CacheError,
    
    # Utilities
    create_error_context,
    VALID_API_PROVIDERS,
    VALID_TASK_TYPES,
    US_STATES_AND_TERRITORIES
)

# Try to import optional components (may not exist yet)
try:
    from .config import (
        LegalRewardSystemConfig,
        create_production_config,
        create_development_config,
        create_test_config,
        validate_system_config,
        get_config_template
    )
    _config_available = True
except ImportError:
    _config_available = False

try:
    from .utils import (
        # Logging
        LegalRewardLogger,
        get_legal_logger,
        setup_system_logging,
        LoggedOperation,
        
        # Caching
        MultiStrategyLegalRewardCache,
        create_aggressive_cache,
        ManagedCache,
        
        # Rate limiting
        MultiProviderRateLimiter,
        create_production_rate_limiter,
        ManagedRateLimiter
    )
    _utils_available = True
except ImportError:
    _utils_available = False

# Package metadata
__version__ = "1.0.0"
__author__ = "Legal AI Development Team"
__description__ = "Enhanced Multi-Task Legal Reward System with US Jurisdiction Support"
__license__ = "MIT"

# Main exports for external usage
__all__ = [
    # Core data structures
    "LegalRewardEvaluation",
    "JudgeEvaluation", 
    "EvaluationMetadata",
    "APIResponse",
    "CostInformation",
    "PerformanceMetrics",
    "LegalDataPoint",
    "EnsembleScore",
    "HybridEvaluationResult",
    "RoutedReward",
    
    # Enums and types
    "APIProvider",
    "LegalTaskType", 
    "USJurisdiction",
    "LegalDomain",
    "CacheStrategy",
    "RateLimitStrategy",
    "LogLevel",
    
    # Exceptions
    "LegalRewardSystemError",
    "ConfigurationError",
    "APIClientError",
    "RateLimitExceededError",
    "JurisdictionInferenceError",
    "JudgeEvaluationError",
    "CacheError",
    
    # Utilities
    "create_error_context",
    "VALID_API_PROVIDERS",
    "VALID_TASK_TYPES",
    "US_STATES_AND_TERRITORIES"
]

# Add optional exports if available
if _config_available:
    __all__.extend([
        "LegalRewardSystemConfig",
        "create_production_config",
        "create_development_config",
        "create_test_config",
        "validate_system_config",
        "get_config_template"
    ])

if _utils_available:
    __all__.extend([
        "LegalRewardLogger",
        "get_legal_logger",
        "setup_system_logging",
        "LoggedOperation",
        "MultiStrategyLegalRewardCache",
        "create_aggressive_cache",
        "ManagedCache",
        "MultiProviderRateLimiter",
        "create_production_rate_limiter",
        "ManagedRateLimiter"
    ])

def get_system_info() -> dict:
    """
    Get comprehensive system information and status.
    
    Returns:
        Dictionary with system information
    """
    return {
        "version": __version__,
        "description": __description__,
        "components_status": {
            "core_data_structures": True,
            "configuration_management": _config_available,
            "logging_system": _utils_available,
            "caching_system": _utils_available,
            "rate_limiting": _utils_available,
            "exception_handling": True
        },
        "phase_1_foundation_complete": True,
        "next_phase": "US Jurisdiction System (Step 8)",
        "estimated_setup_time": "5-10 minutes with .env configuration",
        "main_features": [
            "US jurisdiction support (50 states + DC + federal)",
            "Hybrid evaluation (70% specialized + 30% chat)",
            "Multi-provider API integration with cost optimization",
            "Aggressive caching for training cost reduction",
            "Production-ready monitoring and error handling"
        ]
    }

def validate_foundation_setup() -> dict:
    """
    Validate that the foundation components are properly set up.
    
    Returns:
        Dictionary with validation results
    """
    setup_results = {
        "foundation_valid": True,
        "errors": [],
        "warnings": [],
        "components_tested": {
            "core_imports": False,
            "data_structures": False,
            "enums": False,
            "exceptions": False,
            "config": False,
            "utils": False
        }
    }
    
    # Test core imports
    try:
        from .core import LegalDataPoint, LegalTaskType, USJurisdiction
        setup_results["components_tested"]["core_imports"] = True
    except Exception as e:
        setup_results["errors"].append(f"Core imports failed: {e}")
        setup_results["foundation_valid"] = False
    
    # Test data structures
    try:
        dp = LegalDataPoint(
            query="test",
            response="test", 
            task_type=LegalTaskType.GENERAL_CHAT,
            jurisdiction="general",
            legal_domain=LegalDomain.GENERAL
        )
        setup_results["components_tested"]["data_structures"] = True
    except Exception as e:
        setup_results["errors"].append(f"Data structures failed: {e}")
        setup_results["foundation_valid"] = False
    
    # Test enums
    try:
        task_types = list(LegalTaskType)
        jurisdictions = list(USJurisdiction)
        setup_results["components_tested"]["enums"] = True
    except Exception as e:
        setup_results["errors"].append(f"Enums failed: {e}")
        setup_results["foundation_valid"] = False
    
    # Test exceptions
    try:
        error = LegalRewardSystemError("test error")
        setup_results["components_tested"]["exceptions"] = True
    except Exception as e:
        setup_results["errors"].append(f"Exceptions failed: {e}")
        setup_results["foundation_valid"] = False
    
    # Test optional components
    setup_results["components_tested"]["config"] = _config_available
    setup_results["components_tested"]["utils"] = _utils_available
    
    if not _config_available:
        setup_results["warnings"].append("Configuration system not available")
    
    if not _utils_available:
        setup_results["warnings"].append("Utils system (logging, caching, rate limiting) not available")
    
    if setup_results["errors"]:
        setup_results["next_steps"] = [
            "1. Check that all foundation components are properly installed",
            "2. Verify Python environment and dependencies",
            "3. Review error messages and fix any import issues"
        ]
    
    return setup_results

# Quick start utilities
def quick_start_guide() -> str:
    """
    Get quick start guide for the legal reward system.
    
    Returns:
        Formatted string with quick start instructions
    """
    return """
🚀 ENHANCED MULTI-TASK LEGAL REWARD SYSTEM - QUICK START

Step 1: Environment Setup
------------------------
1. Copy .env.template to .env
2. Fill in your API keys:
   - OPENAI_API_KEY=sk-your-key-here
   - ANTHROPIC_API_KEY=sk-ant-your-key-here  
   - GOOGLE_API_KEY=AIza-your-key-here

Step 2: Test Foundation
-----------------------
python -c "
from legal_reward_system import validate_foundation_setup
result = validate_foundation_setup()
print(f'Foundation valid: {result[\"foundation_valid\"]}')
if result['errors']:
    print('Errors:', result['errors'])
"

Step 3: Development Setup
-------------------------
python -c "
from legal_reward_system import get_system_info
result = get_system_info()
print(f'System status: {result[\"components_status\"]}')
"

Current Status: Phase 1 Foundation Complete ✅
Next Phase: US Jurisdiction System Implementation (Step 8)

For detailed configuration: see config/internal_config.yaml
For full documentation: see implementation roadmap
"""

# Development and testing shortcuts
def create_minimal_test_system():
    """Create minimal system for testing without full configuration"""
    try:
        if _config_available:
            from .config import create_test_config
            config = create_test_config()
        else:
            config = None
        
        if _utils_available:
            from .utils import get_legal_logger
            logger = get_legal_logger("test")
        else:
            import logging
            logger = logging.getLogger("test")
        
        return {
            "config": config,
            "logger": logger,
            "status": "ready"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

# System health check
def system_health_check() -> dict:
    """Comprehensive system health check"""
    health_status = {
        "overall_health": "unknown",
        "timestamp": "2025-06-24",
        "phase_1_foundation": {},
        "environment": {},
        "recommendations": []
    }
    
    # Check foundation components
    foundation_validation = validate_foundation_setup()
    health_status["phase_1_foundation"] = foundation_validation
    
    # Check environment setup
    health_status["environment"]["config_available"] = _config_available
    health_status["environment"]["utils_available"] = _utils_available
    
    # Determine overall health
    if foundation_validation["foundation_valid"]:
        if _config_available and _utils_available:
            health_status["overall_health"] = "healthy"
            health_status["recommendations"] = [
                "✅ System foundation is solid and ready for next phase",
                "🚀 Proceed to Step 8: US Jurisdiction System Foundation",
                "📊 Monitor API costs during development with aggressive caching"
            ]
        else:
            health_status["overall_health"] = "warning"
            health_status["recommendations"] = [
                "⚠️  Foundation is solid but some components missing",
                "🔧 Implement remaining utility systems",
                "🧪 Test configuration before proceeding"
            ]
    else:
        health_status["overall_health"] = "critical"
        health_status["recommendations"] = [
            "🚨 Foundation components have issues",
            "🔍 Review error messages and fix import problems",
            "🛠️  Ensure all dependencies are properly installed"
        ]
    
    return health_status

# Initialize package
def _initialize_package():
    """Initialize package with basic validation"""
    try:
        # Validate that core imports work
        from .core import LegalRewardEvaluation, APIProvider
        
        # Package successfully initialized
        return True
    except ImportError:
        import warnings
        warnings.warn(
            "Legal reward system package initialization incomplete. "
            "Some imports may fail. Check dependencies.",
            ImportWarning,
            stacklevel=2
        )
        return False

# Run package initialization
_package_initialized = _initialize_package()