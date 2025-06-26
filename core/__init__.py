"""
Core Package for Multi-Task Legal Reward System

This package contains the foundational components of the legal reward system:
- Data structures for legal data points, ensemble scores, and rewards
- Enumerations for task types, jurisdictions, and system configuration
- Custom exceptions for structured error handling

Main exports provide a clean API for the rest of the system to import
core functionality without needing to know internal module structure.
"""

# Data Structures - Main system data types
from .data_structures import (
    # Core data structures
    LegalDataPoint,
    EnsembleScore, 
    HybridEvaluationResult,
    RoutedReward,
    LegalRewardEvaluation,
    JudgeEvaluation,
    EvaluationMetadata,
    APIResponse,
    CostInformation,
    PerformanceMetrics,
    
    # Utility functions
    validate_score_range,
    create_fallback_ensemble_score,
    aggregate_ensemble_scores,
    
    # Type aliases
    ScoreDict,
    MetadataDict,
    JurisdictionInfo
)

# Enumerations - System type definitions
from .enums import (
    # Main legal system enums
    LegalTaskType,
    LegalDomain,
    USJurisdiction,
    USJurisdictionLevel,
    
    # System configuration enums
    APIProvider,
    EvaluationMethod,
    CacheStrategy,
    RateLimitStrategy,
    LogLevel,
    
    # Default values
    DEFAULT_LEGAL_TASK_TYPE,
    DEFAULT_LEGAL_DOMAIN,
    DEFAULT_US_JURISDICTION,
    DEFAULT_JURISDICTION_LEVEL,
    DEFAULT_API_PROVIDER,
    DEFAULT_EVALUATION_METHOD,
    DEFAULT_CACHE_STRATEGY,
    DEFAULT_RATE_LIMIT_STRATEGY,
    DEFAULT_LOG_LEVEL,
    
    # Validation sets
    VALID_TASK_TYPES,
    VALID_LEGAL_DOMAINS,
    VALID_US_JURISDICTIONS,
    VALID_JURISDICTION_LEVELS,
    VALID_API_PROVIDERS,
    VALID_EVALUATION_METHODS,
    
    # Utility functions
    get_all_enum_values,
    validate_enum_value,
    get_enum_mapping
)

# Exceptions - Structured error handling
from .exceptions import (
    # Base exception
    LegalRewardSystemError,
    
    # Domain-specific exceptions
    JurisdictionInferenceError,
    JudgeEvaluationError,
    HybridEvaluationError,
    RoutingError,
    VERLIntegrationError,
    SystemValidationError,
    
    # API-related exceptions
    APIClientError,
    RateLimitExceededError,
    APIProviderError,
    APIResponseError,
    
    # System exceptions
    CacheError,
    ConfigurationError,
    
    # Supporting structures
    ErrorContext,
    
    # Utility functions
    wrap_exception,
    create_error_context,
    handle_api_error,
    log_exception,
    get_recovery_suggestions,
    
    # Exception mappings
    EXCEPTION_TYPES,
    ERROR_CODES
)

# Create US_STATES_AND_TERRITORIES from USJurisdiction enum
US_STATES_AND_TERRITORIES = [jurisdiction.value for jurisdiction in USJurisdiction]

# Create validation sets
VALID_API_PROVIDERS = [provider.value for provider in APIProvider] 
VALID_TASK_TYPES = [task_type.value for task_type in LegalTaskType]

# Package metadata
__version__ = "1.0.0"
__author__ = "Legal AI Development Team"
__description__ = "Core components for Multi-Task Legal Reward System"

# Main exports for easy importing
__all__ = [
    # Data Structures
    "LegalDataPoint",
    "EnsembleScore", 
    "HybridEvaluationResult",
    "RoutedReward",
    "LegalRewardEvaluation",
    "JudgeEvaluation",
    "EvaluationMetadata",
    "APIResponse",
    "CostInformation",
    "PerformanceMetrics",
    "validate_score_range",
    "create_fallback_ensemble_score",
    "aggregate_ensemble_scores",
    "ScoreDict",
    "MetadataDict",
    "JurisdictionInfo",
    
    # Enumerations
    "LegalTaskType",
    "LegalDomain", 
    "USJurisdiction",
    "USJurisdictionLevel",
    "APIProvider",
    "EvaluationMethod",
    "CacheStrategy", 
    "RateLimitStrategy",
    "LogLevel",
    
    # Default values
    "DEFAULT_LEGAL_TASK_TYPE",
    "DEFAULT_LEGAL_DOMAIN",
    "DEFAULT_US_JURISDICTION",
    "DEFAULT_JURISDICTION_LEVEL",
    "DEFAULT_API_PROVIDER",
    "DEFAULT_EVALUATION_METHOD",
    "DEFAULT_CACHE_STRATEGY",
    "DEFAULT_RATE_LIMIT_STRATEGY", 
    "DEFAULT_LOG_LEVEL",
    
    # Validation
    "VALID_TASK_TYPES",
    "VALID_LEGAL_DOMAINS",
    "VALID_US_JURISDICTIONS", 
    "VALID_JURISDICTION_LEVELS",
    "VALID_API_PROVIDERS",
    "VALID_EVALUATION_METHODS",
    "get_all_enum_values",
    "validate_enum_value",
    "get_enum_mapping",
    
    # Exceptions
    "LegalRewardSystemError",
    "JurisdictionInferenceError", 
    "JudgeEvaluationError",
    "HybridEvaluationError",
    "RoutingError",
    "VERLIntegrationError",
    "SystemValidationError",
    "APIClientError",
    "RateLimitExceededError",
    "APIProviderError", 
    "APIResponseError",
    "CacheError",
    "ConfigurationError",
    "ErrorContext",
    "wrap_exception",
    "create_error_context",
    "handle_api_error",
    "log_exception",
    "get_recovery_suggestions",
    "EXCEPTION_TYPES",
    "ERROR_CODES",
    
    # Utilities
    "US_STATES_AND_TERRITORIES"
]

# Package-level utility functions
def get_core_version() -> str:
    """Get the version of the core package"""
    return __version__

def validate_core_imports() -> bool:
    """
    Validate that all core imports are working properly.
    
    Returns:
        True if all core components can be imported successfully
    """
    try:
        # Test data structure creation
        test_data_point = LegalDataPoint(
            query="test query",
            response="test response", 
            task_type=LegalTaskType.GENERAL_CHAT,
            jurisdiction="general",
            legal_domain=LegalDomain.GENERAL
        )
        
        # Test enum operations
        task_types = get_all_enum_values(LegalTaskType)
        validated_task = validate_enum_value("general_chat", LegalTaskType)
        
        # Test exception creation
        test_error = LegalRewardSystemError("test error")
        
        return True
        
    except Exception:
        return False

def get_core_summary() -> dict:
    """
    Get summary information about the core package.
    
    Returns:
        Dictionary with core package information
    """
    return {
        "version": __version__,
        "description": __description__,
        "components": {
            "data_structures": [
                "LegalDataPoint", "EnsembleScore", 
                "HybridEvaluationResult", "RoutedReward"
            ],
            "enums": [
                "LegalTaskType", "LegalDomain", "USJurisdiction", 
                "USJurisdictionLevel", "APIProvider", "EvaluationMethod"
            ],
            "exceptions": [
                "LegalRewardSystemError", "JurisdictionInferenceError",
                "JudgeEvaluationError", "APIClientError", "ConfigurationError"
            ]
        },
        "total_exports": len(__all__),
        "validation_passed": validate_core_imports()
    }

# Module initialization
def _initialize_core_package():
    """Initialize core package with validation checks"""
    if not validate_core_imports():
        import warnings
        warnings.warn(
            "Core package imports failed validation. Some functionality may not work correctly.",
            ImportWarning,
            stacklevel=2
        )

# Run initialization when package is imported
_initialize_core_package()