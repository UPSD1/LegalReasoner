"""
Utils Package for Multi-Task Legal Reward System

This package contains utility modules that provide foundational services
for the legal reward system:
- Enhanced logging with cost tracking and performance monitoring
- Aggressive caching system for API cost optimization
- Smart rate limiting with multi-provider fallback
- Performance monitoring and optimization utilities

Main exports provide a clean API for system components to import
utility functionality without needing to know internal module structure.
"""

# Enhanced Logging System
from utils.logging import (
    LegalRewardLogger,
    APIRequestLog,
    PerformanceMetric,
    CostTracker,
    PerformanceTracker,
    LegalRewardJSONFormatter,
    get_legal_logger,
    setup_system_logging,
    get_system_logging_summary,
    LoggedOperation
)

# Aggressive Caching System
from utils.cache import (
    MultiStrategyLegalRewardCache,
    CacheEntry,
    CacheKeyGenerator,
    CacheCompressor,
    PersistentCacheStorage,
    create_cache_with_strategy,
    create_aggressive_cache,
    get_strategy_configurations,
    ManagedCache
)

# Smart Rate Limiting System
from utils.rate_limiter import (
    MultiProviderRateLimiter,
    TokenBucketLimiter,
    TokenBucket,
    IntelligentBackoffStrategy,
    RateLimitConfig,
    RateLimitStatus,
    create_production_rate_limiter,
    create_development_rate_limiter,
    create_aggressive_rate_limiter,
    estimate_tokens_from_content,
    get_task_complexity,
    calculate_optimal_batch_size,
    ManagedRateLimiter
)

# Package metadata
__version__ = "1.0.0"
__author__ = "Legal AI Development Team"
__description__ = "Utility modules for Multi-Task Legal Reward System"

# Main exports for easy importing
__all__ = [
    # Enhanced Logging
    "LegalRewardLogger",
    "APIRequestLog", 
    "PerformanceMetric",
    "CostTracker",
    "PerformanceTracker",
    "LegalRewardJSONFormatter",
    "get_legal_logger",
    "setup_system_logging",
    "get_system_logging_summary",
    "LoggedOperation",
    
    # Aggressive Caching
    "MultiStrategyLegalRewardCache",
    "CacheEntry",
    "CacheKeyGenerator",
    "CacheCompressor", 
    "PersistentCacheStorage",
    "create_cache_with_strategy",
    "create_aggressive_cache",
    "get_strategy_configurations",
    "ManagedCache",
    
    # Smart Rate Limiting
    "MultiProviderRateLimiter",
    "TokenBucketLimiter",
    "TokenBucket",
    "IntelligentBackoffStrategy",
    "RateLimitConfig",
    "RateLimitStatus",
    "create_production_rate_limiter",
    "create_development_rate_limiter",
    "create_aggressive_rate_limiter",
    "estimate_tokens_from_content",
    "get_task_complexity",
    "calculate_optimal_batch_size",
    "ManagedRateLimiter"
]


# Package-level utility functions
def get_utils_version() -> str:
    """Get the version of the utils package"""
    return __version__

def validate_utils_imports() -> bool:
    """
    Validate that all utils imports are working properly.
    
    Returns:
        True if all utility components can be imported successfully
    """
    try:
        # Test logging
        logger = get_legal_logger("test_logger")
        
        # Test caching
        cache_config = {"strategy": "disabled", "cache_dir": "/tmp/test"}
        test_cache = MultiStrategyLegalRewardCache(cache_config)
        
        # Test rate limiting
        rate_limiter_config = {
            "openai": {"requests_per_minute": 10, "tokens_per_minute": 1000}
        }
        test_rate_limiter = MultiProviderRateLimiter(rate_limiter_config)
        
        return True
        
    except Exception:
        return False

def get_utils_summary() -> dict:
    """
    Get summary information about the utils package.
    
    Returns:
        Dictionary with utils package information
    """
    return {
        "version": __version__,
        "description": __description__,
        "modules": {
            "logging": [
                "LegalRewardLogger", "CostTracker", "PerformanceTracker",
                "LegalRewardJSONFormatter", "LoggedOperation"
            ],
            "cache": [
                "MultiStrategyLegalRewardCache", "CacheKeyGenerator", 
                "CacheCompressor", "PersistentCacheStorage"
            ],
            "rate_limiter": [
                "MultiProviderRateLimiter", "TokenBucketLimiter",
                "IntelligentBackoffStrategy", "TokenBucket"
            ]
        },
        "factory_functions": [
            "create_aggressive_cache", "create_production_rate_limiter",
            "get_legal_logger", "setup_system_logging"
        ],
        "context_managers": [
            "LoggedOperation", "ManagedCache", "ManagedRateLimiter"
        ],
        "total_exports": len(__all__),
        "validation_passed": validate_utils_imports()
    }

def create_default_system_utilities(config: dict) -> dict:
    """
    Create default system utilities with provided configuration.
    
    Args:
        config: Configuration dictionary for utilities
        
    Returns:
        Dictionary containing configured utility instances
    """
    utilities = {}
    
    try:
        # Create logger
        logger_config = config.get("logging", {})
        utilities["logger"] = get_legal_logger("system", logger_config)
        
        # Create cache
        cache_config = config.get("caching", {"strategy": "aggressive"})
        utilities["cache"] = create_aggressive_cache(cache_config)
        
        # Create rate limiter
        rate_limiting_config = config.get("rate_limiting", {})
        utilities["rate_limiter"] = create_production_rate_limiter(rate_limiting_config)
        
        utilities["logger"].info("Default system utilities created successfully")
        
    except Exception as e:
        # Fallback utilities if configuration fails
        utilities = {
            "logger": get_legal_logger("system_fallback"),
            "cache": MultiStrategyLegalRewardCache({"strategy": "disabled"}),
            "rate_limiter": create_development_rate_limiter()
        }
        utilities["logger"].error(f"Failed to create configured utilities, using fallbacks: {e}")
    
    return utilities

def cleanup_system_utilities(utilities: dict):
    """
    Clean up system utilities resources.
    
    Args:
        utilities: Dictionary of utility instances to cleanup
    """
    try:
        # Cleanup cache
        if "cache" in utilities and hasattr(utilities["cache"], "close"):
            utilities["cache"].close()
        
        # Cleanup logger
        if "logger" in utilities and hasattr(utilities["logger"], "cleanup"):
            utilities["logger"].cleanup()
        
        # Rate limiter doesn't need explicit cleanup
        
    except Exception as e:
        print(f"Warning: Error during utility cleanup: {e}")

# Performance optimization utilities
def estimate_system_throughput(concurrent_evaluations: int = 10,
                              avg_evaluation_time: float = 2.0,
                              cache_hit_rate: float = 0.7) -> dict:
    """
    Estimate system throughput based on configuration parameters.
    
    Args:
        concurrent_evaluations: Number of concurrent evaluations
        avg_evaluation_time: Average evaluation time in seconds
        cache_hit_rate: Expected cache hit rate (0.0-1.0)
        
    Returns:
        Dictionary with throughput estimates
    """
    # Base throughput without caching
    base_throughput = concurrent_evaluations / avg_evaluation_time
    
    # Cache impact (cache hits are much faster)
    cache_speedup = 1 + (cache_hit_rate * 10)  # Assume cache is 10x faster
    effective_throughput = base_throughput * cache_speedup
    
    # Daily and hourly estimates
    hourly_evaluations = effective_throughput * 3600
    daily_evaluations = hourly_evaluations * 24
    
    return {
        "base_throughput_per_second": base_throughput,
        "effective_throughput_per_second": effective_throughput,
        "hourly_evaluations": hourly_evaluations,
        "daily_evaluations": daily_evaluations,
        "cache_impact_multiplier": cache_speedup,
        "configuration": {
            "concurrent_evaluations": concurrent_evaluations,
            "avg_evaluation_time": avg_evaluation_time,
            "cache_hit_rate": cache_hit_rate
        }
    }

def estimate_monthly_costs(evaluations_per_day: int = 10000,
                          avg_cost_per_evaluation: float = 0.02,
                          cache_hit_rate: float = 0.7) -> dict:
    """
    Estimate monthly API costs based on usage patterns.
    
    Args:
        evaluations_per_day: Expected evaluations per day
        avg_cost_per_evaluation: Average cost per evaluation in USD
        cache_hit_rate: Expected cache hit rate (0.0-1.0)
        
    Returns:
        Dictionary with cost estimates
    """
    # Calculate base costs
    monthly_evaluations = evaluations_per_day * 30
    base_monthly_cost = monthly_evaluations * avg_cost_per_evaluation
    
    # Apply cache savings
    cache_savings_rate = cache_hit_rate * 0.9  # Conservative estimate
    effective_monthly_cost = base_monthly_cost * (1 - cache_savings_rate)
    
    return {
        "monthly_evaluations": monthly_evaluations,
        "base_monthly_cost": base_monthly_cost,
        "effective_monthly_cost": effective_monthly_cost,
        "cache_savings": base_monthly_cost - effective_monthly_cost,
        "cache_savings_percentage": cache_savings_rate * 100,
        "configuration": {
            "evaluations_per_day": evaluations_per_day,
            "avg_cost_per_evaluation": avg_cost_per_evaluation,
            "cache_hit_rate": cache_hit_rate
        }
    }

# Module initialization
def _initialize_utils_package():
    """Initialize utils package with validation checks"""
    if not validate_utils_imports():
        import warnings
        warnings.warn(
            "Utils package imports failed validation. Some functionality may not work correctly.",
            ImportWarning,
            stacklevel=2
        )

# Run initialization when package is imported
_initialize_utils_package()
