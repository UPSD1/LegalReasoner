"""
Configuration Package for Multi-Task Legal Reward System

This package provides comprehensive configuration management for the enhanced
multi-task legal reward system with US jurisdiction support:
- YAML configuration file loading and validation
- Environment variable integration for sensitive data
- Configuration validation with helpful error messages
- Property-based access to configuration sections
- Runtime configuration updates and monitoring

Main exports provide a clean API for system components to access
configuration functionality.
"""

# Main configuration management
from .settings import (
    LegalRewardSystemConfig,
    ConfigValidator,
    ConfigValidationResult,
    EnvironmentManager,
    create_production_config,
    create_development_config,
    create_test_config,
    validate_system_config,
    get_config_template
)

# Package metadata
__version__ = "1.0.0"
__author__ = "Legal AI Development Team"
__description__ = "Configuration management for Multi-Task Legal Reward System"

# Main exports for easy importing
__all__ = [
    # Main configuration
    "LegalRewardSystemConfig",
    
    # Configuration validation
    "ConfigValidator",
    "ConfigValidationResult",
    
    # Environment management
    "EnvironmentManager",
    
    # Factory functions
    "create_production_config",
    "create_development_config", 
    "create_test_config",
    
    # Utilities
    "validate_system_config",
    "get_config_template"
]

# Package-level utility functions
def get_config_version() -> str:
    """Get the version of the config package"""
    return __version__

def validate_config_imports() -> bool:
    """
    Validate that all config imports are working properly.
    
    Returns:
        True if all configuration components can be imported successfully
    """
    try:
        # Test basic configuration validation
        validation_result = ConfigValidationResult(is_valid=True)
        
        # Test environment manager
        env_manager = EnvironmentManager()
        
        # Test config validator
        config_validator = ConfigValidator()
        
        # Test config template
        template = get_config_template()
        
        return True
        
    except Exception:
        return False

def get_config_summary() -> dict:
    """
    Get summary information about the config package.
    
    Returns:
        Dictionary with config package information
    """
    return {
        "version": __version__,
        "description": __description__,
        "components": {
            "configuration_management": True,
            "environment_integration": True,
            "validation_system": True,
            "factory_functions": True
        },
        "total_exports": len(__all__),
        "validation_passed": validate_config_imports()
    }

def create_minimal_config() -> dict:
    """
    Create minimal configuration for testing or emergency fallback.
    
    Returns:
        Dictionary with minimal working configuration
    """
    return {
        "api_providers": {
            "openai": {
                "model": "gpt-4-turbo",
                "rate_limit_rpm": 100,
                "rate_limit_tpm": 10000,
                "cost_per_1k_input_tokens": 0.01,
                "cost_per_1k_output_tokens": 0.03,
                "fallback_priority": 1,
                "max_concurrent_requests": 3
            }
        },
        "caching": {
            "enabled": False,
            "strategy": "disabled"
        },
        "hybrid_evaluation": {
            "specialized_weight": 0.7,
            "general_chat_weight": 0.3,
            "jurisdiction_failure_penalty": 0.2
        },
        "task_weights": {
            "judicial_reasoning": 1.5,
            "precedent_analysis": 1.3,
            "opinion_generation": 1.1,
            "general_chat": 1.0
        },
        "cost_optimization": {
            "max_monthly_api_budget": 100
        },
        "logging": {
            "level": "INFO",
            "structured_logging": False,
            "console_output": True,
            "file_output": False
        },
        "jurisdiction": {
            "enable_inference": True,
            "require_compliance_check": True,
            "default_jurisdiction": "general"
        },
        "performance": {
            "max_concurrent_evaluations": 3
        }
    }

def get_environment_setup_guide() -> dict:
    """
    Get guide for setting up environment variables.
    
    Returns:
        Dictionary with environment setup instructions
    """
    return {
        "required_environment_variables": {
            "OPENAI_API_KEY": {
                "description": "OpenAI API key for GPT-4 access",
                "example": "sk-...",
                "required": True
            },
            "ANTHROPIC_API_KEY": {
                "description": "Anthropic API key for Claude access", 
                "example": "sk-ant-...",
                "required": True
            },
            "GOOGLE_API_KEY": {
                "description": "Google API key for Gemini access",
                "example": "AIza...",
                "required": True
            }
        },
        "optional_environment_variables": {
            "LEGAL_REWARD_LOG_LEVEL": {
                "description": "Override log level",
                "default": "INFO",
                "options": ["DEBUG", "INFO", "WARNING", "ERROR"]
            },
            "LEGAL_REWARD_CACHE_DIR": {
                "description": "Override cache directory",
                "default": "/tmp/legal_reward_cache",
                "example": "/path/to/cache"
            },
            "LEGAL_REWARD_MAX_BUDGET": {
                "description": "Override monthly API budget (USD)",
                "default": "5000",
                "example": "1000"
            },
            "LEGAL_REWARD_CONFIG_PATH": {
                "description": "Override configuration file path",
                "default": "config/internal_config.yaml",
                "example": "/path/to/config.yaml"
            }
        },
        "setup_instructions": [
            "1. Create API accounts with OpenAI, Anthropic, and Google",
            "2. Generate API keys for each provider",
            "3. Set required environment variables in your shell or .env file",
            "4. Optionally set environment overrides for system behavior",
            "5. Test configuration with validate_system_config()"
        ]
    }

def check_system_readiness() -> dict:
    """
    Check if the system is ready for operation.
    
    Returns:
        Dictionary with readiness status and recommendations
    """
    readiness_status = {
        "overall_ready": False,
        "checks": {},
        "blocking_issues": [],
        "recommendations": []
    }
    
    try:
        # Check environment variables
        env_manager = EnvironmentManager()
        env_validation = env_manager.validate_required_env_vars()
        
        readiness_status["checks"]["environment_variables"] = {
            "status": "pass" if env_validation.is_valid else "fail",
            "details": env_validation.get_summary()
        }
        
        if not env_validation.is_valid:
            readiness_status["blocking_issues"].extend(env_validation.missing_required)
        
        # Check configuration file availability
        try:
            config = LegalRewardSystemConfig()
            readiness_status["checks"]["configuration_file"] = {
                "status": "pass",
                "details": f"Configuration loaded from {config.config_path}"
            }
        except Exception as e:
            readiness_status["checks"]["configuration_file"] = {
                "status": "fail", 
                "details": str(e)
            }
            readiness_status["blocking_issues"].append("configuration_file_invalid")
        
        # Overall readiness
        readiness_status["overall_ready"] = len(readiness_status["blocking_issues"]) == 0
        
        # Generate recommendations
        if not readiness_status["overall_ready"]:
            if "environment_variables" in [issue for issue in readiness_status["blocking_issues"] if "API_KEY" in issue]:
                readiness_status["recommendations"].append("Set up required API keys using the environment setup guide")
            
            if "configuration_file_invalid" in readiness_status["blocking_issues"]:
                readiness_status["recommendations"].append("Create or fix configuration file using get_config_template()")
        
    except Exception as e:
        readiness_status["checks"]["system_check"] = {
            "status": "error",
            "details": str(e)
        }
        readiness_status["blocking_issues"].append("system_check_failed")
    
    return readiness_status

def generate_config_file(output_path: str, include_comments: bool = True) -> bool:
    """
    Generate a new configuration file with default values.
    
    Args:
        output_path: Path where to save the configuration file
        include_comments: Whether to include explanatory comments
        
    Returns:
        True if file was created successfully
    """
    try:
        import yaml
        from pathlib import Path
        
        # Get configuration template
        config_template = get_config_template()
        
        # Create output directory if needed
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Add header comment if requested
        content = ""
        if include_comments:
            content = """# Enhanced Multi-Task Legal Reward System Configuration
# Generated configuration file with default values
# 
# Required Environment Variables:
# - OPENAI_API_KEY: Your OpenAI API key
# - ANTHROPIC_API_KEY: Your Anthropic API key  
# - GOOGLE_API_KEY: Your Google API key
#
# For full configuration options, see the complete internal_config.yaml

"""
        
        # Write YAML configuration
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
            yaml.dump(config_template, f, default_flow_style=False, indent=2)
        
        return True
        
    except Exception:
        return False

# Module initialization
def _initialize_config_package():
    """Initialize config package with validation checks"""
    if not validate_config_imports():
        import warnings
        warnings.warn(
            "Config package imports failed validation. Some functionality may not work correctly.",
            ImportWarning,
            stacklevel=2
        )

# Run initialization when package is imported
_initialize_config_package()
