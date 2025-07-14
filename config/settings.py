"""
Configuration Management System for Multi-Task Legal Reward System

This module provides comprehensive configuration management with:
- YAML configuration file loading and validation
- Environment variable integration for secrets
- Configuration validation and type checking
- Property-based access to configuration sections
- Dynamic configuration updates during runtime
- Production-ready security and compliance features

Key Features:
- Complete US jurisdiction support configuration
- Multi-provider API configuration with cost optimization
- Aggressive caching settings for maximum cost savings
- Hybrid evaluation configuration (70% specialized + 30% chat)
- Smart rate limiting with provider fallback chains
- Comprehensive logging and monitoring settings
- Environment variable overrides for sensitive data
- Configuration validation with helpful error messages
"""

import os
import yaml
import copy
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, field

# Import core components
from core import (
    APIProvider, LegalTaskType, USJurisdiction, CacheStrategy,
    RateLimitStrategy, LogLevel, ConfigurationError, 
    create_error_context, VALID_API_PROVIDERS, VALID_TASK_TYPES
)


@dataclass
class ConfigValidationResult:
    """Result of configuration validation"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    missing_required: List[str] = field(default_factory=list)
    invalid_values: List[str] = field(default_factory=list)
    
    def has_critical_errors(self) -> bool:
        """Check if there are critical errors that prevent system operation"""
        return not self.is_valid or len(self.missing_required) > 0
    
    def get_summary(self) -> str:
        """Get human-readable validation summary"""
        if self.is_valid:
            warning_text = f" ({len(self.warnings)} warnings)" if self.warnings else ""
            return f"Configuration valid{warning_text}"
        else:
            return f"Configuration invalid: {len(self.errors)} errors, {len(self.missing_required)} missing required fields"


class EnvironmentManager:
    """
    Manage environment variable integration for sensitive configuration data.
    
    Handles secure loading of API keys and other sensitive configuration
    from environment variables while providing helpful error messages
    for missing required variables.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.EnvironmentManager")
        
        # Required environment variables for system operation
        self.required_env_vars = [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY", 
            "GOOGLE_API_KEY"
        ]
        
        # Optional environment variables with defaults
        self.optional_env_vars = {
            "LEGAL_REWARD_LOG_LEVEL": "INFO",
            "LEGAL_REWARD_CACHE_DIR": "/tmp/legal_reward_cache",
            "LEGAL_REWARD_MAX_BUDGET": "1000",
            "LEGAL_REWARD_CONFIG_PATH": None,
            "LEGAL_REWARD_CACHE_STRATEGY": "aggressive",
            "LEGAL_REWARD_MAX_CONCURRENT": "10"
        }
        
        # Environment variable mappings to config paths
        self.env_config_mappings = {
            "LEGAL_REWARD_LOG_LEVEL": ["logging", "level"],
            "LEGAL_REWARD_CACHE_DIR": ["caching", "cache_dir"],
            "LEGAL_REWARD_MAX_BUDGET": ["cost_optimization", "max_monthly_api_budget"],
            "LEGAL_REWARD_CACHE_STRATEGY": ["caching", "strategy"],
            "LEGAL_REWARD_MAX_CONCURRENT": ["performance", "max_concurrent_evaluations"]
        }
    
    def validate_required_env_vars(self) -> ConfigValidationResult:
        """
        Validate that all required environment variables are present.
        
        Returns:
            ConfigValidationResult with validation status
        """
        missing_vars = []
        errors = []
        
        for var_name in self.required_env_vars:
            if not os.getenv(var_name):
                missing_vars.append(var_name)
                errors.append(f"Missing required environment variable: {var_name}")
        
        is_valid = len(missing_vars) == 0
        
        if not is_valid:
            self.logger.error(f"Missing required environment variables: {missing_vars}")
        
        return ConfigValidationResult(
            is_valid=is_valid,
            errors=errors,
            missing_required=missing_vars
        )
    
    def get_api_keys(self) -> Dict[str, str]:
        """
        Get API keys from environment variables.
        
        Returns:
            Dictionary mapping provider names to API keys
            
        Raises:
            ConfigurationError: If required API keys are missing
        """
        validation_result = self.validate_required_env_vars()
        if not validation_result.is_valid:
            raise ConfigurationError(
                f"Missing required API keys: {validation_result.missing_required}",
                error_context=create_error_context("environment", "get_api_keys"),
                is_recoverable=False
            )
        
        return {
            "openai": os.getenv("OPENAI_API_KEY"),
            "anthropic": os.getenv("ANTHROPIC_API_KEY"),
            "google": os.getenv("GOOGLE_API_KEY")
        }
    
    def apply_environment_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides to configuration.
        
        Args:
            config: Base configuration dictionary
            
        Returns:
            Configuration with environment overrides applied
        """
        # Create a copy to avoid modifying original config
        updated_config = copy.deepcopy(config)
        
        # Apply environment variable overrides
        for env_var, config_path in self.env_config_mappings.items():
            env_value = os.getenv(env_var)
            if env_value:
                self._set_nested_config_value(updated_config, config_path, env_value)
                self.logger.info(f"Applied environment override: {env_var} -> {'.'.join(config_path)}")
        
        return updated_config
    
    def _set_nested_config_value(self, config: Dict[str, Any], path: List[str], value: str):
        """Set a nested configuration value from a dot-separated path"""
        current = config
        
        # Navigate to the parent of the target key
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Set the final value with appropriate type conversion
        final_key = path[-1]
        current[final_key] = self._convert_env_value(value, final_key)
    
    def _convert_env_value(self, value: str, key: str) -> Union[str, int, float, bool]:
        """Convert environment variable string to appropriate type"""
        # Boolean values
        if value.lower() in ("true", "false"):
            return value.lower() == "true"
        
        # Numeric values
        if key in ["max_monthly_api_budget", "max_concurrent_evaluations"]:
            try:
                if "." in value:
                    return float(value)
                else:
                    return int(value)
            except ValueError:
                return value
        
        # Default to string
        return value
    
    def get_environment_summary(self) -> Dict[str, Any]:
        """Get summary of environment variable status"""
        api_keys_present = {
            var: bool(os.getenv(var)) for var in self.required_env_vars
        }
        
        overrides_applied = {
            var: bool(os.getenv(var)) for var in self.env_config_mappings.keys()
        }
        
        return {
            "required_api_keys": api_keys_present,
            "all_api_keys_present": all(api_keys_present.values()),
            "optional_overrides": overrides_applied,
            "total_overrides_applied": sum(overrides_applied.values())
        }


class ConfigValidator:
    """
    Comprehensive configuration validation with legal system specific checks.
    
    Validates configuration values, types, ranges, and logical consistency
    to ensure the legal reward system can operate correctly.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ConfigValidator")
        
        # Required configuration sections
        self.required_sections = [
            "api_providers", "caching", "rate_limiting", "hybrid_evaluation",
            "task_weights", "cost_optimization", "logging", "jurisdiction"
        ]
        
        # Required API provider fields
        self.required_provider_fields = [
            "model", "rate_limit_rpm", "rate_limit_tpm", "cost_per_1k_input_tokens",
            "cost_per_1k_output_tokens", "fallback_priority"
        ]
        
        # Valid ranges for numeric values
        self.numeric_ranges = {
            "rate_limit_rpm": (1, 10000),
            "rate_limit_tpm": (100, 1000000),
            "max_concurrent_requests": (1, 100),
            "cost_per_1k_input_tokens": (0.0, 1.0),
            "cost_per_1k_output_tokens": (0.0, 1.0),
            "fallback_priority": (1, 10),
            "max_cache_size_gb": (0.1, 1000),
            "cache_ttl_hours": (1, 8760),  # 1 hour to 1 year
            "specialized_weight": (0.0, 1.0),
            "general_chat_weight": (0.0, 1.0),
            "jurisdiction_failure_penalty": (0.0, 1.0),
            "max_monthly_api_budget": (10, 100000)
        }
    
    def validate_configuration(self, config: Dict[str, Any]) -> ConfigValidationResult:
        """
        Perform comprehensive configuration validation.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            ConfigValidationResult with detailed validation information
        """
        result = ConfigValidationResult(is_valid=True)
        
        # Validate required sections
        self._validate_required_sections(config, result)
        
        # Validate API providers
        self._validate_api_providers(config, result)
        
        # Validate caching configuration
        self._validate_caching_config(config, result)
        
        # Validate hybrid evaluation configuration  
        self._validate_hybrid_evaluation_config(config, result)
        
        # Validate task weights
        self._validate_task_weights(config, result)
        
        # Validate cost optimization
        self._validate_cost_optimization(config, result)
        
        # Validate logging configuration
        self._validate_logging_config(config, result)
        
        # Validate jurisdiction configuration
        self._validate_jurisdiction_config(config, result)
        
        # Check logical consistency
        self._validate_logical_consistency(config, result)
        
        # Update overall validity
        result.is_valid = len(result.errors) == 0 and len(result.missing_required) == 0
        
        if not result.is_valid:
            self.logger.error(f"Configuration validation failed: {result.get_summary()}")
        elif result.warnings:
            self.logger.warning(f"Configuration validation completed with warnings: {len(result.warnings)} warnings")
        else:
            self.logger.info("Configuration validation passed successfully")
        
        return result
    
    def _validate_required_sections(self, config: Dict[str, Any], result: ConfigValidationResult):
        """Validate that all required configuration sections are present"""
        for section in self.required_sections:
            if section not in config:
                result.missing_required.append(section)
                result.errors.append(f"Missing required configuration section: {section}")
    
    def _validate_api_providers(self, config: Dict[str, Any], result: ConfigValidationResult):
        """Validate API provider configuration"""
        if "api_providers" not in config:
            return
        
        providers = config["api_providers"]
        
        # Check that all required providers are present
        for provider_name in ["openai", "anthropic", "google"]:
            if provider_name not in providers:
                result.errors.append(f"Missing API provider configuration: {provider_name}")
                continue
            
            provider_config = providers[provider_name]
            
            # Validate required fields
            for field in self.required_provider_fields:
                if field not in provider_config:
                    result.errors.append(f"Missing required field '{field}' for provider '{provider_name}'")
                    continue
                
                # Validate numeric ranges
                if field in self.numeric_ranges:
                    value = provider_config[field]
                    min_val, max_val = self.numeric_ranges[field]
                    if not isinstance(value, (int, float)) or not (min_val <= value <= max_val):
                        result.errors.append(f"Invalid value for {provider_name}.{field}: {value} (expected {min_val}-{max_val})")
            
            # Validate model names
            valid_models = {
                "openai": ["gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
                "anthropic": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-sonnet-20240229"],
                "google": ["gemini-1.5-pro", "gemini-1.0-pro"]
            }
            
            model = provider_config.get("model", "")
            if model not in valid_models.get(provider_name, []):
                result.warnings.append(f"Unrecognized model for {provider_name}: {model}")
    
    def _validate_caching_config(self, config: Dict[str, Any], result: ConfigValidationResult):
        """Validate caching configuration"""
        if "caching" not in config:
            return
        
        caching = config["caching"]
        
        # Validate cache strategy
        strategy = caching.get("strategy", "")
        valid_strategies = ["aggressive", "balanced", "conservative", "disabled"]
        if strategy not in valid_strategies:
            result.errors.append(f"Invalid cache strategy: {strategy} (expected one of {valid_strategies})")
        
        # Validate numeric values
        for field in ["max_cache_size_gb", "cache_ttl_hours"]:
            if field in caching and field in self.numeric_ranges:
                value = caching[field]
                min_val, max_val = self.numeric_ranges[field]
                if not isinstance(value, (int, float)) or not (min_val <= value <= max_val):
                    result.errors.append(f"Invalid caching.{field}: {value} (expected {min_val}-{max_val})")
        
        # Validate cache directory
        cache_dir = caching.get("cache_dir", "")
        if cache_dir:
            try:
                Path(cache_dir).parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                result.warnings.append(f"Cache directory may not be writable: {cache_dir} ({e})")
    
    def _validate_hybrid_evaluation_config(self, config: Dict[str, Any], result: ConfigValidationResult):
        """Validate hybrid evaluation configuration"""
        if "hybrid_evaluation" not in config:
            return
        
        hybrid = config["hybrid_evaluation"]
        
        # Validate weights
        specialized_weight = hybrid.get("specialized_weight", 0.7)
        general_chat_weight = hybrid.get("general_chat_weight", 0.3)
        
        # Check that weights sum to 1.0 (within tolerance)
        weight_sum = specialized_weight + general_chat_weight
        if abs(weight_sum - 1.0) > 0.01:
            result.errors.append(f"Hybrid evaluation weights must sum to 1.0, got {weight_sum}")
        
        # Validate individual weight ranges
        for weight_name in ["specialized_weight", "general_chat_weight", "jurisdiction_failure_penalty"]:
            if weight_name in hybrid and weight_name in self.numeric_ranges:
                value = hybrid[weight_name]
                min_val, max_val = self.numeric_ranges[weight_name]
                if not isinstance(value, (int, float)) or not (min_val <= value <= max_val):
                    result.errors.append(f"Invalid hybrid_evaluation.{weight_name}: {value} (expected {min_val}-{max_val})")
    
    def _validate_task_weights(self, config: Dict[str, Any], result: ConfigValidationResult):
        """Validate task difficulty weights"""
        if "task_weights" not in config:
            return
        
        task_weights = config["task_weights"]
        
        # Validate that all task types have weights
        for task_type in VALID_TASK_TYPES:
            if task_type not in task_weights:
                result.warnings.append(f"Missing task weight for: {task_type}")
            else:
                weight = task_weights[task_type]
                if not isinstance(weight, (int, float)) or weight <= 0 or weight > 5.0:
                    result.errors.append(f"Invalid task weight for {task_type}: {weight} (expected 0.1-5.0)")
    
    def _validate_cost_optimization(self, config: Dict[str, Any], result: ConfigValidationResult):
        """Validate cost optimization configuration"""
        if "cost_optimization" not in config:
            return
        
        cost_config = config["cost_optimization"]
        
        # Validate budget
        budget = cost_config.get("max_monthly_api_budget", 0)
        if "max_monthly_api_budget" in self.numeric_ranges:
            min_val, max_val = self.numeric_ranges["max_monthly_api_budget"]
            if not isinstance(budget, (int, float)) or not (min_val <= budget <= max_val):
                result.errors.append(f"Invalid max_monthly_api_budget: {budget} (expected {min_val}-{max_val})")
        
        # Validate budget alert thresholds
        thresholds = cost_config.get("budget_alert_thresholds", [])
        if thresholds:
            for threshold in thresholds:
                if not isinstance(threshold, (int, float)) or not (0.0 <= threshold <= 1.0):
                    result.errors.append(f"Invalid budget alert threshold: {threshold} (expected 0.0-1.0)")
    
    def _validate_logging_config(self, config: Dict[str, Any], result: ConfigValidationResult):
        """Validate logging configuration"""
        if "logging" not in config:
            return
        
        logging_config = config["logging"]
        
        # Validate log level
        level = logging_config.get("level", "INFO")
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if level not in valid_levels:
            result.errors.append(f"Invalid log level: {level} (expected one of {valid_levels})")
        
        # Validate file size
        max_size = logging_config.get("max_file_size_mb", 100)
        if not isinstance(max_size, (int, float)) or max_size <= 0:
            result.errors.append(f"Invalid max_file_size_mb: {max_size} (expected positive number)")
    
    def _validate_jurisdiction_config(self, config: Dict[str, Any], result: ConfigValidationResult):
        """Validate US jurisdiction configuration"""
        if "jurisdiction" not in config:
            return
        
        jurisdiction = config["jurisdiction"]
        
        # Validate inference confidence threshold
        threshold = jurisdiction.get("inference_confidence_threshold", 0.7)
        if not isinstance(threshold, (int, float)) or not (0.0 <= threshold <= 1.0):
            result.errors.append(f"Invalid inference_confidence_threshold: {threshold} (expected 0.0-1.0)")
        
        # Validate default jurisdiction
        default_jurisdiction = jurisdiction.get("default_jurisdiction", "general")
        valid_defaults = ["general", "federal"] + [j.value for j in USJurisdiction]
        if default_jurisdiction not in valid_defaults:
            result.warnings.append(f"Unusual default jurisdiction: {default_jurisdiction}")
    
    def _validate_logical_consistency(self, config: Dict[str, Any], result: ConfigValidationResult):
        """Validate logical consistency across configuration sections"""
        # Check that cache TTL makes sense with cost optimization
        if "caching" in config and "cost_optimization" in config:
            cache_enabled = config["caching"].get("enabled", True)
            ttl_hours = config["caching"].get("cache_ttl_hours", 168)
            
            if cache_enabled and ttl_hours < 24:
                result.warnings.append("Short cache TTL may reduce cost optimization effectiveness")
        
        # Check that rate limits are reasonable for concurrent requests
        if "api_providers" in config:
            for provider_name, provider_config in config["api_providers"].items():
                rpm = provider_config.get("rate_limit_rpm", 0)
                concurrent = provider_config.get("max_concurrent_requests", 0)
                
                if concurrent > 0 and rpm > 0:
                    # Rough check: concurrent requests shouldn't exceed rate limit capacity
                    if concurrent > rpm / 2:  # Allow some headroom
                        result.warnings.append(f"High concurrent requests vs rate limit for {provider_name}: "
                                             f"{concurrent} concurrent, {rpm} RPM")


class LegalRewardSystemConfig:
    """
    Comprehensive configuration management for the legal reward system.
    
    Provides centralized configuration loading, validation, and access with
    environment variable integration, runtime updates, and production-ready
    error handling for the enhanced multi-task legal reward system.
    
    Features:
    - YAML configuration file loading with validation
    - Environment variable overrides for sensitive data
    - Property-based access to configuration sections
    - Runtime configuration updates and validation
    - Comprehensive error handling with helpful messages
    - US jurisdiction system configuration
    - Multi-provider API configuration with cost optimization
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(f"{__name__}.LegalRewardSystemConfig")
        
        # Configuration management components
        self.environment_manager = EnvironmentManager()
        self.config_validator = ConfigValidator()
        
        # Load configuration
        self.config_path = config_path or self._get_default_config_path()
        self.config = self._load_config()
        
        # Apply environment overrides and validate
        self.config = self.environment_manager.apply_environment_overrides(self.config)
        self._validate_config()
        
        # Add API keys from environment
        self._setup_api_keys()
        
        self.logger.info(f"Legal reward system configuration loaded successfully from: {self.config_path}")
    
    def _get_default_config_path(self) -> Path:
        """Get default configuration file path"""
        # Check environment variable first
        env_path = os.getenv("LEGAL_REWARD_CONFIG_PATH")
        if env_path:
            return Path(env_path)
        
        # Look for config file in standard locations
        possible_paths = [
            Path("config/internal_config.yaml"),
            Path("legal_reward_system/config/internal_config.yaml"),
            Path("/etc/legal_reward_system/config.yaml"),
            Path.home() / ".legal_reward_system" / "config.yaml"
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        # Default to the first option
        return possible_paths[0]
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file with comprehensive error handling"""
        try:
            if not self.config_path.exists():
                raise ConfigurationError(
                    f"Configuration file not found: {self.config_path}",
                    config_file=str(self.config_path),
                    error_context=create_error_context("configuration", "load_config"),
                    is_recoverable=False
                )
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if not isinstance(config, dict):
                raise ConfigurationError(
                    f"Configuration file must contain a dictionary, got {type(config)}",
                    config_file=str(self.config_path),
                    error_context=create_error_context("configuration", "load_config"),
                    is_recoverable=False
                )
            
            self.logger.info(f"Loaded configuration from {self.config_path}")
            return config
            
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Invalid YAML in configuration file: {e}",
                config_file=str(self.config_path),
                error_context=create_error_context("configuration", "parse_yaml"),
                original_exception=e,
                is_recoverable=False
            )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to load configuration file: {e}",
                config_file=str(self.config_path),
                error_context=create_error_context("configuration", "load_config"),
                original_exception=e,
                is_recoverable=False
            )
    
    def _validate_config(self):
        """Validate configuration with comprehensive error reporting"""
        # Validate environment variables first
        env_validation = self.environment_manager.validate_required_env_vars()
        if not env_validation.is_valid:
            raise ConfigurationError(
                f"Missing required environment variables: {env_validation.missing_required}",
                error_context=create_error_context("configuration", "validate_environment"),
                is_recoverable=False
            )
        
        # Validate configuration structure and values
        config_validation = self.config_validator.validate_configuration(self.config)
        if config_validation.has_critical_errors():
            error_details = "; ".join(config_validation.errors + 
                                    [f"Missing: {m}" for m in config_validation.missing_required])
            raise ConfigurationError(
                f"Configuration validation failed: {error_details}",
                config_file=str(self.config_path),
                error_context=create_error_context("configuration", "validate_config"),
                is_recoverable=False
            )
        
        # Log warnings if any
        if config_validation.warnings:
            for warning in config_validation.warnings:
                self.logger.warning(f"Configuration warning: {warning}")
    
    def _setup_api_keys(self):
        """Setup API keys from environment variables"""
        try:
            api_keys = self.environment_manager.get_api_keys()
            
            # Add API keys to provider configurations
            for provider_name, api_key in api_keys.items():
                if provider_name in self.config.get("api_providers", {}):
                    self.config["api_providers"][provider_name]["api_key"] = api_key
            
            self.logger.info("API keys loaded successfully from environment variables")
            
        except ConfigurationError:
            # Re-raise configuration errors
            raise
        except Exception as e:
            raise ConfigurationError(
                f"Failed to setup API keys: {e}",
                error_context=create_error_context("configuration", "setup_api_keys"),
                original_exception=e,
                is_recoverable=False
            )
    
    # Property-based access to configuration sections
    
    @property
    def api_providers(self) -> Dict[str, Any]:
        """Get API provider configurations"""
        return self.config.get("api_providers", {})
    
    @property  
    def caching_config(self) -> Dict[str, Any]:
        """Get caching configuration"""
        return self.config.get("caching", {})
    
    @property
    def rate_limiting_config(self) -> Dict[str, Any]:
        """Get rate limiting configuration"""
        return self.config.get("rate_limiting", {})
    
    @property
    def hybrid_evaluation_config(self) -> Dict[str, Any]:
        """Get hybrid evaluation configuration"""
        return self.config.get("hybrid_evaluation", {})
    
    @property
    def task_weights(self) -> Dict[str, float]:
        """Get task difficulty weights"""
        return self.config.get("task_weights", {})
    
    @property
    def cost_optimization(self) -> Dict[str, Any]:
        """Get cost optimization configuration"""
        return self.config.get("cost_optimization", {})
    
    @property
    def logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.config.get("logging", {})
    
    @property
    def jurisdiction_config(self) -> Dict[str, Any]:
        """Get US jurisdiction configuration"""
        return self.config.get("jurisdiction", {})
    
    @property
    def performance_config(self) -> Dict[str, Any]:
        """Get performance configuration"""
        return self.config.get("performance", {})
    
    @property
    def judge_ensembles_config(self) -> Dict[str, Any]:
        """Get judge ensemble configuration"""
        return self.config.get("judge_ensembles", {})
    
    @property
    def verl_integration_config(self) -> Dict[str, Any]:
        """Get VERL integration configuration"""
        return self.config.get("verl_integration", {})
    
    @property
    def security_config(self) -> Dict[str, Any]:
        """Get security configuration"""
        return self.config.get("security", {})
    
    @property
    def system_metadata(self) -> Dict[str, Any]:
        """Get system metadata"""
        return self.config.get("system", {})
    
    # Provider-specific configuration access
    
    def get_provider_config(self, provider: APIProvider) -> Dict[str, Any]:
        """
        Get configuration for specific API provider.
        
        Args:
            provider: API provider enum
            
        Returns:
            Provider configuration dictionary
            
        Raises:
            ConfigurationError: If provider not configured
        """
        provider_name = provider.value
        providers = self.api_providers
        
        if provider_name not in providers:
            raise ConfigurationError(
                f"No configuration found for API provider: {provider_name}",
                config_key=f"api_providers.{provider_name}",
                error_context=create_error_context("configuration", "get_provider_config")
            )
        
        return providers[provider_name]
    
    def get_task_weight(self, task_type: LegalTaskType) -> float:
        """
        Get task difficulty weight for specific task type.
        
        Args:
            task_type: Legal task type enum
            
        Returns:
            Task difficulty weight
        """
        task_name = task_type.value
        weights = self.task_weights
        
        if task_name not in weights:
            # Return default weight based on task type
            default_weight = task_type.get_default_difficulty_weight()
            self.logger.warning(f"No weight configured for {task_name}, using default: {default_weight}")
            return default_weight
        
        return weights[task_name]
    
    # Runtime configuration updates
    
    def update_task_weight(self, task_type: LegalTaskType, weight: float):
        """
        Update task difficulty weight at runtime.
        
        Args:
            task_type: Legal task type to update
            weight: New weight value
            
        Raises:
            ConfigurationError: If weight is invalid
        """
        if not isinstance(weight, (int, float)) or weight <= 0 or weight > 5.0:
            raise ConfigurationError(
                f"Invalid task weight: {weight} (expected 0.1-5.0)",
                config_key=f"task_weights.{task_type.value}",
                error_context=create_error_context("configuration", "update_task_weight")
            )
        
        task_name = task_type.value
        old_weight = self.task_weights.get(task_name, 1.0)
        
        if "task_weights" not in self.config:
            self.config["task_weights"] = {}
        
        self.config["task_weights"][task_name] = weight
        
        self.logger.info(f"Updated task weight for {task_name}: {old_weight} -> {weight}")
    
    def update_api_budget(self, new_budget: float):
        """
        Update monthly API budget at runtime.
        
        Args:
            new_budget: New budget amount in USD
            
        Raises:
            ConfigurationError: If budget is invalid
        """
        if not isinstance(new_budget, (int, float)) or new_budget < 10:
            raise ConfigurationError(
                f"Invalid API budget: {new_budget} (expected >= 10)",
                config_key="cost_optimization.max_monthly_api_budget",
                error_context=create_error_context("configuration", "update_api_budget")
            )
        
        old_budget = self.cost_optimization.get("max_monthly_api_budget", 1000)
        
        if "cost_optimization" not in self.config:
            self.config["cost_optimization"] = {}
        
        self.config["cost_optimization"]["max_monthly_api_budget"] = new_budget
        
        self.logger.info(f"Updated API budget: ${old_budget} -> ${new_budget}")
    
    # Configuration introspection and debugging
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive configuration summary for monitoring and debugging.
        
        Returns:
            Configuration summary with key metrics and status
        """
        # Environment status
        env_summary = self.environment_manager.get_environment_summary()
        
        # Configuration sections status
        sections_present = {
            section: section in self.config 
            for section in self.config_validator.required_sections
        }
        
        # API providers status
        providers_configured = {
            provider: provider in self.api_providers
            for provider in ["openai", "anthropic", "google"]
        }
        
        # Cache and performance settings
        cache_strategy = self.caching_config.get("strategy", "unknown")
        max_concurrent = self.performance_config.get("max_concurrent_evaluations", 10)
        monthly_budget = self.cost_optimization.get("max_monthly_api_budget", 5000)
        
        return {
            "config_file": str(self.config_path),
            "config_valid": True,  # If we got here, config is valid
            "environment_status": env_summary,
            "sections_present": sections_present,
            "providers_configured": providers_configured,
            "cache_strategy": cache_strategy,
            "max_concurrent_evaluations": max_concurrent,
            "monthly_api_budget": monthly_budget,
            "total_task_types": len(self.task_weights),
            "hybrid_evaluation_enabled": self.hybrid_evaluation_config.get("require_jurisdiction_compliance", True),
            "jurisdiction_inference_enabled": self.jurisdiction_config.get("enable_inference", True),
            "system_version": self.system_metadata.get("version", "unknown")
        }
    
    def export_config(self, include_secrets: bool = False) -> Dict[str, Any]:
        """
        Export current configuration for debugging or backup.
        
        Args:
            include_secrets: Whether to include API keys (default: False)
            
        Returns:
            Configuration dictionary
        """
        exported_config = copy.deepcopy(self.config)
        
        if not include_secrets:
            # Remove API keys from export
            for provider_name in ["openai", "anthropic", "google"]:
                if (provider_name in exported_config.get("api_providers", {}) and
                    "api_key" in exported_config["api_providers"][provider_name]):
                    exported_config["api_providers"][provider_name]["api_key"] = "[REDACTED]"
        
        return exported_config
    
    def validate_runtime_config(self) -> ConfigValidationResult:
        """
        Re-validate current configuration (useful after runtime updates).
        
        Returns:
            ConfigValidationResult with current validation status
        """
        return self.config_validator.validate_configuration(self.config)


# Factory functions for different environments

def create_production_config(config_path: Optional[str] = None) -> LegalRewardSystemConfig:
    """
    Create production configuration with validation and security checks.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Production-ready LegalRewardSystemConfig
    """
    return LegalRewardSystemConfig(config_path)


def create_development_config() -> LegalRewardSystemConfig:
    """
    Create development configuration with relaxed settings.
    
    Returns:
        Development LegalRewardSystemConfig
    """
    # For development, we might want to override some settings
    config = LegalRewardSystemConfig()
    
    # Apply development overrides
    config.config.setdefault("development", {})["test_mode"] = False
    config.config["logging"]["level"] = "DEBUG"
    config.config["caching"]["cache_ttl_hours"] = 24  # Shorter for development
    
    return config


def create_test_config() -> LegalRewardSystemConfig:
    """
    Create test configuration with minimal settings and mock APIs.
    
    Returns:
        Test LegalRewardSystemConfig
    """
    config = LegalRewardSystemConfig()
    
    # Apply test overrides
    config.config.setdefault("development", {})["test_mode"] = True
    config.config["logging"]["level"] = "WARNING"  # Reduce test noise
    config.config["caching"]["enabled"] = False   # No caching in tests
    config.config["cost_optimization"]["max_monthly_api_budget"] = 100
    
    return config


def validate_system_config(config_path: Optional[str] = None) -> ConfigValidationResult:
    """
    Standalone configuration validation without creating full config object.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        ConfigValidationResult with validation status
    """
    try:
        config = LegalRewardSystemConfig(config_path)
        return config.validate_runtime_config()
    except ConfigurationError as e:
        return ConfigValidationResult(
            is_valid=False,
            errors=[str(e)],
            missing_required=["configuration_file"]
        )


# Configuration utilities

def get_config_template() -> Dict[str, Any]:
    """
    Get configuration template for creating new configuration files.
    
    Returns:
        Dictionary with minimal required configuration structure
    """
    return {
        "api_providers": {
            "openai": {
                "model": "gpt-4-turbo",
                "rate_limit_rpm": 500,
                "rate_limit_tpm": 30000,
                "cost_per_1k_input_tokens": 0.01,
                "cost_per_1k_output_tokens": 0.03,
                "fallback_priority": 1
            },
            "anthropic": {
                "model": "claude-3-5-sonnet-20241022",
                "rate_limit_rpm": 400,
                "rate_limit_tpm": 40000,
                "cost_per_1k_input_tokens": 0.003,
                "cost_per_1k_output_tokens": 0.015,
                "fallback_priority": 2
            },
            "google": {
                "model": "gemini-1.5-pro",
                "rate_limit_rpm": 300,
                "rate_limit_tpm": 32000,
                "cost_per_1k_input_tokens": 0.00125,
                "cost_per_1k_output_tokens": 0.005,
                "fallback_priority": 3
            }
        },
        "caching": {
            "enabled": True,
            "strategy": "aggressive",
            "cache_dir": "/tmp/legal_reward_cache",
            "max_cache_size_gb": 10,
            "cache_ttl_hours": 168
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
        "logging": {
            "level": "INFO",
            "structured_logging": True
        },
        "jurisdiction": {
            "enable_inference": True,
            "require_compliance_check": True
        }
    }
