"""
Enhanced Configuration Management for Legal Reward System

This module provides comprehensive configuration management for the Enhanced 
Multi-Task Legal Reward System, including the new prompt template system
integration and all system components.

Updated to support:
- Prompt template loading and management
- US jurisdiction-aware configuration
- Enhanced API client configuration
- Hybrid evaluation system settings
- VERL integration configuration

Author: Legal Reward System Team
Version: 1.0.1 (Updated for prompt template system)
"""

import os
import yaml
import json
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Import prompt template system
from .prompts import (
    UnifiedPromptTemplateSystem,
    PromptTemplateConfig,
    get_prompt_system_info,
    validate_prompt_system
)

# Import other system components
from ..jurisdiction.us_system import USJurisdiction
from ..core.enums import LegalTaskType


class ConfigurationError(Exception):
    """Configuration-related errors"""
    pass


class EnvironmentType(Enum):
    """System environment types"""
    PRODUCTION = "production"
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"


@dataclass
class APIProviderConfig:
    """Configuration for individual API providers"""
    enabled: bool = True
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    rate_limit_requests_per_minute: int = 60
    rate_limit_tokens_per_minute: int = 50000
    timeout_seconds: int = 30
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    cost_per_1k_tokens: float = 0.002
    
    # Model-specific settings
    default_model: Optional[str] = None
    available_models: List[str] = field(default_factory=list)
    context_window: int = 4096
    max_tokens: int = 1000


@dataclass
class PromptTemplateSystemConfig:
    """Configuration for the prompt template system"""
    enabled: bool = True
    cache_enabled: bool = True
    cache_size_limit: int = 1000
    validation_enabled: bool = True
    gating_enabled: bool = True
    jurisdiction_validation: bool = True
    
    # Performance settings
    parallel_loading: bool = True
    memory_optimization: bool = True
    
    # Quality settings
    prompt_quality_validation: bool = True
    jurisdiction_accuracy_validation: bool = True
    
    # Gating settings
    gating_threshold: float = 3.0
    gating_failure_penalty: float = 0.5
    
    # Template management
    auto_reload_templates: bool = False
    template_validation_on_load: bool = True


@dataclass
class HybridEvaluationConfig:
    """Configuration for hybrid evaluation system"""
    enabled: bool = True
    specialized_weight: float = 0.7
    general_chat_weight: float = 0.3
    
    # Jurisdiction compliance gating
    jurisdiction_compliance_enabled: bool = True
    jurisdiction_compliance_threshold: float = 3.0
    jurisdiction_failure_penalty: float = 0.5
    
    # Performance optimization
    parallel_evaluation: bool = True
    cache_hybrid_results: bool = True
    
    # Quality assurance
    minimum_ensemble_confidence: float = 0.5
    require_jurisdiction_compliance: bool = True


@dataclass
class JurisdictionSystemConfig:
    """Configuration for US jurisdiction system"""
    enabled: bool = True
    inference_enabled: bool = True
    compliance_checking_enabled: bool = True
    
    # Supported jurisdictions
    supported_jurisdictions: List[USJurisdiction] = field(default_factory=lambda: list(USJurisdiction))
    default_jurisdiction: USJurisdiction = USJurisdiction.GENERAL
    
    # Inference settings
    inference_confidence_threshold: float = 0.7
    fallback_to_general: bool = True
    
    # Compliance settings
    strict_compliance_mode: bool = True
    compliance_validation_enabled: bool = True


@dataclass
class CostOptimizationConfig:
    """Configuration for cost optimization"""
    enabled: bool = True
    max_monthly_budget: float = 5000.0
    
    # Provider routing
    prefer_cheaper_models: bool = True
    use_fallback_chain: bool = True
    complexity_routing_enabled: bool = True
    
    # Budget management
    budget_alert_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.75, 0.9, 0.95])
    cost_tracking_enabled: bool = True
    
    # Performance vs cost balance
    cost_performance_balance: float = 0.7  # 0.0 = pure cost, 1.0 = pure performance


@dataclass
class LegalRewardSystemConfig:
    """
    Comprehensive configuration for the Enhanced Multi-Task Legal Reward System.
    
    This configuration class integrates all system components including the new
    prompt template system, jurisdiction awareness, and hybrid evaluation.
    """
    
    # Environment and basic settings
    environment: EnvironmentType = EnvironmentType.DEVELOPMENT
    system_name: str = "Enhanced Multi-Task Legal Reward System"
    version: str = "1.0.1"
    
    # API Provider configurations
    openai: APIProviderConfig = field(default_factory=lambda: APIProviderConfig(
        default_model="gpt-4",
        available_models=["gpt-4", "gpt-4-32k", "gpt-3.5-turbo"],
        context_window=8192,
        cost_per_1k_tokens=0.03
    ))
    
    anthropic: APIProviderConfig = field(default_factory=lambda: APIProviderConfig(
        default_model="claude-3-sonnet-20240229",
        available_models=["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
        context_window=200000,
        cost_per_1k_tokens=0.015
    ))
    
    google: APIProviderConfig = field(default_factory=lambda: APIProviderConfig(
        default_model="gemini-pro",
        available_models=["gemini-pro", "gemini-pro-vision"],
        context_window=32768,
        cost_per_1k_tokens=0.001
    ))
    
    # System component configurations
    prompt_template_system: PromptTemplateSystemConfig = field(default_factory=PromptTemplateSystemConfig)
    hybrid_evaluation: HybridEvaluationConfig = field(default_factory=HybridEvaluationConfig)
    jurisdiction_system: JurisdictionSystemConfig = field(default_factory=JurisdictionSystemConfig)
    cost_optimization: CostOptimizationConfig = field(default_factory=CostOptimizationConfig)
    
    # Task difficulty weights
    task_weights: Dict[str, float] = field(default_factory=lambda: {
        "judicial_reasoning": 1.5,
        "precedent_analysis": 1.3,
        "opinion_generation": 1.1,
        "general_chat": 1.0
    })
    
    # Logging and monitoring
    logging_level: str = "INFO"
    log_file_path: Optional[str] = None
    performance_monitoring_enabled: bool = True
    
    # Caching configuration
    cache_enabled: bool = True
    cache_strategy: str = "multi_strategy"
    cache_ttl_seconds: int = 3600
    cache_max_size: int = 1000
    
    # Rate limiting
    global_rate_limit_enabled: bool = True
    rate_limit_requests_per_minute: int = 300
    rate_limit_tokens_per_minute: int = 150000
    
    # VERL integration
    verl_integration_enabled: bool = True
    verl_batch_size: int = 32
    verl_timeout_seconds: int = 300
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        self._validate_configuration()
        self._setup_api_keys()
        self._initialize_prompt_system()
    
    def _validate_configuration(self):
        """Validate configuration consistency and requirements"""
        
        # Validate hybrid evaluation weights
        if abs(self.hybrid_evaluation.specialized_weight + self.hybrid_evaluation.general_chat_weight - 1.0) > 0.01:
            raise ConfigurationError("Hybrid evaluation weights must sum to 1.0")
        
        # Validate task weights
        if not all(weight > 0 for weight in self.task_weights.values()):
            raise ConfigurationError("All task weights must be positive")
        
        # Validate cost optimization
        if self.cost_optimization.max_monthly_budget <= 0:
            raise ConfigurationError("Monthly budget must be positive")
        
        # Validate jurisdiction system
        if not self.jurisdiction_system.supported_jurisdictions:
            raise ConfigurationError("At least one jurisdiction must be supported")
        
        # Validate prompt template system
        if self.prompt_template_system.gating_threshold < 0 or self.prompt_template_system.gating_threshold > 10:
            raise ConfigurationError("Gating threshold must be between 0 and 10")
    
    def _setup_api_keys(self):
        """Setup API keys from environment variables"""
        
        # OpenAI
        if not self.openai.api_key:
            self.openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # Anthropic
        if not self.anthropic.api_key:
            self.anthropic.api_key = os.getenv("ANTHROPIC_API_KEY")
        
        # Google
        if not self.google.api_key:
            self.google.api_key = os.getenv("GOOGLE_API_KEY")
        
        # Validate required API keys for production
        if self.environment == EnvironmentType.PRODUCTION:
            if not any([self.openai.api_key, self.anthropic.api_key, self.google.api_key]):
                raise ConfigurationError("At least one API key must be configured for production")
    
    def _initialize_prompt_system(self):
        """Initialize the prompt template system"""
        
        if self.prompt_template_system.enabled:
            try:
                # Validate prompt system
                validation_result = validate_prompt_system()
                if not validation_result["system_valid"]:
                    if self.environment == EnvironmentType.PRODUCTION:
                        raise ConfigurationError(f"Prompt system validation failed: {validation_result['issues']}")
                    else:
                        import warnings
                        warnings.warn(f"Prompt system validation issues: {validation_result['issues']}")
                
            except Exception as e:
                if self.environment == EnvironmentType.PRODUCTION:
                    raise ConfigurationError(f"Failed to initialize prompt system: {str(e)}")
                else:
                    import warnings
                    warnings.warn(f"Could not initialize prompt system: {str(e)}")
    
    def get_api_provider_config(self, provider_name: str) -> APIProviderConfig:
        """Get configuration for a specific API provider"""
        
        provider_configs = {
            "openai": self.openai,
            "anthropic": self.anthropic,
            "google": self.google
        }
        
        if provider_name not in provider_configs:
            raise ConfigurationError(f"Unknown API provider: {provider_name}")
        
        return provider_configs[provider_name]
    
    def get_enabled_providers(self) -> List[str]:
        """Get list of enabled API providers"""
        
        enabled_providers = []
        
        if self.openai.enabled and self.openai.api_key:
            enabled_providers.append("openai")
        
        if self.anthropic.enabled and self.anthropic.api_key:
            enabled_providers.append("anthropic")
        
        if self.google.enabled and self.google.api_key:
            enabled_providers.append("google")
        
        return enabled_providers
    
    def get_prompt_template_config(self) -> PromptTemplateConfig:
        """Get prompt template configuration for the UnifiedPromptTemplateSystem"""
        
        config = PromptTemplateConfig()
        config.cache_enabled = self.prompt_template_system.cache_enabled
        config.cache_size_limit = self.prompt_template_system.cache_size_limit
        config.validation_enabled = self.prompt_template_system.validation_enabled
        config.gating_enabled = self.prompt_template_system.gating_enabled
        config.jurisdiction_validation = self.prompt_template_system.jurisdiction_validation
        config.parallel_loading = self.prompt_template_system.parallel_loading
        config.memory_optimization = self.prompt_template_system.memory_optimization
        config.prompt_quality_validation = self.prompt_template_system.prompt_quality_validation
        config.jurisdiction_accuracy_validation = self.prompt_template_system.jurisdiction_accuracy_validation
        
        return config
    
    def get_task_weight(self, task_type: LegalTaskType) -> float:
        """Get difficulty weight for a task type"""
        return self.task_weights.get(task_type.value, 1.0)
    
    def is_jurisdiction_supported(self, jurisdiction: USJurisdiction) -> bool:
        """Check if a jurisdiction is supported"""
        return jurisdiction in self.jurisdiction_system.supported_jurisdictions
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        
        return {
            "system_name": self.system_name,
            "version": self.version,
            "environment": self.environment.value,
            "enabled_providers": self.get_enabled_providers(),
            "supported_jurisdictions": [j.value for j in self.jurisdiction_system.supported_jurisdictions],
            "supported_task_types": [t.value for t in LegalTaskType],
            "prompt_system_enabled": self.prompt_template_system.enabled,
            "hybrid_evaluation_enabled": self.hybrid_evaluation.enabled,
            "jurisdiction_compliance_enabled": self.hybrid_evaluation.jurisdiction_compliance_enabled,
            "cost_optimization_enabled": self.cost_optimization.enabled,
            "verl_integration_enabled": self.verl_integration_enabled
        }


def load_config_from_file(file_path: str) -> LegalRewardSystemConfig:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        file_path: Path to configuration file
        
    Returns:
        Loaded configuration
    """
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise ConfigurationError(f"Configuration file not found: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            if file_path.suffix.lower() in ['.yml', '.yaml']:
                config_data = yaml.safe_load(f)
            elif file_path.suffix.lower() == '.json':
                config_data = json.load(f)
            else:
                raise ConfigurationError(f"Unsupported configuration file format: {file_path.suffix}")
        
        return create_config_from_dict(config_data)
    
    except Exception as e:
        raise ConfigurationError(f"Failed to load configuration from {file_path}: {str(e)}")


def create_config_from_dict(config_dict: Dict[str, Any]) -> LegalRewardSystemConfig:
    """
    Create configuration from dictionary.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Configuration object
    """
    
    try:
        # Convert environment string to enum
        if "environment" in config_dict:
            config_dict["environment"] = EnvironmentType(config_dict["environment"])
        
        # Convert jurisdiction strings to enums
        if "jurisdiction_system" in config_dict:
            js_config = config_dict["jurisdiction_system"]
            if "supported_jurisdictions" in js_config:
                js_config["supported_jurisdictions"] = [
                    USJurisdiction(j) if isinstance(j, str) else j 
                    for j in js_config["supported_jurisdictions"]
                ]
            if "default_jurisdiction" in js_config:
                js_config["default_jurisdiction"] = USJurisdiction(js_config["default_jurisdiction"])
        
        # Create nested configuration objects
        nested_configs = ["openai", "anthropic", "google", "prompt_template_system", 
                         "hybrid_evaluation", "jurisdiction_system", "cost_optimization"]
        
        for config_name in nested_configs:
            if config_name in config_dict:
                config_class = {
                    "openai": APIProviderConfig,
                    "anthropic": APIProviderConfig,
                    "google": APIProviderConfig,
                    "prompt_template_system": PromptTemplateSystemConfig,
                    "hybrid_evaluation": HybridEvaluationConfig,
                    "jurisdiction_system": JurisdictionSystemConfig,
                    "cost_optimization": CostOptimizationConfig
                }[config_name]
                
                config_dict[config_name] = config_class(**config_dict[config_name])
        
        return LegalRewardSystemConfig(**config_dict)
    
    except Exception as e:
        raise ConfigurationError(f"Failed to create configuration from dictionary: {str(e)}")


def create_default_config(environment: EnvironmentType = EnvironmentType.DEVELOPMENT) -> LegalRewardSystemConfig:
    """
    Create default configuration for specified environment.
    
    Args:
        environment: Target environment
        
    Returns:
        Default configuration
    """
    
    config = LegalRewardSystemConfig(environment=environment)
    
    # Environment-specific adjustments
    if environment == EnvironmentType.PRODUCTION:
        config.logging_level = "WARNING"
        config.performance_monitoring_enabled = True
        config.cost_optimization.enabled = True
        config.prompt_template_system.validation_enabled = True
        config.hybrid_evaluation.require_jurisdiction_compliance = True
        
    elif environment == EnvironmentType.DEVELOPMENT:
        config.logging_level = "DEBUG"
        config.performance_monitoring_enabled = True
        config.cost_optimization.enabled = False
        config.prompt_template_system.validation_enabled = True
        config.hybrid_evaluation.require_jurisdiction_compliance = False
        
    elif environment == EnvironmentType.TESTING:
        config.logging_level = "ERROR"
        config.performance_monitoring_enabled = False
        config.cost_optimization.enabled = False
        config.prompt_template_system.cache_enabled = False
        config.cache_enabled = False
        
    return config


def save_config_to_file(config: LegalRewardSystemConfig, file_path: str, format: str = "yaml"):
    """
    Save configuration to file.
    
    Args:
        config: Configuration to save
        file_path: Target file path
        format: File format ("yaml" or "json")
    """
    
    # Convert configuration to dictionary
    config_dict = _config_to_dict(config)
    
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(file_path, 'w') as f:
            if format.lower() in ['yml', 'yaml']:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif format.lower() == 'json':
                json.dump(config_dict, f, indent=2, default=str)
            else:
                raise ConfigurationError(f"Unsupported format: {format}")
    
    except Exception as e:
        raise ConfigurationError(f"Failed to save configuration to {file_path}: {str(e)}")


def _config_to_dict(config: LegalRewardSystemConfig) -> Dict[str, Any]:
    """Convert configuration object to dictionary"""
    
    result = {}
    
    for field_name, field_value in config.__dict__.items():
        if isinstance(field_value, Enum):
            result[field_name] = field_value.value
        elif hasattr(field_value, '__dict__'):
            # Nested configuration object
            nested_dict = {}
            for nested_name, nested_value in field_value.__dict__.items():
                if isinstance(nested_value, Enum):
                    nested_dict[nested_name] = nested_value.value
                elif isinstance(nested_value, list) and nested_value and isinstance(nested_value[0], Enum):
                    nested_dict[nested_name] = [item.value for item in nested_value]
                else:
                    nested_dict[nested_name] = nested_value
            result[field_name] = nested_dict
        else:
            result[field_name] = field_value
    
    return result


def validate_configuration(config: LegalRewardSystemConfig) -> Dict[str, Any]:
    """
    Comprehensive configuration validation.
    
    Args:
        config: Configuration to validate
        
    Returns:
        Validation results
    """
    
    validation_results = {
        "config_valid": True,
        "issues": [],
        "warnings": [],
        "component_status": {}
    }
    
    try:
        # Validate API providers
        enabled_providers = config.get_enabled_providers()
        if not enabled_providers:
            validation_results["config_valid"] = False
            validation_results["issues"].append("No API providers enabled")
        else:
            validation_results["component_status"]["api_providers"] = {
                "enabled_count": len(enabled_providers),
                "enabled_providers": enabled_providers
            }
        
        # Validate prompt template system
        if config.prompt_template_system.enabled:
            try:
                prompt_validation = validate_prompt_system()
                validation_results["component_status"]["prompt_system"] = prompt_validation
                if not prompt_validation["system_valid"]:
                    validation_results["warnings"].extend(prompt_validation["issues"])
            except Exception as e:
                validation_results["warnings"].append(f"Prompt system validation error: {str(e)}")
        
        # Validate jurisdiction system
        supported_count = len(config.jurisdiction_system.supported_jurisdictions)
        validation_results["component_status"]["jurisdiction_system"] = {
            "supported_jurisdictions": supported_count,
            "default_jurisdiction": config.jurisdiction_system.default_jurisdiction.value
        }
        
        if supported_count == 0:
            validation_results["config_valid"] = False
            validation_results["issues"].append("No jurisdictions supported")
        
        # Validate task weights
        task_weight_issues = []
        for task, weight in config.task_weights.items():
            if weight <= 0:
                task_weight_issues.append(f"Invalid weight for {task}: {weight}")
        
        if task_weight_issues:
            validation_results["config_valid"] = False
            validation_results["issues"].extend(task_weight_issues)
        
        validation_results["component_status"]["task_weights"] = {
            "weight_count": len(config.task_weights),
            "valid_weights": len(task_weight_issues) == 0
        }
        
        # Validate budget settings
        if config.cost_optimization.enabled:
            if config.cost_optimization.max_monthly_budget <= 0:
                validation_results["config_valid"] = False
                validation_results["issues"].append("Invalid monthly budget")
            
            validation_results["component_status"]["cost_optimization"] = {
                "enabled": True,
                "monthly_budget": config.cost_optimization.max_monthly_budget,
                "tracking_enabled": config.cost_optimization.cost_tracking_enabled
            }
    
    except Exception as e:
        validation_results["config_valid"] = False
        validation_results["issues"].append(f"Configuration validation error: {str(e)}")
    
    return validation_results


# Default configuration instance
default_config = create_default_config()


# Configuration loading shortcuts
def load_default_config(environment: str = "development") -> LegalRewardSystemConfig:
    """Load default configuration for environment"""
    env_type = EnvironmentType(environment.lower())
    return create_default_config(env_type)


def load_config_from_env() -> LegalRewardSystemConfig:
    """Load configuration with environment variable overrides"""
    
    config = create_default_config()
    
    # Override with environment variables
    if os.getenv("LEGAL_REWARD_ENVIRONMENT"):
        config.environment = EnvironmentType(os.getenv("LEGAL_REWARD_ENVIRONMENT").lower())
    
    if os.getenv("LEGAL_REWARD_LOG_LEVEL"):
        config.logging_level = os.getenv("LEGAL_REWARD_LOG_LEVEL")
    
    # Budget override
    if os.getenv("LEGAL_REWARD_MAX_BUDGET"):
        config.cost_optimization.max_monthly_budget = float(os.getenv("LEGAL_REWARD_MAX_BUDGET"))
    
    return config


# Example usage and testing
if __name__ == "__main__":
    # Create and validate default configuration
    config = create_default_config(EnvironmentType.DEVELOPMENT)
    
    print("Legal Reward System Configuration:")
    print("=" * 50)
    
    # Show system info
    system_info = config.get_system_info()
    print(f"System: {system_info['system_name']} v{system_info['version']}")
    print(f"Environment: {system_info['environment']}")
    print(f"Enabled Providers: {system_info['enabled_providers']}")
    print(f"Supported Jurisdictions: {len(system_info['supported_jurisdictions'])}")
    
    # Validate configuration
    validation = validate_configuration(config)
    print(f"\nConfiguration Valid: {'YES' if validation['config_valid'] else 'NO'}")
    
    if validation['issues']:
        print(f"Issues: {validation['issues']}")
    
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")
    
    # Show prompt system status
    if config.prompt_template_system.enabled:
        try:
            prompt_info = get_prompt_system_info()
            print(f"\nPrompt System:")
            print(f"Total Prompt Types: {prompt_info['capabilities']['total_prompt_types']}")
            print(f"Jurisdiction Coverage: {prompt_info['capabilities']['jurisdiction_coverage']}")
            print(f"VERL Compatible: {prompt_info['integration']['verl_compatible']}")
        except Exception as e:
            print(f"\nPrompt System Error: {str(e)}")
    
    print("\nConfiguration loaded successfully!")
