"""
Legal Reward System - Test Configuration and Environment Setup
===========================================================

This module provides comprehensive configuration management for the Legal Reward System
test suite, including environment-specific settings, test data configuration, and
performance thresholds.

Features:
- Environment-specific test configurations
- Performance benchmark thresholds
- Mock data generation settings
- API testing configurations
- CI/CD integration settings
- Test execution parameters

Author: Legal AI Development Team
Version: 1.0.0
"""

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from enum import Enum

# Add yaml import with fallback
try:
    import yaml
except ImportError:
    yaml = None
    logging.warning("PyYAML not installed. YAML config files will not be supported.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestEnvironment(Enum):
    """Test environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    CI_CD = "ci_cd"
    PRODUCTION = "production"
    LOCAL = "local"


class TestMode(Enum):
    """Test execution modes"""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    STRESS = "stress"
    VERL = "verl"
    FULL = "full"
    QUICK = "quick"


@dataclass
class PerformanceThresholds:
    """Performance test thresholds"""
    max_response_time_seconds: float = 3.0
    min_throughput_ops_per_second: float = 10.0
    min_success_rate_percent: float = 95.0
    max_error_rate_percent: float = 5.0
    max_memory_usage_mb: float = 2048.0
    min_gpu_utilization_percent: float = 70.0
    min_cache_hit_rate_percent: float = 60.0
    min_cost_reduction_percent: float = 70.0
    max_training_cost_per_cycle: float = 1500.0
    max_evaluation_time_seconds: float = 2.0


@dataclass
class APITestConfiguration:
    """API testing configuration"""
    mock_mode: bool = True
    rate_limit_testing: bool = True
    timeout_seconds: float = 30.0
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    cost_tracking_enabled: bool = True
    provider_fallback_testing: bool = True
    api_keys_required: List[str] = field(default_factory=lambda: [
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"
    ])


@dataclass
class MockDataConfiguration:
    """Mock data generation configuration"""
    num_legal_queries: int = 1000
    num_jurisdictions: int = 53  # 50 states + DC + federal + general
    legal_domains: List[str] = field(default_factory=lambda: [
        "constitutional", "contract", "tort", "criminal", "civil_procedure",
        "evidence", "corporate", "intellectual_property", "family", "employment"
    ])
    task_types: List[str] = field(default_factory=lambda: [
        "judicial_reasoning", "precedent_analysis", "opinion_generation", "general_chat"
    ])
    response_length_range: tuple = (50, 500)
    quality_score_range: tuple = (0.6, 0.95)
    complexity_levels: List[str] = field(default_factory=lambda: ["simple", "medium", "complex"])


@dataclass
class LoadTestConfiguration:
    """Load testing configuration"""
    concurrent_users_range: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 50, 100])
    batch_sizes: List[int] = field(default_factory=lambda: [8, 16, 32, 64, 128])
    test_duration_seconds: float = 60.0
    ramp_up_time_seconds: float = 10.0
    think_time_seconds: float = 1.0
    max_users_stress_test: int = 500
    stress_test_step_size: int = 25


@dataclass
class VERLTestConfiguration:
    """VERL integration test configuration"""
    batch_sizes: List[int] = field(default_factory=lambda: [8, 16, 32, 64, 128])
    sequence_lengths: List[int] = field(default_factory=lambda: [128, 256, 512])
    training_simulation_samples: int = 10000
    epochs: int = 3
    learning_rate: float = 1e-5
    gradient_accumulation_steps: int = 4
    mixed_precision: bool = True
    distributed_training: bool = True
    gpu_count: int = 8
    checkpoint_testing: bool = True


@dataclass
class CITestConfiguration:
    """CI/CD test configuration"""
    quick_test_timeout_minutes: int = 10
    full_test_timeout_minutes: int = 60
    performance_test_timeout_minutes: int = 30
    parallel_execution: bool = True
    max_parallel_jobs: int = 4
    artifact_retention_days: int = 30
    notification_channels: List[str] = field(default_factory=lambda: ["slack", "email"])
    required_coverage_percent: float = 90.0


@dataclass
class TestConfiguration:
    """Complete test configuration"""
    environment: TestEnvironment = TestEnvironment.DEVELOPMENT
    test_mode: TestMode = TestMode.FULL
    performance_thresholds: PerformanceThresholds = field(default_factory=PerformanceThresholds)
    api_config: APITestConfiguration = field(default_factory=APITestConfiguration)
    mock_data_config: MockDataConfiguration = field(default_factory=MockDataConfiguration)
    load_test_config: LoadTestConfiguration = field(default_factory=LoadTestConfiguration)
    verl_config: VERLTestConfiguration = field(default_factory=VERLTestConfiguration)
    ci_config: CITestConfiguration = field(default_factory=CITestConfiguration)
    
    # General settings
    verbose_logging: bool = True
    save_test_artifacts: bool = True
    generate_html_reports: bool = True
    enable_performance_profiling: bool = False
    test_data_directory: str = "./test_data"
    results_directory: str = "./test_results"
    logs_directory: str = "./test_logs"


class TestConfigurationManager:
    """Manager for test configurations"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "test_config.yaml"
        self.config = TestConfiguration()
        self._load_configuration()
    
    def _load_configuration(self):
        """Load configuration from file and environment variables"""
        # Load from file if exists
        if os.path.exists(self.config_file):
            self._load_from_file()
        
        # Override with environment variables
        self._load_from_environment()
        
        # Validate configuration
        self._validate_configuration()
    
    def _load_from_file(self):
        """Load configuration from YAML file"""
        try:
            with open(self.config_file, 'r') as f:
                if self.config_file.endswith('.yaml') or self.config_file.endswith('.yml'):
                    if yaml is None:
                        logger.warning("PyYAML not available. Treating YAML file as JSON.")
                        config_dict = json.load(f)
                    else:
                        config_dict = yaml.safe_load(f)
                else:
                    config_dict = json.load(f)
            
            # Update configuration with loaded values
            self._update_config_from_dict(config_dict)
            logger.info(f"Configuration loaded from {self.config_file}")
            
        except Exception as e:
            logger.warning(f"Failed to load configuration from {self.config_file}: {e}")
    
    def _update_config_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary"""
        for key, value in config_dict.items():
            if hasattr(self.config, key):
                if isinstance(value, dict):
                    # Handle nested configuration objects
                    current_value = getattr(self.config, key)
                    if hasattr(current_value, '__dict__'):
                        for sub_key, sub_value in value.items():
                            if hasattr(current_value, sub_key):
                                setattr(current_value, sub_key, sub_value)
                else:
                    setattr(self.config, key, value)
    
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        env_mappings = {
            "TEST_ENVIRONMENT": ("environment", lambda x: TestEnvironment(x)),
            "TEST_MODE": ("test_mode", lambda x: TestMode(x)),
            "MOCK_MODE": ("api_config.mock_mode", lambda x: x.lower() == 'true'),
            "VERBOSE_LOGGING": ("verbose_logging", lambda x: x.lower() == 'true'),
            "PERFORMANCE_TESTING": ("enable_performance_profiling", lambda x: x.lower() == 'true'),
            "MAX_RESPONSE_TIME": ("performance_thresholds.max_response_time_seconds", float),
            "MIN_THROUGHPUT": ("performance_thresholds.min_throughput_ops_per_second", float),
            "MIN_SUCCESS_RATE": ("performance_thresholds.min_success_rate_percent", float),
            "CI_TIMEOUT_MINUTES": ("ci_config.full_test_timeout_minutes", int),
            "PARALLEL_JOBS": ("ci_config.max_parallel_jobs", int)
        }
        
        for env_var, (config_path, converter) in env_mappings.items():
            if env_var in os.environ:
                try:
                    value = converter(os.environ[env_var])
                    self._set_nested_config(config_path, value)
                    logger.info(f"Configuration override from {env_var}: {config_path} = {value}")
                except Exception as e:
                    logger.warning(f"Failed to set {config_path} from {env_var}: {e}")
    
    def _set_nested_config(self, path: str, value: Any):
        """Set nested configuration value using dot notation"""
        parts = path.split('.')
        current = self.config
        
        for part in parts[:-1]:
            current = getattr(current, part)
        
        setattr(current, parts[-1], value)
    
    def _validate_configuration(self):
        """Validate configuration values"""
        # Validate performance thresholds
        thresholds = self.config.performance_thresholds
        if thresholds.max_response_time_seconds <= 0:
            raise ValueError("Max response time must be positive")
        if thresholds.min_throughput_ops_per_second <= 0:
            raise ValueError("Min throughput must be positive")
        if not 0 <= thresholds.min_success_rate_percent <= 100:
            raise ValueError("Success rate must be between 0 and 100")
        
        # Validate directories
        for directory in [self.config.test_data_directory, self.config.results_directory, self.config.logs_directory]:
            os.makedirs(directory, exist_ok=True)
        
        logger.info("Configuration validation completed successfully")
    
    def get_config(self) -> TestConfiguration:
        """Get current configuration"""
        return self.config
    
    def save_configuration(self, output_file: Optional[str] = None):
        """Save current configuration to file"""
        output_file = output_file or self.config_file
        
        # Convert to dictionary
        config_dict = asdict(self.config)
        
        # Convert enums to strings
        config_dict["environment"] = self.config.environment.value
        config_dict["test_mode"] = self.config.test_mode.value
        
        try:
            with open(output_file, 'w') as f:
                if (output_file.endswith('.yaml') or output_file.endswith('.yml')) and yaml is not None:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_dict, f, indent=2)
            
            logger.info(f"Configuration saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to {output_file}: {e}")
    
    def get_environment_config(self, environment: TestEnvironment) -> TestConfiguration:
        """Get configuration optimized for specific environment"""
        config = TestConfiguration()
        
        if environment == TestEnvironment.CI_CD:
            # Optimize for CI/CD environment
            config.api_config.mock_mode = True
            config.load_test_config.concurrent_users_range = [1, 5, 10]
            config.load_test_config.test_duration_seconds = 30.0
            config.verl_config.training_simulation_samples = 100
            config.mock_data_config.num_legal_queries = 100
            config.enable_performance_profiling = False
            config.verbose_logging = False
            
        elif environment == TestEnvironment.STAGING:
            # Staging environment with limited resources
            config.api_config.mock_mode = False
            config.load_test_config.max_users_stress_test = 100
            config.verl_config.training_simulation_samples = 1000
            config.performance_thresholds.min_throughput_ops_per_second = 5.0
            
        elif environment == TestEnvironment.PRODUCTION:
            # Production-like testing
            config.api_config.mock_mode = False
            config.enable_performance_profiling = True
            config.load_test_config.test_duration_seconds = 300.0
            config.performance_thresholds.min_success_rate_percent = 99.0
            
        elif environment == TestEnvironment.LOCAL:
            # Local development environment
            config.api_config.mock_mode = True
            config.verbose_logging = True
            config.save_test_artifacts = True
            config.generate_html_reports = True
            
        config.environment = environment
        return config


class TestDataGenerator:
    """Generate test data based on configuration"""
    
    def __init__(self, config: MockDataConfiguration):
        self.config = config
    
    def generate_legal_test_cases(self) -> List[Dict[str, Any]]:
        """Generate legal test cases"""
        test_cases = []
        
        # Predefined high-quality legal scenarios
        base_scenarios = [
            {
                "query": "What is the burden of proof in criminal cases under federal law?",
                "domain": "criminal",
                "jurisdiction": "federal",
                "task_type": "judicial_reasoning",
                "complexity": "medium",
                "expected_score": 0.92
            },
            {
                "query": "Find precedents for contract interpretation in California courts",
                "domain": "contract", 
                "jurisdiction": "california",
                "task_type": "precedent_analysis",
                "complexity": "complex",
                "expected_score": 0.88
            },
            {
                "query": "Draft an opinion on the admissibility of digital evidence",
                "domain": "evidence",
                "jurisdiction": "federal",
                "task_type": "opinion_generation",
                "complexity": "complex",
                "expected_score": 0.85
            },
            {
                "query": "Explain the concept of proximate cause in tort law",
                "domain": "tort",
                "jurisdiction": "general",
                "task_type": "general_chat",
                "complexity": "simple",
                "expected_score": 0.78
            },
            {
                "query": "What are the requirements for establishing a corporation in Delaware?",
                "domain": "corporate",
                "jurisdiction": "delaware",
                "task_type": "judicial_reasoning",
                "complexity": "medium",
                "expected_score": 0.83
            }
        ]
        
        # Generate variations of base scenarios
        for i in range(self.config.num_legal_queries):
            base_scenario = base_scenarios[i % len(base_scenarios)]
            
            # Create variation
            test_case = {
                "id": f"test_case_{i+1}",
                "query": f"[Variation {i+1}] {base_scenario['query']}",
                "response": self._generate_response(base_scenario),
                "legal_domain": base_scenario["domain"],
                "jurisdiction": base_scenario["jurisdiction"],
                "task_type": base_scenario["task_type"],
                "complexity": base_scenario["complexity"],
                "expected_score": base_scenario["expected_score"] + (i % 10) * 0.01 - 0.05,  # Add slight variation
                "metadata": {
                    "generated": True,
                    "base_scenario": base_scenario["query"][:50] + "...",
                    "variation_number": i + 1
                }
            }
            
            test_cases.append(test_case)
        
        return test_cases
    
    def _generate_response(self, scenario: Dict[str, Any]) -> str:
        """Generate appropriate response based on scenario"""
        domain = scenario["domain"]
        complexity = scenario["complexity"]
        
        # Base responses by domain
        base_responses = {
            "criminal": "Under federal criminal law, the prosecution must prove guilt beyond a reasonable doubt, which is the highest standard of evidence in the legal system.",
            "contract": "Contract interpretation in California follows established principles including the plain meaning rule and consideration of extrinsic evidence when terms are ambiguous.",
            "evidence": "The admissibility of evidence is governed by the Federal Rules of Evidence, particularly Rule 401 for relevance and Rule 403 for prejudicial impact.",
            "tort": "Tort law establishes liability for civil wrongs, with key elements including duty, breach, causation, and damages varying by jurisdiction.",
            "corporate": "Corporate formation requires compliance with state-specific requirements including filing articles of incorporation and establishing proper governance structures."
        }
        
        base_response = base_responses.get(domain, "This legal matter requires careful analysis of applicable law and precedent.")
        
        # Adjust response length based on complexity
        if complexity == "simple":
            return base_response
        elif complexity == "medium":
            return f"{base_response} Additional considerations include jurisdictional variations and recent case law developments that may impact the analysis."
        else:  # complex
            return f"{base_response} A comprehensive analysis must consider multiple factors including statutory requirements, case law precedents, regulatory guidelines, and potential constitutional implications. The specific facts and circumstances of each case will determine the applicable legal framework and potential outcomes."
    
    def generate_jurisdiction_test_data(self) -> Dict[str, Any]:
        """Generate jurisdiction-specific test data"""
        us_states = [
            "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
            "connecticut", "delaware", "florida", "georgia", "hawaii", "idaho",
            "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana",
            "maine", "maryland", "massachusetts", "michigan", "minnesota",
            "mississippi", "missouri", "montana", "nebraska", "nevada",
            "new_hampshire", "new_jersey", "new_mexico", "new_york",
            "north_carolina", "north_dakota", "ohio", "oklahoma", "oregon",
            "pennsylvania", "rhode_island", "south_carolina", "south_dakota",
            "tennessee", "texas", "utah", "vermont", "virginia", "washington",
            "west_virginia", "wisconsin", "wyoming", "district_of_columbia"
        ]
        
        return {
            "us_states": us_states,
            "federal": "federal",
            "general": "general",
            "total_jurisdictions": len(us_states) + 2,
            "jurisdiction_test_cases": [
                {
                    "jurisdiction": jurisdiction,
                    "test_queries": [
                        f"What is the statute of limitations for contracts in {jurisdiction}?",
                        f"How does {jurisdiction} handle product liability cases?",
                        f"What are the employment law requirements in {jurisdiction}?"
                    ]
                }
                for jurisdiction in us_states[:10]  # Sample for first 10 states
            ]
        }
    
    def generate_performance_test_data(self) -> Dict[str, Any]:
        """Generate data for performance testing"""
        return {
            "load_test_scenarios": [
                {
                    "name": "Light Load",
                    "concurrent_users": 5,
                    "operations_per_user": 10,
                    "expected_throughput": 8.0
                },
                {
                    "name": "Medium Load",
                    "concurrent_users": 20,
                    "operations_per_user": 15,
                    "expected_throughput": 15.0
                },
                {
                    "name": "Heavy Load",
                    "concurrent_users": 50,
                    "operations_per_user": 20,
                    "expected_throughput": 25.0
                },
                {
                    "name": "Stress Test",
                    "concurrent_users": 100,
                    "operations_per_user": 10,
                    "expected_throughput": 30.0
                }
            ],
            "batch_size_tests": [8, 16, 32, 64, 128],
            "performance_targets": {
                "response_time_p95": 3.0,
                "throughput_minimum": 10.0,
                "error_rate_maximum": 5.0,
                "memory_usage_maximum": 2048
            }
        }


class EnvironmentSetup:
    """Environment setup and validation"""
    
    @staticmethod
    def setup_test_environment(config: TestConfiguration) -> bool:
        """Set up test environment"""
        try:
            # Create directories
            directories = [
                config.test_data_directory,
                config.results_directory,
                config.logs_directory
            ]
            
            for directory in directories:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Created directory: {directory}")
            
            # Set up logging
            EnvironmentSetup._setup_logging(config)
            
            # Validate API keys if not in mock mode
            if not config.api_config.mock_mode:
                EnvironmentSetup._validate_api_keys(config.api_config.api_keys_required)
            
            # Generate test data
            EnvironmentSetup._generate_test_data(config)
            
            logger.info("Test environment setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set up test environment: {e}")
            return False
    
    @staticmethod
    def _setup_logging(config: TestConfiguration):
        """Set up logging configuration"""
        log_level = logging.INFO if config.verbose_logging else logging.WARNING
        log_file = os.path.join(config.logs_directory, "test_execution.log")
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    @staticmethod
    def _validate_api_keys(required_keys: List[str]):
        """Validate required API keys"""
        missing_keys = []
        for key in required_keys:
            if not os.getenv(key):
                missing_keys.append(key)
        
        if missing_keys:
            logger.warning(f"Missing API keys: {missing_keys}")
            logger.info("Set environment variables or enable mock mode for testing")
    
    @staticmethod
    def _generate_test_data(config: TestConfiguration):
        """Generate test data files"""
        data_generator = TestDataGenerator(config.mock_data_config)
        
        # Generate legal test cases
        test_cases = data_generator.generate_legal_test_cases()
        test_cases_file = os.path.join(config.test_data_directory, "legal_test_cases.json")
        with open(test_cases_file, 'w') as f:
            json.dump(test_cases, f, indent=2)
        logger.info(f"Generated {len(test_cases)} legal test cases")
        
        # Generate jurisdiction test data
        jurisdiction_data = data_generator.generate_jurisdiction_test_data()
        jurisdiction_file = os.path.join(config.test_data_directory, "jurisdiction_test_data.json")
        with open(jurisdiction_file, 'w') as f:
            json.dump(jurisdiction_data, f, indent=2)
        logger.info("Generated jurisdiction test data")
        
        # Generate performance test data
        performance_data = data_generator.generate_performance_test_data()
        performance_file = os.path.join(config.test_data_directory, "performance_test_data.json")
        with open(performance_file, 'w') as f:
            json.dump(performance_data, f, indent=2)
        logger.info("Generated performance test data")


# ================================================================
# CONFIGURATION TEMPLATES
# ================================================================

def create_default_config_file():
    """Create default configuration file"""
    config = TestConfiguration()
    config_manager = TestConfigurationManager()
    config_manager.config = config
    config_manager.save_configuration("test_config_default.yaml")
    print("Default configuration saved to test_config_default.yaml")


def create_ci_config_file():
    """Create CI/CD optimized configuration file"""
    config_manager = TestConfigurationManager()
    ci_config = config_manager.get_environment_config(TestEnvironment.CI_CD)
    config_manager.config = ci_config
    config_manager.save_configuration("test_config_ci.yaml")
    print("CI/CD configuration saved to test_config_ci.yaml")


def create_production_config_file():
    """Create production testing configuration file"""
    config_manager = TestConfigurationManager()
    prod_config = config_manager.get_environment_config(TestEnvironment.PRODUCTION)
    config_manager.config = prod_config
    config_manager.save_configuration("test_config_production.yaml")
    print("Production configuration saved to test_config_production.yaml")


# ================================================================
# MAIN EXECUTION
# ================================================================

def main():
    """Main execution for configuration setup"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Legal Reward System Test Configuration")
    parser.add_argument("--create-default", action="store_true", help="Create default config file")
    parser.add_argument("--create-ci", action="store_true", help="Create CI/CD config file")
    parser.add_argument("--create-production", action="store_true", help="Create production config file")
    parser.add_argument("--setup-environment", action="store_true", help="Set up test environment")
    parser.add_argument("--config-file", default="test_config.yaml", help="Configuration file to use")
    
    args = parser.parse_args()
    
    if args.create_default:
        create_default_config_file()
    elif args.create_ci:
        create_ci_config_file()
    elif args.create_production:
        create_production_config_file()
    elif args.setup_environment:
        config_manager = TestConfigurationManager(args.config_file)
        config = config_manager.get_config()
        success = EnvironmentSetup.setup_test_environment(config)
        if success:
            print("✅ Test environment setup completed successfully")
        else:
            print("❌ Test environment setup failed")
            exit(1)
    else:
        print("No action specified. Use --help for available options.")


if __name__ == "__main__":
    main()