# Legal Reward System
## Multi-Task Legal AI Reward Function for VERL Training


A production-ready, multi-task legal reward system designed for training legal AI agents using Group Relative Policy Optimization (GRPO) within the VERL framework. This system evaluates legal reasoning across four specialized task types with comprehensive US jurisdiction support.

## 🏛️ System Overview

The Legal Reward System is built on research into search-augmented reinforcement learning for legal reasoning, implementing a hybrid evaluation approach that balances specialized legal accuracy (70%) with general conversational quality (30%). The system supports the complete US legal jurisdiction system with automatic inference and compliance gating.

### Key Innovation: Hybrid Evaluation Framework

```
Final Reward = (Specialized Legal Score × 70%) + (General Chat Score × 30%) × Task Weight

With Jurisdiction Compliance Gating:
✅ Compliant: Full reward calculation
❌ Non-compliant: Score × 0.2 (80% penalty for safety)
```

## 📋 Supported Legal Task Types

### 1. **Judicial Reasoning** (`JUDICIAL_REASONING`)
Evaluation of judicial decision-making, legal analysis, and court opinion reasoning.

**Specialized Judge Ensemble:**
- **Legal Analysis Judge** (30%) - Evaluates analytical rigor and legal reasoning quality
- **Jurisdiction Accuracy Judge** (25%) - Validates proper jurisdiction and applicable law
- **Citation Quality Judge** (20%) - Assesses legal citation accuracy and relevance
- **Precedent Application Judge** (15%) - Reviews proper use of legal precedents
- **Writing Quality Judge** (10%) - Evaluates professional legal writing standards

### 2. **Precedent Analysis** (`PRECEDENT_ANALYSIS`)
Assessment of case law analysis, analogical reasoning, and precedent application.

**Specialized Judge Ensemble:**
- **Case Law Accuracy Judge** (30%) - Validates factual accuracy of case references
- **Analogical Reasoning Judge** (25%) - Evaluates quality of legal analogies
- **Citation Quality Judge** (20%) - Assesses citation format and relevance
- **Precedent Hierarchy Judge** (15%) - Reviews understanding of precedent hierarchy
- **Distinguishing Analysis Judge** (10%) - Evaluates case distinction analysis

### 3. **Opinion Generation** (`OPINION_GENERATION`)
Evaluation of legal opinion writing, advocacy, and client-focused communication.

**Specialized Judge Ensemble:**
- **Argument Strength Judge** (25%) - Assesses persuasive argument construction
- **Advocacy Effectiveness Judge** (25%) - Evaluates advocacy quality and strategy
- **Client Focus Judge** (20%) - Reviews client-centered approach
- **Professional Writing Judge** (15%) - Validates professional writing standards
- **Strategic Positioning Judge** (15%) - Assesses strategic legal positioning

### 4. **General Chat** (`GENERAL_CHAT`)
Assessment of general legal consultation and conversational assistance.

**Specialized Judge Ensemble:**
- **Helpfulness Judge** (25%) - Evaluates practical value and assistance quality
- **Legal Ethics Judge** (25%) - Ensures compliance with legal ethics standards
- **Clarity Judge** (25%) - Assesses communication clarity and accessibility
- **Jurisdiction Compliance Judge** (25%) - Critical safety gating mechanism

## 🗺️ US Jurisdiction Support

### Complete Coverage
- **Federal System**: Supreme Court, Circuit Courts, District Courts, Federal Agencies
- **All 50 States**: Complete state court systems and state-specific law
- **Special Jurisdictions**: District of Columbia, US territories, specialized courts

### Automatic Jurisdiction Inference
The system automatically detects jurisdiction context through:
- Court identifier analysis
- Geographic reference parsing
- Legal authority citation analysis
- Jurisdiction-specific legal terminology recognition

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/legal-ai/legal-reward-system.git
cd legal-reward-system

# Install dependencies
pip install -r requirements.txt

# Minimal installation
pip install -e .

# Full installation with all features
pip install -e ".[all]"
```

### Environment Setup

```bash
# Required API keys for production
export OPENAI_API_KEY="your_openai_api_key"
export ANTHROPIC_API_KEY="your_anthropic_api_key"
export GOOGLE_API_KEY="your_google_api_key"

# Optional: Enable development mode
export LEGAL_REWARD_DEV_MODE=true
export MOCK_MODE=true  # For testing without API keys
```

### Basic Usage

```python
from legal_reward_system.verl_integration import multi_task_legal_reward_function

# VERL-compatible reward function (main entry point)
reward_score = multi_task_legal_reward_function(
    data_source="legal_training_data",
    solution_str="Model generated legal response...",
    ground_truth="Expected legal response or query...",
    extra_info={
        "task_type": "judicial_reasoning",
        "jurisdiction": "federal",
        "legal_domain": "constitutional"
    }
)

print(f"Reward Score: {reward_score}")  # Range: 0.0 - 15.0
```

### VERL Integration

```python
# VERL training configuration
from legal_reward_system.factory import create_production_legal_reward_router

# Production setup
router = create_production_legal_reward_router()

# VERL configuration file
VERL_CONFIG = {
    "trainer": {
        "algorithm": "GRPO",
        "reward_model": {
            "custom_reward_function": {
                "path": "legal_reward_system.verl_integration",
                "name": "multi_task_legal_reward_function"
            }
        }
    }
}
```

## 📊 Performance & Cost Optimization

### Performance Benchmarks
- **Evaluation Time**: <3 seconds per hybrid evaluation
- **Throughput**: 10+ evaluations/second (8 A100 GPUs)
- **Success Rate**: 95%+ evaluation success rate
- **Accuracy**: 90%+ jurisdiction inference accuracy

### Cost Optimization Features
- **60-80% API Cost Reduction** through intelligent caching
- **Multi-Provider Fallback**: OpenAI → Anthropic → Google
- **Real-time Budget Tracking**: Configurable cost limits
- **Batch Processing**: Concurrent evaluation with rate limiting

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   VERL Integration                      │
│              (verl_integration.py)                      │
├─────────────────────────────────────────────────────────┤
│           Multi-Task Legal Reward Router                │
│                 (routing/router.py)                     │
├─────────────────────────────────────────────────────────┤
│    Hybrid Evaluation System (70% Specialized + 30%)    │
│              (routing/hybrid_evaluation.py)             │
├─────────────────────────────────────────────────────────┤
│  Judge Ensembles                    │  US Jurisdiction  │
│  - Judicial Reasoning              │  - Inference Engine│
│  - Precedent Analysis              │  - Compliance Judge │
│  - Opinion Generation              │  - All 50 States   │
│  - Enhanced General Chat           │  - Federal System  │
├─────────────────────────────────────────────────────────┤
│              API Client Layer                           │
│     (GPT-4 + Claude + Gemini with Cost Optimization)   │
├─────────────────────────────────────────────────────────┤
│           Utilities Foundation                          │
│    Caching + Rate Limiting + Logging + Configuration   │
└─────────────────────────────────────────────────────────┘
```

### Core Components

#### Main Entry Points
- **`verl_integration.py`** - Primary VERL interface and reward function
- **`factory.py`** - System setup and configuration factory functions
- **`routing/router.py`** - Multi-task legal reward router (orchestration)

#### Specialized Systems
- **`jurisdiction/`** - US legal jurisdiction system (all 50 states + federal)
- **`judges/`** - Judge ensembles for specialized legal evaluation
- **`routing/`** - Hybrid evaluation and task routing logic
- **`config/`** - Configuration management and legal prompts
- **`utils/`** - Caching, rate limiting, logging, and performance monitoring

## 🧪 Testing & Validation

### Run Test Suite

```bash
# Quick validation (5 minutes)
python test_runner.py --mode=quick

# Full comprehensive testing (20-30 minutes)
python test_runner.py --mode=full --output=html

# Performance benchmarking
python test_runner.py --mode=performance

# Integration testing
python test_runner.py --mode=integration
```

### Test Coverage
- **Unit Tests**: All 23 core components (100% coverage)
- **Integration Tests**: VERL compatibility, API integration, cost optimization
- **Performance Tests**: Throughput, latency, cost efficiency validation
- **Specialized Tests**: US jurisdiction accuracy, legal compliance validation

## 📈 Production Deployment

### System Requirements
- **Python**: 3.9+
- **Memory**: 8GB+ RAM recommended
- **GPU**: Optional (8 A100 GPUs recommended for training)
- **Storage**: 10GB+ for caching and logs

### Production Configuration

```python
from legal_reward_system.config import create_production_config
from legal_reward_system.factory import create_production_legal_reward_router

# Production setup with full optimization
config = create_production_config()
router = create_production_legal_reward_router(config)

# Key production settings
PRODUCTION_CONFIG = {
    "enable_caching": True,
    "enable_cost_optimization": True,
    "max_cost_per_evaluation": 0.05,  # $0.05 per evaluation
    "require_jurisdiction_compliance": True,
    "max_concurrent_evaluations": 10,
    "enable_performance_monitoring": True
}
```

### Monitoring & Logging

The system includes comprehensive monitoring with:
- **Performance Metrics**: Response times, success rates, cost tracking
- **Legal Compliance**: Jurisdiction accuracy, ethics compliance
- **Error Tracking**: Detailed error logs with context and recovery
- **Cost Analytics**: Real-time API cost tracking and budget alerts

## 📚 Documentation

### Configuration Options

```python
# Development vs Production modes
config = create_development_config()  # Fast setup, reduced accuracy
config = create_production_config()   # Full accuracy, optimized performance

# Router configuration
router_config = RouterConfig(
    specialized_weight=0.7,           # 70% specialized evaluation
    general_chat_weight=0.3,          # 30% general chat evaluation
    jurisdiction_failure_penalty=0.2, # 80% penalty for non-compliance
    enable_cost_optimization=True,
    max_concurrent_evaluations=10
)
```

### Task Difficulty Weights

The system automatically adjusts scores based on legal task complexity:

```python
# Default task difficulty weights
TASK_WEIGHTS = {
    LegalTaskType.JUDICIAL_REASONING: 1.2,    # +20% for complexity
    LegalTaskType.PRECEDENT_ANALYSIS: 1.1,   # +10% for research depth
    LegalTaskType.OPINION_GENERATION: 1.0,   # Baseline difficulty
    LegalTaskType.GENERAL_CHAT: 0.9          # -10% for accessibility
}
```

### Jurisdiction Inference Examples

```python
# Automatic jurisdiction detection
examples = [
    "Federal constitutional law" → USJurisdiction.FEDERAL
    "California contract dispute" → USJurisdiction.CALIFORNIA
    "New York family court" → USJurisdiction.NEW_YORK
    "Supreme Court precedent" → USJurisdiction.FEDERAL
]
```

## 🤝 Integration Examples

### Custom Judge Ensemble

```python
from legal_reward_system.judges.base import BaseJudgeEnsemble
from legal_reward_system.core import LegalTaskType

class CustomLegalJudge(BaseJudgeEnsemble):
    def __init__(self, config):
        super().__init__(
            ensemble_name="CustomLegalEnsemble",
            task_type=LegalTaskType.CUSTOM,
            config=config
        )
    
    async def evaluate_ensemble(self, data_point):
        # Custom evaluation logic
        return ensemble_score
```

## 🔧 Development

### Project Structure

```
legal_reward_system/
├── __init__.py                     # Package initialization
├── core/                          # Core data structures and types
│   ├── data_structures.py         # LegalDataPoint, EvaluationResult
│   ├── enums.py                   # Task types, jurisdictions, providers
│   └── exceptions.py              # Custom exception classes
├── jurisdiction/                  # US jurisdiction system
│   ├── us_system.py               # US jurisdiction definitions
│   ├── inference_engine.py       # Automatic jurisdiction detection
│   └── compliance_judge.py        # Jurisdiction compliance validation
├── judges/                        # Evaluation judge ensembles
│   ├── base.py                    # Base judge ensemble interface
│   ├── api_client.py              # Cost-optimized API client
│   ├── general_chat.py            # General conversation evaluation
│   └── specialized/               # Task-specific judge ensembles
├── routing/                       # Task routing and evaluation
│   ├── router.py                  # Multi-task reward router
│   ├── hybrid_evaluation.py      # Hybrid evaluation system
│   └── task_weights.py            # Task difficulty management
├── config/                        # Configuration management
│   ├── __init__.py                # Configuration classes
│   └── prompts/                   # Legal evaluation prompts
├── utils/                         # Utility systems
│   ├── logging.py                 # Enhanced logging system
│   ├── cache.py                   # Multi-layer caching
│   ├── rate_limiter.py            # API rate limiting
│   └── performance.py             # Performance monitoring
├── verl_integration.py            # VERL integration interface
└── factory.py                     # System factory functions

tests/
├── unit/                          # Unit tests for all components
├── integration/                   # Integration and workflow tests
└── performance/                   # Performance and cost validation

docs/
├── architecture.md                # System architecture details
├── api-reference.md               # Complete API documentation
└── deployment.md                  # Production deployment guide
```

### Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/new-judge-ensemble`
3. **Run tests**: `python test_runner.py --mode=full`
4. **Submit a pull request** with comprehensive test coverage

### Code Quality Standards

- **Type Hints**: All functions include comprehensive type annotations
- **Documentation**: Docstrings for all classes and public methods
- **Testing**: 95%+ test coverage required for all new features
- **Performance**: All components must meet performance benchmarks
- **Legal Compliance**: All legal evaluation components require expert review

## 📄 Legal & Ethical Considerations

### Professional Responsibility
This system is designed to assist in legal AI training and evaluation. It includes:
- **Jurisdiction Compliance Gating** for safety
- **Legal Ethics Evaluation** components
- **Professional Standards** validation
- **Appropriate Boundary** recognition

### Important Disclaimers
- This system is for training AI models, not for providing legal advice
- All evaluations should be reviewed by qualified legal professionals
- Jurisdiction compliance features are safety mechanisms, not legal guarantees
- The system is designed to complement, not replace, human legal expertise

## 📞 Support & Community

### website
- **Website**: [Website](https://lexlens.app/)
- **contact**: [email](daa238@cornell.edu)

### Getting Help
- **Issues**: [GitHub Issues](https://github.com/UPSD1/LegalReasoner/issues)
- **Discussions**: [GitHub Discussions](https://github.com/UPSD1/LegalReasoner/discussions)
- **Feature Requests**: Use GitHub Issues with `enhancement` label

### Contributing to Legal AI Research
This project is part of ongoing research into search-augmented reinforcement learning for legal reasoning. Contributions that advance the field of legal AI are highly welcomed, particularly in:
- Additional legal task type support
- International jurisdiction systems
- Legal domain specialization
- Evaluation methodology improvements

## 🏆 Acknowledgments

This work builds upon research in:
- **Search-Augmented Reinforcement Learning** for improved legal reasoning
- **LegalBench**: Collaborative benchmark for legal reasoning evaluation
- **VERL Framework**: Volcano Engine Reinforcement Learning platform
- **Multi-Task Learning**: Specialized evaluation across legal domains

### Research Foundation
Based on research into search-augmented reinforcement learning for legal reasoning, implementing Group Relative Policy Optimization (GRPO) with specialized legal reward functions for training more reliable and accurate legal AI agents.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔖 Citation

If you use this system in your research, please cite:

```bibtex
@software{legal_reward_system,
  title={Multi-Task Legal Reward System for VERL Training},
  author={Legal AI Research Team},
  year={2025},
  url={https://github.com/UPSD1/LegalReasoner},
  note={Production-ready legal AI reward function with comprehensive US jurisdiction support}
}
```

---

**Built with ❤️ for the Legal AI Research Community**