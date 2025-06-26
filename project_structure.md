## PROJECT ORGANIZATION

```
legal_reward_system/
├── __init__.py                           # Main exports: MultiTaskLegalRewardFunction
├── verl_integration.py                   # MAIN ENTRY POINT - VERL interface
├── core/
│   ├── __init__.py
│   ├── data_structures.py                # LegalDataPoint, EnsembleScore, RoutedReward
│   ├── enums.py                         # LegalTaskType, LegalDomain, USJurisdiction
│   └── exceptions.py                    # Custom exceptions
├── jurisdiction/
│   ├── __init__.py
│   ├── us_system.py                     # US jurisdiction enums and utilities
│   ├── inference_engine.py             # USJurisdictionInferenceEngine  
│   └── compliance_judge.py             # JurisdictionComplianceJudge
├── judges/
│   ├── __init__.py
│   ├── base.py                         # BaseJudgeEnsemble abstract class
│   ├── api_client.py                   # Unified API client with AGGRESSIVE rate limiting & caching
│   ├── general_chat.py                 # EnhancedGeneralChatEnsemble
│   ├── specialized/
│   │   ├── __init__.py
│   │   ├── judicial_reasoning.py        # JudicialReasoningEnsemble
│   │   ├── precedent_analysis.py        # PrecedentAnalysisEnsemble
│   │   └── opinion_generation.py        # OpinionGenerationEnsemble
│   └── mock.py                         # Mock judges for testing & development
├── routing/
│   ├── __init__.py
│   ├── router.py                       # MultiTaskLegalRewardRouter
│   ├── hybrid_evaluation.py           # Hybrid evaluation logic (70%/30%)
│   └── task_weights.py                 # TaskDifficultyWeightManager
├── config/
│   ├── __init__.py
│   ├── settings.py                     # Configuration management
│   ├── internal_config.yaml            # YOUR INTERNAL CONFIG (all settings)
│   └── prompts/                        # US legal prompt templates
│       ├── __init__.py
│       ├── judicial_reasoning.py
│       ├── precedent_analysis.py
│       ├── opinion_generation.py
│       ├── general_chat.py
│       └── jurisdiction_compliance.py
├── utils/
│   ├── __init__.py
│   ├── logging.py                      # Enhanced logging
│   ├── performance.py                  # Performance monitoring
│   ├── rate_limiter.py                 # API rate limiting (per provider)
│   └── cache.py                        # AGGRESSIVE caching system
└── factory.py                          # create_internal_legal_reward_system()

tests/
├── __init__.py
├── unit/
│   ├── test_core/
│   │   ├── test_data_structures.py
│   │   ├── test_enums.py
│   │   └── test_exceptions.py
│   ├── test_jurisdiction/
│   │   ├── test_us_system.py
│   │   ├── test_inference_engine.py
│   │   └── test_compliance_judge.py
│   ├── test_judges/
│   │   ├── test_base.py
│   │   ├── test_api_client.py
│   │   ├── test_general_chat.py
│   │   └── test_mock.py
│   ├── test_routing/
│   │   ├── test_router.py
│   │   ├── test_hybrid_evaluation.py
│   │   └── test_task_weights.py
│   └── test_utils/
│       ├── test_cache.py
│       ├── test_rate_limiter.py
│       ├── test_logging.py
│       └── test_performance.py
├── integration/
│   ├── test_verl_integration.py        # VERL compatibility tests
│   ├── test_api_judges.py              # API-based judge tests  
│   ├── test_cost_optimization.py       # Cost analysis & caching effectiveness
│   ├── test_rate_limiting.py           # Rate limiting validation
│   └── test_end_to_end_workflow.py     # Complete system tests
└── performance/
    ├── test_throughput.py              # Performance benchmarks
    └── test_training_simulation.py     # Full training cycle simulation

# Additional development files (not part of main package)
requirements.txt                        # Python dependencies
setup.py                               # Package setup configuration
README.md                              # Project documentation
.gitignore                             # Git ignore patterns
.env.example                          # Environment variable template
```

