"""
US Legal Prompt Templates Package

This package provides sophisticated, jurisdiction-aware prompt templates for
evaluating legal responses across all aspects of the Enhanced Multi-Task Legal
Reward System. These templates form the foundation for professional-grade
legal evaluation with comprehensive US jurisdiction support.

Package Components:
- judicial_reasoning.py: Formal judicial analysis evaluation templates
- precedent_analysis.py: Case law and precedent evaluation templates  
- opinion_generation.py: Legal advocacy and opinion evaluation templates
- general_chat.py: Enhanced general chat evaluation templates (with gating)
- jurisdiction_compliance.py: Critical gating evaluation templates

Key Features:
- Complete US jurisdiction coverage (all 50 states + DC + federal)
- Professional-grade legal evaluation standards
- Jurisdiction-specific legal context integration
- Critical gating functionality for jurisdiction compliance
- Optimized for GRPO training with VERL integration

Author: Legal Reward System Team
Version: 1.0.0
"""

from typing import Dict, List, Optional, Union, Any
from enum import Enum

# Import all prompt template modules
from .judicial_reasoning import (
    JudicialReasoningPromptType,
    get_judicial_reasoning_prompt,
    get_all_judicial_reasoning_prompts,
    validate_judicial_reasoning_prompt,
    JudicialReasoningPromptInfo,
    JUDICIAL_REASONING_PROMPT_INFO
)

from .precedent_analysis import (
    PrecedentAnalysisPromptType,
    get_precedent_analysis_prompt,
    get_all_precedent_analysis_prompts,
    validate_precedent_analysis_prompt,
    PRECEDENT_ANALYSIS_PROMPT_INFO
)

from .opinion_generation import (
    OpinionGenerationPromptType,
    get_opinion_generation_prompt,
    get_all_opinion_generation_prompts,
    validate_opinion_generation_prompt,
    OPINION_GENERATION_PROMPT_INFO
)

from .general_chat import (
    GeneralChatPromptType,
    GeneralChatEvaluationContext,
    get_general_chat_prompt,
    get_all_general_chat_prompts,
    validate_general_chat_prompt,
    GeneralChatPromptManager,
    GENERAL_CHAT_PROMPT_INFO
)

from .jurisdiction_compliance import (
    JurisdictionCompliancePromptType,
    JurisdictionComplianceContext,
    get_jurisdiction_compliance_prompt,
    get_all_jurisdiction_compliance_prompts,
    validate_jurisdiction_compliance_prompt,
    JurisdictionComplianceManager,
    get_gating_threshold,
    assess_gating_failure,
    JURISDICTION_COMPLIANCE_PROMPT_INFO
)

# Import core types for integration
from ...jurisdiction.us_system import USJurisdiction
from ...core.enums import LegalTaskType

# Package version and metadata
__version__ = "1.0.0"
__author__ = "Legal Reward System Team"
__description__ = "Professional US legal prompt templates with jurisdiction awareness"

# Main exports
__all__ = [
    # Prompt type enums
    "JudicialReasoningPromptType",
    "PrecedentAnalysisPromptType", 
    "OpinionGenerationPromptType",
    "GeneralChatPromptType",
    "JurisdictionCompliancePromptType",
    
    # Context classes
    "GeneralChatEvaluationContext",
    "JurisdictionComplianceContext",
    
    # Individual prompt functions
    "get_judicial_reasoning_prompt",
    "get_precedent_analysis_prompt",
    "get_opinion_generation_prompt", 
    "get_general_chat_prompt",
    "get_jurisdiction_compliance_prompt",
    
    # Bulk prompt functions
    "get_all_judicial_reasoning_prompts",
    "get_all_precedent_analysis_prompts",
    "get_all_opinion_generation_prompts",
    "get_all_general_chat_prompts",
    "get_all_jurisdiction_compliance_prompts",
    
    # Validation functions
    "validate_judicial_reasoning_prompt",
    "validate_precedent_analysis_prompt",
    "validate_opinion_generation_prompt",
    "validate_general_chat_prompt",
    "validate_jurisdiction_compliance_prompt",
    
    # Management classes
    "GeneralChatPromptManager",
    "JurisdictionComplianceManager",
    
    # Gating functions
    "get_gating_threshold",
    "assess_gating_failure",
    
    # Unified prompt system
    "UnifiedPromptTemplateSystem",
    "PromptTemplateConfig",
    "get_prompt_for_task_type",
    "validate_prompt_system",
    "get_prompt_system_info"
]


class PromptTemplateConfig:
    """Configuration for prompt template system"""
    
    def __init__(self):
        self.cache_enabled = True
        self.cache_size_limit = 1000  # Number of cached prompts
        self.validation_enabled = True
        self.gating_enabled = True
        self.jurisdiction_validation = True
        
        # Performance settings
        self.parallel_loading = True
        self.memory_optimization = True
        
        # Quality settings
        self.prompt_quality_validation = True
        self.jurisdiction_accuracy_validation = True


class UnifiedPromptTemplateSystem:
    """
    Unified system for managing all legal prompt templates with caching,
    validation, and performance optimization.
    """
    
    def __init__(self, config: Optional[PromptTemplateConfig] = None):
        self.config = config or PromptTemplateConfig()
        
        # Initialize managers
        self.general_chat_manager = GeneralChatPromptManager()
        self.jurisdiction_compliance_manager = JurisdictionComplianceManager()
        
        # Cache for prompt templates
        self._prompt_cache = {} if self.config.cache_enabled else None
        
        # Track system performance
        self.performance_stats = {
            "prompts_generated": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "validation_failures": 0,
            "gating_failures": 0
        }
    
    def get_prompt_for_task_type(self, task_type: LegalTaskType, 
                                prompt_type: str,
                                jurisdiction: USJurisdiction,
                                query: str,
                                response: str,
                                **kwargs) -> str:
        """
        Get a prompt for any task type with unified interface.
        
        Args:
            task_type: Legal task type
            prompt_type: Specific prompt type within the task
            jurisdiction: US jurisdiction
            query: Original legal query
            response: Response to evaluate
            **kwargs: Additional context
            
        Returns:
            Formatted prompt template
        """
        
        self.performance_stats["prompts_generated"] += 1
        
        # Check cache first
        cache_key = f"{task_type.value}_{prompt_type}_{jurisdiction.value}"
        if self.config.cache_enabled and cache_key in self._prompt_cache:
            self.performance_stats["cache_hits"] += 1
            cached_template = self._prompt_cache[cache_key]
            return self._substitute_dynamic_content(cached_template, query, response, **kwargs)
        
        self.performance_stats["cache_misses"] += 1
        
        # Generate prompt based on task type
        if task_type == LegalTaskType.JUDICIAL_REASONING:
            prompt = self._get_judicial_reasoning_prompt(prompt_type, jurisdiction, query, response, **kwargs)
        elif task_type == LegalTaskType.PRECEDENT_ANALYSIS:
            prompt = self._get_precedent_analysis_prompt(prompt_type, jurisdiction, query, response, **kwargs)
        elif task_type == LegalTaskType.OPINION_GENERATION:
            prompt = self._get_opinion_generation_prompt(prompt_type, jurisdiction, query, response, **kwargs)
        elif task_type == LegalTaskType.GENERAL_CHAT:
            prompt = self._get_general_chat_prompt(prompt_type, jurisdiction, query, response, **kwargs)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
        
        # Cache the template (without dynamic content)
        if self.config.cache_enabled:
            self._cache_prompt_template(cache_key, prompt, query, response)
        
        return prompt
    
    def get_jurisdiction_compliance_prompt(self, prompt_type: str,
                                         jurisdiction: USJurisdiction,
                                         task_type: LegalTaskType,
                                         query: str,
                                         response: str,
                                         **kwargs) -> str:
        """Get jurisdiction compliance prompt (gating function)"""
        
        # Create context
        context = JurisdictionComplianceContext(
            query=query,
            response=response,
            jurisdiction=jurisdiction,
            task_type=task_type,
            legal_domain=kwargs.get("legal_domain", "General"),
            complexity_level=kwargs.get("complexity_level", "standard"),
            federal_implications=kwargs.get("federal_implications", False)
        )
        
        # Get prompt type enum
        compliance_prompt_type = JurisdictionCompliancePromptType(prompt_type)
        
        return get_jurisdiction_compliance_prompt(compliance_prompt_type, context)
    
    def _get_judicial_reasoning_prompt(self, prompt_type: str, jurisdiction: USJurisdiction,
                                     query: str, response: str, **kwargs) -> str:
        """Get judicial reasoning prompt"""
        reasoning_prompt_type = JudicialReasoningPromptType(prompt_type)
        jurisdiction_context = kwargs.get("jurisdiction_context", f"{jurisdiction.value} legal analysis")
        return get_judicial_reasoning_prompt(reasoning_prompt_type, jurisdiction, response, query, jurisdiction_context)
    
    def _get_precedent_analysis_prompt(self, prompt_type: str, jurisdiction: USJurisdiction,
                                     query: str, response: str, **kwargs) -> str:
        """Get precedent analysis prompt"""
        precedent_prompt_type = PrecedentAnalysisPromptType(prompt_type)
        jurisdiction_context = kwargs.get("jurisdiction_context", f"{jurisdiction.value} precedent analysis")
        return get_precedent_analysis_prompt(precedent_prompt_type, jurisdiction, response, query, jurisdiction_context)
    
    def _get_opinion_generation_prompt(self, prompt_type: str, jurisdiction: USJurisdiction,
                                     query: str, response: str, **kwargs) -> str:
        """Get opinion generation prompt"""
        opinion_prompt_type = OpinionGenerationPromptType(prompt_type)
        jurisdiction_context = kwargs.get("jurisdiction_context", f"{jurisdiction.value} legal advocacy")
        return get_opinion_generation_prompt(opinion_prompt_type, jurisdiction, response, query, jurisdiction_context)
    
    def _get_general_chat_prompt(self, prompt_type: str, jurisdiction: USJurisdiction,
                               query: str, response: str, **kwargs) -> str:
        """Get general chat prompt"""
        # Create context
        context = GeneralChatEvaluationContext(
            query=query,
            response=response,
            jurisdiction=jurisdiction,
            legal_domain=kwargs.get("legal_domain", "General"),
            task_context=kwargs.get("task_context"),
            user_type=kwargs.get("user_type", "general_public")
        )
        
        chat_prompt_type = GeneralChatPromptType(prompt_type)
        return get_general_chat_prompt(chat_prompt_type, context)
    
    def _cache_prompt_template(self, cache_key: str, full_prompt: str, 
                             query: str, response: str):
        """Cache prompt template with dynamic content placeholders"""
        if not self.config.cache_enabled:
            return
        
        # Replace dynamic content with placeholders for caching
        template = full_prompt.replace(query, "{query}").replace(response, "{response}")
        
        # Manage cache size
        if len(self._prompt_cache) >= self.config.cache_size_limit:
            # Remove oldest entries (simple FIFO)
            oldest_key = next(iter(self._prompt_cache))
            del self._prompt_cache[oldest_key]
        
        self._prompt_cache[cache_key] = template
    
    def _substitute_dynamic_content(self, template: str, query: str, 
                                  response: str, **kwargs) -> str:
        """Substitute dynamic content in cached template"""
        return template.format(query=query, response=response, **kwargs)
    
    def validate_prompt_system(self) -> Dict[str, Any]:
        """
        Validate the entire prompt system for completeness and accuracy.
        
        Returns:
            Validation results with status and any issues found
        """
        
        validation_results = {
            "system_valid": True,
            "component_status": {},
            "issues": [],
            "statistics": {}
        }
        
        # Validate each component
        components = [
            ("judicial_reasoning", JUDICIAL_REASONING_PROMPT_INFO),
            ("precedent_analysis", PRECEDENT_ANALYSIS_PROMPT_INFO),
            ("opinion_generation", OPINION_GENERATION_PROMPT_INFO),
            ("general_chat", GENERAL_CHAT_PROMPT_INFO),
            ("jurisdiction_compliance", JURISDICTION_COMPLIANCE_PROMPT_INFO)
        ]
        
        total_templates = 0
        total_jurisdictions = 0
        
        for component_name, prompt_info in components:
            try:
                # Validate component
                component_valid = True
                component_issues = []
                
                # Check template count
                template_count = prompt_info.get("template_count", 0)
                if template_count == 0:
                    component_valid = False
                    component_issues.append("No templates found")
                
                # Check jurisdiction coverage
                jurisdiction_count = prompt_info.get("jurisdiction_count", 0)
                if jurisdiction_count < 6:  # At least federal + 5 major states
                    component_valid = False
                    component_issues.append("Insufficient jurisdiction coverage")
                
                validation_results["component_status"][component_name] = {
                    "valid": component_valid,
                    "template_count": template_count,
                    "jurisdiction_count": jurisdiction_count,
                    "issues": component_issues
                }
                
                if not component_valid:
                    validation_results["system_valid"] = False
                    validation_results["issues"].extend([f"{component_name}: {issue}" for issue in component_issues])
                
                total_templates += template_count
                total_jurisdictions = max(total_jurisdictions, jurisdiction_count)
                
            except Exception as e:
                validation_results["system_valid"] = False
                validation_results["component_status"][component_name] = {
                    "valid": False,
                    "error": str(e)
                }
                validation_results["issues"].append(f"{component_name}: Validation error - {str(e)}")
        
        # Overall statistics
        validation_results["statistics"] = {
            "total_templates": total_templates,
            "total_jurisdictions": total_jurisdictions,
            "components_validated": len(components),
            "valid_components": sum(1 for status in validation_results["component_status"].values() if status.get("valid", False))
        }
        
        return validation_results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get prompt system performance statistics"""
        stats = self.performance_stats.copy()
        
        # Calculate derived metrics
        total_requests = stats["cache_hits"] + stats["cache_misses"]
        if total_requests > 0:
            stats["cache_hit_rate"] = stats["cache_hits"] / total_requests
        else:
            stats["cache_hit_rate"] = 0.0
        
        # Add manager stats
        if self.config.cache_enabled:
            stats["cache_size"] = len(self._prompt_cache) if self._prompt_cache else 0
            stats["general_chat_cache"] = self.general_chat_manager.get_cache_stats()
        
        stats["gating_failures_stats"] = self.jurisdiction_compliance_manager.get_failure_statistics()
        
        return stats


def get_prompt_for_task_type(task_type: LegalTaskType, prompt_type: str,
                           jurisdiction: USJurisdiction, query: str, 
                           response: str, **kwargs) -> str:
    """
    Convenience function to get prompt for any task type.
    
    This creates a temporary UnifiedPromptTemplateSystem instance.
    For production use, create a persistent instance for better performance.
    """
    
    system = UnifiedPromptTemplateSystem()
    return system.get_prompt_for_task_type(task_type, prompt_type, jurisdiction, 
                                         query, response, **kwargs)


def validate_prompt_system() -> Dict[str, Any]:
    """
    Convenience function to validate the entire prompt system.
    
    Returns:
        Validation results
    """
    
    system = UnifiedPromptTemplateSystem()
    return system.validate_prompt_system()


def get_prompt_system_info() -> Dict[str, Any]:
    """
    Get comprehensive information about the prompt template system.
    
    Returns:
        System information including capabilities and statistics
    """
    
    return {
        "package_version": __version__,
        "package_description": __description__,
        
        # Component information
        "components": {
            "judicial_reasoning": JUDICIAL_REASONING_PROMPT_INFO,
            "precedent_analysis": PRECEDENT_ANALYSIS_PROMPT_INFO,
            "opinion_generation": OPINION_GENERATION_PROMPT_INFO,
            "general_chat": GENERAL_CHAT_PROMPT_INFO,
            "jurisdiction_compliance": JURISDICTION_COMPLIANCE_PROMPT_INFO
        },
        
        # System capabilities
        "capabilities": {
            "total_prompt_types": sum(info.get("template_count", 0) for info in [
                JUDICIAL_REASONING_PROMPT_INFO,
                PRECEDENT_ANALYSIS_PROMPT_INFO,
                OPINION_GENERATION_PROMPT_INFO,
                GENERAL_CHAT_PROMPT_INFO,
                JURISDICTION_COMPLIANCE_PROMPT_INFO
            ]),
            "jurisdiction_coverage": len(USJurisdiction),
            "task_type_coverage": len(LegalTaskType),
            "gating_enabled": True,
            "caching_supported": True,
            "validation_supported": True
        },
        
        # Integration points
        "integration": {
            "verl_compatible": True,
            "grpo_optimized": True,
            "api_client_integration": True,
            "jurisdiction_system_integration": True,
            "hybrid_evaluation_support": True
        }
    }


# Initialize validation on import for early error detection
def _validate_package_integrity():
    """Validate package integrity on import"""
    try:
        validation_result = validate_prompt_system()
        if not validation_result["system_valid"]:
            import warnings
            warnings.warn(f"Prompt system validation issues: {validation_result['issues']}")
    except Exception as e:
        import warnings
        warnings.warn(f"Could not validate prompt system: {str(e)}")


# Run validation check
_validate_package_integrity()


# Example usage
if __name__ == "__main__":
    # Create unified system
    prompt_system = UnifiedPromptTemplateSystem()
    
    # Test prompt generation
    test_prompt = prompt_system.get_prompt_for_task_type(
        task_type=LegalTaskType.JUDICIAL_REASONING,
        prompt_type="legal_accuracy",
        jurisdiction=USJurisdiction.CALIFORNIA,
        query="Analyze the contract formation requirements",
        response="Contract formation requires offer, acceptance, and consideration...",
        jurisdiction_context="California contract law analysis"
    )
    
    print("Sample Unified Prompt System Output:")
    print("=" * 60)
    print(test_prompt[:300] + "...")
    
    # Validate system
    validation = prompt_system.validate_prompt_system()
    print(f"\nSystem Validation: {'PASSED' if validation['system_valid'] else 'FAILED'}")
    print(f"Components: {validation['statistics']['valid_components']}/{validation['statistics']['components_validated']}")
    
    # Show system info
    system_info = get_prompt_system_info()
    print(f"\nSystem Capabilities:")
    print(f"Total Prompt Types: {system_info['capabilities']['total_prompt_types']}")
    print(f"Jurisdiction Coverage: {system_info['capabilities']['jurisdiction_coverage']}")
    print(f"VERL Compatible: {system_info['integration']['verl_compatible']}")
