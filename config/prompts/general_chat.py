"""
General Chat Prompt Templates for Enhanced Legal AI System

This module provides sophisticated prompt templates for evaluating general legal
chat responses with US jurisdiction awareness. These templates are used by the 
EnhancedGeneralChatEnsemble and serve as the 30% chat quality component in 
hybrid evaluation for specialized tasks.

Key Evaluation Components:
- Helpfulness (25%): Practical utility and completeness
- Legal Ethics (25%): Professional responsibility and compliance  
- Clarity (25%): Communication effectiveness and comprehensibility
- Jurisdiction Compliance (25%): US legal system accuracy (CRITICAL GATING)

Author: Legal Reward System Team
Version: 1.0.0
"""

from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass

# Import jurisdiction system
from ...jurisdiction.us_system import USJurisdiction


class GeneralChatPromptType(Enum):
    """Types of general chat evaluation prompts"""
    HELPFULNESS = "helpfulness"
    LEGAL_ETHICS = "legal_ethics"
    CLARITY = "clarity"
    JURISDICTION_COMPLIANCE = "jurisdiction_compliance"


@dataclass
class GeneralChatEvaluationContext:
    """Context for general chat evaluation"""
    query: str
    response: str
    jurisdiction: USJurisdiction
    legal_domain: str
    task_context: Optional[str] = None
    user_type: str = "general_public"  # legal_professional, law_student, general_public


# Base prompt templates for each evaluation component
BASE_PROMPTS = {
    GeneralChatPromptType.HELPFULNESS: """
You are a distinguished legal expert evaluating the helpfulness and practical utility of a legal AI response within the {jurisdiction} legal system.

**Jurisdiction Context**: {jurisdiction_context}
**Legal Domain**: {legal_domain}
**User Context**: {user_type}

**Original Legal Query**: {query}

**Legal AI Response to Evaluate**: 
{response}

**Evaluation Criteria - Helpfulness (Scale 0-10)**:

**Exceptional (9-10)**:
- Provides comprehensive, actionable guidance appropriate for the legal question
- Demonstrates deep understanding of practical legal implications in {jurisdiction}
- Offers specific next steps or resources when appropriate
- Addresses the user's underlying needs beyond the literal question
- Provides appropriate scope - neither oversimplified nor unnecessarily complex
- Includes relevant warnings about legal complexity when needed
- **{jurisdiction} Helpfulness Standards**: {helpfulness_standards}

**Proficient (7-8)**:
- Provides useful, relevant information that addresses the query
- Generally understands practical implications in {jurisdiction} 
- Offers some actionable guidance or next steps
- Appropriate level of detail for the question asked
- Some consideration of user's practical needs

**Developing (5-6)**:
- Provides basic information but limited practical utility
- Basic understanding of {jurisdiction} legal context
- Limited actionable guidance
- May be too general or miss key practical considerations
- Partially addresses the user's needs

**Inadequate (0-4)**:
- Provides little practical value to the user
- Poor understanding of {jurisdiction} legal implications
- No actionable guidance or inappropriate advice
- Significantly over or under-scoped for the question
- Fails to address user's underlying legal needs

**{jurisdiction} Specific Helpfulness Considerations**:
{helpfulness_context}

**Required Response Format**:
{{"score": X.X, "reasoning": "Comprehensive evaluation of practical utility, actionable guidance, and appropriate scope within {jurisdiction} legal framework"}}
""",

    GeneralChatPromptType.LEGAL_ETHICS: """
You are a distinguished legal ethics expert evaluating the ethical compliance and professional responsibility standards of a legal AI response within the {jurisdiction} legal system.

**Jurisdiction Context**: {jurisdiction_context}
**Applicable Ethics Rules**: {ethics_framework}
**Professional Standards**: {professional_standards}

**Original Legal Query**: {query}

**Legal AI Response to Evaluate**: 
{response}

**Evaluation Criteria - Legal Ethics (Scale 0-10)**:

**Exceptional (9-10)**:
- Fully compliant with {jurisdiction} professional responsibility rules
- Appropriate disclaimers about attorney-client relationships and legal advice
- Demonstrates awareness of unauthorized practice of law boundaries
- Maintains confidentiality principles and professional discretion
- Encourages consultation with licensed attorneys when appropriate
- Truthful and accurate representation of legal information
- **{jurisdiction} Ethics Standards**: {ethics_standards}

**Proficient (7-8)**:
- Generally compliant with professional responsibility standards
- Includes basic disclaimers about legal advice limitations
- Shows awareness of practice boundaries
- Maintains professional tone and appropriate boundaries
- Generally truthful in legal information presentation

**Developing (5-6)**:
- Some ethical considerations addressed but incompletely
- Basic disclaimers present but may be insufficient
- Limited awareness of professional boundaries
- Some concerns about unauthorized practice implications
- Generally truthful but may have minor accuracy issues

**Inadequate (0-4)**:
- Significant ethical violations or professional responsibility breaches
- Inappropriate attorney-client relationship implications
- Unauthorized practice of law concerns
- Missing critical disclaimers or warnings
- Misleading or inaccurate legal information
- Violations of confidentiality or professional standards

**{jurisdiction} Professional Responsibility Context**:
{ethics_context}

**Required Response Format**:
{{"score": X.X, "reasoning": "Detailed evaluation of ethical compliance, professional boundaries, and responsibility standards within {jurisdiction} legal framework"}}
""",

    GeneralChatPromptType.CLARITY: """
You are a distinguished legal communication expert evaluating the clarity, comprehensibility, and communication effectiveness of a legal AI response for the {jurisdiction} legal system.

**Jurisdiction Context**: {jurisdiction_context}
**Communication Standards**: {communication_framework}
**Target Audience**: {user_type}

**Original Legal Query**: {query}

**Legal AI Response to Evaluate**: 
{response}

**Evaluation Criteria - Clarity (Scale 0-10)**:

**Exceptional (9-10)**:
- Crystal clear explanation of complex legal concepts
- Excellent organization with logical flow of information
- Appropriate use of legal terminology with plain language explanations
- Perfect audience targeting for {user_type} in {jurisdiction}
- Complex ideas broken down into understandable components
- Effective use of examples or analogies when helpful
- **{jurisdiction} Communication Standards**: {clarity_standards}

**Proficient (7-8)**:
- Clear and understandable explanation of legal concepts
- Good organization and logical structure
- Appropriate balance of legal terminology and accessibility
- Generally well-targeted for the intended audience
- Most complex ideas explained clearly

**Developing (5-6)**:
- Generally understandable but some unclear passages
- Basic organization but could be more logical
- Some overuse or underuse of legal terminology
- Partially appropriate for target audience
- Some complex concepts not well explained

**Inadequate (0-4)**:
- Unclear or confusing explanation of legal concepts
- Poor organization and illogical structure
- Inappropriate use of legal jargon or oversimplification
- Poorly targeted for intended audience
- Complex ideas not broken down effectively
- Difficult to follow or understand

**{jurisdiction} Communication Context**:
{clarity_context}

**Required Response Format**:
{{"score": X.X, "reasoning": "Comprehensive evaluation of clarity, organization, and communication effectiveness for {user_type} within {jurisdiction} legal system"}}
""",

    GeneralChatPromptType.JURISDICTION_COMPLIANCE: """
You are a distinguished {jurisdiction} legal expert evaluating the jurisdictional accuracy and compliance of a legal AI response within the {jurisdiction} legal system.

**Critical Assessment**: This is a GATING evaluation - significant jurisdictional errors should result in low scores that may disqualify the entire response.

**Jurisdiction Context**: {jurisdiction_context}
**Legal Framework**: {legal_framework}
**Jurisdictional Requirements**: {jurisdiction_requirements}

**Original Legal Query**: {query}

**Legal AI Response to Evaluate**: 
{response}

**Evaluation Criteria - Jurisdiction Compliance (Scale 0-10)**:

**Exceptional (9-10)**:
- Demonstrates sophisticated understanding of {jurisdiction} legal system
- Accurate application of jurisdiction-specific laws, procedures, and standards
- Proper recognition of {jurisdiction} unique legal approaches and requirements
- Correct understanding of state vs. federal law applications where relevant
- Accurate representation of {jurisdiction} court system and procedures
- **{jurisdiction} Compliance Excellence**: {compliance_standards}

**Proficient (7-8)**:
- Good understanding of {jurisdiction} legal framework
- Generally accurate application of jurisdiction-specific requirements
- Correct basic understanding of {jurisdiction} legal system
- Appropriate recognition of jurisdictional boundaries
- Generally accurate procedural and substantive law references

**Developing (5-6)**:
- Basic understanding of {jurisdiction} but with some gaps
- Some inaccuracies in jurisdiction-specific applications
- Limited recognition of unique {jurisdiction} requirements
- Some confusion between state and federal law applications
- Basic procedural understanding with some errors

**Inadequate (0-4)**:
- Significant misunderstanding of {jurisdiction} legal system
- Major inaccuracies in jurisdiction-specific law application
- Failure to recognize {jurisdiction} unique requirements
- Serious confusion between different jurisdictional frameworks
- Dangerous misinformation about {jurisdiction} legal procedures
- **CRITICAL**: May render entire response unreliable for {jurisdiction} use

**{jurisdiction} Specific Compliance Requirements**:
{compliance_context}

**Gating Function Notice**: Scores below 3.0 indicate serious jurisdictional compliance failures that may disqualify the response for use in {jurisdiction} legal contexts.

**Required Response Format**:
{{"score": X.X, "reasoning": "Critical evaluation of jurisdictional accuracy, legal framework compliance, and {jurisdiction}-specific requirement adherence"}}
"""
}


# Jurisdiction-specific contexts for general chat evaluation
JURISDICTION_SPECIFIC_CONTEXTS = {
    USJurisdiction.FEDERAL: {
        "jurisdiction_context": "Federal legal system covering constitutional law, federal statutes, and federal court procedures",
        "legal_domain": "Federal law",
        "helpfulness_standards": "Federal legal guidance emphasizing constitutional compliance and federal jurisdictional clarity",
        "helpfulness_context": "Federal legal help must consider constitutional implications and federal vs. state jurisdictional boundaries",
        "ethics_framework": "Federal court professional responsibility rules and federal attorney conduct standards",
        "professional_standards": "Federal court ethics rules and constitutional constraints on legal practice",
        "ethics_standards": "Federal ethics emphasizing constitutional compliance and federal court professional responsibility",
        "ethics_context": "Federal ethics require attention to constitutional constraints and federal court professional conduct rules",
        "communication_framework": "Federal legal communication standards for constitutional and federal statutory matters",
        "clarity_standards": "Federal communication emphasizing constitutional precision and federal law clarity",
        "clarity_context": "Federal communication must be precise on constitutional matters and federal jurisdictional boundaries",
        "legal_framework": "U.S. Constitution, federal statutes, federal court rules, and federal case law",
        "jurisdiction_requirements": "Must comply with federal constitutional constraints and federal court jurisdictional requirements",
        "compliance_standards": "Federal compliance requires constitutional accuracy and federal jurisdictional precision",
        "compliance_context": "Federal compliance must address constitutional implications and federal vs. state law distinctions"
    },
    
    USJurisdiction.CALIFORNIA: {
        "jurisdiction_context": "California state legal system with progressive legal approaches and comprehensive consumer protections",
        "legal_domain": "California state law",
        "helpfulness_standards": "California legal guidance emphasizing consumer protection and progressive legal development",
        "helpfulness_context": "California legal help often involves broader protections than federal minimums and unique state approaches",
        "ethics_framework": "California State Bar professional responsibility rules and California attorney conduct standards",
        "professional_standards": "California State Bar ethics rules and California-specific professional conduct requirements",
        "ethics_standards": "California ethics emphasizing consumer protection and progressive professional responsibility",
        "ethics_context": "California ethics include strong consumer protection focus and progressive professional responsibility approaches",
        "communication_framework": "California legal communication standards emphasizing accessibility and consumer protection",
        "clarity_standards": "California communication emphasizing consumer accessibility and progressive legal clarity",
        "clarity_context": "California communication should consider diverse population and strong consumer protection framework",
        "legal_framework": "California Constitution, California codes, California case law, and relevant federal law",
        "jurisdiction_requirements": "Must comply with California's progressive legal standards and consumer protection requirements",
        "compliance_standards": "California compliance requires understanding of progressive legal development and consumer protection focus",
        "compliance_context": "California compliance must address state's progressive legal approaches and comprehensive consumer protections"
    },
    
    USJurisdiction.NEW_YORK: {
        "jurisdiction_context": "New York state legal system with sophisticated commercial law and complex procedural requirements",
        "legal_domain": "New York state law",
        "helpfulness_standards": "New York legal guidance emphasizing commercial sophistication and procedural precision",
        "helpfulness_context": "New York legal help often involves complex commercial matters and sophisticated procedural requirements",
        "ethics_framework": "New York State Bar professional responsibility rules and New York attorney conduct standards",
        "professional_standards": "New York State Bar ethics rules and New York commercial practice professional standards",
        "ethics_standards": "New York ethics emphasizing commercial practice sophistication and professional responsibility",
        "ethics_context": "New York ethics include sophisticated commercial practice considerations and complex professional responsibility requirements",
        "communication_framework": "New York legal communication standards emphasizing commercial precision and procedural clarity",
        "clarity_standards": "New York communication emphasizing commercial sophistication and procedural precision",
        "clarity_context": "New York communication should reflect sophisticated commercial practice and complex procedural requirements",
        "legal_framework": "New York Constitution, New York statutes, New York case law, and relevant federal law",
        "jurisdiction_requirements": "Must comply with New York's sophisticated commercial law standards and procedural requirements",
        "compliance_standards": "New York compliance requires commercial law sophistication and procedural precision",
        "compliance_context": "New York compliance must address sophisticated commercial practice and complex procedural framework"
    },
    
    USJurisdiction.TEXAS: {
        "jurisdiction_context": "Texas state legal system emphasizing property rights, business-friendly approaches, and state sovereignty",
        "legal_domain": "Texas state law",
        "helpfulness_standards": "Texas legal guidance emphasizing practical business solutions and property rights",
        "helpfulness_context": "Texas legal help often emphasizes business-friendly approaches and practical property rights solutions",
        "ethics_framework": "Texas State Bar professional responsibility rules and Texas attorney conduct standards",
        "professional_standards": "Texas State Bar ethics rules and Texas business practice professional standards",
        "ethics_standards": "Texas ethics emphasizing business practice and practical professional responsibility",
        "ethics_context": "Texas ethics emphasize practical business considerations and straightforward professional responsibility approaches",
        "communication_framework": "Texas legal communication standards emphasizing practical clarity and business application",
        "clarity_standards": "Texas communication emphasizing practical business clarity and straightforward legal explanation",
        "clarity_context": "Texas communication should be practical, business-focused, and straightforward in legal explanation",
        "legal_framework": "Texas Constitution, Texas codes, Texas case law, and relevant federal law",
        "jurisdiction_requirements": "Must comply with Texas business-friendly standards and property rights emphasis",
        "compliance_standards": "Texas compliance requires understanding of business-friendly approaches and property rights focus",
        "compliance_context": "Texas compliance must address business-friendly legal environment and property rights emphasis"
    },
    
    USJurisdiction.FLORIDA: {
        "jurisdiction_context": "Florida state legal system balancing traditional and modern approaches with unique procedural requirements",
        "legal_domain": "Florida state law",
        "helpfulness_standards": "Florida legal guidance balancing traditional legal principles with modern practical applications",
        "helpfulness_context": "Florida legal help balances traditional legal approaches with modern practical needs and unique state requirements",
        "ethics_framework": "Florida Bar professional responsibility rules and Florida attorney conduct standards",
        "professional_standards": "Florida Bar ethics rules and Florida practice professional standards",
        "ethics_standards": "Florida ethics balancing traditional professional responsibility with modern practice needs",
        "ethics_context": "Florida ethics balance traditional professional responsibility approaches with modern practice considerations",
        "communication_framework": "Florida legal communication standards balancing formality with practical accessibility",
        "clarity_standards": "Florida communication balancing professional formality with practical clarity",
        "clarity_context": "Florida communication should balance traditional legal formality with practical accessibility and clarity",
        "legal_framework": "Florida Constitution, Florida statutes, Florida case law, and relevant federal law",
        "jurisdiction_requirements": "Must comply with Florida's balanced traditional-modern approach and unique procedural requirements",
        "compliance_standards": "Florida compliance requires understanding of balanced traditional-modern legal approaches",
        "compliance_context": "Florida compliance must address balanced traditional-modern framework and unique state procedural requirements"
    },
    
    USJurisdiction.GENERAL: {
        "jurisdiction_context": "General U.S. legal system covering widely applicable legal principles and common law traditions",
        "legal_domain": "General U.S. law",
        "helpfulness_standards": "General U.S. legal guidance applicable across jurisdictions with universal legal principles",
        "helpfulness_context": "General legal help should apply broadly across U.S. jurisdictions without state-specific requirements",
        "ethics_framework": "General U.S. professional responsibility principles and widely applicable attorney conduct standards",
        "professional_standards": "General U.S. ethics principles applicable across jurisdictions",
        "ethics_standards": "General U.S. ethics emphasizing widely applicable professional responsibility principles",
        "ethics_context": "General ethics should apply broadly across U.S. jurisdictions without state-specific requirements",
        "communication_framework": "General U.S. legal communication standards applicable across jurisdictions",
        "clarity_standards": "General U.S. communication emphasizing universal clarity and broad applicability",
        "clarity_context": "General communication should be clear and applicable across diverse U.S. legal contexts",
        "legal_framework": "U.S. Constitution, generally applicable federal law, and common law principles",
        "jurisdiction_requirements": "Must comply with general U.S. legal principles applicable across jurisdictions",
        "compliance_standards": "General compliance requires accuracy on widely applicable U.S. legal principles",
        "compliance_context": "General compliance must address broadly applicable U.S. legal framework without state-specific requirements"
    }
}


def get_general_chat_prompt(prompt_type: GeneralChatPromptType,
                           context: GeneralChatEvaluationContext) -> str:
    """
    Get a formatted general chat evaluation prompt.
    
    Args:
        prompt_type: Type of general chat evaluation
        context: Evaluation context including query, response, jurisdiction
        
    Returns:
        Formatted evaluation prompt
    """
    
    # Get base prompt template
    base_template = BASE_PROMPTS[prompt_type]
    
    # Get jurisdiction-specific context
    jurisdiction_context_dict = JURISDICTION_SPECIFIC_CONTEXTS.get(
        context.jurisdiction, 
        JURISDICTION_SPECIFIC_CONTEXTS[USJurisdiction.GENERAL]
    )
    
    # Format the prompt with all context
    formatted_prompt = base_template.format(
        query=context.query,
        response=context.response,
        jurisdiction=context.jurisdiction.value,
        legal_domain=context.legal_domain,
        user_type=context.user_type,
        **jurisdiction_context_dict
    )
    
    return formatted_prompt


def get_all_general_chat_prompts(context: GeneralChatEvaluationContext) -> Dict[str, str]:
    """
    Get all general chat evaluation prompts for a jurisdiction.
    
    Args:
        context: Evaluation context
        
    Returns:
        Dictionary mapping prompt types to formatted prompts
    """
    
    prompts = {}
    
    for prompt_type in GeneralChatPromptType:
        prompts[prompt_type.value] = get_general_chat_prompt(prompt_type, context)
    
    return prompts


def validate_general_chat_prompt(prompt_type: GeneralChatPromptType,
                                jurisdiction: USJurisdiction) -> bool:
    """
    Validate that a prompt type is available for a jurisdiction.
    
    Args:
        prompt_type: Prompt type to validate
        jurisdiction: Jurisdiction to check
        
    Returns:
        True if prompt is available
    """
    
    return (prompt_type in BASE_PROMPTS and 
            jurisdiction in JURISDICTION_SPECIFIC_CONTEXTS)


# Enhanced prompt template utilities
class GeneralChatPromptManager:
    """Manager for general chat prompt templates with caching and optimization"""
    
    def __init__(self):
        self._prompt_cache = {}
        
    def get_cached_prompt(self, prompt_type: GeneralChatPromptType, 
                         context: GeneralChatEvaluationContext) -> str:
        """Get prompt with caching for performance optimization"""
        
        cache_key = f"{prompt_type.value}_{context.jurisdiction.value}_{context.user_type}"
        
        if cache_key not in self._prompt_cache:
            self._prompt_cache[cache_key] = get_general_chat_prompt(prompt_type, context)
        
        # Dynamic substitution for query/response specific content
        cached_template = self._prompt_cache[cache_key]
        return cached_template.format(
            query=context.query,
            response=context.response
        )
    
    def clear_cache(self):
        """Clear prompt cache"""
        self._prompt_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            "cached_prompts": len(self._prompt_cache),
            "memory_usage_estimate": sum(len(p) for p in self._prompt_cache.values())
        }


# Prompt template metadata and validation
GENERAL_CHAT_PROMPT_INFO = {
    "template_count": len(BASE_PROMPTS),
    "jurisdiction_count": len(JURISDICTION_SPECIFIC_CONTEXTS),
    "supported_jurisdictions": list(JURISDICTION_SPECIFIC_CONTEXTS.keys()),
    "prompt_types": [pt.value for pt in GeneralChatPromptType],
    "description": "Professional general chat evaluation prompts with US jurisdiction awareness and gating functionality",
    "critical_components": ["jurisdiction_compliance"],  # GATING component
    "evaluation_weights": {
        "helpfulness": 0.25,
        "legal_ethics": 0.25,
        "clarity": 0.25,
        "jurisdiction_compliance": 0.25  # CRITICAL GATING
    }
}


def get_prompt_info() -> Dict[str, any]:
    """Get information about general chat prompt templates"""
    return GENERAL_CHAT_PROMPT_INFO.copy()


# Example usage and testing
if __name__ == "__main__":
    # Test context creation
    test_context = GeneralChatEvaluationContext(
        query="What should I know about contract formation in my state?",
        response="Contract formation requires offer, acceptance, and consideration...",
        jurisdiction=USJurisdiction.CALIFORNIA,
        legal_domain="Contract Law",
        user_type="general_public"
    )
    
    # Get a specific prompt
    helpfulness_prompt = get_general_chat_prompt(
        GeneralChatPromptType.HELPFULNESS,
        test_context
    )
    
    print("Sample General Chat Helpfulness Prompt:")
    print("=" * 60)
    print(helpfulness_prompt[:500] + "...")
    
    # Test jurisdiction compliance (CRITICAL GATING)
    compliance_prompt = get_general_chat_prompt(
        GeneralChatPromptType.JURISDICTION_COMPLIANCE,
        test_context
    )
    
    print(f"\nJurisdiction Compliance Prompt (GATING):")
    print("=" * 60)
    print(compliance_prompt[:300] + "...")
    
    # Show prompt info
    print(f"\nGeneral Chat Prompt Template Info:")
    info = get_prompt_info()
    print(f"Template Count: {info['template_count']}")
    print(f"Supported Jurisdictions: {len(info['supported_jurisdictions'])}")
    print(f"Critical Components: {info['critical_components']}")
    print(f"Evaluation Weights: {info['evaluation_weights']}")
