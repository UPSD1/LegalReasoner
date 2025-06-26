"""
Judicial Reasoning Prompt Templates for Multi-Task Legal Reward System

This module contains professional-grade prompt templates for evaluating judicial
reasoning and formal legal analysis. Templates are designed for US jurisdiction-aware
evaluation with state-specific legal context and professional legal standards.

Key Features:
- US jurisdiction-specific prompt variations
- Professional legal evaluation criteria
- FIRAC structure assessment templates
- Constitutional analysis prompts
- State-specific legal context integration
"""

from typing import Dict, Optional
from enum import Enum
from ...core import USJurisdiction


class JudicialReasoningPromptType(Enum):
    """Types of judicial reasoning evaluation prompts"""
    LEGAL_ACCURACY = "legal_accuracy"
    FIRAC_STRUCTURE = "firac_structure"
    PRECEDENT_APPLICATION = "precedent_application"
    CONSTITUTIONAL_ANALYSIS = "constitutional_analysis"
    JUDICIAL_TONE = "judicial_tone"


# Base templates that work for all jurisdictions
BASE_PROMPTS = {
    JudicialReasoningPromptType.LEGAL_ACCURACY: """
You are a distinguished legal expert evaluating the legal accuracy and doctrinal correctness of a judicial reasoning response in the context of {jurisdiction_context}.

**Legal Context**: {jurisdiction_context}
**Applicable Legal Framework**: {legal_framework}

**Original Legal Query**: {prompt}

**Judicial Response to Evaluate**: 
{response}

**Evaluation Criteria - Legal Accuracy (Scale 0-10)**:

**Exceptional (9-10)**: 
- All legal doctrines and principles cited are completely accurate
- Statutory interpretation demonstrates deep understanding of legal text
- Legal precedents are correctly identified and applied
- Factual legal statements are precise and verifiable
- Complex legal issues are identified with sophistication
- {jurisdiction_specific_accuracy}

**Proficient (7-8)**:
- Most legal doctrines and principles are accurate with minor gaps
- Statutory interpretation is generally sound with small omissions
- Precedents are mostly correctly applied
- Legal statements are largely accurate
- Key legal issues are properly identified
- Meets {jurisdiction} professional standards

**Developing (5-6)**:
- Some legal concepts are accurate but significant gaps exist
- Statutory interpretation shows basic understanding but lacks depth
- Precedent application has notable errors
- Some factual inaccuracies in legal statements
- Misses some important legal issues

**Inadequate (0-4)**:
- Major legal inaccuracies or fundamental misunderstandings
- Incorrect statutory interpretation
- Misapplication of precedents
- Significant factual errors in legal statements
- Fails to identify critical legal issues

**Special Considerations for {jurisdiction}**:
{jurisdiction_considerations}

**Required Response Format**:
{{"score": X.X, "reasoning": "Detailed assessment of legal accuracy including specific examples of strengths and weaknesses, with reference to {jurisdiction} legal standards"}}
""",

    JudicialReasoningPromptType.FIRAC_STRUCTURE: """
You are a distinguished judicial writing expert evaluating the FIRAC structure and legal reasoning flow of a judicial response in {jurisdiction_context}.

**Jurisdiction Context**: {jurisdiction_context}
**Expected Analytical Framework**: FIRAC (Facts, Issues, Rules, Application, Conclusion)

**Original Legal Query**: {prompt}

**Judicial Response to Evaluate**: 
{response}

**Evaluation Criteria - FIRAC Structure (Scale 0-10)**:

**Exceptional (9-10)**: 
- **Facts**: Clear, relevant facts precisely identified and stated
- **Issues**: Legal issues properly framed and articulated
- **Rules**: Applicable legal rules accurately stated with authority
- **Application**: Logical, thorough application of rules to facts
- **Conclusion**: Clear, justified conclusion following from analysis
- Seamless logical flow with excellent transitions
- Demonstrates mastery of {jurisdiction} judicial reasoning standards

**Proficient (7-8)**:
- All FIRAC elements present and generally well-executed
- Good logical organization with clear structure
- Minor gaps in transitions or reasoning flow
- Meets professional {jurisdiction} judicial writing standards
- Conclusion supported by analysis

**Developing (5-6)**:
- Most FIRAC elements present but some are underdeveloped
- Basic logical organization but weak transitions
- Some confusion in rule application
- Conclusion somewhat supported but reasoning unclear

**Inadequate (0-4)**:
- Missing critical FIRAC elements
- Poor logical organization and flow
- Weak or incorrect rule application
- Unsupported or missing conclusion
- Does not meet {jurisdiction} judicial standards

**{jurisdiction} Specific FIRAC Expectations**:
{firac_expectations}

**Required Response Format**:
{{"score": X.X, "reasoning": "Detailed evaluation of each FIRAC component and overall reasoning flow, with specific examples and {jurisdiction} context"}}
""",

    JudicialReasoningPromptType.PRECEDENT_APPLICATION: """
You are a distinguished legal precedent expert evaluating the application of case law and precedents in a judicial reasoning response for {jurisdiction_context}.

**Jurisdiction Context**: {jurisdiction_context}
**Precedent Hierarchy**: {precedent_hierarchy}
**Key Legal Authorities**: {key_authorities}

**Original Legal Query**: {prompt}

**Judicial Response to Evaluate**: 
{response}

**Evaluation Criteria - Precedent Application (Scale 0-10)**:

**Exceptional (9-10)**: 
- Highly relevant and authoritative precedents selected
- Accurate representation of case holdings and rationale  
- Sophisticated analogical reasoning between cases
- Proper distinction between binding and persuasive authority
- Excellent integration of precedents into legal argument
- Demonstrates deep understanding of {jurisdiction} case law
- Professional-level citation and precedent usage

**Proficient (7-8)**:
- Relevant precedents appropriately selected and applied
- Generally accurate case descriptions and holdings
- Good analogical reasoning with precedents
- Understands precedent hierarchy in {jurisdiction}
- Proper integration of case law into analysis

**Developing (5-6)**:
- Some relevant precedents used but selection could be better
- Basic understanding of case holdings with minor inaccuracies
- Limited analogical reasoning
- Some confusion about precedent hierarchy
- Precedents not well integrated into argument

**Inadequate (0-4)**:
- Irrelevant or inappropriate precedent selection
- Significant inaccuracies in case descriptions
- Poor or no analogical reasoning
- Misunderstanding of precedent hierarchy
- Precedents poorly integrated or misapplied

**{jurisdiction} Precedent Considerations**:
{precedent_considerations}

**Required Response Format**:
{{"score": X.X, "reasoning": "Comprehensive evaluation of precedent usage, case law accuracy, and analogical reasoning within {jurisdiction} legal framework"}}
""",

    JudicialReasoningPromptType.CONSTITUTIONAL_ANALYSIS: """
You are a distinguished constitutional law expert evaluating the constitutional analysis and federal/state jurisdictional understanding in a judicial response for {jurisdiction_context}.

**Jurisdiction Context**: {jurisdiction_context}
**Constitutional Framework**: {constitutional_framework}
**Federal/State Considerations**: {federal_state_considerations}

**Original Legal Query**: {prompt}

**Judicial Response to Evaluate**: 
{response}

**Evaluation Criteria - Constitutional Analysis (Scale 0-10)**:

**Exceptional (9-10)**: 
- Accurate identification of constitutional issues when present
- Sophisticated application of constitutional principles and doctrines
- Clear understanding of federal vs. state law boundaries
- Proper analysis of constitutional hierarchy and supremacy
- Advanced constitutional interpretation methodology
- Excellent handling of {jurisdiction} constitutional provisions
- Professional-level constitutional reasoning

**Proficient (7-8)**:
- Proper identification of constitutional issues
- Good application of constitutional principles
- Sound understanding of federal/state distinctions
- Appropriate constitutional analysis methodology
- Meets {jurisdiction} constitutional analysis standards

**Developing (5-6)**:
- Basic identification of constitutional issues
- Limited application of constitutional principles
- Some confusion about federal/state boundaries
- Simple constitutional analysis approach
- Meets minimum constitutional reasoning requirements

**Inadequate (0-4)**:
- Fails to identify relevant constitutional issues
- Misapplication of constitutional principles
- Confusion about federal/state law distinctions
- Poor constitutional analysis methodology
- Below {jurisdiction} constitutional standards

**{jurisdiction} Constitutional Considerations**:
{constitutional_considerations}

**Special Note**: If no constitutional issues are present in the response, evaluate whether this omission is appropriate and justified.

**Required Response Format**:
{{"score": X.X, "reasoning": "Thorough assessment of constitutional analysis quality, federal/state law understanding, and {jurisdiction} constitutional compliance"}}
""",

    JudicialReasoningPromptType.JUDICIAL_TONE: """
You are a distinguished judicial writing expert evaluating the tone, style, and professional appropriateness of a judicial reasoning response in {jurisdiction_context}.

**Jurisdiction Context**: {jurisdiction_context}
**Expected Tone Standards**: {tone_standards}
**Professional Requirements**: {professional_requirements}

**Original Legal Query**: {prompt}

**Judicial Response to Evaluate**: 
{response}

**Evaluation Criteria - Judicial Tone and Style (Scale 0-10)**:

**Exceptional (9-10)**: 
- Impeccably professional and formal judicial tone throughout
- Objective, impartial language befitting judicial authority
- Sophisticated legal terminology used appropriately
- Clear, authoritative writing style with excellent flow
- Consistent professional tone from beginning to end
- Exemplifies {jurisdiction} judicial writing excellence
- Writing quality suitable for published judicial opinions

**Proficient (7-8)**:
- Professional and appropriate judicial tone
- Generally objective and impartial language
- Proper legal terminology and vocabulary
- Clear writing style with good organization
- Maintains professional standards throughout
- Meets {jurisdiction} judicial writing expectations

**Developing (5-6)**:
- Mostly appropriate tone with some informal elements
- Generally professional but occasional lapses
- Basic legal terminology usage
- Adequate writing clarity
- Some inconsistency in professional tone

**Inadequate (0-4)**:
- Inappropriate or unprofessional tone
- Biased or non-objective language
- Poor use of legal terminology
- Unclear or disorganized writing
- Fails to meet {jurisdiction} professional standards

**{jurisdiction} Judicial Writing Standards**:
{writing_standards}

**Required Response Format**:
{{"score": X.X, "reasoning": "Detailed evaluation of judicial tone, writing style, and professional appropriateness with specific examples and {jurisdiction} context"}}
"""
}

# Jurisdiction-specific template variations
JURISDICTION_SPECIFIC_PROMPTS = {
    USJurisdiction.FEDERAL: {
        "legal_framework": "Federal statutes, constitutional law, federal regulations, and federal case law",
        "jurisdiction_specific_accuracy": "Demonstrates mastery of federal legal doctrine and constitutional principles",
        "jurisdiction_considerations": "Federal courts require strict adherence to constitutional analysis and federal statutory interpretation",
        "firac_expectations": "Federal judicial opinions require rigorous constitutional analysis and clear statutory interpretation",
        "precedent_hierarchy": "Supreme Court > Circuit Courts > District Courts",
        "key_authorities": "U.S. Constitution, federal statutes, Supreme Court and Circuit Court precedents",
        "precedent_considerations": "Must distinguish between circuit splits and binding Supreme Court precedent",
        "constitutional_framework": "U.S. Constitution, Bill of Rights, constitutional amendments",
        "federal_state_considerations": "Federal law supremacy, preemption analysis, constitutional boundaries",
        "constitutional_considerations": "Requires sophisticated understanding of constitutional doctrine and federal supremacy",
        "tone_standards": "Formal federal judicial tone appropriate for federal court opinions",
        "professional_requirements": "Must meet federal judicial writing standards for published opinions",
        "writing_standards": "Federal courts expect precise, formal, and authoritative judicial writing"
    },
    
    USJurisdiction.CALIFORNIA: {
        "legal_framework": "California statutes, California Constitution, California case law, and relevant federal law",
        "jurisdiction_specific_accuracy": "Demonstrates expertise in California legal doctrine and state-specific requirements",
        "jurisdiction_considerations": "California law often provides broader protections than federal minimums, requires understanding of California's unique legal approaches",
        "firac_expectations": "California judicial opinions emphasize practical application and clear policy reasoning",
        "precedent_hierarchy": "California Supreme Court > California Courts of Appeal > Superior Courts",
        "key_authorities": "California Constitution, California codes, California Supreme Court and appellate decisions",
        "precedent_considerations": "California precedent is binding within the state; federal precedent applicable for federal constitutional issues",
        "constitutional_framework": "California Constitution, which often provides broader rights than federal Constitution",
        "federal_state_considerations": "California law may exceed federal minimums; state constitutional protections may be broader",
        "constitutional_considerations": "Must understand California's independent state constitutional analysis",
        "tone_standards": "Professional California judicial tone balancing formality with accessibility",
        "professional_requirements": "Must meet California judicial Council writing standards",
        "writing_standards": "California courts value clear, practical reasoning accessible to diverse audiences"
    },
    
    USJurisdiction.NEW_YORK: {
        "legal_framework": "New York statutes, New York Constitution, New York case law, and relevant federal law",
        "jurisdiction_specific_accuracy": "Demonstrates mastery of New York's unique legal traditions and commercial law expertise",
        "jurisdiction_considerations": "New York law, especially commercial and corporate law, is influential nationwide",
        "firac_expectations": "New York judicial opinions emphasize thorough legal analysis and commercial law precision",
        "precedent_hierarchy": "New York Court of Appeals > Appellate Division > Trial Courts",
        "key_authorities": "New York Constitution, New York statutes, Court of Appeals and Appellate Division decisions",
        "precedent_considerations": "New York precedent binding within state; New York commercial law often persuasive nationally",
        "constitutional_framework": "New York Constitution and its unique provisions",
        "federal_state_considerations": "New York law particularly important in commercial and corporate contexts",
        "constitutional_considerations": "New York constitutional analysis with attention to commercial law implications",
        "tone_standards": "Sophisticated New York judicial tone reflecting the state's legal prominence",
        "professional_requirements": "Must meet New York's high judicial writing standards",
        "writing_standards": "New York courts expect precise, sophisticated analysis especially in commercial matters"
    },
    
    USJurisdiction.TEXAS: {
        "legal_framework": "Texas statutes, Texas Constitution, Texas case law, and relevant federal law",
        "jurisdiction_specific_accuracy": "Demonstrates understanding of Texas legal traditions and state-specific approaches",
        "jurisdiction_considerations": "Texas law emphasizes property rights, business-friendly approaches, and state sovereignty",
        "firac_expectations": "Texas judicial opinions value clear reasoning and practical applications",
        "precedent_hierarchy": "Texas Supreme Court/Texas Court of Criminal Appeals > Texas Courts of Appeals > District Courts",
        "key_authorities": "Texas Constitution, Texas codes, Texas Supreme Court and appellate court decisions",
        "precedent_considerations": "Texas precedent binding within state; dual supreme court system for civil and criminal matters",
        "constitutional_framework": "Texas Constitution with its unique provisions and structure",
        "federal_state_considerations": "Texas emphasizes state rights and limited federal intervention where possible",
        "constitutional_considerations": "Texas constitutional analysis with attention to state sovereignty principles",
        "tone_standards": "Professional Texas judicial tone emphasizing clarity and practical application",
        "professional_requirements": "Must meet Texas judicial writing standards",
        "writing_standards": "Texas courts value straightforward, practical legal reasoning"
    },
    
    USJurisdiction.FLORIDA: {
        "legal_framework": "Florida statutes, Florida Constitution, Florida case law, and relevant federal law",
        "jurisdiction_specific_accuracy": "Demonstrates understanding of Florida legal approaches and state-specific requirements",
        "jurisdiction_considerations": "Florida law balances traditional and modern approaches, with unique procedural requirements",
        "firac_expectations": "Florida judicial opinions emphasize clear application of law to facts",
        "precedent_hierarchy": "Florida Supreme Court > District Courts of Appeal > Circuit Courts",
        "key_authorities": "Florida Constitution, Florida statutes, Florida Supreme Court and DCA decisions",
        "precedent_considerations": "Florida precedent binding within state; attention to district court jurisdiction",
        "constitutional_framework": "Florida Constitution and its specific provisions",
        "federal_state_considerations": "Florida law applications with federal constitutional compliance",
        "constitutional_considerations": "Florida constitutional analysis with state-specific considerations",
        "tone_standards": "Professional Florida judicial tone balancing formality with practicality",
        "professional_requirements": "Must meet Florida judicial writing standards",
        "writing_standards": "Florida courts expect clear, practical legal analysis"
    },
    
    USJurisdiction.GENERAL: {
        "legal_framework": "General U.S. legal principles, common law traditions, and widely applicable precedents",
        "jurisdiction_specific_accuracy": "Demonstrates sound understanding of general U.S. legal principles",
        "jurisdiction_considerations": "Analysis should apply to general U.S. legal contexts without jurisdiction-specific requirements",
        "firac_expectations": "Standard judicial reasoning format appropriate for general U.S. legal analysis",
        "precedent_hierarchy": "General understanding of hierarchical court systems and precedent authority",
        "key_authorities": "Generally applicable U.S. legal sources and widely accepted precedents",
        "precedent_considerations": "Focus on widely applicable precedents and general legal principles",
        "constitutional_framework": "U.S. Constitution and generally applicable constitutional principles",
        "federal_state_considerations": "General understanding of federal and state law interactions",
        "constitutional_considerations": "General constitutional analysis applicable across U.S. jurisdictions",
        "tone_standards": "Professional judicial tone appropriate for general U.S. legal analysis",
        "professional_requirements": "General U.S. professional legal writing standards",
        "writing_standards": "Clear, professional legal writing meeting general U.S. judicial standards"
    }
}


def get_judicial_reasoning_prompt(prompt_type: JudicialReasoningPromptType,
                                jurisdiction: USJurisdiction,
                                response: str,
                                query: str,
                                jurisdiction_context: str) -> str:
    """
    Get a formatted judicial reasoning evaluation prompt.
    
    Args:
        prompt_type: Type of judicial reasoning evaluation
        jurisdiction: US jurisdiction for context
        response: Legal response to evaluate
        query: Original legal query
        jurisdiction_context: Additional jurisdiction context
        
    Returns:
        Formatted evaluation prompt
    """
    
    # Get base prompt template
    base_template = BASE_PROMPTS[prompt_type]
    
    # Get jurisdiction-specific context
    jurisdiction_context_dict = JURISDICTION_SPECIFIC_PROMPTS.get(
        jurisdiction, 
        JURISDICTION_SPECIFIC_PROMPTS[USJurisdiction.GENERAL]
    )
    
    # Format the prompt with all context
    formatted_prompt = base_template.format(
        jurisdiction_context=jurisdiction_context,
        response=response,
        prompt=query,
        jurisdiction=jurisdiction.value,
        **jurisdiction_context_dict
    )
    
    return formatted_prompt


def get_all_judicial_reasoning_prompts(jurisdiction: USJurisdiction,
                                     response: str,
                                     query: str,
                                     jurisdiction_context: str) -> Dict[str, str]:
    """
    Get all judicial reasoning prompts for a jurisdiction.
    
    Args:
        jurisdiction: US jurisdiction for context
        response: Legal response to evaluate
        query: Original legal query
        jurisdiction_context: Additional jurisdiction context
        
    Returns:
        Dictionary mapping prompt types to formatted prompts
    """
    
    prompts = {}
    
    for prompt_type in JudicialReasoningPromptType:
        prompts[prompt_type.value] = get_judicial_reasoning_prompt(
            prompt_type, jurisdiction, response, query, jurisdiction_context
        )
    
    return prompts


def validate_judicial_reasoning_prompt(prompt_type: JudicialReasoningPromptType,
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
            jurisdiction in JURISDICTION_SPECIFIC_PROMPTS)


# Prompt template metadata
JUDICIAL_REASONING_PROMPT_INFO = {
    "template_count": len(BASE_PROMPTS),
    "jurisdiction_count": len(JURISDICTION_SPECIFIC_PROMPTS),
    "supported_jurisdictions": list(JURISDICTION_SPECIFIC_PROMPTS.keys()),
    "prompt_types": [pt.value for pt in JudicialReasoningPromptType],
    "description": "Professional judicial reasoning evaluation prompts with US jurisdiction awareness"
}


def get_prompt_info() -> Dict[str, any]:
    """Get information about judicial reasoning prompt templates"""
    return JUDICIAL_REASONING_PROMPT_INFO.copy()


# Example usage
if __name__ == "__main__":
    # Test prompt generation
    test_response = "The court finds that defendant's conduct constitutes a breach..."
    test_query = "Analyze whether the defendant's conduct constitutes breach of contract"
    test_jurisdiction = USJurisdiction.CALIFORNIA
    test_context = "California state court contract law analysis"
    
    # Get a specific prompt
    legal_accuracy_prompt = get_judicial_reasoning_prompt(
        JudicialReasoningPromptType.LEGAL_ACCURACY,
        test_jurisdiction,
        test_response,
        test_query,
        test_context
    )
    
    print("Sample Judicial Reasoning Prompt:")
    print("=" * 50)
    print(legal_accuracy_prompt[:500] + "...")
    
    # Show prompt info
    print(f"\nPrompt Template Info:")
    info = get_prompt_info()
    print(f"Template Count: {info['template_count']}")
    print(f"Supported Jurisdictions: {len(info['supported_jurisdictions'])}")
    print(f"Prompt Types: {info['prompt_types']}")
