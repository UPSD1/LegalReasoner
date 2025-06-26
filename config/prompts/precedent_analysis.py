"""
Precedent Analysis Prompt Templates for Multi-Task Legal Reward System

This module contains professional-grade prompt templates for evaluating precedent
analysis, case law reasoning, and analogical legal analysis. Templates are designed 
for US jurisdiction-aware evaluation with state-specific case law context and 
professional legal research standards.

Key Features:
- US jurisdiction-specific case law context
- Professional precedent analysis criteria
- Citation quality assessment templates
- Analogical reasoning evaluation prompts
- Court hierarchy and authority understanding
"""

from typing import Dict, Optional
from enum import Enum
from ...core import USJurisdiction


class PrecedentAnalysisPromptType(Enum):
    """Types of precedent analysis evaluation prompts"""
    CASE_LAW_ACCURACY = "case_law_accuracy"
    ANALOGICAL_REASONING = "analogical_reasoning"
    DISTINGUISHING_FACTORS = "distinguishing_factors"
    HIERARCHY_UNDERSTANDING = "hierarchy_understanding"
    CITATION_QUALITY = "citation_quality"


# Base templates for precedent analysis evaluation
BASE_PROMPTS = {
    PrecedentAnalysisPromptType.CASE_LAW_ACCURACY: """
You are a distinguished legal research expert evaluating the accuracy of case law identification and precedent description in a legal analysis for {jurisdiction_context}.

**Jurisdiction Context**: {jurisdiction_context}
**Court System**: {court_hierarchy}
**Key Legal Authorities**: {key_authorities}

**Original Legal Query**: {prompt}

**Legal Response to Evaluate**: 
{response}

**Evaluation Criteria - Case Law Accuracy (Scale 0-10)**:

**Exceptional (9-10)**: 
- Precedents cited are highly relevant and accurately described
- Case holdings and legal principles are precisely represented
- Factual summaries of cases are accurate and complete
- Legal reasoning from cases is correctly understood and explained
- Case citations demonstrate deep research and understanding
- Perfect accuracy in representing {jurisdiction} case law
- Professional-level legal research quality

**Proficient (7-8)**:
- Most precedents are relevant and accurately described
- Generally accurate representation of case holdings
- Minor gaps in case descriptions or legal principles
- Good understanding of legal reasoning from cases
- Meets professional {jurisdiction} legal research standards

**Developing (5-6)**:
- Some precedents relevant but descriptions lack accuracy
- Basic understanding of case holdings with notable gaps
- Some inaccuracies in factual case summaries
- Limited grasp of legal reasoning from precedents
- Falls short of professional research standards

**Inadequate (0-4)**:
- Irrelevant precedents or significant inaccuracies
- Major misrepresentation of case holdings
- Substantial errors in factual case descriptions
- Poor understanding of legal reasoning from cases
- Does not meet {jurisdiction} legal research standards

**{jurisdiction} Case Law Considerations**:
{case_law_considerations}

**Required Response Format**:
{{"score": X.X, "reasoning": "Comprehensive evaluation of case law accuracy, precedent relevance, and research quality within {jurisdiction} legal framework with specific examples"}}
""",

    PrecedentAnalysisPromptType.ANALOGICAL_REASONING: """
You are a distinguished legal reasoning expert evaluating the quality of analogical reasoning and case-to-case comparisons in a legal analysis for {jurisdiction_context}.

**Jurisdiction Context**: {jurisdiction_context}
**Analogical Framework**: {analogical_framework}
**Precedent Standards**: {precedent_standards}

**Original Legal Query**: {prompt}

**Legal Response to Evaluate**: 
{response}

**Evaluation Criteria - Analogical Reasoning (Scale 0-10)**:

**Exceptional (9-10)**: 
- Sophisticated and compelling analogical connections between cases
- Deep analysis of relevant similarities and their legal significance
- Excellent identification of key factors that make cases comparable
- Strong logical reasoning about how precedents support arguments
- Nuanced understanding of analogical strength and limitations
- Demonstrates mastery of {jurisdiction} analogical reasoning traditions
- Professional-level comparative legal analysis

**Proficient (7-8)**:
- Good analogical connections with sound logical basis
- Clear identification of relevant case similarities
- Appropriate reasoning about precedent application
- Understanding of analogical strength within {jurisdiction} context
- Meets professional analogical reasoning standards

**Developing (5-6)**:
- Basic analogical reasoning with some logical connections
- Limited identification of relevant similarities
- Superficial analysis of case comparisons
- Some understanding of precedent application
- Meets minimum analogical reasoning requirements

**Inadequate (0-4)**:
- Poor or illogical analogical connections
- Failure to identify relevant case similarities
- Weak or no comparative analysis between cases
- Misunderstanding of how precedents relate to current situation
- Falls below {jurisdiction} reasoning standards

**{jurisdiction} Analogical Reasoning Standards**:
{analogical_standards}

**Required Response Format**:
{{"score": X.X, "reasoning": "Detailed assessment of analogical reasoning quality, case comparison strength, and logical connections within {jurisdiction} legal tradition"}}
""",

    PrecedentAnalysisPromptType.DISTINGUISHING_FACTORS: """
You are a distinguished legal analysis expert evaluating the identification and analysis of distinguishing factors between cases in a legal analysis for {jurisdiction_context}.

**Jurisdiction Context**: {jurisdiction_context}
**Distinguishing Framework**: {distinguishing_framework}
**Legal Limitation Standards**: {limitation_standards}

**Original Legal Query**: {prompt}

**Legal Response to Evaluate**: 
{response}

**Evaluation Criteria - Distinguishing Factors Analysis (Scale 0-10)**:

**Exceptional (9-10)**: 
- Excellent identification of material differences between cases
- Sophisticated analysis of why certain precedents may not apply
- Clear recognition of factual distinctions affecting legal outcomes
- Strong understanding of legal distinctions that limit precedent scope
- Comprehensive analysis balancing similarities and differences
- Demonstrates expertise in {jurisdiction} case distinction methods
- Professional-level precedent limitation analysis

**Proficient (7-8)**:
- Good identification of material case differences
- Clear explanation of precedent limitations
- Sound recognition of factual and legal distinctions
- Appropriate analysis of case applicability
- Meets professional {jurisdiction} distinction standards

**Developing (5-6)**:
- Basic identification of some case differences
- Limited analysis of precedent limitations
- Superficial recognition of distinguishing factors
- Some understanding of case applicability limits
- Meets minimum distinction analysis requirements

**Inadequate (0-4)**:
- Poor identification of material differences
- Failure to recognize precedent limitations
- Weak analysis of distinguishing factors
- Misunderstanding of case applicability
- Below {jurisdiction} legal analysis standards

**{jurisdiction} Case Distinction Methods**:
{distinction_methods}

**Required Response Format**:
{{"score": X.X, "reasoning": "Thorough evaluation of distinguishing factors analysis, precedent limitation understanding, and case distinction quality within {jurisdiction} framework"}}
""",

    PrecedentAnalysisPromptType.HIERARCHY_UNDERSTANDING: """
You are a distinguished judicial authority expert evaluating understanding of precedent hierarchy and the distinction between binding and persuasive authority in a legal analysis for {jurisdiction_context}.

**Jurisdiction Context**: {jurisdiction_context}
**Court Hierarchy**: {court_hierarchy}
**Authority Framework**: {authority_framework}

**Original Legal Query**: {prompt}

**Legal Response to Evaluate**: 
{response}

**Evaluation Criteria - Hierarchy Understanding (Scale 0-10)**:

**Exceptional (9-10)**: 
- Perfect recognition of binding vs. persuasive authority
- Sophisticated understanding of court hierarchy and precedent strength
- Excellent treatment of higher court vs. lower court decisions
- Clear recognition of jurisdictional limits on precedent applicability
- Deep understanding of how hierarchy affects legal analysis
- Masterful application of {jurisdiction} precedent hierarchy rules
- Professional-level authority recognition and application

**Proficient (7-8)**:
- Good recognition of binding and persuasive authority
- Sound understanding of court hierarchy within {jurisdiction}
- Appropriate treatment of different court levels
- Clear recognition of jurisdictional precedent limits
- Meets professional hierarchy understanding standards

**Developing (5-6)**:
- Basic understanding of authority types
- Limited grasp of court hierarchy implications
- Some recognition of precedent strength differences
- Minimal understanding of jurisdictional limits
- Meets basic hierarchy comprehension requirements

**Inadequate (0-4)**:
- Poor understanding of binding vs. persuasive authority
- Confusion about court hierarchy and precedent strength
- Inappropriate treatment of different authority levels
- Misunderstanding of jurisdictional precedent rules
- Below {jurisdiction} professional standards

**{jurisdiction} Hierarchy Specific Rules**:
{hierarchy_rules}

**Required Response Format**:
{{"score": X.X, "reasoning": "Comprehensive assessment of precedent hierarchy understanding, authority recognition, and jurisdictional awareness within {jurisdiction} legal system"}}
""",

    PrecedentAnalysisPromptType.CITATION_QUALITY: """
You are a distinguished legal citation expert evaluating the quality, format, and appropriateness of legal citations in a legal analysis for {jurisdiction_context}.

**Jurisdiction Context**: {jurisdiction_context}
**Citation Standards**: {citation_standards}
**Format Requirements**: {format_requirements}

**Original Legal Query**: {prompt}

**Legal Response to Evaluate**: 
{response}

**Evaluation Criteria - Citation Quality (Scale 0-10)**:

**Exceptional (9-10)**: 
- Perfect legal citation format and style throughout
- Complete citation information (case name, court, year, reporter, etc.)
- Accurate citations with precise details and pinpoint references
- Appropriate use of pinpoint citations when needed for specific points
- Consistent citation format following {jurisdiction} conventions
- Professional-level citation quality suitable for court filings
- Excellent integration of citations within legal analysis

**Proficient (7-8)**:
- Good citation format with minor formatting issues
- Generally complete citation information
- Mostly accurate citations with few errors
- Appropriate citation usage for legal support
- Meets professional {jurisdiction} citation standards

**Developing (5-6)**:
- Basic citation format with notable inconsistencies
- Some missing citation information
- Several citation inaccuracies or format errors
- Limited use of pinpoint citations where needed
- Meets minimum citation requirements

**Inadequate (0-4)**:
- Poor citation format with major inconsistencies
- Significant missing citation information
- Multiple citation inaccuracies and format errors
- Inappropriate or missing citations for legal assertions
- Below {jurisdiction} professional citation standards

**{jurisdiction} Citation Requirements**:
{citation_requirements}

**Required Response Format**:
{{"score": X.X, "reasoning": "Detailed evaluation of citation format, accuracy, completeness, and professional appropriateness within {jurisdiction} legal citation standards"}}
"""
}

# Jurisdiction-specific context for precedent analysis
JURISDICTION_SPECIFIC_CONTEXTS = {
    USJurisdiction.FEDERAL: {
        "court_hierarchy": "U.S. Supreme Court > Federal Circuit Courts > Federal District Courts",
        "key_authorities": "U.S. Constitution, federal statutes, Supreme Court precedents, circuit court decisions",
        "case_law_considerations": "Federal precedent applies nationwide; circuit splits create complexity until Supreme Court resolution",
        "analogical_framework": "Federal courts emphasize constitutional principles and statutory interpretation in analogical reasoning",
        "precedent_standards": "Federal precedent hierarchy strictly observed; Supreme Court precedent binding on all lower courts",
        "analogical_standards": "Federal analogical reasoning focuses on constitutional principles and federal statutory interpretation",
        "distinguishing_framework": "Federal courts carefully distinguish precedent based on constitutional and statutory differences",
        "limitation_standards": "Federal precedent limitation based on constitutional scope and statutory authority",
        "distinction_methods": "Federal case distinction emphasizes constitutional analysis and federal jurisdiction limits",
        "authority_framework": "Supreme Court binding; circuit precedent binding within circuit; district court persuasive only",
        "hierarchy_rules": "Strict federal hierarchy: Supreme Court > Circuit > District; circuit splits resolved by Supreme Court",
        "citation_standards": "Federal citation follows Bluebook format with emphasis on federal reporters and pinpoint citations",
        "format_requirements": "Bluebook citation format required for federal court practice",
        "citation_requirements": "Federal citations must include reporter, court, year, and pinpoint citations for specific holdings"
    },
    
    USJurisdiction.CALIFORNIA: {
        "court_hierarchy": "California Supreme Court > California Courts of Appeal > California Superior Courts",
        "key_authorities": "California Constitution, California codes, California Supreme Court and appellate decisions",
        "case_law_considerations": "California precedent binding within state; California often leads in progressive legal development",
        "analogical_framework": "California courts emphasize practical application and policy considerations in analogical reasoning",
        "precedent_standards": "California Supreme Court precedent binding statewide; Court of Appeal precedent binding in district",
        "analogical_standards": "California analogical reasoning balances legal doctrine with practical policy implications",
        "distinguishing_framework": "California courts distinguish cases based on factual differences and policy considerations",
        "limitation_standards": "California precedent limitation considers both legal doctrine and underlying policy rationale",
        "distinction_methods": "California case distinction emphasizes practical differences and policy implications",
        "authority_framework": "California Supreme Court binding; Court of Appeal binding within district; trial court persuasive",
        "hierarchy_rules": "California hierarchy: Supreme Court > Courts of Appeal > Superior Courts; published opinions binding",
        "citation_standards": "California citation follows California Style Manual with emphasis on official reporters",
        "format_requirements": "California Style Manual format preferred for California state court practice",
        "citation_requirements": "California citations emphasize official reporters, parallel citations, and California-specific format"
    },
    
    USJurisdiction.NEW_YORK: {
        "court_hierarchy": "New York Court of Appeals > Appellate Division > New York trial courts",
        "key_authorities": "New York Constitution, New York statutes, Court of Appeals and Appellate Division decisions",
        "case_law_considerations": "New York precedent particularly influential in commercial and corporate law nationally",
        "analogical_framework": "New York courts emphasize sophisticated legal reasoning and commercial law precision",
        "precedent_standards": "Court of Appeals precedent binding statewide; Appellate Division precedent binding in department",
        "analogical_standards": "New York analogical reasoning emphasizes legal sophistication and commercial law expertise",
        "distinguishing_framework": "New York courts distinguish cases with attention to commercial and corporate law nuances",
        "limitation_standards": "New York precedent limitation considers commercial law implications and legal sophistication",
        "distinction_methods": "New York case distinction emphasizes legal sophistication and commercial law precision",
        "authority_framework": "Court of Appeals binding; Appellate Division binding within department; trial court persuasive",
        "hierarchy_rules": "New York hierarchy: Court of Appeals > Appellate Division > trial courts; published opinions binding",
        "citation_standards": "New York citation follows standard legal citation with New York reporter preferences",
        "format_requirements": "Standard legal citation format adapted for New York practice requirements",
        "citation_requirements": "New York citations emphasize official state reporters and precise pinpoint citations"
    },
    
    USJurisdiction.TEXAS: {
        "court_hierarchy": "Texas Supreme Court/Court of Criminal Appeals > Texas Courts of Appeals > Texas District Courts",
        "key_authorities": "Texas Constitution, Texas codes, Texas Supreme Court and appellate court decisions",
        "case_law_considerations": "Texas has dual supreme courts for civil and criminal matters; emphasizes state sovereignty",
        "analogical_framework": "Texas courts emphasize practical application and business-friendly legal reasoning",
        "precedent_standards": "Supreme Court precedent binding; Court of Appeals precedent binding within district",
        "analogical_standards": "Texas analogical reasoning focuses on practical application and economic considerations",
        "distinguishing_framework": "Texas courts distinguish cases with attention to practical and economic factors",
        "limitation_standards": "Texas precedent limitation considers practical application and state sovereignty principles",
        "distinction_methods": "Texas case distinction emphasizes practical differences and economic implications",
        "authority_framework": "Supreme Court binding; Court of Appeals binding within district; trial court persuasive",
        "hierarchy_rules": "Texas dual hierarchy: Supreme Court (civil) and Court of Criminal Appeals (criminal) at top",
        "citation_standards": "Texas citation follows standard format with attention to Texas-specific reporters",
        "format_requirements": "Standard legal citation adapted for Texas dual court system",
        "citation_requirements": "Texas citations must account for dual supreme court system and appropriate jurisdiction"
    },
    
    USJurisdiction.FLORIDA: {
        "court_hierarchy": "Florida Supreme Court > District Courts of Appeal > Florida Circuit Courts",
        "key_authorities": "Florida Constitution, Florida statutes, Florida Supreme Court and DCA decisions",
        "case_law_considerations": "Florida precedent binding within state; attention to district court jurisdictional boundaries",
        "analogical_framework": "Florida courts balance traditional legal reasoning with modern practical applications",
        "precedent_standards": "Florida Supreme Court binding; DCA precedent binding within district",
        "analogical_standards": "Florida analogical reasoning balances traditional doctrine with practical considerations",
        "distinguishing_framework": "Florida courts distinguish cases based on factual and legal differences",
        "limitation_standards": "Florida precedent limitation considers both legal doctrine and practical applications",
        "distinction_methods": "Florida case distinction emphasizes clear factual and legal differences",
        "authority_framework": "Supreme Court binding; DCA binding within district; circuit court persuasive",
        "hierarchy_rules": "Florida hierarchy: Supreme Court > District Courts of Appeal > Circuit Courts",
        "citation_standards": "Florida citation follows standard format with Florida reporter preferences",
        "format_requirements": "Standard legal citation format for Florida court practice",
        "citation_requirements": "Florida citations emphasize official reporters and district court jurisdiction awareness"
    },
    
    USJurisdiction.GENERAL: {
        "court_hierarchy": "General U.S. court hierarchy principles with appellate court structure",
        "key_authorities": "Generally applicable U.S. legal precedents and widely recognized case law",
        "case_law_considerations": "Focus on widely applicable precedents and general legal principles",
        "analogical_framework": "Standard U.S. analogical reasoning principles applicable across jurisdictions",
        "precedent_standards": "General U.S. precedent hierarchy and authority principles",
        "analogical_standards": "Standard analogical reasoning principles used across U.S. jurisdictions",
        "distinguishing_framework": "General principles of case distinction applicable in U.S. legal system",
        "limitation_standards": "Standard precedent limitation principles used in U.S. legal analysis",
        "distinction_methods": "General case distinction methods applicable across U.S. jurisdictions",
        "authority_framework": "General understanding of binding vs. persuasive authority in U.S. legal system",
        "hierarchy_rules": "Standard U.S. court hierarchy principles: appellate courts bind trial courts",
        "citation_standards": "Standard U.S. legal citation format following generally accepted principles",
        "format_requirements": "Standard legal citation format used across U.S. jurisdictions",
        "citation_requirements": "Standard U.S. citation requirements with proper case identification and pinpoint citations"
    }
}


def get_precedent_analysis_prompt(prompt_type: PrecedentAnalysisPromptType,
                                jurisdiction: USJurisdiction,
                                response: str,
                                query: str,
                                jurisdiction_context: str) -> str:
    """
    Get a formatted precedent analysis evaluation prompt.
    
    Args:
        prompt_type: Type of precedent analysis evaluation
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
    jurisdiction_context_dict = JURISDICTION_SPECIFIC_CONTEXTS.get(
        jurisdiction, 
        JURISDICTION_SPECIFIC_CONTEXTS[USJurisdiction.GENERAL]
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


def get_all_precedent_analysis_prompts(jurisdiction: USJurisdiction,
                                     response: str,
                                     query: str,
                                     jurisdiction_context: str) -> Dict[str, str]:
    """
    Get all precedent analysis prompts for a jurisdiction.
    
    Args:
        jurisdiction: US jurisdiction for context
        response: Legal response to evaluate
        query: Original legal query
        jurisdiction_context: Additional jurisdiction context
        
    Returns:
        Dictionary mapping prompt types to formatted prompts
    """
    
    prompts = {}
    
    for prompt_type in PrecedentAnalysisPromptType:
        prompts[prompt_type.value] = get_precedent_analysis_prompt(
            prompt_type, jurisdiction, response, query, jurisdiction_context
        )
    
    return prompts


def validate_precedent_analysis_prompt(prompt_type: PrecedentAnalysisPromptType,
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


# Prompt template metadata
PRECEDENT_ANALYSIS_PROMPT_INFO = {
    "template_count": len(BASE_PROMPTS),
    "jurisdiction_count": len(JURISDICTION_SPECIFIC_CONTEXTS),
    "supported_jurisdictions": list(JURISDICTION_SPECIFIC_CONTEXTS.keys()),
    "prompt_types": [pt.value for pt in PrecedentAnalysisPromptType],
    "description": "Professional precedent analysis evaluation prompts with US jurisdiction case law awareness"
}


def get_prompt_info() -> Dict[str, any]:
    """Get information about precedent analysis prompt templates"""
    return PRECEDENT_ANALYSIS_PROMPT_INFO.copy()


# Example usage
if __name__ == "__main__":
    # Test prompt generation
    test_response = "This case is analogous to Smith v. Jones where the court held..."
    test_query = "Analyze relevant precedent for contract unconscionability claim"
    test_jurisdiction = USJurisdiction.CALIFORNIA
    test_context = "California state court contract law precedent analysis"
    
    # Get a specific prompt
    analogical_reasoning_prompt = get_precedent_analysis_prompt(
        PrecedentAnalysisPromptType.ANALOGICAL_REASONING,
        test_jurisdiction,
        test_response,
        test_query,
        test_context
    )
    
    print("Sample Precedent Analysis Prompt:")
    print("=" * 50)
    print(analogical_reasoning_prompt[:500] + "...")
    
    # Show prompt info
    print(f"\nPrompt Template Info:")
    info = get_prompt_info()
    print(f"Template Count: {info['template_count']}")
    print(f"Supported Jurisdictions: {len(info['supported_jurisdictions'])}")
    print(f"Prompt Types: {info['prompt_types']}")
