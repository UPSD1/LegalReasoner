"""
Opinion Generation Prompt Templates for Multi-Task Legal Reward System

This module contains professional-grade prompt templates for evaluating legal
opinion generation, advocacy writing, and persuasive legal communication. 
Templates are designed for US jurisdiction-aware evaluation with state-specific
advocacy standards and professional legal writing requirements.

Key Features:
- US jurisdiction-specific advocacy standards
- Professional legal writing evaluation criteria
- Client advocacy assessment templates
- Persuasive communication evaluation prompts
- Legal research quality assessment
"""

from typing import Dict, Optional
from enum import Enum
from ...core import USJurisdiction


class OpinionGenerationPromptType(Enum):
    """Types of opinion generation evaluation prompts"""
    ARGUMENT_STRENGTH = "argument_strength"
    PERSUASIVENESS = "persuasiveness"
    LEGAL_RESEARCH_QUALITY = "legal_research_quality"
    CLIENT_ADVOCACY = "client_advocacy"
    PROFESSIONAL_WRITING = "professional_writing"


# Base templates for opinion generation evaluation
BASE_PROMPTS = {
    OpinionGenerationPromptType.ARGUMENT_STRENGTH: """
You are a distinguished legal advocacy expert evaluating the logical strength and legal foundation of arguments in a legal opinion for {jurisdiction_context}.

**Jurisdiction Context**: {jurisdiction_context}
**Legal Standards**: {legal_standards}
**Research Sources**: {research_sources}
**Advocacy Framework**: {advocacy_framework}

**Original Legal Query**: {prompt}

**Legal Opinion to Evaluate**: 
{response}

**Evaluation Criteria - Argument Strength (Scale 0-10)**:

**Exceptional (9-10)**: 
- Arguments demonstrate exceptional logical force and coherence
- Rock-solid legal foundation supporting every argument presented
- Crystal-clear reasoning chain from premises to well-supported conclusions
- Masterful use of legal principles, doctrines, and authorities
- Compelling evidence and legal support for all assertions
- Perfect logical consistency throughout argument structure
- Demonstrates mastery of {jurisdiction} legal argument standards
- Professional-level advocacy suitable for complex litigation

**Proficient (7-8)**:
- Strong logical arguments with solid legal foundation
- Clear reasoning connecting premises to conclusions
- Good use of legal principles and supporting authorities
- Most arguments well-supported with appropriate evidence
- Meets professional {jurisdiction} advocacy standards
- Suitable for standard legal practice

**Developing (5-6)**:
- Basic logical structure with some strong arguments
- Limited legal foundation for some assertions
- Some gaps in reasoning or logical connections
- Adequate use of legal principles but lacking depth
- Meets minimum argument strength requirements

**Inadequate (0-4)**:
- Weak logical structure with poorly supported arguments
- Insufficient legal foundation for key assertions
- Major gaps in reasoning or logical inconsistencies
- Poor use of legal principles and authorities
- Falls below {jurisdiction} professional advocacy standards

**{jurisdiction} Argument Standards**:
{argument_standards}

**Required Response Format**:
{{"score": X.X, "reasoning": "Comprehensive evaluation of argument strength, logical foundation, and legal support quality within {jurisdiction} advocacy standards with specific examples"}}
""",

    OpinionGenerationPromptType.PERSUASIVENESS: """
You are a distinguished legal communication expert evaluating the rhetorical effectiveness and persuasive quality of a legal opinion in {jurisdiction_context}.

**Jurisdiction Context**: {jurisdiction_context}
**Advocacy Style**: {advocacy_style}
**Persuasion Standards**: {persuasion_standards}
**Communication Framework**: {communication_framework}

**Original Legal Query**: {prompt}

**Legal Opinion to Evaluate**: 
{response}

**Evaluation Criteria - Persuasiveness (Scale 0-10)**:

**Exceptional (9-10)**: 
- Exceptionally compelling and persuasive presentation throughout
- Masterful rhetorical strategies perfectly suited for legal advocacy
- Powerful opening that immediately engages and compelling conclusion
- Excellent persuasive organization with optimal argument sequencing
- Highly effective language choices that enhance persuasive impact
- Perfect tone balancing authority, credibility, and persuasive appeal
- Demonstrates mastery of {jurisdiction} persuasive advocacy traditions
- Professional-level persuasive writing suitable for high-stakes litigation

**Proficient (7-8)**:
- Strong persuasive presentation with compelling arguments
- Good rhetorical strategies appropriate for legal context
- Effective organization enhancing persuasive impact
- Appropriate language and tone for legal advocacy
- Meets professional {jurisdiction} persuasive writing standards

**Developing (5-6)**:
- Basic persuasive elements with some compelling moments
- Limited rhetorical strategies or persuasive techniques
- Adequate organization but missed persuasive opportunities
- Some effective language but lacks consistent persuasive impact
- Meets minimum persuasive communication requirements

**Inadequate (0-4)**:
- Weak or ineffective persuasive presentation
- Poor rhetorical strategies inappropriate for legal context
- Disorganized presentation that undermines persuasive goals
- Language and tone that detract from persuasive impact
- Falls below {jurisdiction} professional persuasive standards

**{jurisdiction} Persuasive Advocacy Standards**:
{persuasive_standards}

**Required Response Format**:
{{"score": X.X, "reasoning": "Detailed assessment of persuasive effectiveness, rhetorical quality, and communication impact within {jurisdiction} advocacy traditions"}}
""",

    OpinionGenerationPromptType.LEGAL_RESEARCH_QUALITY: """
You are a distinguished legal research expert evaluating the depth, relevance, and quality of legal research supporting arguments in a legal opinion for {jurisdiction_context}.

**Jurisdiction Context**: {jurisdiction_context}
**Research Standards**: {research_standards}
**Authority Framework**: {authority_framework}
**Source Requirements**: {source_requirements}

**Original Legal Query**: {prompt}

**Legal Opinion to Evaluate**: 
{response}

**Evaluation Criteria - Legal Research Quality (Scale 0-10)**:

**Exceptional (9-10)**: 
- Exceptionally thorough and comprehensive legal research demonstrated
- Highly relevant and authoritative legal sources expertly integrated
- Outstanding quality and authority of legal authorities referenced
- Complete coverage of all relevant legal issues and authorities
- Masterful integration of research seamlessly supporting arguments
- Current, accurate, and precisely relevant legal sources throughout
- Demonstrates expertise in {jurisdiction} legal research methods
- Professional-level research quality suitable for complex legal matters

**Proficient (7-8)**:
- Thorough legal research with relevant, authoritative sources
- Good integration of research supporting legal arguments
- Appropriate authority level for legal sources cited
- Adequate coverage of relevant legal issues
- Meets professional {jurisdiction} legal research standards

**Developing (5-6)**:
- Basic legal research with some relevant sources
- Limited integration of research into arguments
- Some appropriate authorities but gaps in coverage
- Minimal coverage of relevant legal issues
- Meets minimum legal research requirements

**Inadequate (0-4)**:
- Inadequate or superficial legal research
- Irrelevant or inappropriate legal sources
- Poor integration of research into arguments
- Significant gaps in coverage of legal issues
- Falls below {jurisdiction} professional research standards

**{jurisdiction} Research Standards**:
{research_standards_detail}

**Required Response Format**:
{{"score": X.X, "reasoning": "Comprehensive evaluation of legal research depth, source quality, and research integration within {jurisdiction} professional research standards"}}
""",

    OpinionGenerationPromptType.CLIENT_ADVOCACY: """
You are a distinguished client advocacy expert evaluating how effectively a legal opinion advances client interests and serves as strategic legal advocacy in {jurisdiction_context}.

**Jurisdiction Context**: {jurisdiction_context}
**Client Advocacy Standards**: {client_standards}
**Strategic Framework**: {strategic_framework}
**Ethical Boundaries**: {ethical_boundaries}

**Original Legal Query**: {prompt}

**Legal Opinion to Evaluate**: 
{response}

**Evaluation Criteria - Client Advocacy (Scale 0-10)**:

**Exceptional (9-10)**: 
- Exceptionally effective advancement of client interests and objectives
- Masterful strategic positioning of arguments to maximize client benefit
- Outstanding advocacy that strongly positions client for favorable outcomes
- Excellent anticipation and strategic addressing of opposing arguments
- Powerful client advocacy while scrupulously maintaining ethical boundaries
- Consistently client-focused approach throughout analysis and recommendations
- Demonstrates mastery of {jurisdiction} client advocacy traditions
- Professional-level strategic advocacy suitable for high-stakes representation

**Proficient (7-8)**:
- Strong advancement of client interests with good strategic positioning
- Clear advocacy for client's legal position and objectives
- Appropriate anticipation of opposing arguments
- Good balance of strong advocacy with ethical considerations
- Meets professional {jurisdiction} client advocacy standards

**Developing (5-6)**:
- Basic advancement of client interests with limited strategic thinking
- Some advocacy for client position but lacks strategic depth
- Minimal anticipation of opposing arguments
- Adequate ethical boundaries but limited advocacy strength
- Meets minimum client advocacy requirements

**Inadequate (0-4)**:
- Poor advancement of client interests or strategic positioning
- Weak advocacy that fails to effectively serve client objectives
- No anticipation of opposing arguments or strategic considerations
- Ethical issues or inadequate client advocacy
- Falls below {jurisdiction} professional advocacy standards

**{jurisdiction} Client Advocacy Standards**:
{client_advocacy_standards}

**Required Response Format**:
{{"score": X.X, "reasoning": "Thorough assessment of client advocacy effectiveness, strategic positioning, and professional representation quality within {jurisdiction} ethical and advocacy framework"}}
""",

    OpinionGenerationPromptType.PROFESSIONAL_WRITING: """
You are a distinguished legal writing expert evaluating the professional quality, clarity, and appropriateness of legal writing in a legal opinion for {jurisdiction_context}.

**Jurisdiction Context**: {jurisdiction_context}
**Writing Standards**: {writing_standards}
**Professional Requirements**: {professional_requirements}
**Style Framework**: {style_framework}

**Original Legal Query**: {prompt}

**Legal Opinion to Evaluate**: 
{response}

**Evaluation Criteria - Professional Writing Quality (Scale 0-10)**:

**Exceptional (9-10)**: 
- Exceptionally clear, professional, and polished legal writing throughout
- Perfect grammar, syntax, and sophisticated legal terminology usage
- Outstanding organization with seamless logical flow of ideas
- Ideal tone and formality perfectly suited for professional legal documents
- Exceptional clarity and precision while maintaining legal sophistication
- Professional formatting and presentation meeting highest standards
- Demonstrates mastery of {jurisdiction} legal writing excellence
- Writing quality suitable for publication or high-profile legal matters

**Proficient (7-8)**:
- Clear, professional legal writing with good organization
- Generally correct grammar and appropriate legal terminology
- Good logical flow with effective transitions between ideas
- Appropriate tone and formality for legal professional context
- Meets professional {jurisdiction} legal writing standards

**Developing (5-6)**:
- Basic professional writing with some clarity issues
- Generally acceptable grammar with some terminology problems
- Adequate organization but limited logical flow
- Mostly appropriate tone with some inconsistencies
- Meets minimum professional writing requirements

**Inadequate (0-4)**:
- Poor professional writing quality with significant clarity problems
- Grammar, syntax, or terminology issues affecting comprehension
- Poor organization and logical flow of ideas
- Inappropriate tone or formality for legal professional context
- Falls below {jurisdiction} professional writing standards

**{jurisdiction} Legal Writing Standards**:
{writing_standards_detail}

**Required Response Format**:
{{"score": X.X, "reasoning": "Detailed evaluation of professional writing quality, clarity, organization, and appropriateness within {jurisdiction} legal writing standards with specific examples"}}
"""
}

# Jurisdiction-specific context for opinion generation
JURISDICTION_SPECIFIC_CONTEXTS = {
    USJurisdiction.FEDERAL: {
        "legal_standards": "Federal legal standards requiring constitutional analysis and federal statutory interpretation",
        "research_sources": "Federal statutes, constitutional precedents, Supreme Court and circuit court decisions",
        "advocacy_framework": "Federal court advocacy emphasizing constitutional principles and federal jurisdiction",
        "argument_standards": "Federal arguments must demonstrate constitutional foundation and federal jurisdictional basis",
        "advocacy_style": "Formal federal court advocacy with emphasis on constitutional and federal law",
        "persuasion_standards": "Federal persuasive standards emphasizing legal precision and constitutional analysis",
        "communication_framework": "Formal federal legal communication appropriate for federal court practice",
        "persuasive_standards": "Federal courts expect sophisticated constitutional and statutory argument development",
        "research_standards": "Federal legal research emphasizing constitutional law and federal statutory interpretation",
        "authority_framework": "Federal research requires Supreme Court, circuit court, and federal statutory authorities",
        "source_requirements": "Federal sources including constitutional precedents and federal statutory interpretation",
        "research_standards_detail": "Federal research must emphasize constitutional law, federal statutes, and federal case law hierarchy",
        "client_standards": "Federal client advocacy within constitutional and federal jurisdictional framework",
        "strategic_framework": "Federal strategic advocacy considering constitutional implications and federal court procedures",
        "ethical_boundaries": "Federal ethical requirements under federal court rules and constitutional constraints",
        "client_advocacy_standards": "Federal client advocacy must balance strong representation with constitutional compliance",
        "writing_standards": "Formal federal legal writing appropriate for federal court practice and constitutional analysis",
        "professional_requirements": "Federal professional writing standards for federal court filings and constitutional analysis",
        "style_framework": "Formal federal legal writing style emphasizing precision and constitutional analysis",
        "writing_standards_detail": "Federal writing must meet federal court standards for constitutional and statutory analysis"
    },
    
    USJurisdiction.CALIFORNIA: {
        "legal_standards": "California legal standards emphasizing progressive legal development and practical application",
        "research_sources": "California statutes, California Constitution, California Supreme Court and appellate decisions",
        "advocacy_framework": "California advocacy balancing legal doctrine with practical policy considerations",
        "argument_standards": "California arguments should address both legal doctrine and underlying policy implications",
        "advocacy_style": "Professional California advocacy balancing formality with practical accessibility",
        "persuasion_standards": "California persuasive standards emphasizing both legal reasoning and practical impact",
        "communication_framework": "California legal communication balancing professional standards with practical clarity",
        "persuasive_standards": "California courts value arguments that address both legal doctrine and practical policy implications",
        "research_standards": "California legal research emphasizing state constitutional law and progressive legal development",
        "authority_framework": "California research requires state constitutional, statutory, and case law authorities",
        "source_requirements": "California sources including state constitutional analysis and progressive legal authorities",
        "research_standards_detail": "California research emphasizes state constitutional analysis and practical policy considerations",
        "client_standards": "California client advocacy considering progressive legal standards and practical outcomes",
        "strategic_framework": "California strategic advocacy considering practical policy implications and client objectives",
        "ethical_boundaries": "California ethical requirements under California Rules of Professional Conduct",
        "client_advocacy_standards": "California advocacy balances strong client representation with progressive legal standards",
        "writing_standards": "Professional California legal writing balancing formality with practical accessibility",
        "professional_requirements": "California professional writing standards emphasizing clarity and practical application",
        "style_framework": "California legal writing style balancing professional formality with practical communication",
        "writing_standards_detail": "California writing should be professionally clear and accessible while maintaining legal precision"
    },
    
    USJurisdiction.NEW_YORK: {
        "legal_standards": "New York legal standards emphasizing sophisticated legal analysis and commercial law expertise",
        "research_sources": "New York statutes, New York Constitution, Court of Appeals and Appellate Division decisions",
        "advocacy_framework": "New York advocacy emphasizing legal sophistication and commercial law precision",
        "argument_standards": "New York arguments must demonstrate sophisticated legal reasoning and commercial law understanding",
        "advocacy_style": "Sophisticated New York advocacy reflecting the state's legal prominence and commercial expertise",
        "persuasion_standards": "New York persuasive standards emphasizing legal sophistication and commercial law precision",
        "communication_framework": "Sophisticated New York legal communication appropriate for complex commercial matters",
        "persuasive_standards": "New York courts expect sophisticated argument development with commercial law expertise",
        "research_standards": "New York legal research emphasizing sophisticated legal analysis and commercial law",
        "authority_framework": "New York research requires sophisticated state and commercial law authorities",
        "source_requirements": "New York sources including sophisticated state law analysis and commercial precedents",
        "research_standards_detail": "New York research must demonstrate sophisticated legal analysis especially in commercial contexts",
        "client_standards": "New York client advocacy emphasizing sophisticated legal strategy and commercial considerations",
        "strategic_framework": "New York strategic advocacy considering sophisticated legal and commercial implications",
        "ethical_boundaries": "New York ethical requirements under New York Rules of Professional Conduct",
        "client_advocacy_standards": "New York advocacy must demonstrate sophisticated legal strategy and commercial law expertise",
        "writing_standards": "Sophisticated New York legal writing reflecting the state's legal prominence",
        "professional_requirements": "New York professional writing standards emphasizing legal sophistication",
        "style_framework": "Sophisticated New York legal writing style appropriate for complex legal matters",
        "writing_standards_detail": "New York writing must demonstrate sophisticated legal analysis and commercial law precision"
    },
    
    USJurisdiction.TEXAS: {
        "legal_standards": "Texas legal standards emphasizing practical application and business-friendly approaches",
        "research_sources": "Texas statutes, Texas Constitution, Texas Supreme Court and appellate court decisions",
        "advocacy_framework": "Texas advocacy emphasizing practical solutions and business-friendly legal approaches",
        "argument_standards": "Texas arguments should emphasize practical application and business considerations",
        "advocacy_style": "Professional Texas advocacy emphasizing clarity and practical business solutions",
        "persuasion_standards": "Texas persuasive standards emphasizing practical reasoning and business considerations",
        "communication_framework": "Texas legal communication emphasizing practical clarity and business applications",
        "persuasive_standards": "Texas courts value practical arguments that consider business implications and economic factors",
        "research_standards": "Texas legal research emphasizing practical application and business-friendly precedents",
        "authority_framework": "Texas research requires state law authorities with attention to business implications",
        "source_requirements": "Texas sources including business-friendly precedents and practical legal authorities",
        "research_standards_detail": "Texas research should emphasize practical legal solutions and business considerations",
        "client_standards": "Texas client advocacy emphasizing practical solutions and business-friendly outcomes",
        "strategic_framework": "Texas strategic advocacy considering practical business implications and economic factors",
        "ethical_boundaries": "Texas ethical requirements under Texas Disciplinary Rules of Professional Conduct",
        "client_advocacy_standards": "Texas advocacy emphasizes practical client solutions and business-friendly approaches",
        "writing_standards": "Professional Texas legal writing emphasizing clarity and practical application",
        "professional_requirements": "Texas professional writing standards emphasizing practical communication",
        "style_framework": "Texas legal writing style emphasizing straightforward practical reasoning",
        "writing_standards_detail": "Texas writing should be clear, practical, and focused on business-friendly solutions"
    },
    
    USJurisdiction.FLORIDA: {
        "legal_standards": "Florida legal standards balancing traditional and modern legal approaches",
        "research_sources": "Florida statutes, Florida Constitution, Florida Supreme Court and DCA decisions",
        "advocacy_framework": "Florida advocacy balancing traditional legal reasoning with modern practical applications",
        "argument_standards": "Florida arguments should balance traditional legal doctrine with practical considerations",
        "advocacy_style": "Professional Florida advocacy balancing formality with practical accessibility",
        "persuasion_standards": "Florida persuasive standards balancing traditional legal reasoning with practical impact",
        "communication_framework": "Florida legal communication balancing professional standards with practical clarity",
        "persuasive_standards": "Florida courts value balanced arguments addressing both legal tradition and practical application",
        "research_standards": "Florida legal research balancing traditional legal analysis with modern practical considerations",
        "authority_framework": "Florida research requires state authorities with attention to practical applications",
        "source_requirements": "Florida sources including balanced traditional and modern legal authorities",
        "research_standards_detail": "Florida research should balance traditional legal analysis with practical considerations",
        "client_standards": "Florida client advocacy balancing traditional legal approaches with practical client needs",
        "strategic_framework": "Florida strategic advocacy considering both traditional legal factors and practical outcomes",
        "ethical_boundaries": "Florida ethical requirements under Florida Rules of Professional Conduct",
        "client_advocacy_standards": "Florida advocacy balances traditional legal approaches with practical client service",
        "writing_standards": "Professional Florida legal writing balancing formality with practical clarity",
        "professional_requirements": "Florida professional writing standards emphasizing balanced formal and practical communication",
        "style_framework": "Florida legal writing style balancing traditional formality with modern accessibility",
        "writing_standards_detail": "Florida writing should balance professional formality with practical clarity and accessibility"
    },
    
    USJurisdiction.GENERAL: {
        "legal_standards": "General U.S. legal standards applicable across jurisdictions",
        "research_sources": "Generally applicable U.S. legal authorities and widely recognized precedents",
        "advocacy_framework": "Standard U.S. legal advocacy principles applicable across jurisdictions",
        "argument_standards": "Standard U.S. argument development principles applicable in general legal practice",
        "advocacy_style": "Professional legal advocacy appropriate for general U.S. legal practice",
        "persuasion_standards": "Standard U.S. persuasive communication principles for legal advocacy",
        "communication_framework": "Professional legal communication standards applicable across U.S. jurisdictions",
        "persuasive_standards": "Standard U.S. persuasive argument development appropriate for general legal practice",
        "research_standards": "General U.S. legal research standards applicable across jurisdictions",
        "authority_framework": "Standard U.S. legal research methodology using generally applicable authorities",
        "source_requirements": "Generally applicable U.S. legal sources and widely recognized authorities",
        "research_standards_detail": "Standard U.S. legal research methodology applicable across jurisdictions",
        "client_standards": "Standard U.S. client advocacy principles applicable in general legal practice",
        "strategic_framework": "Standard U.S. strategic advocacy considerations applicable across jurisdictions",
        "ethical_boundaries": "General U.S. legal ethical requirements under standard professional conduct rules",
        "client_advocacy_standards": "Standard U.S. client advocacy principles applicable in general legal practice",
        "writing_standards": "Professional legal writing standards applicable across U.S. jurisdictions",
        "professional_requirements": "Standard U.S. professional legal writing requirements",
        "style_framework": "Professional legal writing style appropriate for general U.S. legal practice",
        "writing_standards_detail": "Standard U.S. professional legal writing appropriate for general legal practice"
    }
}


def get_opinion_generation_prompt(prompt_type: OpinionGenerationPromptType,
                                jurisdiction: USJurisdiction,
                                response: str,
                                query: str,
                                jurisdiction_context: str) -> str:
    """
    Get a formatted opinion generation evaluation prompt.
    
    Args:
        prompt_type: Type of opinion generation evaluation
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


def get_all_opinion_generation_prompts(jurisdiction: USJurisdiction,
                                     response: str,
                                     query: str,
                                     jurisdiction_context: str) -> Dict[str, str]:
    """
    Get all opinion generation prompts for a jurisdiction.
    
    Args:
        jurisdiction: US jurisdiction for context
        response: Legal response to evaluate
        query: Original legal query
        jurisdiction_context: Additional jurisdiction context
        
    Returns:
        Dictionary mapping prompt types to formatted prompts
    """
    
    prompts = {}
    
    for prompt_type in OpinionGenerationPromptType:
        prompts[prompt_type.value] = get_opinion_generation_prompt(
            prompt_type, jurisdiction, response, query, jurisdiction_context
        )
    
    return prompts


def validate_opinion_generation_prompt(prompt_type: OpinionGenerationPromptType,
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
OPINION_GENERATION_PROMPT_INFO = {
    "template_count": len(BASE_PROMPTS),
    "jurisdiction_count": len(JURISDICTION_SPECIFIC_CONTEXTS),
    "supported_jurisdictions": list(JURISDICTION_SPECIFIC_CONTEXTS.keys()),
    "prompt_types": [pt.value for pt in OpinionGenerationPromptType],
    "description": "Professional opinion generation evaluation prompts with US jurisdiction advocacy awareness"
}


def get_prompt_info() -> Dict[str, any]:
    """Get information about opinion generation prompt templates"""
    return OPINION_GENERATION_PROMPT_INFO.copy()


# Example usage
if __name__ == "__main__":
    # Test prompt generation
    test_response = "We strongly recommend pursuing litigation. Our legal research shows..."
    test_query = "Provide legal opinion on pursuing breach of contract claim"
    test_jurisdiction = USJurisdiction.CALIFORNIA
    test_context = "California state court contract litigation opinion"
    
    # Get a specific prompt
    argument_strength_prompt = get_opinion_generation_prompt(
        OpinionGenerationPromptType.ARGUMENT_STRENGTH,
        test_jurisdiction,
        test_response,
        test_query,
        test_context
    )
    
    print("Sample Opinion Generation Prompt:")
    print("=" * 50)
    print(argument_strength_prompt[:500] + "...")
    
    # Show prompt info
    print(f"\nPrompt Template Info:")
    info = get_prompt_info()
    print(f"Template Count: {info['template_count']}")
    print(f"Supported Jurisdictions: {len(info['supported_jurisdictions'])}")
    print(f"Prompt Types: {info['prompt_types']}")
