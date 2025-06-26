"""
Jurisdiction Compliance Prompt Templates for Enhanced Legal AI System

This module provides sophisticated prompt templates for evaluating US jurisdiction
compliance in legal responses. These templates are used by the JurisdictionComplianceJudge
and serve as the critical gating component throughout the system.

CRITICAL FUNCTION: Jurisdiction compliance acts as a gating mechanism - responses
with poor jurisdiction compliance (score < 3.0) may be penalized or rejected entirely.

Key Evaluation Areas:
- Legal Framework Accuracy: Correct application of jurisdiction-specific laws
- Procedural Compliance: Understanding of jurisdiction court systems and procedures  
- Substantive Law Accuracy: Correct application of jurisdiction substantive law
- Constitutional Framework: Federal vs. state constitutional considerations
- Professional Standards: Jurisdiction-specific professional responsibility requirements

Author: Legal Reward System Team
Version: 1.0.0
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Import jurisdiction system
from ...jurisdiction.us_system import USJurisdiction
from ...core.enums import LegalTaskType


class JurisdictionCompliancePromptType(Enum):
    """Types of jurisdiction compliance evaluation prompts"""
    LEGAL_FRAMEWORK_ACCURACY = "legal_framework_accuracy"
    PROCEDURAL_COMPLIANCE = "procedural_compliance"
    SUBSTANTIVE_LAW_ACCURACY = "substantive_law_accuracy"
    CONSTITUTIONAL_FRAMEWORK = "constitutional_framework"
    PROFESSIONAL_STANDARDS = "professional_standards"


@dataclass
class JurisdictionComplianceContext:
    """Context for jurisdiction compliance evaluation"""
    query: str
    response: str
    jurisdiction: USJurisdiction
    task_type: LegalTaskType
    legal_domain: str
    complexity_level: str = "standard"  # standard, complex, expert
    federal_implications: bool = False
    interstate_considerations: bool = False


# Base prompt templates for jurisdiction compliance evaluation
BASE_PROMPTS = {
    JurisdictionCompliancePromptType.LEGAL_FRAMEWORK_ACCURACY: """
You are a distinguished {jurisdiction} legal expert conducting a CRITICAL GATING EVALUATION of legal framework accuracy. This evaluation determines whether the response meets {jurisdiction} jurisdictional requirements.

**CRITICAL NOTICE**: This is a gating evaluation. Significant jurisdictional errors will result in low scores that may disqualify the entire response.

**Jurisdiction**: {jurisdiction}
**Legal Framework**: {legal_framework}
**Task Type**: {task_type}
**Federal Implications**: {federal_implications}

**Original Legal Query**: {query}

**Legal Response to Evaluate**: 
{response}

**CRITICAL EVALUATION CRITERIA - Legal Framework Accuracy (Scale 0-10)**:

**EXCEPTIONAL (9-10) - Framework Mastery**:
- Perfect understanding and application of {jurisdiction} legal framework
- Accurate citation and application of {jurisdiction} statutes, regulations, and case law
- Sophisticated understanding of {jurisdiction} unique legal approaches and requirements
- Correct integration of federal law where applicable to {jurisdiction} practice
- Demonstrates expert-level knowledge of {jurisdiction} legal system structure
- **{jurisdiction} Framework Excellence**: {framework_excellence}

**PROFICIENT (7-8) - Framework Competence**:
- Good understanding of {jurisdiction} legal framework
- Generally accurate application of {jurisdiction} law and procedures
- Correct basic understanding of jurisdiction-specific requirements
- Appropriate integration of federal law considerations
- Solid grasp of {jurisdiction} legal system fundamentals

**DEVELOPING (5-6) - Framework Gaps**:
- Basic understanding of {jurisdiction} framework but with notable gaps
- Some inaccuracies in jurisdiction-specific law application
- Limited understanding of {jurisdiction} unique requirements
- Some confusion about federal vs. state law integration
- Incomplete grasp of {jurisdiction} legal system structure

**INADEQUATE (0-4) - CRITICAL FRAMEWORK FAILURES**:
- Significant misunderstanding of {jurisdiction} legal framework
- Major inaccuracies in jurisdiction-specific law application
- Dangerous misinformation about {jurisdiction} legal requirements
- Serious confusion between different jurisdictional frameworks
- **GATING FAILURE**: Response unreliable for {jurisdiction} legal use

**{jurisdiction} Legal Framework Requirements**:
{framework_requirements}

**Gating Threshold**: Scores below 3.0 indicate critical framework failures that disqualify the response.

**Required Response Format**:
{{"score": X.X, "reasoning": "Critical assessment of {jurisdiction} legal framework accuracy, noting any gating failures or excellent compliance"}}
""",

    JurisdictionCompliancePromptType.PROCEDURAL_COMPLIANCE: """
You are a distinguished {jurisdiction} procedural law expert conducting a CRITICAL GATING EVALUATION of procedural compliance and court system understanding.

**CRITICAL NOTICE**: This is a gating evaluation focusing on {jurisdiction} procedural accuracy and court system competence.

**Jurisdiction**: {jurisdiction}
**Court System**: {court_system}
**Procedural Framework**: {procedural_framework}
**Task Type**: {task_type}

**Original Legal Query**: {query}

**Legal Response to Evaluate**: 
{response}

**CRITICAL EVALUATION CRITERIA - Procedural Compliance (Scale 0-10)**:

**EXCEPTIONAL (9-10) - Procedural Mastery**:
- Perfect understanding of {jurisdiction} court system hierarchy and procedures
- Accurate knowledge of {jurisdiction} civil and criminal procedure rules
- Sophisticated understanding of jurisdiction-specific procedural requirements
- Correct application of {jurisdiction} filing deadlines, rules, and procedures
- Expert knowledge of {jurisdiction} appellate procedures and jurisdiction
- **{jurisdiction} Procedural Excellence**: {procedural_excellence}

**PROFICIENT (7-8) - Procedural Competence**:
- Good understanding of {jurisdiction} court system and basic procedures
- Generally accurate knowledge of major procedural requirements
- Correct understanding of basic {jurisdiction} procedural rules
- Appropriate awareness of procedural deadlines and requirements
- Solid grasp of court system hierarchy

**DEVELOPING (5-6) - Procedural Gaps**:
- Basic understanding of {jurisdiction} procedures but with gaps
- Some inaccuracies in procedural rule application
- Limited knowledge of jurisdiction-specific procedural requirements
- Some confusion about court system hierarchy or procedures
- Incomplete understanding of procedural deadlines

**INADEQUATE (0-4) - CRITICAL PROCEDURAL FAILURES**:
- Significant misunderstanding of {jurisdiction} court system
- Major inaccuracies in procedural rule application
- Dangerous misinformation about procedural requirements
- Serious errors in court system hierarchy or jurisdiction
- **GATING FAILURE**: Procedural errors that could harm legal outcomes

**{jurisdiction} Procedural Requirements**:
{procedural_requirements}

**Gating Threshold**: Scores below 3.0 indicate critical procedural failures.

**Required Response Format**:
{{"score": X.X, "reasoning": "Critical assessment of {jurisdiction} procedural compliance and court system understanding"}}
""",

    JurisdictionCompliancePromptType.SUBSTANTIVE_LAW_ACCURACY: """
You are a distinguished {jurisdiction} substantive law expert conducting a CRITICAL GATING EVALUATION of substantive law accuracy and application.

**CRITICAL NOTICE**: This is a gating evaluation focusing on {jurisdiction} substantive law correctness and proper application.

**Jurisdiction**: {jurisdiction}
**Legal Domain**: {legal_domain}
**Substantive Framework**: {substantive_framework}
**Complexity Level**: {complexity_level}

**Original Legal Query**: {query}

**Legal Response to Evaluate**: 
{response}

**CRITICAL EVALUATION CRITERIA - Substantive Law Accuracy (Scale 0-10)**:

**EXCEPTIONAL (9-10) - Substantive Mastery**:
- Perfect accuracy in {jurisdiction} substantive law application
- Sophisticated understanding of {jurisdiction} statutory interpretation and case law
- Expert knowledge of {jurisdiction} unique substantive legal doctrines
- Correct application of {jurisdiction} elements, standards, and tests
- Demonstrates mastery of {jurisdiction} substantive law developments
- **{jurisdiction} Substantive Excellence**: {substantive_excellence}

**PROFICIENT (7-8) - Substantive Competence**:
- Good accuracy in {jurisdiction} substantive law application
- Generally correct understanding of major legal doctrines
- Appropriate application of {jurisdiction} legal standards
- Correct understanding of basic elements and requirements
- Solid grasp of {jurisdiction} substantive law fundamentals

**DEVELOPING (5-6) - Substantive Gaps**:
- Basic accuracy in substantive law but with notable gaps
- Some errors in legal doctrine application
- Limited understanding of jurisdiction-specific substantive requirements
- Some confusion about legal elements or standards
- Incomplete grasp of {jurisdiction} substantive law principles

**INADEQUATE (0-4) - CRITICAL SUBSTANTIVE FAILURES**:
- Significant errors in {jurisdiction} substantive law application
- Major misunderstanding of legal doctrines and requirements
- Dangerous misinformation about substantive legal standards
- Serious errors in element analysis or legal tests
- **GATING FAILURE**: Substantive errors that could mislead or harm

**{jurisdiction} Substantive Law Requirements**:
{substantive_requirements}

**Gating Threshold**: Scores below 3.0 indicate critical substantive law failures.

**Required Response Format**:
{{"score": X.X, "reasoning": "Critical assessment of {jurisdiction} substantive law accuracy and proper application"}}
""",

    JurisdictionCompliancePromptType.CONSTITUTIONAL_FRAMEWORK: """
You are a distinguished constitutional law expert evaluating constitutional framework understanding for {jurisdiction} legal response compliance.

**CRITICAL NOTICE**: This is a gating evaluation focusing on constitutional framework accuracy and federal/state law integration.

**Jurisdiction**: {jurisdiction}
**Constitutional Framework**: {constitutional_framework}
**Federal/State Integration**: {federal_state_framework}
**Constitutional Implications**: {constitutional_implications}

**Original Legal Query**: {query}

**Legal Response to Evaluate**: 
{response}

**CRITICAL EVALUATION CRITERIA - Constitutional Framework (Scale 0-10)**:

**EXCEPTIONAL (9-10) - Constitutional Mastery**:
- Perfect understanding of {jurisdiction} constitutional framework
- Sophisticated federal and state constitutional integration
- Expert knowledge of constitutional supremacy and preemption principles
- Correct application of constitutional analysis methodology
- Demonstrates mastery of {jurisdiction} constitutional interpretation
- **{jurisdiction} Constitutional Excellence**: {constitutional_excellence}

**PROFICIENT (7-8) - Constitutional Competence**:
- Good understanding of {jurisdiction} constitutional framework
- Generally correct federal/state constitutional integration
- Appropriate understanding of constitutional principles
- Correct basic constitutional analysis approach
- Solid grasp of constitutional interpretation principles

**DEVELOPING (5-6) - Constitutional Gaps**:
- Basic constitutional understanding but with gaps
- Some confusion in federal/state constitutional integration
- Limited understanding of constitutional analysis methodology
- Some errors in constitutional principle application
- Incomplete constitutional framework understanding

**INADEQUATE (0-4) - CRITICAL CONSTITUTIONAL FAILURES**:
- Significant constitutional framework misunderstanding
- Major errors in federal/state constitutional integration
- Dangerous constitutional analysis errors
- Serious misapplication of constitutional principles
- **GATING FAILURE**: Constitutional errors that could undermine legal analysis

**{jurisdiction} Constitutional Requirements**:
{constitutional_requirements}

**Gating Threshold**: Scores below 3.0 indicate critical constitutional failures.

**Required Response Format**:
{{"score": X.X, "reasoning": "Critical assessment of constitutional framework understanding and federal/state integration"}}
""",

    JurisdictionCompliancePromptType.PROFESSIONAL_STANDARDS: """
You are a distinguished {jurisdiction} professional responsibility expert conducting a CRITICAL GATING EVALUATION of professional standards compliance.

**CRITICAL NOTICE**: This is a gating evaluation focusing on {jurisdiction} professional responsibility and ethical requirements.

**Jurisdiction**: {jurisdiction}
**Professional Framework**: {professional_framework}
**Ethical Standards**: {ethical_standards}
**Professional Context**: {professional_context}

**Original Legal Query**: {query}

**Legal Response to Evaluate**: 
{response}

**CRITICAL EVALUATION CRITERIA - Professional Standards (Scale 0-10)**:

**EXCEPTIONAL (9-10) - Professional Excellence**:
- Perfect compliance with {jurisdiction} professional responsibility standards
- Sophisticated understanding of {jurisdiction} ethical requirements
- Expert application of professional conduct rules
- Correct understanding of {jurisdiction} attorney regulation
- Demonstrates mastery of professional responsibility principles
- **{jurisdiction} Professional Excellence**: {professional_excellence}

**PROFICIENT (7-8) - Professional Competence**:
- Good compliance with {jurisdiction} professional standards
- Generally correct understanding of ethical requirements
- Appropriate application of professional conduct principles
- Correct basic understanding of attorney regulation
- Solid grasp of professional responsibility fundamentals

**DEVELOPING (5-6) - Professional Gaps**:
- Basic professional standards understanding but with gaps
- Some confusion about ethical requirements
- Limited understanding of professional conduct applications
- Some errors in professional responsibility principles
- Incomplete professional standards compliance

**INADEQUATE (0-4) - CRITICAL PROFESSIONAL FAILURES**:
- Significant professional standards violations
- Major misunderstanding of ethical requirements
- Dangerous professional responsibility errors
- Serious violations of attorney conduct principles
- **GATING FAILURE**: Professional errors that could harm legal practice

**{jurisdiction} Professional Requirements**:
{professional_requirements}

**Gating Threshold**: Scores below 3.0 indicate critical professional standards failures.

**Required Response Format**:
{{"score": X.X, "reasoning": "Critical assessment of {jurisdiction} professional standards compliance and ethical requirements"}}
"""
}


# Comprehensive jurisdiction-specific contexts for compliance evaluation
JURISDICTION_SPECIFIC_COMPLIANCE_CONTEXTS = {
    USJurisdiction.FEDERAL: {
        "legal_framework": "U.S. Constitution, federal statutes, federal regulations, Supreme Court and circuit court precedents",
        "framework_excellence": "Federal framework mastery requires constitutional supremacy understanding and federal court expertise",
        "framework_requirements": "Federal compliance requires constitutional accuracy, federal statute interpretation, and federal court procedural knowledge",
        "court_system": "Federal district courts, circuit courts of appeals, Supreme Court, specialized federal courts",
        "procedural_framework": "Federal Rules of Civil Procedure, Federal Rules of Criminal Procedure, Federal Rules of Evidence",
        "procedural_excellence": "Federal procedural mastery requires expertise in federal court rules and federal jurisdiction",
        "procedural_requirements": "Federal procedural compliance requires federal court rule accuracy and jurisdictional understanding",
        "substantive_framework": "Federal constitutional law, federal statutory interpretation, federal administrative law",
        "substantive_excellence": "Federal substantive mastery requires constitutional analysis and federal statutory interpretation expertise",
        "substantive_requirements": "Federal substantive compliance requires constitutional accuracy and federal law interpretation",
        "constitutional_framework": "U.S. Constitution supremacy, federal preemption, constitutional interpretation methodology",
        "federal_state_framework": "Federal supremacy principles, preemption analysis, constitutional constraints on states",
        "constitutional_implications": "Federal constitutional analysis with supremacy clause and preemption considerations",
        "constitutional_excellence": "Federal constitutional mastery requires supremacy understanding and preemption analysis",
        "constitutional_requirements": "Federal constitutional compliance requires supremacy accuracy and preemption understanding",
        "professional_framework": "Federal court admission requirements, federal ethics rules, federal professional responsibility",
        "ethical_standards": "Federal court professional responsibility rules and federal attorney conduct standards",
        "professional_context": "Federal court practice with federal ethics requirements and constitutional constraints",
        "professional_excellence": "Federal professional mastery requires federal court ethics and constitutional compliance",
        "professional_requirements": "Federal professional compliance requires federal court ethics and constitutional understanding"
    },
    
    USJurisdiction.CALIFORNIA: {
        "legal_framework": "California Constitution, California codes, California case law, relevant federal law",
        "framework_excellence": "California framework mastery requires understanding of progressive legal development and consumer protection focus",
        "framework_requirements": "California compliance requires state constitutional understanding and progressive legal approach recognition",
        "court_system": "California Supreme Court, California Courts of Appeal, Superior Courts, specialized California courts",
        "procedural_framework": "California Code of Civil Procedure, California Rules of Court, California Evidence Code",
        "procedural_excellence": "California procedural mastery requires state-specific procedure understanding and consumer protection integration",
        "procedural_requirements": "California procedural compliance requires state procedure accuracy and consumer protection awareness",
        "substantive_framework": "California constitutional law, California statutory interpretation, California consumer protection law",
        "substantive_excellence": "California substantive mastery requires progressive legal understanding and consumer protection expertise",
        "substantive_requirements": "California substantive compliance requires progressive legal approach and consumer protection accuracy",
        "constitutional_framework": "California Constitution broader protections, federal/state constitutional integration",
        "federal_state_framework": "California state law broader protections, federal constitutional compliance",
        "constitutional_implications": "California constitutional analysis with broader protections and federal integration",
        "constitutional_excellence": "California constitutional mastery requires broader protection understanding and federal integration",
        "constitutional_requirements": "California constitutional compliance requires broader protection accuracy and federal understanding",
        "professional_framework": "California State Bar rules, California professional responsibility, California attorney regulation",
        "ethical_standards": "California State Bar ethics rules and California professional conduct standards",
        "professional_context": "California legal practice with state bar requirements and consumer protection focus",
        "professional_excellence": "California professional mastery requires state bar expertise and consumer protection understanding",
        "professional_requirements": "California professional compliance requires state bar accuracy and consumer protection awareness"
    },
    
    USJurisdiction.NEW_YORK: {
        "legal_framework": "New York Constitution, New York statutes, New York case law, relevant federal law",
        "framework_excellence": "New York framework mastery requires commercial law sophistication and procedural complexity understanding",
        "framework_requirements": "New York compliance requires sophisticated commercial law understanding and complex procedural knowledge",
        "court_system": "New York Court of Appeals, Appellate Divisions, Supreme Court, specialized New York courts",
        "procedural_framework": "New York Civil Practice Law and Rules (CPLR), New York Rules of Professional Conduct",
        "procedural_excellence": "New York procedural mastery requires CPLR expertise and commercial litigation sophistication",
        "procedural_requirements": "New York procedural compliance requires CPLR accuracy and commercial procedure understanding",
        "substantive_framework": "New York constitutional law, New York commercial law, New York statutory interpretation",
        "substantive_excellence": "New York substantive mastery requires commercial law expertise and sophisticated legal analysis",
        "substantive_requirements": "New York substantive compliance requires commercial law accuracy and sophisticated analysis",
        "constitutional_framework": "New York Constitution, commercial law constitutional implications, federal integration",
        "federal_state_framework": "New York state law with federal commercial law integration",
        "constitutional_implications": "New York constitutional analysis with commercial law and federal considerations",
        "constitutional_excellence": "New York constitutional mastery requires commercial implications and federal integration",
        "constitutional_requirements": "New York constitutional compliance requires commercial accuracy and federal understanding",
        "professional_framework": "New York State Bar rules, New York Rules of Professional Conduct, commercial practice standards",
        "ethical_standards": "New York State Bar ethics rules and commercial practice professional standards",
        "professional_context": "New York legal practice with commercial sophistication and complex professional requirements",
        "professional_excellence": "New York professional mastery requires commercial practice expertise and sophisticated ethics",
        "professional_requirements": "New York professional compliance requires commercial practice accuracy and complex ethics understanding"
    },
    
    USJurisdiction.TEXAS: {
        "legal_framework": "Texas Constitution, Texas codes, Texas case law, relevant federal law",
        "framework_excellence": "Texas framework mastery requires business-friendly approach understanding and property rights focus",
        "framework_requirements": "Texas compliance requires business-friendly legal understanding and property rights accuracy",
        "court_system": "Texas Supreme Court, Texas Court of Criminal Appeals, Texas Courts of Appeals, District Courts",
        "procedural_framework": "Texas Rules of Civil Procedure, Texas Rules of Criminal Procedure, Texas Rules of Evidence",
        "procedural_excellence": "Texas procedural mastery requires dual supreme court understanding and business-friendly procedures",
        "procedural_requirements": "Texas procedural compliance requires dual court system accuracy and business procedure understanding",
        "substantive_framework": "Texas constitutional law, Texas business law, Texas property law",
        "substantive_excellence": "Texas substantive mastery requires business law expertise and property rights understanding",
        "substantive_requirements": "Texas substantive compliance requires business law accuracy and property rights understanding",
        "constitutional_framework": "Texas Constitution, state sovereignty principles, federal/state integration",
        "federal_state_framework": "Texas state rights emphasis with federal constitutional compliance",
        "constitutional_implications": "Texas constitutional analysis with state sovereignty and federal integration",
        "constitutional_excellence": "Texas constitutional mastery requires state sovereignty understanding and federal integration",
        "constitutional_requirements": "Texas constitutional compliance requires state sovereignty accuracy and federal understanding",
        "professional_framework": "Texas State Bar rules, Texas professional responsibility, business practice standards",
        "ethical_standards": "Texas State Bar ethics rules and business practice professional standards",
        "professional_context": "Texas legal practice with business focus and practical professional requirements",
        "professional_excellence": "Texas professional mastery requires business practice expertise and practical ethics",
        "professional_requirements": "Texas professional compliance requires business practice accuracy and practical ethics understanding"
    },
    
    USJurisdiction.FLORIDA: {
        "legal_framework": "Florida Constitution, Florida statutes, Florida case law, relevant federal law",
        "framework_excellence": "Florida framework mastery requires balanced traditional-modern approach and unique procedural understanding",
        "framework_requirements": "Florida compliance requires balanced legal approach understanding and unique procedure accuracy",
        "court_system": "Florida Supreme Court, District Courts of Appeal, Circuit Courts, County Courts",
        "procedural_framework": "Florida Rules of Civil Procedure, Florida Rules of Criminal Procedure, Florida Evidence Code",
        "procedural_excellence": "Florida procedural mastery requires traditional-modern balance and unique state procedures",
        "procedural_requirements": "Florida procedural compliance requires balanced approach accuracy and unique procedure understanding",
        "substantive_framework": "Florida constitutional law, Florida statutory interpretation, traditional-modern legal integration",
        "substantive_excellence": "Florida substantive mastery requires traditional-modern balance and practical legal application",
        "substantive_requirements": "Florida substantive compliance requires balanced approach accuracy and practical application",
        "constitutional_framework": "Florida Constitution, traditional-modern constitutional interpretation, federal integration",
        "federal_state_framework": "Florida state law traditional-modern balance with federal constitutional compliance",
        "constitutional_implications": "Florida constitutional analysis with traditional-modern balance and federal considerations",
        "constitutional_excellence": "Florida constitutional mastery requires traditional-modern understanding and federal integration",
        "constitutional_requirements": "Florida constitutional compliance requires balanced approach accuracy and federal understanding",
        "professional_framework": "Florida Bar rules, Florida professional responsibility, traditional-modern practice standards",
        "ethical_standards": "Florida Bar ethics rules and traditional-modern professional standards",
        "professional_context": "Florida legal practice with traditional-modern balance and practical requirements",
        "professional_excellence": "Florida professional mastery requires traditional-modern expertise and practical ethics",
        "professional_requirements": "Florida professional compliance requires balanced approach accuracy and practical ethics"
    },
    
    USJurisdiction.GENERAL: {
        "legal_framework": "General U.S. legal principles, common law traditions, widely applicable precedents",
        "framework_excellence": "General framework mastery requires broad U.S. legal principle understanding",
        "framework_requirements": "General compliance requires broadly applicable legal principle accuracy",
        "court_system": "General U.S. court system hierarchy, federal and state court integration",
        "procedural_framework": "General U.S. procedural principles, common procedural approaches",
        "procedural_excellence": "General procedural mastery requires broad procedural principle understanding",
        "procedural_requirements": "General procedural compliance requires broadly applicable procedure accuracy",
        "substantive_framework": "General U.S. substantive law principles, common law doctrines",
        "substantive_excellence": "General substantive mastery requires broad substantive principle understanding",
        "substantive_requirements": "General substantive compliance requires broadly applicable substantive accuracy",
        "constitutional_framework": "U.S. Constitution, general constitutional principles, federal/state integration",
        "federal_state_framework": "General federal/state law integration principles",
        "constitutional_implications": "General constitutional analysis applicable across jurisdictions",
        "constitutional_excellence": "General constitutional mastery requires broad constitutional principle understanding",
        "constitutional_requirements": "General constitutional compliance requires broadly applicable constitutional accuracy",
        "professional_framework": "General U.S. professional responsibility principles, widely applicable ethics",
        "ethical_standards": "General U.S. professional responsibility standards",
        "professional_context": "General U.S. legal practice with broadly applicable professional requirements",
        "professional_excellence": "General professional mastery requires broad professional principle understanding",
        "professional_requirements": "General professional compliance requires broadly applicable professional accuracy"
    }
}


def get_jurisdiction_compliance_prompt(prompt_type: JurisdictionCompliancePromptType,
                                     context: JurisdictionComplianceContext) -> str:
    """
    Get a formatted jurisdiction compliance evaluation prompt.
    
    Args:
        prompt_type: Type of jurisdiction compliance evaluation
        context: Evaluation context including query, response, jurisdiction
        
    Returns:
        Formatted evaluation prompt for critical gating assessment
    """
    
    # Get base prompt template
    base_template = BASE_PROMPTS[prompt_type]
    
    # Get jurisdiction-specific context
    jurisdiction_context_dict = JURISDICTION_SPECIFIC_COMPLIANCE_CONTEXTS.get(
        context.jurisdiction, 
        JURISDICTION_SPECIFIC_COMPLIANCE_CONTEXTS[USJurisdiction.GENERAL]
    )
    
    # Format the prompt with all context
    formatted_prompt = base_template.format(
        query=context.query,
        response=context.response,
        jurisdiction=context.jurisdiction.value,
        task_type=context.task_type.value,
        legal_domain=context.legal_domain,
        complexity_level=context.complexity_level,
        federal_implications=context.federal_implications,
        **jurisdiction_context_dict
    )
    
    return formatted_prompt


def get_all_jurisdiction_compliance_prompts(context: JurisdictionComplianceContext) -> Dict[str, str]:
    """
    Get all jurisdiction compliance evaluation prompts.
    
    Args:
        context: Evaluation context
        
    Returns:
        Dictionary mapping prompt types to formatted prompts
    """
    
    prompts = {}
    
    for prompt_type in JurisdictionCompliancePromptType:
        prompts[prompt_type.value] = get_jurisdiction_compliance_prompt(prompt_type, context)
    
    return prompts


def validate_jurisdiction_compliance_prompt(prompt_type: JurisdictionCompliancePromptType,
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
            jurisdiction in JURISDICTION_SPECIFIC_COMPLIANCE_CONTEXTS)


def get_gating_threshold() -> float:
    """Get the gating threshold for jurisdiction compliance"""
    return 3.0  # Scores below 3.0 indicate critical failures


def assess_gating_failure(score: float) -> Tuple[bool, str]:
    """
    Assess if a jurisdiction compliance score represents a gating failure.
    
    Args:
        score: Jurisdiction compliance score (0-10)
        
    Returns:
        Tuple of (is_gating_failure, failure_message)
    """
    
    threshold = get_gating_threshold()
    
    if score < threshold:
        return True, f"GATING FAILURE: Jurisdiction compliance score {score} below threshold {threshold}"
    
    return False, ""


# Enhanced jurisdiction compliance management
class JurisdictionComplianceManager:
    """Manager for jurisdiction compliance evaluation with gating logic"""
    
    def __init__(self):
        self.gating_threshold = get_gating_threshold()
        self.failure_history = []
        
    def evaluate_gating_compliance(self, scores: Dict[str, float], 
                                 jurisdiction: USJurisdiction) -> Dict[str, any]:
        """
        Evaluate overall jurisdiction compliance with gating logic.
        
        Args:
            scores: Dictionary of component scores
            jurisdiction: Jurisdiction being evaluated
            
        Returns:
            Compliance evaluation with gating assessment
        """
        
        # Calculate weighted average (equal weights)
        total_score = sum(scores.values()) / len(scores) if scores else 0.0
        
        # Assess gating failure
        is_gating_failure, failure_message = assess_gating_failure(total_score)
        
        # Track failure history
        if is_gating_failure:
            self.failure_history.append({
                "jurisdiction": jurisdiction,
                "score": total_score,
                "component_scores": scores.copy(),
                "failure_message": failure_message
            })
        
        return {
            "overall_score": total_score,
            "component_scores": scores,
            "is_gating_failure": is_gating_failure,
            "failure_message": failure_message,
            "jurisdiction": jurisdiction,
            "threshold": self.gating_threshold,
            "compliance_level": self._get_compliance_level(total_score)
        }
    
    def _get_compliance_level(self, score: float) -> str:
        """Get compliance level description"""
        if score >= 9.0:
            return "exceptional"
        elif score >= 7.0:
            return "proficient" 
        elif score >= 5.0:
            return "developing"
        elif score >= self.gating_threshold:
            return "minimal"
        else:
            return "gating_failure"
    
    def get_failure_statistics(self) -> Dict[str, any]:
        """Get statistics on gating failures"""
        if not self.failure_history:
            return {"total_failures": 0, "failure_rate": 0.0}
        
        jurisdiction_failures = {}
        for failure in self.failure_history:
            jurisdiction = failure["jurisdiction"]
            if jurisdiction not in jurisdiction_failures:
                jurisdiction_failures[jurisdiction] = 0
            jurisdiction_failures[jurisdiction] += 1
        
        return {
            "total_failures": len(self.failure_history),
            "jurisdiction_failures": jurisdiction_failures,
            "average_failure_score": sum(f["score"] for f in self.failure_history) / len(self.failure_history),
            "most_problematic_jurisdiction": max(jurisdiction_failures.items(), key=lambda x: x[1])[0] if jurisdiction_failures else None
        }


# Prompt template metadata
JURISDICTION_COMPLIANCE_PROMPT_INFO = {
    "template_count": len(BASE_PROMPTS),
    "jurisdiction_count": len(JURISDICTION_SPECIFIC_COMPLIANCE_CONTEXTS),
    "supported_jurisdictions": list(JURISDICTION_SPECIFIC_COMPLIANCE_CONTEXTS.keys()),
    "prompt_types": [pt.value for pt in JurisdictionCompliancePromptType],
    "description": "Critical gating evaluation prompts for US jurisdiction compliance",
    "gating_threshold": get_gating_threshold(),
    "critical_function": "GATING - scores below threshold may disqualify responses",
    "evaluation_components": [
        "legal_framework_accuracy",
        "procedural_compliance", 
        "substantive_law_accuracy",
        "constitutional_framework",
        "professional_standards"
    ]
}


def get_prompt_info() -> Dict[str, any]:
    """Get information about jurisdiction compliance prompt templates"""
    return JURISDICTION_COMPLIANCE_PROMPT_INFO.copy()


# Example usage and testing
if __name__ == "__main__":
    # Test context creation
    test_context = JurisdictionComplianceContext(
        query="What are the requirements for filing a breach of contract claim?",
        response="To file a breach of contract claim, you must establish...",
        jurisdiction=USJurisdiction.CALIFORNIA,
        task_type=LegalTaskType.JUDICIAL_REASONING,
        legal_domain="Contract Law",
        complexity_level="standard",
        federal_implications=False
    )
    
    # Get a specific prompt (legal framework accuracy)
    framework_prompt = get_jurisdiction_compliance_prompt(
        JurisdictionCompliancePromptType.LEGAL_FRAMEWORK_ACCURACY,
        test_context
    )
    
    print("Sample Jurisdiction Compliance Prompt (GATING):")
    print("=" * 70)
    print(framework_prompt[:500] + "...")
    
    # Test gating assessment
    test_score = 2.5  # Below threshold
    is_failure, message = assess_gating_failure(test_score)
    
    print(f"\nGating Assessment Test:")
    print(f"Score: {test_score}")
    print(f"Is Gating Failure: {is_failure}")
    print(f"Message: {message}")
    
    # Show prompt info
    print(f"\nJurisdiction Compliance Prompt Info:")
    info = get_prompt_info()
    print(f"Template Count: {info['template_count']}")
    print(f"Gating Threshold: {info['gating_threshold']}")
    print(f"Critical Function: {info['critical_function']}")
    print(f"Evaluation Components: {len(info['evaluation_components'])}")
