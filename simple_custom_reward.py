from typing import Optional, Dict
import os
from openai import OpenAI


def make_llm_request(prompt: str, model: str = "gpt-4.1-mini", temperature: float = 0.1) -> str:
    """
    Make a request to OpenAI using the new Responses API.
    
    Args:
        prompt (str): The prompt to send to the LLM
        model (str): The model to use (default: "gpt-4.1-mini")
        temperature (float): Sampling temperature (default: 0.1 for consistency)
    
    Returns:
        str: The LLM's response text
    
    Raises:
        Exception: If the API request fails
    """
    try:
        # Initialize the OpenAI client
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Make request using the new Responses API
        response = client.responses.create(
            model=model,
            input=prompt,
            temperature=temperature
        )
        
        # Extract and return the text output
        return response.output_text
        
    except Exception as e:
        raise Exception(f"LLM request failed: {str(e)}")


def extract_score_from_response(response_text: str) -> float:
    """
    Extract numerical score from LLM response text.
    
    Args:
        response_text (str): The full LLM response
        
    Returns:
        float: Extracted score between 0.01 and 1.0 (never 0)
    """
    import re
    
    # Debug: Print the actual response
    print(f"DEBUG - Raw LLM Response: '{response_text.strip()}'")
    
    # Clean the response text
    cleaned_text = response_text.strip().lower()
    
    # Try to extract a decimal number between 0 and 1
    patterns = [
        r'\b(1\.0+)\b',                    # Exactly 1.0
        r'\b(0\.[0-9]+)\b',                # Decimal between 0 and 1
        r'\b([01])\b',                     # Just 0 or 1
        r'score.*?([0-9]*\.?[0-9]+)',      # "score: 0.7" pattern
        r'([0-9]*\.?[0-9]+).*score',       # "0.7 score" pattern
        r'([0-9]*\.?[0-9]+)',              # Any number
    ]
    
    extracted_score = None
    
    for pattern in patterns:
        matches = re.findall(pattern, cleaned_text)
        if matches:
            try:
                # Try each match to find a valid score
                for match in matches:
                    potential_score = float(match)
                    if 0.0 <= potential_score <= 1.0:
                        extracted_score = potential_score
                        print(f"DEBUG - Extracted score: {extracted_score}")
                        break
                if extracted_score is not None:
                    break
            except ValueError:
                continue
    
    # If no score found, return minimum penalty score
    if extracted_score is None:
        print(f"WARNING: Could not extract valid score from response: '{response_text.strip()}'")
        return 0.0
    
    # Ensure we never return exactly 0 (reserve 0 for system failures)
    if extracted_score == 0.0:
        print("DEBUG - Converting 0.0 to minimum penalty score 0.01")
        return 0.01  # Minimum penalty for actual predictions
    
    return max(0.01, min(1.0, extracted_score))  # Ensure between 0.01 and 1.0


def evaluate_response(
    data_source: str, 
    solution_str: str, 
    ground_truth: str, 
    extra_info: Optional[Dict] = None
) -> float:
    """
    Main interface function for the reward system.
    
    Evaluates the similarity/relatedness between solution_str and ground_truth
    using task-specific evaluation functions.
    
    Args:
        data_source (str): Source of the data
        solution_str (str): The generated solution/response to evaluate
        ground_truth (str): The expected/reference answer
        extra_info (Optional[Dict]): Additional information, may contain 'task_type'
    
    Returns:
        float: Score between 0 and 1 (0 = lowest, 1 = highest)
    """
    # Extract task_type from extra_info if available
    task_type = None
    if extra_info and isinstance(extra_info, dict):
        task_type = extra_info.get('task_type')
    
    # Route to appropriate evaluation function based on task_type
    if task_type == 'general_chat':
        return evaluate_general_chat(solution_str, ground_truth)
    elif task_type == 'precedent_analysis':
        return evaluate_precedent_analysis(solution_str, ground_truth)
    elif task_type == 'opinion_generation':
        return evaluate_opinion_generation(solution_str, ground_truth)
    elif task_type == 'judicial_reasoning':
        return evaluate_judicial_reasoning(solution_str, ground_truth)
    else:
        # Default to miscellaneous evaluation for unknown/missing task_types
        return evaluate_miscellaneous(solution_str, ground_truth)


def evaluate_general_chat(solution_str: str, ground_truth: str) -> float:
    """
    Evaluate legal responses to common legal questions for training assessment.
    
    Args:
        solution_str (str): The AI agent's predicted legal response
        ground_truth (str): The human expert's legal response
    
    Returns:
        float: Score between 0.0 and 1.0
    """
    prompt = f"""
You are a senior legal expert evaluating an AI legal assistant's training performance. Focus on SEMANTIC SIMILARITY and PRACTICAL EQUIVALENCE, not exact wording.

PREDICTED RESPONSE (AI Agent): "{solution_str}"
GROUND TRUTH (Human Expert): "{ground_truth}"

**SCORING PHILOSOPHY**: 
Responses with the same core meaning should score 0.7-0.9 even if worded differently. Reserve very low scores (0.01-0.10) only for fundamentally wrong or dangerous legal advice.

**EVALUATION CRITERIA:**

**LEGAL ACCURACY & CORRECTNESS (40%)**
- Core legal information accuracy (same legal facts = high score)
- Absence of dangerous misinformation
- Practical legal equivalence

**SEMANTIC SIMILARITY (30%)**
- Same meaning expressed differently = HIGH SCORE (0.7-0.9)
- Similar intent and core message preservation  
- Functional equivalence for legal purposes

**LEGAL REASONING & METHODOLOGY (20%)**
- Sound legal approach and analysis
- Appropriate legal principles application

**PROFESSIONAL STANDARDS (10%)**
- Professional legal tone and clarity
- Ethical considerations

**DETAILED SCORING GUIDE:**
- **0.85-1.0**: Same legal meaning/advice (even if different words)
- **0.70-0.84**: Similar legal advice with minor variations
- **0.50-0.69**: Adequate but notable differences in approach/completeness
- **0.30-0.49**: Significant legal deficiencies but not dangerous
- **0.10-0.29**: Poor legal advice with major errors
- **0.01-0.09**: Dangerous/fundamentally wrong legal information only

CRITICAL: Return ONLY a decimal number between 0.01 and 1.0. Focus on semantic equivalence over exact wording.
"""
    
    try:
        response = make_llm_request(prompt)
        return extract_score_from_response(response)
    except Exception as e:
        print(f"ERROR in evaluate_general_chat: {e}")
        return 0.0


def evaluate_precedent_analysis(solution_str: str, ground_truth: str) -> float:
    """
    Evaluate precedent analysis responses for legal accuracy and completeness.
    
    Args:
        solution_str (str): The AI agent's predicted precedent analysis
        ground_truth (str): The human expert's precedent analysis
    
    Returns:
        float: Score between 0 and 1
    """
    prompt = f"""
You are a senior appellate court judge and legal scholar evaluating an AI legal assistant's precedent analysis capabilities. You are comparing the AI agent's precedent analysis against a human expert's analysis to assess training quality.

PREDICTED ANALYSIS (AI Agent): "{solution_str}"
GROUND TRUTH (Human Expert): "{ground_truth}"

Evaluate the AI agent's precedent analysis across these specialized legal dimensions:

**CONTROLLING PRECEDENT IDENTIFICATION (30%)**
- Accurate identification of binding/controlling precedents
- Correct understanding of jurisdictional hierarchy and authority
- Proper recognition of mandatory vs. persuasive authority
- Understanding of which court decisions are binding on the current case

**PRECEDENT RELEVANCE & WEIGHT ASSESSMENT (25%)**
- Accurate evaluation of precedent strength and relevance
- Proper assessment of factual similarity to current case
- Correct analysis of how precedent weight affects current matter
- Understanding of precedent hierarchy and influence

**LEGAL REASONING & CASE DISTINCTION (20%)**
- Sound analytical approach to comparing case facts and holdings
- Ability to distinguish cases based on material differences
- Proper application of precedent to current legal issues
- Logical reasoning about precedent applicability

**INFLUENTIAL PRECEDENT RECOGNITION (15%)**
- Identification of landmark and foundational cases
- Recognition of precedents that have shaped legal doctrine
- Understanding of precedent evolution and development
- Awareness of circuit splits and conflicting authorities

**COMPLETENESS & METHODOLOGY (10%)**
- Comprehensive coverage of relevant precedent landscape
- Systematic approach to precedent research and analysis
- Inclusion of both supporting and distinguishing precedents
- Proper citation format and case identification

**SCORING GUIDELINES:**
- 0.9-1.0: Expert-level precedent analysis, comprehensive and accurate OR Same legal meaning (even if different words)
- 0.7-0.8: Strong analysis with minor gaps in precedent evaluation OR Similar legal point with minor variations
- 0.5-0.6: Adequate but missing key precedents or weight assessments OR Adequate but notable differences in approach/completeness
- 0.3-0.4: Poor analysis with significant precedent errors or omissions OR Significant legal deficiencies but not dangerous
- 0.0-0.2: Fundamentally flawed understanding of precedent system

Focus on whether the AI agent demonstrates proper understanding of:
- Case law hierarchy and binding authority
- Precedent strength and applicability analysis
- Legal reasoning in precedent application
- Recognition of influential and controlling cases

Provide ONLY a numerical score between 0 and 1 as a decimal number.
"""
    try:
        response = make_llm_request(prompt)
        return extract_score_from_response(response)
    except (ValueError, Exception):
        return 0.0


def evaluate_opinion_generation(solution_str: str, ground_truth: str) -> float:
    """
    Evaluate opinion generation responses for legal advocacy and argumentation quality.
    
    Args:
        solution_str (str): The AI agent's predicted legal opinion/argument
        ground_truth (str): The human expert's legal opinion/argument
    
    Returns:
        float: Score between 0.0 and 1.0
    """
    prompt = f"""
You are a senior litigation attorney and legal writing expert evaluating an AI legal assistant's advocacy writing capabilities. You are comparing the AI agent's legal opinion/argument against a human expert's work to assess training quality in legal advocacy.

PREDICTED OPINION (AI Agent): "{solution_str}"
GROUND TRUTH (Human Expert): "{ground_truth}"

Evaluate the AI agent's legal opinion/argument across these specialized advocacy dimensions:

**LEGAL ARGUMENT CONSTRUCTION (30%)**
- Logical structure and persuasive argument flow
- Clear identification and framing of legal issues
- Strategic positioning for plaintiff or defendant perspective
- Coherent progression from facts to legal conclusions
- Effective use of legal reasoning and analysis

**FACT APPLICATION & INTEGRATION (25%)**
- Skillful weaving of facts into legal arguments
- Strategic selection and presentation of favorable facts
- Effective fact-to-law application and connection
- Addressing adverse facts and counter-arguments
- Contextual fact analysis supporting legal position

**PERSUASIVE ADVOCACY TECHNIQUES (20%)**
- Compelling and convincing argumentation style
- Strategic emphasis and positioning of key points
- Effective use of legal precedent to support arguments
- Persuasive language and rhetorical effectiveness
- Professional yet forceful advocacy tone

**LEGAL STANDARDS & DOCTRINE APPLICATION (15%)**
- Correct application of relevant legal standards
- Proper understanding of burden of proof requirements
- Accurate citation and use of controlling law
- Integration of statutory and case law authorities
- Understanding of procedural and substantive requirements

**STRATEGIC ADVOCACY & COMPLETENESS (10%)**
- Comprehensive coverage of relevant legal arguments
- Strategic anticipation of opposing arguments
- Effective organization for maximum persuasive impact
- Professional legal writing standards and clarity
- Appropriate advocacy positioning (plaintiff/defendant perspective)

**SCORING GUIDELINES:**
- 0.9-1.0: Expert-level legal advocacy, highly persuasive and comprehensive OR Same legal meaning (even if different words)
- 0.7-0.8: Strong advocacy with minor weaknesses in argument or strategy OR Similar legal point with minor variations
- 0.5-0.6: Adequate argumentation but lacks persuasive force or completeness OR Adequate but notable differences in approach/completeness
- 0.3-0.4: Poor advocacy with significant argumentative flaws OR Significant legal deficiencies but not dangerous
- 0.0-0.2: Fundamentally weak or incorrect legal argumentation

Focus on whether the AI agent demonstrates proper understanding of:
- Legal advocacy writing and persuasive argumentation
- Strategic case positioning and argument construction
- Effective fact-to-law integration and application
- Professional litigation writing standards

Provide ONLY a numerical score between 0 and 1 as a decimal number.
"""
    
    try:
        response = make_llm_request(prompt)
        return extract_score_from_response(response)
    except (ValueError, Exception):
        return 0.0


def evaluate_judicial_reasoning(solution_str: str, ground_truth: str) -> float:
    """
    Evaluate judicial reasoning responses for understanding legal decision-making psychology.
    
    Args:
        solution_str (str): The AI agent's predicted judicial reasoning analysis
        ground_truth (str): The human expert's judicial reasoning analysis
    
    Returns:
        float: Score between 0 and 1
    """
    prompt = f"""
You are a judicial psychology expert and veteran legal practitioner evaluating an AI legal assistant's ability to understand and analyze the thought processes of legal decision-makers. You are comparing the AI agent's judicial reasoning analysis against a human expert's analysis.

PREDICTED REASONING ANALYSIS (AI Agent): "{solution_str}"
GROUND TRUTH (Human Expert): "{ground_truth}"

Evaluate the AI agent's judicial reasoning analysis across these specialized psychological and analytical dimensions:

**DECISION-MAKING PSYCHOLOGY UNDERSTANDING (30%)**
- Accurate insight into judge's/lawyer's thought process and mental reasoning
- Understanding of psychological factors influencing legal decisions
- Recognition of cognitive patterns and decision-making frameworks
- Insight into how legal practitioners process information and reach conclusions
- Understanding of professional mindset and judicial temperament

**FACTUAL INFLUENCE ANALYSIS (25%)**
- Identification of key facts that influenced the ruling or decision
- Understanding of which evidence carried the most weight
- Analysis of how specific facts shaped the legal practitioner's thinking
- Recognition of fact patterns that triggered particular legal reasoning
- Assessment of factual elements that were decisive vs. peripheral

**COUNTERFACTUAL REASONING (20%)**
- Analysis of what could have changed the outcome or ruling
- Understanding of alternative scenarios and their likely impact
- Identification of critical decision points and alternative paths
- Recognition of factors that could have swayed reasoning in opposite direction
- Strategic understanding of case vulnerabilities and strengths

**LEGAL PRACTITIONER MINDSET PENETRATION (15%)**
- Getting into the "head" of judges, lawyers, paralegals, and other legal professionals
- Understanding professional perspectives, biases, and priorities
- Recognition of institutional and procedural influences on reasoning
- Insight into how different legal roles approach decision-making
- Understanding of professional constraints and considerations

**REASONING PROCESS MAPPING (10%)**
- Logical reconstruction of the decision-making sequence
- Understanding of how legal principles were weighed and applied
- Analysis of the progression from facts to legal conclusions
- Recognition of reasoning gaps, assumptions, and implicit factors
- Comprehensive mapping of the entire thought process

**SCORING GUIDELINES:**
- 0.9-1.0: Expert-level psychological insight into legal decision-making processes OR Same legal meaning (even if different words)
- 0.7-0.8: Strong understanding with minor gaps in reasoning analysis OR Similar legal point with minor variations
- 0.5-0.6: Adequate but lacks depth in psychological or factual insight OR Adequate but notable differences in approach/completeness
- 0.3-0.4: Poor analysis with significant misunderstanding of decision-making OR Significant legal deficiencies but not dangerous
- 0.0-0.2: Fundamentally flawed understanding of judicial reasoning psychology

Focus on whether the AI agent demonstrates proper understanding of:
- How legal practitioners think and make decisions
- What factors truly influence legal reasoning and outcomes
- The psychology behind judicial and legal decision-making
- Strategic analysis of decision points and alternative outcomes

Provide ONLY a numerical score between 0 and 1 as a decimal number.
"""
    
    try:
        response = make_llm_request(prompt)
        return extract_score_from_response(response)
    except (ValueError, Exception):
        return 0.0


def evaluate_miscellaneous(solution_str: str, ground_truth: str) -> float:
    """
    General evaluation function for unknown or miscellaneous task types.
    
    Args:
        solution_str (str): The generated response
        ground_truth (str): The expected response
    
    Returns:
        float: Score between 0 and 1
    """
    prompt = f"""
You are an expert evaluator for general text comparison. Compare the following two responses and rate their similarity and quality.

SOLUTION: "{solution_str}"
GROUND TRUTH: "{ground_truth}"

Evaluate based on:
1. Semantic similarity and meaning preservation
2. Factual accuracy and correctness
3. Completeness and thoroughness
4. Overall quality and coherence
5. Appropriateness for the given context

Provide a score between 0.01 and 1.0 (where 0.0 = completely different/incorrect, 1.0 = identical/excellent).
Return ONLY the numerical score as a decimal number.
"""
    
    try:
        response = make_llm_request(prompt)
        return extract_score_from_response(response)
    except (ValueError, Exception):
        return 0.0


# Configuration for the reward system
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_TEMPERATURE = 0.4
FALLBACK_SCORE = 0.0

# Example usage and testing:
if __name__ == "__main__":
    
    print("=== COMPREHENSIVE REWARD SYSTEM TESTING ===")
    print("Testing all evaluation modes with various scenarios...")
    print()
    
    # ================================
    # IDENTITY TESTS (Should score 0.95-1.0)
    # ================================
    print("🔍 TESTING IDENTICAL STRINGS (Expected: 0.95-1.0)")
    print("=" * 60)
    
    test_cases_identical = [
        {
            "name": "Precedent Analysis - Identical",
            "task_type": "precedent_analysis",
            "solution": "The controlling precedent is Brown v. Board which establishes binding authority for this jurisdiction.",
            "ground_truth": "The controlling precedent is Brown v. Board which establishes binding authority for this jurisdiction."
        },
        {
            "name": "Opinion Generation - Identical", 
            "task_type": "opinion_generation",
            "solution": "Plaintiff has a strong case based on contract breach and should prevail on summary judgment.",
            "ground_truth": "Plaintiff has a strong case based on contract breach and should prevail on summary judgment."
        },
        {
            "name": "Judicial Reasoning - Identical",
            "task_type": "judicial_reasoning", 
            "solution": "The judge was influenced by the defendant's credibility issues and the strength of physical evidence.",
            "ground_truth": "The judge was influenced by the defendant's credibility issues and the strength of physical evidence."
        }
    ]
    
    for test in test_cases_identical:
        try:
            score = evaluate_response(
                data_source="legal_test",
                solution_str=test["solution"],
                ground_truth=test["ground_truth"],
                extra_info={"task_type": test["task_type"]}
            )
            print(f"   {test['name']}: {score:.3f} {'✅' if score >= 0.95 else '❌'}")
        except Exception as e:
            print(f"   {test['name']}: ERROR - {e}")
    
    print()
    
    # ================================
    # SEMANTIC SIMILARITY TESTS (Should score 0.8-0.9)
    # ================================
    print("🔍 TESTING SEMANTIC SIMILARITY (Expected: 0.8-0.9)")
    print("=" * 60)
    
    test_cases_similar = [
        {
            "name": "Precedent Analysis - Same Conclusion",
            "task_type": "precedent_analysis",
            "solution": "Case X is the controlling authority that binds this court's decision.",
            "ground_truth": "Case X provides binding precedent that controls the outcome here."
        },
        {
            "name": "Opinion Generation - Same Position",
            "task_type": "opinion_generation", 
            "solution": "Defendant should win this motion because plaintiff lacks standing.",
            "ground_truth": "The court should grant defendant's motion due to plaintiff's standing deficiency."
        },
        {
            "name": "Judicial Reasoning - Same Insight",
            "task_type": "judicial_reasoning",
            "solution": "The judge prioritized witness testimony over documentary evidence.",
            "ground_truth": "Witness credibility carried more weight than the documents in the judge's analysis."
        },
        {
            "name": "General Chat - Legal Greeting",
            "task_type": "general_chat",
            "solution": "I can help you with your legal question today.",
            "ground_truth": "How may I assist with your legal matter?"
        }
    ]
    
    for test in test_cases_similar:
        try:
            score = evaluate_response(
                data_source="legal_test",
                solution_str=test["solution"],
                ground_truth=test["ground_truth"],
                extra_info={"task_type": test["task_type"]}
            )
            print(f"   {test['name']}: {score:.3f} {'✅' if score >= 0.8 else '❌'}")
        except Exception as e:
            print(f"   {test['name']}: ERROR - {e}")
    
    print()
    
    # ================================
    # DIFFERENT BUT REASONABLE TESTS (Should score 0.6-0.8)
    # ================================
    print("🔍 TESTING DIFFERENT BUT REASONABLE (Expected: 0.6-0.8)")
    print("=" * 60)
    
    test_cases_different = [
        {
            "name": "Precedent Analysis - Different Cases",
            "task_type": "precedent_analysis",
            "solution": "Smith v. Jones from the Second Circuit provides persuasive authority for our position.",
            "ground_truth": "The controlling precedent from our jurisdiction is Brown v. Davis which mandates this result."
        },
        {
            "name": "Opinion Generation - Different Strategy",
            "task_type": "opinion_generation",
            "solution": "We should focus on the constitutional arguments and due process violations.",
            "ground_truth": "The strongest approach is to emphasize contract law and breach of terms."
        },
        {
            "name": "Judicial Reasoning - Different Factors",
            "task_type": "judicial_reasoning",
            "solution": "The judge was swayed by policy considerations and public interest.",
            "ground_truth": "The decision was driven by strict legal precedent and statutory interpretation."
        }
    ]
    
    for test in test_cases_different:
        try:
            score = evaluate_response(
                data_source="legal_test",
                solution_str=test["solution"],
                ground_truth=test["ground_truth"],
                extra_info={"task_type": test["task_type"]}
            )
            print(f"   {test['name']}: {score:.3f} {'✅' if 0.6 <= score <= 0.8 else '❌'}")
        except Exception as e:
            print(f"   {test['name']}: ERROR - {e}")
    
    print()
    
    # ================================
    # POOR QUALITY TESTS (Should score 0.2-0.4)
    # ================================
    print("🔍 TESTING POOR QUALITY RESPONSES (Expected: 0.2-0.4)")
    print("=" * 60)
    
    test_cases_poor = [
        {
            "name": "Precedent Analysis - Major Errors",
            "task_type": "precedent_analysis",
            "solution": "There are no relevant precedents and courts can decide however they want.",
            "ground_truth": "The controlling precedent from Brown v. Board clearly establishes the legal standard."
        },
        {
            "name": "Opinion Generation - Weak Argument",
            "task_type": "opinion_generation",
            "solution": "We might win because the other side seems wrong and unfair.",
            "ground_truth": "Plaintiff has strong grounds for relief based on clear statutory violations and established precedent."
        },
        {
            "name": "Judicial Reasoning - Confused Analysis",
            "task_type": "judicial_reasoning",
            "solution": "The judge probably just flipped a coin or had a bad day.",
            "ground_truth": "The judge carefully weighed the credibility of witnesses against the documentary evidence."
        }
    ]
    
    for test in test_cases_poor:
        try:
            score = evaluate_response(
                data_source="legal_test",
                solution_str=test["solution"],
                ground_truth=test["ground_truth"],
                extra_info={"task_type": test["task_type"]}
            )
            print(f"   {test['name']}: {score:.3f} {'✅' if 0.2 <= score <= 0.4 else '❌'}")
        except Exception as e:
            print(f"   {test['name']}: ERROR - {e}")
    
    print()
    
    # ================================
    # DANGEROUS/WRONG TESTS (Should score 0.01-0.2)
    # ================================
    print("🔍 TESTING DANGEROUS/WRONG RESPONSES (Expected: 0.01-0.2)")
    print("=" * 60)
    
    test_cases_dangerous = [
        {
            "name": "Precedent Analysis - Completely Wrong",
            "task_type": "precedent_analysis",
            "solution": "Precedents don't matter and you can ignore all previous court decisions.",
            "ground_truth": "The controlling precedent from Brown v. Board clearly establishes the legal standard."
        },
        {
            "name": "Opinion Generation - Opposite Position",
            "task_type": "opinion_generation",
            "solution": "Defendant should definitely lose and pay maximum damages immediately.",
            "ground_truth": "Defendant has strong defenses and should prevail on all counts."
        },
        {
            "name": "Judicial Reasoning - Harmful Analysis",
            "task_type": "judicial_reasoning",
            "solution": "Judges are corrupt and make decisions based on bribes and personal bias.",
            "ground_truth": "The judge carefully considered the legal arguments and evidence presented."
        },
        {
            "name": "General Chat - Wrong Legal Advice",
            "task_type": "general_chat",
            "solution": "You don't need a lawyer, just ignore all court orders and legal notices.",
            "ground_truth": "I recommend consulting with a qualified attorney about your legal matter."
        }
    ]
    
    for test in test_cases_dangerous:
        try:
            score = evaluate_response(
                data_source="legal_test",
                solution_str=test["solution"],
                ground_truth=test["ground_truth"],
                extra_info={"task_type": test["task_type"]}
            )
            print(f"   {test['name']}: {score:.3f} {'✅' if score <= 0.2 else '❌'}")
        except Exception as e:
            print(f"   {test['name']}: ERROR - {e}")
    
    print()
    
    # ================================
    # MISCELLANEOUS MODE TESTS
    # ================================
    print("🔍 TESTING MISCELLANEOUS MODE (Various Expected Scores)")
    print("=" * 60)
    
    test_cases_misc = [
        {
            "name": "Factual Equivalence - High Score",
            "solution": "The capital of France is Paris.",
            "ground_truth": "Paris is the capital of France.",
            "expected": "0.9+"
        },
        {
            "name": "Greeting Equivalence - High Score", 
            "solution": "Hello, how can I help you today?",
            "ground_truth": "Hi there! How may I assist you?",
            "expected": "0.8+"
        },
        {
            "name": "Wrong Fact - Low Score",
            "solution": "The capital of France is London.",
            "ground_truth": "Paris is the capital of France.",
            "expected": "0.01-0.2"
        },
        {
            "name": "Unrelated Response - Low Score",
            "solution": "I like pizza and video games.",
            "ground_truth": "Please explain the legal doctrine of stare decisis.",
            "expected": "0.01-0.2"
        }
    ]
    
    for test in test_cases_misc:
        try:
            score = evaluate_response(
                data_source="general_test",
                solution_str=test["solution"],
                ground_truth=test["ground_truth"],
                extra_info={}  # Uses miscellaneous
            )
            print(f"   {test['name']}: {score:.3f} (Expected: {test['expected']})")
        except Exception as e:
            print(f"   {test['name']}: ERROR - {e}")
    
    print()



    # print("1. Testing greeting similarity...")
    # try:
    #     score1 = evaluate_response(
    #         data_source="chat_log",
    #         solution_str="Hello, how can I help you today?",
    #         ground_truth="Hi there! How may I assist you?",
    #         extra_info={"task_type": "general_chat"}
    #     )
    #     print(f"   Greeting Score: {score1} (Expected: 0.8-0.9)")
    # except Exception as e:
    #     print(f"   Error: {e}")
    
    # print()
    
    # print("2. Testing factual equivalence...")
    # try:
    #     score2 = evaluate_response(
    #         data_source="general_qa",
    #         solution_str="The capital of France is Paris.",
    #         ground_truth="Paris is the capital of France.",
    #         extra_info={}  # Uses miscellaneous
    #     )
    #     print(f"   Capital Score: {score2} (Expected: 0.9-1.0)")
    # except Exception as e:
    #     print(f"   Error: {e}")
    
    # print()
    
    # print("3. Testing legal similarity...")
    # try:
    #     score3 = evaluate_response(
    #         data_source="legal_database",
    #         solution_str="The court ruled in favor of the plaintiff based on precedent X, applying the doctrine of stare decisis.",
    #         ground_truth="The court should rule for plaintiff citing precedent X and legal principles.",
    #         extra_info={"task_type": "judicial_reasoning"}
    #     )
    #     print(f"   Legal Score: {score3} (Expected: 0.6-0.8)")
    # except Exception as e:
    #     print(f"   Error: {e}")
    
    # print()
    
    # print("4. Testing penalty scoring for bad responses...")
    # try:
    #     score4 = evaluate_response(
    #         data_source="general_qa",
    #         solution_str="The capital of France is London.",  # Wrong answer
    #         ground_truth="Paris is the capital of France.",
    #         extra_info={}
    #     )
    #     print(f"   Wrong Answer Score: {score4} (Expected: 0.01-0.10)")
    # except Exception as e:
    #     print(f"   Error: {e}")
    
  