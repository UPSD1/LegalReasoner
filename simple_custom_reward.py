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
        float: Score between 0 and 1
    """
    prompt = f"""
You are a senior legal expert evaluating an AI legal assistant's training performance. You are comparing the AI agent's predicted response against a human expert's response to assess training quality.

PREDICTED RESPONSE (AI Agent): "{solution_str}"
GROUND TRUTH (Human Expert): "{ground_truth}"

Evaluate the AI agent's response across these critical legal dimensions:

**LEGAL ACCURACY & CORRECTNESS (40%)**
- Factual accuracy of legal information provided
- Correct application of relevant laws and regulations
- Absence of legal misinformation or errors

**LEGAL REASONING & METHODOLOGY (25%)**
- Logical structure and legal reasoning process
- Appropriate use of legal principles and concepts
- Sound analytical approach to the legal question

**COMPLETENESS & RELEVANCE (20%)**
- Coverage of essential legal points and considerations
- Relevance to the specific legal question asked
- Inclusion of important caveats, disclaimers, or limitations

**PROFESSIONAL STANDARDS (10%)**
- Appropriate legal tone and language
- Ethical considerations and professional responsibility
- Clarity and accessibility for the intended audience

**SIMILARITY TO EXPERT RESPONSE (5%)**
- Alignment with human expert's approach and conclusions
- Consistency in legal interpretation and advice

**SCORING GUIDELINES:**
- 0.9-1.0: Excellent legal response, matches expert quality
- 0.7-0.8: Good legal response with minor gaps or differences
- 0.5-0.6: Adequate response but notable legal deficiencies
- 0.3-0.4: Poor response with significant legal errors
- 0.0-0.2: Dangerous or completely incorrect legal information

Consider this evaluation in the context of AI training assessment - focus on whether the agent is learning to provide safe, accurate, and helpful legal information.

Provide ONLY a numerical score between 0 and 1 as a decimal number.
"""
    
    try:
        response = make_llm_request(prompt)
        print(response)
        # Extract numerical score from response
        score = float(response.strip())
        return max(0.0, min(1.0, score))  # Ensure score is between 0 and 1
    except (ValueError, Exception):
        return 0.0  # Return 0 if parsing fails


def evaluate_precedent_analysis(solution_str: str, ground_truth: str) -> float:
    """
    Evaluate precedent analysis responses for legal accuracy and completeness.
    
    Args:
        solution_str (str): The generated precedent analysis
        ground_truth (str): The expected precedent analysis
    
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
- 0.9-1.0: Expert-level precedent analysis, comprehensive and accurate
- 0.7-0.8: Strong analysis with minor gaps in precedent evaluation
- 0.5-0.6: Adequate but missing key precedents or weight assessments
- 0.3-0.4: Poor analysis with significant precedent errors or omissions
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
        score = float(response.strip())
        return max(0.0, min(1.0, score))
    except (ValueError, Exception):
        return 0.0


def evaluate_opinion_generation(solution_str: str, ground_truth: str) -> float:
    """
    Evaluate opinion generation responses for coherence and reasoning quality.
    
    Args:
        solution_str (str): The generated opinion
        ground_truth (str): The expected opinion structure/content
    
    Returns:
        float: Score between 0 and 1
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
- 0.9-1.0: Expert-level legal advocacy, highly persuasive and comprehensive
- 0.7-0.8: Strong advocacy with minor weaknesses in argument or strategy
- 0.5-0.6: Adequate argumentation but lacks persuasive force or completeness
- 0.3-0.4: Poor advocacy with significant argumentative flaws
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
        score = float(response.strip())
        return max(0.0, min(1.0, score))
    except (ValueError, Exception):
        return 0.0


def evaluate_judicial_reasoning(solution_str: str, ground_truth: str) -> float:
    """
    Evaluate judicial reasoning responses for legal soundness and logic.
    
    Args:
        solution_str (str): The generated judicial reasoning
        ground_truth (str): The expected judicial reasoning
    
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
- 0.9-1.0: Expert-level psychological insight into legal decision-making processes
- 0.7-0.8: Strong understanding with minor gaps in reasoning analysis
- 0.5-0.6: Adequate but lacks depth in psychological or factual insight
- 0.3-0.4: Poor analysis with significant misunderstanding of decision-making
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
        score = float(response.strip())
        return max(0.0, min(1.0, score))
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
- 0.9-1.0: Expert-level psychological insight into legal decision-making processes
- 0.7-0.8: Strong understanding with minor gaps in reasoning analysis
- 0.5-0.6: Adequate but lacks depth in psychological or factual insight
- 0.3-0.4: Poor analysis with significant misunderstanding of decision-making
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
        score = float(response.strip())
        return max(0.0, min(1.0, score))
    except (ValueError, Exception):
        return 0.0


# Configuration for the reward system
DEFAULT_MODEL = "gpt-4.1-mini"
DEFAULT_TEMPERATURE = 0.7
FALLBACK_SCORE = 0.0

# Example usage and testing:
if __name__ == "__main__":
    # Set your OpenAI API key
    # os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    
    # Example with task_type specified
    print("Testing judicial reasoning evaluation...")
    score1 = evaluate_response(
        data_source="legal_database",
        solution_str="The court ruled in favor of the plaintiff based on precedent X, applying the doctrine of stare decisis.",
        ground_truth="The court should rule for plaintiff citing precedent X and legal principles.",
        extra_info={"task_type": "judicial_reasoning"}
    )
    print(f"Score: {score1}")
    
    # Example without task_type (will use miscellaneous)
    print("\nTesting miscellaneous evaluation...")
    score2 = evaluate_response(
        data_source="general_qa",
        solution_str="The capital of France is Paris.",
        ground_truth="Paris is the capital of France.",
        extra_info={}
    )
    print(f"Score: {score2}")
    
    # Example with general chat
    print("\nTesting general chat evaluation...")
    score3 = evaluate_response(
        data_source="chat_log",
        solution_str="Hello, how can I help you today?",
        ground_truth="Hi there! How may I assist you?",
        extra_info={"task_type": "general_chat"}
    )
    print(f"Score: {score3}")
    
    print("\nAll evaluation functions are ready to use!")
    print("Make sure to set your OPENAI_API_KEY environment variable.")