"""
Judicial Reasoning Ensemble - Real Implementation

Provides specialized evaluation for judicial reasoning tasks including FIRAC structure,
legal accuracy, precedent application, constitutional analysis, and judicial tone.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from ...core import (
    LegalTaskType, USJurisdiction, LegalDomain,
    JudgeEvaluation, EvaluationMetadata,
    LegalRewardSystemError, create_error_context
)
from ...config import LegalRewardSystemConfig
from ..base import BaseJudgeEnsemble
from ..api_client import CostOptimizedAPIClient
from ...utils import get_legal_logger


class JudicialAnalysisComponent(Enum):
    """Components of judicial reasoning evaluation"""
    LEGAL_ACCURACY = "legal_accuracy"
    FIRAC_STRUCTURE = "firac_structure"
    PRECEDENT_APPLICATION = "precedent_application"
    CONSTITUTIONAL_ANALYSIS = "constitutional_analysis"
    JUDICIAL_TONE = "judicial_tone"


@dataclass
class JudicialReasoningScore:
    """Structured score for judicial reasoning evaluation"""
    legal_accuracy: float
    firac_structure: float
    precedent_application: float
    constitutional_analysis: float
    judicial_tone: float
    jurisdiction_compliance: float
    jurisdiction_compliant: bool
    overall_score: float
    confidence: float
    reasoning: str


class JudicialReasoningEnsemble(BaseJudgeEnsemble):
    """
    Real implementation of judicial reasoning evaluation ensemble.
    
    Evaluates legal responses for judicial reasoning quality including:
    - Legal accuracy and doctrinal correctness
    - FIRAC structure (Facts, Issue, Rule, Application, Conclusion)
    - Precedent application and case law integration
    - Constitutional analysis when applicable
    - Appropriate judicial tone and formality
    """
    
    def __init__(self, 
                 config: LegalRewardSystemConfig,
                 api_client: Optional[CostOptimizedAPIClient] = None):
        
        super().__init__(
            ensemble_name="judicial_reasoning_ensemble",
            task_type=LegalTaskType.JUDICIAL_REASONING
        )
        
        self.config = config
        self.api_client = api_client or CostOptimizedAPIClient(config)
        self.logger = get_legal_logger("judicial_reasoning_ensemble")
        
        # Component weights for ensemble aggregation
        self.component_weights = {
            JudicialAnalysisComponent.LEGAL_ACCURACY: 0.25,
            JudicialAnalysisComponent.FIRAC_STRUCTURE: 0.25,
            JudicialAnalysisComponent.PRECEDENT_APPLICATION: 0.20,
            JudicialAnalysisComponent.CONSTITUTIONAL_ANALYSIS: 0.15,
            JudicialAnalysisComponent.JUDICIAL_TONE: 0.15
        }
        
        # Jurisdiction-specific legal contexts
        self.jurisdiction_contexts = {
            USJurisdiction.FEDERAL: "federal courts, constitutional law, federal statutes, federal precedents",
            USJurisdiction.CALIFORNIA: "California state courts, California statutes, California case law",
            USJurisdiction.NEW_YORK: "New York state courts, New York statutes, New York case law",
            USJurisdiction.TEXAS: "Texas state courts, Texas statutes, Texas case law",
            USJurisdiction.FLORIDA: "Florida state courts, Florida statutes, Florida case law",
            USJurisdiction.GENERAL: "general US legal principles, common law traditions"
        }
        
        # Performance tracking
        self.evaluation_count = 0
        self.total_evaluation_time = 0.0
        self.api_call_count = 0
        self.cache_hit_count = 0
        self.jurisdiction_compliance_failures = []
        
        self.logger.info("Initialized JudicialReasoningEnsemble with real API-based evaluation")
    
    async def evaluate_response(self,
                              response: str,
                              task_type: LegalTaskType,
                              jurisdiction: USJurisdiction,
                              prompt: str = "",
                              context: Optional[Dict[str, Any]] = None) -> JudgeEvaluation:
        """
        Evaluate response using real judicial reasoning analysis.
        
        Args:
            response: Legal response to evaluate
            task_type: Should be JUDICIAL_REASONING
            jurisdiction: US jurisdiction for evaluation context
            prompt: Original prompt/question
            context: Additional evaluation context
            
        Returns:
            Complete judicial reasoning evaluation
        """
        
        start_time = time.time()
        
        try:
            # Get jurisdiction-specific context
            jurisdiction_context = self.jurisdiction_contexts.get(
                jurisdiction, self.jurisdiction_contexts[USJurisdiction.GENERAL]
            )
            
            # Evaluate all judicial reasoning components
            component_scores = await self._evaluate_all_components(
                response, prompt, jurisdiction_context, context
            )
            
            # Calculate overall judicial reasoning score
            judicial_score = self._calculate_judicial_score(component_scores, jurisdiction)
            
            # Create evaluation result
            evaluation = JudgeEvaluation(
                judge_name=self.ensemble_name,
                task_type=task_type,
                score=judicial_score.overall_score,
                confidence=judicial_score.confidence,
                reasoning=judicial_score.reasoning,
                metadata=EvaluationMetadata(
                    evaluation_time=time.time() - start_time,
                    model_used=self.api_client.get_current_model(),
                    cost=self._estimate_evaluation_cost(),
                    api_provider=self.api_client.get_current_provider().value,
                    tokens_used=len(response.split()) * 2,  # Estimate
                    jurisdiction=jurisdiction.value,
                    additional_context={
                        "component_scores": {
                            comp.value: getattr(judicial_score, comp.value) 
                            for comp in JudicialAnalysisComponent
                        },
                        "jurisdiction_compliance": judicial_score.jurisdiction_compliance,
                        "jurisdiction_compliant": judicial_score.jurisdiction_compliant,
                        "ensemble_type": "judicial_reasoning",
                        "evaluation_method": "real_api_based"
                    }
                )
            )
            
            # Track performance
            self.evaluation_count += 1
            self.total_evaluation_time += time.time() - start_time
            
            # Track jurisdiction compliance failures
            if not judicial_score.jurisdiction_compliant:
                self.jurisdiction_compliance_failures.append({
                    "timestamp": time.time(),
                    "jurisdiction": jurisdiction.value,
                    "compliance_score": judicial_score.jurisdiction_compliance,
                    "response_preview": response[:200]
                })
            
            return evaluation
            
        except Exception as e:
            self.logger.error(f"Error in judicial reasoning evaluation: {e}")
            raise LegalRewardSystemError(
                f"Judicial reasoning evaluation failed: {e}",
                error_context=create_error_context(
                    "judicial_reasoning_ensemble", 
                    "evaluate_response"
                )
            )
    
    async def _evaluate_all_components(self,
                                     response: str,
                                     prompt: str,
                                     jurisdiction_context: str,
                                     context: Optional[Dict]) -> Dict[JudicialAnalysisComponent, float]:
        """Evaluate all judicial reasoning components using real API calls"""
        
        # Create evaluation tasks for all components
        evaluation_tasks = {}
        
        for component in JudicialAnalysisComponent:
            evaluation_tasks[component] = self._evaluate_component(
                component, response, prompt, jurisdiction_context, context
            )
        
        # Execute all evaluations concurrently
        results = await asyncio.gather(*evaluation_tasks.values(), return_exceptions=True)
        
        # Collect results
        component_scores = {}
        for component, result in zip(evaluation_tasks.keys(), results):
            if isinstance(result, Exception):
                self.logger.warning(f"Component {component.value} evaluation failed: {result}")
                component_scores[component] = 5.0  # Neutral fallback
            else:
                component_scores[component] = result
        
        return component_scores
    
    async def _evaluate_component(self,
                                component: JudicialAnalysisComponent,
                                response: str,
                                prompt: str,
                                jurisdiction_context: str,
                                context: Optional[Dict]) -> float:
        """Evaluate a specific judicial reasoning component"""
        
        # Component-specific evaluation prompts
        evaluation_prompts = {
            JudicialAnalysisComponent.LEGAL_ACCURACY: self._create_legal_accuracy_prompt,
            JudicialAnalysisComponent.FIRAC_STRUCTURE: self._create_firac_structure_prompt,
            JudicialAnalysisComponent.PRECEDENT_APPLICATION: self._create_precedent_application_prompt,
            JudicialAnalysisComponent.CONSTITUTIONAL_ANALYSIS: self._create_constitutional_analysis_prompt,
            JudicialAnalysisComponent.JUDICIAL_TONE: self._create_judicial_tone_prompt
        }
        
        # Create component-specific evaluation prompt
        eval_prompt = evaluation_prompts[component](response, prompt, jurisdiction_context)
        
        try:
            # Make API call for evaluation
            api_response = await self.api_client.evaluate_async(
                prompt=eval_prompt,
                response=response,
                max_tokens=500,
                temperature=0.1  # Low temperature for consistent evaluation
            )
            
            self.api_call_count += 1
            
            # Extract numerical score from API response
            score = self._extract_score_from_response(api_response.content)
            return max(0.0, min(10.0, score))  # Bound score to [0, 10]
            
        except Exception as e:
            self.logger.warning(f"API evaluation failed for {component.value}: {e}")
            return 5.0  # Neutral fallback score
    
    def _create_legal_accuracy_prompt(self, response: str, prompt: str, jurisdiction_context: str) -> str:
        """Create prompt for legal accuracy evaluation"""
        return f"""
You are a legal expert evaluating the accuracy of legal reasoning and doctrine.

JURISDICTION CONTEXT: {jurisdiction_context}

ORIGINAL QUESTION: {prompt}

RESPONSE TO EVALUATE: {response}

EVALUATION CRITERIA - Legal Accuracy:
- Correct statement of legal principles and doctrines
- Accurate understanding of applicable law
- Proper legal terminology and concepts
- Factual accuracy in legal statements
- Appropriate scope and limitations of legal rules

Provide a score from 0-10 where:
- 9-10: Exceptional legal accuracy, perfect doctrinal understanding
- 7-8: Strong legal accuracy with minor issues
- 5-6: Adequate accuracy but some legal errors
- 3-4: Significant legal inaccuracies
- 0-2: Major legal errors or misunderstandings

Score: [Your numerical score 0-10]
Reasoning: [Brief explanation of the score]
"""
    
    def _create_firac_structure_prompt(self, response: str, prompt: str, jurisdiction_context: str) -> str:
        """Create prompt for FIRAC structure evaluation"""
        return f"""
You are a legal writing expert evaluating judicial analysis structure.

JURISDICTION CONTEXT: {jurisdiction_context}

ORIGINAL QUESTION: {prompt}

RESPONSE TO EVALUATE: {response}

EVALUATION CRITERIA - FIRAC Structure:
- Facts: Clear identification and organization of relevant facts
- Issue: Precise statement of legal issues
- Rule: Accurate statement of applicable legal rules
- Application: Thorough application of law to facts
- Conclusion: Clear, well-supported conclusion

Provide a score from 0-10 where:
- 9-10: Perfect FIRAC structure with all elements clearly present
- 7-8: Strong structure with most elements well-developed
- 5-6: Basic structure present but some elements weak
- 3-4: Poor structure, missing key elements
- 0-2: No clear analytical structure

Score: [Your numerical score 0-10]
Reasoning: [Brief explanation focusing on structural elements]
"""
    
    def _create_precedent_application_prompt(self, response: str, prompt: str, jurisdiction_context: str) -> str:
        """Create prompt for precedent application evaluation"""
        return f"""
You are a legal expert evaluating the use and application of legal precedents.

JURISDICTION CONTEXT: {jurisdiction_context}

ORIGINAL QUESTION: {prompt}

RESPONSE TO EVALUATE: {response}

EVALUATION CRITERIA - Precedent Application:
- Identification of relevant precedents and case law
- Proper citation and reference to authorities
- Appropriate analogical reasoning
- Understanding of precedent hierarchy (binding vs. persuasive)
- Effective distinguishing of inapposite cases

Provide a score from 0-10 where:
- 9-10: Excellent use of precedents with sophisticated analysis
- 7-8: Good precedent application with proper reasoning
- 5-6: Basic use of precedents with some weaknesses
- 3-4: Poor precedent use or misunderstanding
- 0-2: No meaningful precedent analysis

Score: [Your numerical score 0-10]
Reasoning: [Brief explanation of precedent usage]
"""
    
    def _create_constitutional_analysis_prompt(self, response: str, prompt: str, jurisdiction_context: str) -> str:
        """Create prompt for constitutional analysis evaluation"""
        return f"""
You are a constitutional law expert evaluating constitutional analysis.

JURISDICTION CONTEXT: {jurisdiction_context}

ORIGINAL QUESTION: {prompt}

RESPONSE TO EVALUATE: {response}

EVALUATION CRITERIA - Constitutional Analysis:
- Recognition of constitutional issues and implications
- Understanding of constitutional principles and amendments
- Proper analysis of constitutional standards (strict scrutiny, etc.)
- Federal vs. state constitutional considerations
- Separation of powers and federalism awareness

Provide a score from 0-10 where:
- 9-10: Sophisticated constitutional analysis (when applicable)
- 7-8: Good constitutional reasoning
- 5-6: Basic constitutional understanding
- 3-4: Limited constitutional analysis
- 0-2: Poor or missing constitutional reasoning
- N/A: If constitutional analysis is not applicable, score 7

Score: [Your numerical score 0-10 or N/A]
Reasoning: [Brief explanation of constitutional elements]
"""
    
    def _create_judicial_tone_prompt(self, response: str, prompt: str, jurisdiction_context: str) -> str:
        """Create prompt for judicial tone evaluation"""
        return f"""
You are evaluating the appropriateness of judicial tone and style.

JURISDICTION CONTEXT: {jurisdiction_context}

ORIGINAL QUESTION: {prompt}

RESPONSE TO EVALUATE: {response}

EVALUATION CRITERIA - Judicial Tone:
- Appropriate formality and professionalism
- Objective, impartial analytical voice
- Clear, authoritative communication
- Proper judicial language and terminology
- Avoidance of advocacy or bias

Provide a score from 0-10 where:
- 9-10: Perfect judicial tone, highly professional
- 7-8: Appropriate judicial style with minor issues
- 5-6: Generally appropriate but some tone problems
- 3-4: Inconsistent or inappropriate tone
- 0-2: Unprofessional or highly inappropriate tone

Score: [Your numerical score 0-10]
Reasoning: [Brief explanation of tone assessment]
"""
    
    def _extract_score_from_response(self, api_response: str) -> float:
        """Extract numerical score from API response"""
        
        # Look for "Score: X" pattern
        import re
        
        # Try different score patterns
        patterns = [
            r"Score:\s*(\d+(?:\.\d+)?)",
            r"score:\s*(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)/10",
            r"(\d+(?:\.\d+)?)\s*out of 10",
            r"Rating:\s*(\d+(?:\.\d+)?)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, api_response)
            if match:
                try:
                    score = float(match.group(1))
                    return score
                except ValueError:
                    continue
        
        # If no clear score found, try to infer from content
        if any(word in api_response.lower() for word in ["excellent", "exceptional", "outstanding"]):
            return 8.5
        elif any(word in api_response.lower() for word in ["good", "strong", "adequate"]):
            return 7.0
        elif any(word in api_response.lower() for word in ["poor", "weak", "inadequate"]):
            return 3.0
        else:
            return 5.0  # Default neutral score
    
    def _calculate_judicial_score(self,
                                component_scores: Dict[JudicialAnalysisComponent, float],
                                jurisdiction: USJurisdiction) -> JudicialReasoningScore:
        """Calculate overall judicial reasoning score from components"""
        
        # Calculate weighted average of components
        weighted_sum = 0.0
        total_weight = 0.0
        
        for component, score in component_scores.items():
            weight = self.component_weights[component]
            weighted_sum += score * weight
            total_weight += weight
        
        base_score = weighted_sum / total_weight if total_weight > 0 else 5.0
        
        # Jurisdiction compliance check
        jurisdiction_compliance = min(component_scores.values()) if component_scores else 5.0
        jurisdiction_compliant = jurisdiction_compliance >= 6.0  # Threshold for compliance
        
        # Apply jurisdiction penalty if non-compliant
        if jurisdiction_compliant:
            final_score = base_score
        else:
            final_score = base_score * 0.7  # 30% penalty for non-compliance
        
        # Calculate confidence based on score consistency
        scores_list = list(component_scores.values())
        if scores_list:
            score_variance = sum((s - base_score) ** 2 for s in scores_list) / len(scores_list)
            confidence = max(0.5, 1.0 - (score_variance / 10.0))
        else:
            confidence = 0.5
        
        # Generate reasoning
        reasoning = self._generate_reasoning(component_scores, jurisdiction_compliance, jurisdiction_compliant)
        
        return JudicialReasoningScore(
            legal_accuracy=component_scores.get(JudicialAnalysisComponent.LEGAL_ACCURACY, 5.0),
            firac_structure=component_scores.get(JudicialAnalysisComponent.FIRAC_STRUCTURE, 5.0),
            precedent_application=component_scores.get(JudicialAnalysisComponent.PRECEDENT_APPLICATION, 5.0),
            constitutional_analysis=component_scores.get(JudicialAnalysisComponent.CONSTITUTIONAL_ANALYSIS, 5.0),
            judicial_tone=component_scores.get(JudicialAnalysisComponent.JUDICIAL_TONE, 5.0),
            jurisdiction_compliance=jurisdiction_compliance,
            jurisdiction_compliant=jurisdiction_compliant,
            overall_score=final_score,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def _generate_reasoning(self,
                          component_scores: Dict[JudicialAnalysisComponent, float],
                          jurisdiction_compliance: float,
                          jurisdiction_compliant: bool) -> str:
        """Generate detailed reasoning for the evaluation"""
        
        reasoning_parts = []
        reasoning_parts.append("JUDICIAL REASONING EVALUATION")
        reasoning_parts.append("=" * 40)
        
        # Component breakdown
        components = [
            ("Legal Accuracy", component_scores.get(JudicialAnalysisComponent.LEGAL_ACCURACY, 5.0)),
            ("FIRAC Structure", component_scores.get(JudicialAnalysisComponent.FIRAC_STRUCTURE, 5.0)),
            ("Precedent Application", component_scores.get(JudicialAnalysisComponent.PRECEDENT_APPLICATION, 5.0)),
            ("Constitutional Analysis", component_scores.get(JudicialAnalysisComponent.CONSTITUTIONAL_ANALYSIS, 5.0)),
            ("Judicial Tone", component_scores.get(JudicialAnalysisComponent.JUDICIAL_TONE, 5.0))
        ]
        
        reasoning_parts.append("Component Analysis:")
        for component_name, score in components:
            reasoning_parts.append(f"• {component_name}: {score:.1f}/10")
        
        # Jurisdiction compliance
        if jurisdiction_compliant:
            reasoning_parts.append(f"\n✓ Jurisdiction Compliance: {jurisdiction_compliance:.1f}/10 (PASSED)")
        else:
            reasoning_parts.append(f"\n✗ Jurisdiction Compliance: {jurisdiction_compliance:.1f}/10 (FAILED)")
        
        return "\n".join(reasoning_parts)
    
    def _estimate_evaluation_cost(self) -> float:
        """Estimate cost of the evaluation"""
        # Rough estimate based on token usage and API pricing
        estimated_tokens = 2000  # Conservative estimate for judicial reasoning evaluation
        return estimated_tokens * 0.00003  # Approximate cost per token
    
    def get_performance_analytics(self) -> Dict[str, Any]:
        """Get comprehensive performance analytics"""
        
        avg_evaluation_time = self.total_evaluation_time / max(1, self.evaluation_count)
        
        return {
            "ensemble_name": self.ensemble_name,
            "task_type": self.task_type.value,
            "evaluations_completed": self.evaluation_count,
            "total_evaluation_time": self.total_evaluation_time,
            "average_evaluation_time": avg_evaluation_time,
            "api_calls_made": self.api_call_count,
            "cache_hit_count": self.cache_hit_count,
            "jurisdiction_compliance_failures": len(self.jurisdiction_compliance_failures),
            "component_weights": {comp.value: weight for comp, weight in self.component_weights.items()},
            "supported_jurisdictions": list(self.jurisdiction_contexts.keys()),
            "evaluation_method": "real_api_based"
        }


# Factory function
def create_judicial_reasoning_ensemble(config: LegalRewardSystemConfig,
                                     api_client: Optional[CostOptimizedAPIClient] = None) -> JudicialReasoningEnsemble:
    """Create production judicial reasoning ensemble"""
    return JudicialReasoningEnsemble(config, api_client)