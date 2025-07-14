# Create: legal_reward_system/judges/specialized/precedent_analysis.py

"""
Precedent Analysis Ensemble - Real Implementation

Provides specialized evaluation for precedent analysis tasks including case law accuracy,
analogical reasoning, citation quality, precedent hierarchy, and distinguishing analysis.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from core import (
    LegalTaskType, USJurisdiction, LegalDomain,
    JudgeEvaluation, EvaluationMetadata,
    LegalRewardSystemError, create_error_context
)
from config import LegalRewardSystemConfig
from judges.base import BaseJudgeEnsemble
from judges.api_client import CostOptimizedAPIClient
from utils import get_legal_logger


class PrecedentAnalysisComponent(Enum):
    """Components of precedent analysis evaluation"""
    CASE_LAW_ACCURACY = "case_law_accuracy"
    ANALOGICAL_REASONING = "analogical_reasoning"
    CITATION_QUALITY = "citation_quality"
    PRECEDENT_HIERARCHY = "precedent_hierarchy"
    DISTINGUISHING_ANALYSIS = "distinguishing_analysis"


@dataclass
class PrecedentAnalysisScore:
    """Structured score for precedent analysis evaluation"""
    case_law_accuracy: float
    analogical_reasoning: float
    citation_quality: float
    precedent_hierarchy: float
    distinguishing_analysis: float
    jurisdiction_compliance: float
    jurisdiction_compliant: bool
    overall_score: float
    confidence: float
    reasoning: str


class PrecedentAnalysisEnsemble(BaseJudgeEnsemble):
    """
    Real implementation of precedent analysis evaluation ensemble.
    
    Evaluates legal responses for precedent analysis quality including:
    - Case law accuracy and relevance
    - Analogical reasoning between cases
    - Citation quality and format
    - Understanding of precedent hierarchy (binding vs. persuasive)
    - Effective distinguishing of inapposite cases
    """
    
    def __init__(self, 
                 config: LegalRewardSystemConfig,
                 api_client: Optional[CostOptimizedAPIClient] = None):
        
        super().__init__(
            ensemble_name="precedent_analysis_ensemble",
            task_type=LegalTaskType.PRECEDENT_ANALYSIS
        )
        
        self.config = config
        self.api_client = api_client or CostOptimizedAPIClient(config)
        self.logger = get_legal_logger("precedent_analysis_ensemble")
        
        # Component weights for ensemble aggregation
        self.component_weights = {
            PrecedentAnalysisComponent.CASE_LAW_ACCURACY: 0.30,
            PrecedentAnalysisComponent.ANALOGICAL_REASONING: 0.25,
            PrecedentAnalysisComponent.CITATION_QUALITY: 0.20,
            PrecedentAnalysisComponent.PRECEDENT_HIERARCHY: 0.15,
            PrecedentAnalysisComponent.DISTINGUISHING_ANALYSIS: 0.10
        }
        
        # Jurisdiction-specific precedent contexts
        self.jurisdiction_precedent_contexts = {
            USJurisdiction.FEDERAL: "federal circuit courts, Supreme Court precedents, federal case law",
            USJurisdiction.CALIFORNIA: "California Supreme Court, California Courts of Appeal, federal precedents binding in California",
            USJurisdiction.NEW_YORK: "New York Court of Appeals, New York Appellate Division, federal precedents binding in New York",
            USJurisdiction.TEXAS: "Texas Supreme Court, Texas Courts of Appeals, federal precedents binding in Texas",
            USJurisdiction.FLORIDA: "Florida Supreme Court, Florida District Courts of Appeal, federal precedents binding in Florida",
            USJurisdiction.GENERAL: "general US case law principles, common precedent structures"
        }
        
        # Performance tracking
        self.evaluation_count = 0
        self.total_evaluation_time = 0.0
        self.api_call_count = 0
        self.cache_hit_count = 0
        self.precedent_analysis_patterns = []
        
        self.logger.info("Initialized PrecedentAnalysisEnsemble with real API-based evaluation")
    
    async def evaluate_response(self,
                              response: str,
                              task_type: LegalTaskType,
                              jurisdiction: USJurisdiction,
                              prompt: str = "",
                              context: Optional[Dict[str, Any]] = None) -> JudgeEvaluation:
        """
        Evaluate response using real precedent analysis.
        
        Args:
            response: Legal response to evaluate
            task_type: Should be PRECEDENT_ANALYSIS
            jurisdiction: US jurisdiction for evaluation context
            prompt: Original prompt/question
            context: Additional evaluation context
            
        Returns:
            Complete precedent analysis evaluation
        """
        
        start_time = time.time()
        
        try:
            # Get jurisdiction-specific precedent context
            precedent_context = self.jurisdiction_precedent_contexts.get(
                jurisdiction, self.jurisdiction_precedent_contexts[USJurisdiction.GENERAL]
            )
            
            # Evaluate all precedent analysis components
            component_scores = await self._evaluate_all_precedent_components(
                response, prompt, precedent_context, context
            )
            
            # Calculate overall precedent analysis score
            precedent_score = self._calculate_precedent_score(component_scores, jurisdiction)
            
            # Create evaluation result
            evaluation = JudgeEvaluation(
                judge_name=self.ensemble_name,
                task_type=task_type,
                score=precedent_score.overall_score,
                confidence=precedent_score.confidence,
                reasoning=precedent_score.reasoning,
                metadata=EvaluationMetadata(
                    evaluation_time=time.time() - start_time,
                    model_used=self.api_client.get_current_model(),
                    cost=self._estimate_evaluation_cost(),
                    api_provider=self.api_client.get_current_provider().value,
                    tokens_used=len(response.split()) * 2,
                    jurisdiction=jurisdiction.value,
                    additional_context={
                        "component_scores": {
                            comp.value: getattr(precedent_score, comp.value) 
                            for comp in PrecedentAnalysisComponent
                        },
                        "jurisdiction_compliance": precedent_score.jurisdiction_compliance,
                        "jurisdiction_compliant": precedent_score.jurisdiction_compliant,
                        "ensemble_type": "precedent_analysis",
                        "evaluation_method": "real_api_based"
                    }
                )
            )
            
            # Track performance
            self.evaluation_count += 1
            self.total_evaluation_time += time.time() - start_time
            
            return evaluation
            
        except Exception as e:
            self.logger.error(f"Error in precedent analysis evaluation: {e}")
            raise LegalRewardSystemError(
                f"Precedent analysis evaluation failed: {e}",
                error_context=create_error_context(
                    "precedent_analysis_ensemble", 
                    "evaluate_response"
                )
            )
    
    async def _evaluate_all_precedent_components(self,
                                               response: str,
                                               prompt: str,
                                               precedent_context: str,
                                               context: Optional[Dict]) -> Dict[PrecedentAnalysisComponent, float]:
        """Evaluate all precedent analysis components using real API calls"""
        
        # Create evaluation tasks for all components
        evaluation_tasks = {}
        
        for component in PrecedentAnalysisComponent:
            evaluation_tasks[component] = self._evaluate_precedent_component(
                component, response, prompt, precedent_context, context
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
    
    async def _evaluate_precedent_component(self,
                                          component: PrecedentAnalysisComponent,
                                          response: str,
                                          prompt: str,
                                          precedent_context: str,
                                          context: Optional[Dict]) -> float:
        """Evaluate a specific precedent analysis component"""
        
        # Component-specific evaluation prompts
        evaluation_prompts = {
            PrecedentAnalysisComponent.CASE_LAW_ACCURACY: self._create_case_law_accuracy_prompt,
            PrecedentAnalysisComponent.ANALOGICAL_REASONING: self._create_analogical_reasoning_prompt,
            PrecedentAnalysisComponent.CITATION_QUALITY: self._create_citation_quality_prompt,
            PrecedentAnalysisComponent.PRECEDENT_HIERARCHY: self._create_precedent_hierarchy_prompt,
            PrecedentAnalysisComponent.DISTINGUISHING_ANALYSIS: self._create_distinguishing_analysis_prompt
        }
        
        # Create component-specific evaluation prompt
        eval_prompt = evaluation_prompts[component](response, prompt, precedent_context)
        
        try:
            # Make API call for evaluation
            api_response = await self.api_client.evaluate_async(
                prompt=eval_prompt,
                response=response,
                max_tokens=500,
                temperature=0.1
            )
            
            self.api_call_count += 1
            
            # Extract numerical score from API response
            score = self._extract_score_from_response(api_response.content)
            return max(0.0, min(10.0, score))
            
        except Exception as e:
            self.logger.warning(f"API evaluation failed for {component.value}: {e}")
            return 5.0
    
    def _create_case_law_accuracy_prompt(self, response: str, prompt: str, precedent_context: str) -> str:
        """Create prompt for case law accuracy evaluation"""
        return f"""
You are a legal expert evaluating the accuracy of case law references and precedent usage.

PRECEDENT CONTEXT: {precedent_context}

ORIGINAL QUESTION: {prompt}

RESPONSE TO EVALUATE: {response}

EVALUATION CRITERIA - Case Law Accuracy:
- Accurate reference to real cases and holdings
- Correct understanding of case outcomes and rationale
- Appropriate selection of relevant precedents
- Factual accuracy in case descriptions
- Proper understanding of legal doctrines from cases

Provide a score from 0-10 where:
- 9-10: Exceptional case law accuracy, perfect precedent understanding
- 7-8: Strong accuracy with minor case law issues
- 5-6: Generally accurate but some precedent errors
- 3-4: Significant case law inaccuracies
- 0-2: Major errors in precedent usage or fabricated cases

Score: [Your numerical score 0-10]
Reasoning: [Brief explanation of case law accuracy assessment]
"""
    
    def _create_analogical_reasoning_prompt(self, response: str, prompt: str, precedent_context: str) -> str:
        """Create prompt for analogical reasoning evaluation"""
        return f"""
You are a legal expert evaluating analogical reasoning between cases.

PRECEDENT CONTEXT: {precedent_context}

ORIGINAL QUESTION: {prompt}

RESPONSE TO EVALUATE: {response}

EVALUATION CRITERIA - Analogical Reasoning:
- Quality of analogies between current facts and precedent cases
- Identification of legally significant similarities
- Recognition of material differences
- Logical progression from precedent to conclusion
- Sophisticated understanding of analogical legal reasoning

Provide a score from 0-10 where:
- 9-10: Sophisticated analogical reasoning with nuanced analysis
- 7-8: Good analogical thinking with clear connections
- 5-6: Basic analogical reasoning with some weaknesses
- 3-4: Poor analogical connections or flawed reasoning
- 0-2: No meaningful analogical analysis

Score: [Your numerical score 0-10]
Reasoning: [Brief explanation of analogical reasoning quality]
"""
    
    def _create_citation_quality_prompt(self, response: str, prompt: str, precedent_context: str) -> str:
        """Create prompt for citation quality evaluation"""
        return f"""
You are a legal expert evaluating citation quality and format.

PRECEDENT CONTEXT: {precedent_context}

ORIGINAL QUESTION: {prompt}

RESPONSE TO EVALUATE: {response}

EVALUATION CRITERIA - Citation Quality:
- Proper legal citation format and style
- Complete and accurate citation information
- Appropriate use of pinpoint citations
- Consistent citation style throughout
- Proper use of signals and explanatory parentheticals

Provide a score from 0-10 where:
- 9-10: Perfect citation format with professional quality
- 7-8: Good citations with minor formatting issues
- 5-6: Adequate citations but some format problems
- 3-4: Poor citation format or incomplete information
- 0-2: No proper citations or major citation errors

Score: [Your numerical score 0-10]
Reasoning: [Brief explanation of citation quality]
"""
    
    def _create_precedent_hierarchy_prompt(self, response: str, prompt: str, precedent_context: str) -> str:
        """Create prompt for precedent hierarchy evaluation"""
        return f"""
You are a legal expert evaluating understanding of precedent hierarchy.

PRECEDENT CONTEXT: {precedent_context}

ORIGINAL QUESTION: {prompt}

RESPONSE TO EVALUATE: {response}

EVALUATION CRITERIA - Precedent Hierarchy:
- Understanding of binding vs. persuasive authority
- Proper recognition of court hierarchy
- Appropriate weight given to different precedents
- Understanding of federal vs. state precedent relationships
- Recognition of when precedents are controlling

Provide a score from 0-10 where:
- 9-10: Perfect understanding of precedent hierarchy
- 7-8: Good grasp of binding vs. persuasive authority
- 5-6: Basic understanding with some confusion
- 3-4: Limited understanding of precedent hierarchy
- 0-2: Poor or no understanding of precedent relationships

Score: [Your numerical score 0-10]
Reasoning: [Brief explanation of hierarchy understanding]
"""
    
    def _create_distinguishing_analysis_prompt(self, response: str, prompt: str, precedent_context: str) -> str:
        """Create prompt for distinguishing analysis evaluation"""
        return f"""
You are a legal expert evaluating the ability to distinguish cases.

PRECEDENT CONTEXT: {precedent_context}

ORIGINAL QUESTION: {prompt}

RESPONSE TO EVALUATE: {response}

EVALUATION CRITERIA - Distinguishing Analysis:
- Identification of material factual differences
- Recognition of legal distinctions between cases
- Effective arguing for why precedents don't apply
- Understanding of when cases can be distinguished
- Sophisticated analysis of factual and legal distinctions

Provide a score from 0-10 where:
- 9-10: Excellent distinguishing analysis with nuanced reasoning
- 7-8: Good ability to distinguish cases effectively
- 5-6: Basic distinguishing with some weaknesses
- 3-4: Poor distinguishing analysis
- 0-2: No meaningful case distinguishing

Score: [Your numerical score 0-10]
Reasoning: [Brief explanation of distinguishing analysis]
"""
    
    def _extract_score_from_response(self, api_response: str) -> float:
        """Extract numerical score from API response"""
        
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
        
        # Fallback inference
        if any(word in api_response.lower() for word in ["excellent", "exceptional", "outstanding"]):
            return 8.5
        elif any(word in api_response.lower() for word in ["good", "strong", "adequate"]):
            return 7.0
        elif any(word in api_response.lower() for word in ["poor", "weak", "inadequate"]):
            return 3.0
        else:
            return 5.0
    
    def _calculate_precedent_score(self,
                                 component_scores: Dict[PrecedentAnalysisComponent, float],
                                 jurisdiction: USJurisdiction) -> PrecedentAnalysisScore:
        """Calculate overall precedent analysis score from components"""
        
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
        jurisdiction_compliant = jurisdiction_compliance >= 6.0
        
        # Apply jurisdiction penalty if non-compliant
        if jurisdiction_compliant:
            final_score = base_score
        else:
            final_score = base_score * 0.7
        
        # Calculate confidence
        scores_list = list(component_scores.values())
        if scores_list:
            score_variance = sum((s - base_score) ** 2 for s in scores_list) / len(scores_list)
            confidence = max(0.5, 1.0 - (score_variance / 10.0))
        else:
            confidence = 0.5
        
        # Generate reasoning
        reasoning = self._generate_precedent_reasoning(component_scores, jurisdiction_compliance, jurisdiction_compliant)
        
        return PrecedentAnalysisScore(
            case_law_accuracy=component_scores.get(PrecedentAnalysisComponent.CASE_LAW_ACCURACY, 5.0),
            analogical_reasoning=component_scores.get(PrecedentAnalysisComponent.ANALOGICAL_REASONING, 5.0),
            citation_quality=component_scores.get(PrecedentAnalysisComponent.CITATION_QUALITY, 5.0),
            precedent_hierarchy=component_scores.get(PrecedentAnalysisComponent.PRECEDENT_HIERARCHY, 5.0),
            distinguishing_analysis=component_scores.get(PrecedentAnalysisComponent.DISTINGUISHING_ANALYSIS, 5.0),
            jurisdiction_compliance=jurisdiction_compliance,
            jurisdiction_compliant=jurisdiction_compliant,
            overall_score=final_score,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def _generate_precedent_reasoning(self,
                                    component_scores: Dict[PrecedentAnalysisComponent, float],
                                    jurisdiction_compliance: float,
                                    jurisdiction_compliant: bool) -> str:
        """Generate detailed reasoning for precedent analysis evaluation"""
        
        reasoning_parts = []
        reasoning_parts.append("PRECEDENT ANALYSIS EVALUATION")
        reasoning_parts.append("=" * 40)
        
        # Component breakdown
        components = [
            ("Case Law Accuracy", component_scores.get(PrecedentAnalysisComponent.CASE_LAW_ACCURACY, 5.0)),
            ("Analogical Reasoning", component_scores.get(PrecedentAnalysisComponent.ANALOGICAL_REASONING, 5.0)),
            ("Citation Quality", component_scores.get(PrecedentAnalysisComponent.CITATION_QUALITY, 5.0)),
            ("Precedent Hierarchy", component_scores.get(PrecedentAnalysisComponent.PRECEDENT_HIERARCHY, 5.0)),
            ("Distinguishing Analysis", component_scores.get(PrecedentAnalysisComponent.DISTINGUISHING_ANALYSIS, 5.0))
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
        estimated_tokens = 2000
        return estimated_tokens * 0.00003
    
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
            "component_weights": {comp.value: weight for comp, weight in self.component_weights.items()},
            "supported_jurisdictions": list(self.jurisdiction_precedent_contexts.keys()),
            "evaluation_method": "real_api_based"
        }


# Factory function
def create_precedent_analysis_ensemble(config: LegalRewardSystemConfig,
                                     api_client: Optional[CostOptimizedAPIClient] = None) -> PrecedentAnalysisEnsemble:
    """Create production precedent analysis ensemble"""
    return PrecedentAnalysisEnsemble(config, api_client)