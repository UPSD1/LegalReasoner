# Create: legal_reward_system/judges/specialized/opinion_generation.py

"""
Opinion Generation Ensemble - Real Implementation

Provides specialized evaluation for opinion generation tasks including persuasive reasoning,
legal argument structure, client advocacy, writing clarity, and ethical considerations.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any
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


class OpinionGenerationComponent(Enum):
    """Components of opinion generation evaluation"""
    PERSUASIVE_REASONING = "persuasive_reasoning"
    ARGUMENT_STRUCTURE = "argument_structure"
    CLIENT_ADVOCACY = "client_advocacy"
    WRITING_CLARITY = "writing_clarity"
    ETHICAL_CONSIDERATIONS = "ethical_considerations"


@dataclass
class OpinionGenerationScore:
    """Structured score for opinion generation evaluation"""
    persuasive_reasoning: float
    argument_structure: float
    client_advocacy: float
    writing_clarity: float
    ethical_considerations: float
    jurisdiction_compliance: float
    jurisdiction_compliant: bool
    overall_score: float
    confidence: float
    reasoning: str


class OpinionGenerationEnsemble(BaseJudgeEnsemble):
    """
    Real implementation of opinion generation evaluation ensemble.
    
    Evaluates legal responses for opinion generation quality including:
    - Persuasive reasoning and argumentation
    - Clear legal argument structure
    - Effective client advocacy
    - Writing clarity and organization
    - Ethical considerations and professional responsibility
    """
    
    def __init__(self, 
                 config: LegalRewardSystemConfig,
                 api_client: Optional[CostOptimizedAPIClient] = None):
        
        super().__init__(
            ensemble_name="opinion_generation_ensemble",
            task_type=LegalTaskType.OPINION_GENERATION
        )
        
        self.config = config
        self.api_client = api_client or CostOptimizedAPIClient(config)
        self.logger = get_legal_logger("opinion_generation_ensemble")
        
        # Component weights for ensemble aggregation
        self.component_weights = {
            OpinionGenerationComponent.PERSUASIVE_REASONING: 0.30,
            OpinionGenerationComponent.ARGUMENT_STRUCTURE: 0.25,
            OpinionGenerationComponent.CLIENT_ADVOCACY: 0.20,
            OpinionGenerationComponent.WRITING_CLARITY: 0.15,
            OpinionGenerationComponent.ETHICAL_CONSIDERATIONS: 0.10
        }
        
        # Jurisdiction-specific advocacy contexts
        self.jurisdiction_advocacy_contexts = {
            USJurisdiction.FEDERAL: "federal court advocacy, federal rules of evidence and procedure",
            USJurisdiction.CALIFORNIA: "California state court advocacy, California Rules of Court, California law",
            USJurisdiction.NEW_YORK: "New York state court advocacy, New York CPLR, New York practice",
            USJurisdiction.TEXAS: "Texas state court advocacy, Texas Rules of Civil Procedure, Texas law",
            USJurisdiction.FLORIDA: "Florida state court advocacy, Florida Rules of Civil Procedure, Florida law",
            USJurisdiction.GENERAL: "general legal advocacy principles, universal argument techniques"
        }
        
        # Performance tracking
        self.evaluation_count = 0
        self.total_evaluation_time = 0.0
        self.api_call_count = 0
        self.cache_hit_count = 0
        self.advocacy_effectiveness_patterns = []
        
        self.logger.info("Initialized OpinionGenerationEnsemble with real API-based evaluation")
    
    async def evaluate_response(self,
                              response: str,
                              task_type: LegalTaskType,
                              jurisdiction: USJurisdiction,
                              prompt: str = "",
                              context: Optional[Dict[str, Any]] = None) -> JudgeEvaluation:
        """
        Evaluate response using real opinion generation analysis.
        
        Args:
            response: Legal response to evaluate
            task_type: Should be OPINION_GENERATION
            jurisdiction: US jurisdiction for evaluation context
            prompt: Original prompt/question
            context: Additional evaluation context
            
        Returns:
            Complete opinion generation evaluation
        """
        
        start_time = time.time()
        
        try:
            # Get jurisdiction-specific advocacy context
            advocacy_context = self.jurisdiction_advocacy_contexts.get(
                jurisdiction, self.jurisdiction_advocacy_contexts[USJurisdiction.GENERAL]
            )
            
            # Evaluate all opinion generation components
            component_scores = await self._evaluate_all_opinion_components(
                response, prompt, advocacy_context, context
            )
            
            # Calculate overall opinion generation score
            opinion_score = self._calculate_opinion_score(component_scores, jurisdiction)
            
            # Create evaluation result
            evaluation = JudgeEvaluation(
                judge_name=self.ensemble_name,
                task_type=task_type,
                score=opinion_score.overall_score,
                confidence=opinion_score.confidence,
                reasoning=opinion_score.reasoning,
                metadata=EvaluationMetadata(
                    evaluation_time=time.time() - start_time,
                    model_used=self.api_client.get_current_model(),
                    cost=self._estimate_evaluation_cost(),
                    api_provider=self.api_client.get_current_provider().value,
                    tokens_used=len(response.split()) * 2,
                    jurisdiction=jurisdiction.value,
                    additional_context={
                        "component_scores": {
                            comp.value: getattr(opinion_score, comp.value) 
                            for comp in OpinionGenerationComponent
                        },
                        "jurisdiction_compliance": opinion_score.jurisdiction_compliance,
                        "jurisdiction_compliant": opinion_score.jurisdiction_compliant,
                        "ensemble_type": "opinion_generation",
                        "evaluation_method": "real_api_based"
                    }
                )
            )
            
            # Track performance
            self.evaluation_count += 1
            self.total_evaluation_time += time.time() - start_time
            
            return evaluation
            
        except Exception as e:
            self.logger.error(f"Error in opinion generation evaluation: {e}")
            raise LegalRewardSystemError(
                f"Opinion generation evaluation failed: {e}",
                error_context=create_error_context(
                    "opinion_generation_ensemble", 
                    "evaluate_response"
                )
            )
    
    async def _evaluate_all_opinion_components(self,
                                             response: str,
                                             prompt: str,
                                             advocacy_context: str,
                                             context: Optional[Dict]) -> Dict[OpinionGenerationComponent, float]:
        """Evaluate all opinion generation components using real API calls"""
        
        # Create evaluation tasks for all components
        evaluation_tasks = {}
        
        for component in OpinionGenerationComponent:
            evaluation_tasks[component] = self._evaluate_opinion_component(
                component, response, prompt, advocacy_context, context
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
    
    async def _evaluate_opinion_component(self,
                                        component: OpinionGenerationComponent,
                                        response: str,
                                        prompt: str,
                                        advocacy_context: str,
                                        context: Optional[Dict]) -> float:
        """Evaluate a specific opinion generation component"""
        
        # Component-specific evaluation prompts
        evaluation_prompts = {
            OpinionGenerationComponent.PERSUASIVE_REASONING: self._create_persuasive_reasoning_prompt,
            OpinionGenerationComponent.ARGUMENT_STRUCTURE: self._create_argument_structure_prompt,
            OpinionGenerationComponent.CLIENT_ADVOCACY: self._create_client_advocacy_prompt,
            OpinionGenerationComponent.WRITING_CLARITY: self._create_writing_clarity_prompt,
            OpinionGenerationComponent.ETHICAL_CONSIDERATIONS: self._create_ethical_considerations_prompt
        }
        
        # Create component-specific evaluation prompt
        eval_prompt = evaluation_prompts[component](response, prompt, advocacy_context)
        
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
    
    def _create_persuasive_reasoning_prompt(self, response: str, prompt: str, advocacy_context: str) -> str:
        """Create prompt for persuasive reasoning evaluation"""
        return f"""
You are a legal expert evaluating persuasive reasoning and argumentation.

ADVOCACY CONTEXT: {advocacy_context}

ORIGINAL QUESTION: {prompt}

RESPONSE TO EVALUATE: {response}

EVALUATION CRITERIA - Persuasive Reasoning:
- Strength and logic of legal arguments
- Use of compelling evidence and authority
- Anticipation and refutation of counterarguments
- Logical flow from premises to conclusions
- Persuasive force and rhetorical effectiveness

Provide a score from 0-10 where:
- 9-10: Exceptionally persuasive with compelling logic
- 7-8: Strong persuasive arguments with good reasoning
- 5-6: Adequate persuasion but some weak arguments
- 3-4: Limited persuasive power or flawed reasoning
- 0-2: Weak or unconvincing arguments

Score: [Your numerical score 0-10]
Reasoning: [Brief explanation of persuasive strength]
"""
    
    def _create_argument_structure_prompt(self, response: str, prompt: str, advocacy_context: str) -> str:
        """Create prompt for argument structure evaluation"""
        return f"""
You are a legal expert evaluating legal argument structure and organization.

ADVOCACY CONTEXT: {advocacy_context}

ORIGINAL QUESTION: {prompt}

RESPONSE TO EVALUATE: {response}

EVALUATION CRITERIA - Argument Structure:
- Clear introduction and thesis statement
- Logical organization of supporting arguments
- Effective use of headings and transitions
- Strong conclusion that reinforces main points
- Overall coherence and flow

Provide a score from 0-10 where:
- 9-10: Perfect argument structure with excellent organization
- 7-8: Strong structure with good logical flow
- 5-6: Adequate structure but some organizational issues
- 3-4: Poor structure or confusing organization
- 0-2: No clear argument structure

Score: [Your numerical score 0-10]
Reasoning: [Brief explanation of structural quality]
"""
    
    def _create_client_advocacy_prompt(self, response: str, prompt: str, advocacy_context: str) -> str:
        """Create prompt for client advocacy evaluation"""
        return f"""
You are a legal expert evaluating client advocacy effectiveness.

ADVOCACY CONTEXT: {advocacy_context}

ORIGINAL QUESTION: {prompt}

RESPONSE TO EVALUATE: {response}

EVALUATION CRITERIA - Client Advocacy:
- Clear representation of client interests
- Effective presentation of client's position
- Strategic thinking about client goals
- Practical advice and recommendations
- Zealous advocacy within ethical bounds

Provide a score from 0-10 where:
- 9-10: Excellent client advocacy with strategic thinking
- 7-8: Strong advocacy with good client focus
- 5-6: Adequate advocacy but some missed opportunities
- 3-4: Limited advocacy effectiveness
- 0-2: Poor or absent client advocacy

Score: [Your numerical score 0-10]
Reasoning: [Brief explanation of advocacy effectiveness]
"""
    
    def _create_writing_clarity_prompt(self, response: str, prompt: str, advocacy_context: str) -> str:
        """Create prompt for writing clarity evaluation"""
        return f"""
You are a legal expert evaluating writing clarity and communication.

ADVOCACY CONTEXT: {advocacy_context}

ORIGINAL QUESTION: {prompt}

RESPONSE TO EVALUATE: {response}

EVALUATION CRITERIA - Writing Clarity:
- Clear and accessible language
- Proper grammar and syntax
- Effective use of legal terminology
- Concise and direct communication
- Professional tone and style

Provide a score from 0-10 where:
- 9-10: Exceptionally clear and professional writing
- 7-8: Clear writing with good communication
- 5-6: Generally clear but some issues
- 3-4: Unclear or confusing writing
- 0-2: Very poor writing quality

Score: [Your numerical score 0-10]
Reasoning: [Brief explanation of writing quality]
"""
    
    def _create_ethical_considerations_prompt(self, response: str, prompt: str, advocacy_context: str) -> str:
        """Create prompt for ethical considerations evaluation"""
        return f"""
You are a legal expert evaluating ethical considerations and professional responsibility.

ADVOCACY CONTEXT: {advocacy_context}

ORIGINAL QUESTION: {prompt}

RESPONSE TO EVALUATE: {response}

EVALUATION CRITERIA - Ethical Considerations:
- Compliance with professional responsibility rules
- Appropriate confidentiality considerations
- Avoidance of conflicts of interest
- Honest representation of law and facts
- Respect for opposing parties and tribunal

Provide a score from 0-10 where:
- 9-10: Exemplary ethical awareness and compliance
- 7-8: Good ethical considerations
- 5-6: Adequate ethical awareness
- 3-4: Some ethical concerns or oversights
- 0-2: Serious ethical issues

Score: [Your numerical score 0-10]
Reasoning: [Brief explanation of ethical assessment]
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
    
    def _calculate_opinion_score(self,
                               component_scores: Dict[OpinionGenerationComponent, float],
                               jurisdiction: USJurisdiction) -> OpinionGenerationScore:
        """Calculate overall opinion generation score from components"""
        
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
        reasoning = self._generate_opinion_reasoning(component_scores, jurisdiction_compliance, jurisdiction_compliant)
        
        return OpinionGenerationScore(
            persuasive_reasoning=component_scores.get(OpinionGenerationComponent.PERSUASIVE_REASONING, 5.0),
            argument_structure=component_scores.get(OpinionGenerationComponent.ARGUMENT_STRUCTURE, 5.0),
            client_advocacy=component_scores.get(OpinionGenerationComponent.CLIENT_ADVOCACY, 5.0),
            writing_clarity=component_scores.get(OpinionGenerationComponent.WRITING_CLARITY, 5.0),
            ethical_considerations=component_scores.get(OpinionGenerationComponent.ETHICAL_CONSIDERATIONS, 5.0),
            jurisdiction_compliance=jurisdiction_compliance,
            jurisdiction_compliant=jurisdiction_compliant,
            overall_score=final_score,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def _generate_opinion_reasoning(self,
                                  component_scores: Dict[OpinionGenerationComponent, float],
                                  jurisdiction_compliance: float,
                                  jurisdiction_compliant: bool) -> str:
        """Generate detailed reasoning for opinion generation evaluation"""
        
        reasoning_parts = []
        reasoning_parts.append("OPINION GENERATION EVALUATION")
        reasoning_parts.append("=" * 40)
        
        # Component breakdown
        components = [
            ("Persuasive Reasoning", component_scores.get(OpinionGenerationComponent.PERSUASIVE_REASONING, 5.0)),
            ("Argument Structure", component_scores.get(OpinionGenerationComponent.ARGUMENT_STRUCTURE, 5.0)),
            ("Client Advocacy", component_scores.get(OpinionGenerationComponent.CLIENT_ADVOCACY, 5.0)),
            ("Writing Clarity", component_scores.get(OpinionGenerationComponent.WRITING_CLARITY, 5.0)),
            ("Ethical Considerations", component_scores.get(OpinionGenerationComponent.ETHICAL_CONSIDERATIONS, 5.0))
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
            "supported_jurisdictions": list(self.jurisdiction_advocacy_contexts.keys()),
            "evaluation_method": "real_api_based"
        }


# Factory function
def create_opinion_generation_ensemble(config: LegalRewardSystemConfig,
                                     api_client: Optional[CostOptimizedAPIClient] = None) -> OpinionGenerationEnsemble:
    """Create production opinion generation ensemble"""
    return OpinionGenerationEnsemble(config, api_client)