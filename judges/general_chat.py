"""
Enhanced General Chat Ensemble for Multi-Task Legal Reward System

This module implements the enhanced general chat evaluation ensemble that serves
as the foundation of the hybrid evaluation system (30% weight). It includes
jurisdiction compliance gating and comprehensive chat quality assessment.

Key Features:
- Jurisdiction compliance gating (critical for system integrity)
- Multi-judge evaluation (helpfulness, legal ethics, clarity)
- Hybrid integration with specialized evaluation
- Comprehensive legal chat quality assessment
- API-based evaluation with cost optimization
- Production-ready error handling and fallback mechanisms

The enhanced general chat ensemble acts as both a standalone evaluator for
general chat tasks and as the general chat component of the hybrid evaluation
system for specialized legal tasks.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

# Import core components
from core import (
    LegalRewardEvaluation, JudgeEvaluation, EvaluationMetadata,
    APIProvider, LegalTaskType, USJurisdiction, LegalDomain,
    LegalRewardSystemError, create_error_context
)
from jurisdiction import (
    JurisdictionComplianceJudge, JurisdictionComplianceResult,
    create_production_compliance_judge
)
from judges.base import (
    BaseJudge, BaseJudgeEnsemble, JudgeConfig, EvaluationContext,
    JudgeType, EvaluationStrategy, create_general_chat_judge_config,
    create_individual_judge_config
)


@dataclass
class GeneralChatEvaluationResult:
    """
    Result of enhanced general chat evaluation.
    
    Contains comprehensive chat quality assessment with jurisdiction
    compliance gating and detailed component scoring.
    """
    
    # Overall results
    overall_score: float  # 0.0 to 10.0
    confidence: float     # 0.0 to 1.0
    is_gated: bool       # Whether jurisdiction compliance passed
    
    # Component scores
    helpfulness_score: float = 0.0
    legal_ethics_score: float = 0.0
    clarity_score: float = 0.0
    jurisdiction_compliance_score: float = 0.0
    
    # Component details
    component_evaluations: Dict[str, JudgeEvaluation] = field(default_factory=dict)
    jurisdiction_compliance_result: Optional[JurisdictionComplianceResult] = None
    
    # Evaluation metadata
    evaluation_method: str = "enhanced_general_chat"
    processing_time_ms: float = 0.0
    total_cost: float = 0.0
    
    # Quality assessment
    reasoning: str = ""
    recommendations: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    
    def get_component_breakdown(self) -> Dict[str, float]:
        """Get breakdown of component scores"""
        return {
            "helpfulness": self.helpfulness_score,
            "legal_ethics": self.legal_ethics_score,
            "clarity": self.clarity_score,
            "jurisdiction_compliance": self.jurisdiction_compliance_score
        }
    
    def has_strong_performance(self, threshold: float = 7.0) -> bool:
        """Check if evaluation shows strong performance"""
        return self.overall_score >= threshold and self.is_gated
    
    def get_weakest_component(self) -> Tuple[str, float]:
        """Get the weakest performing component"""
        components = self.get_component_breakdown()
        return min(components.items(), key=lambda x: x[1])


class HelpfulnessJudge(BaseJudge):
    """
    Individual judge for evaluating helpfulness in legal responses.
    
    Assesses how helpful, informative, and useful the response is
    for the user's legal question or situation.
    """
    
    def _initialize_judge(self):
        """Initialize helpfulness-specific components"""
        self.evaluation_criteria = [
            "Directly addresses the user's question",
            "Provides useful and actionable information", 
            "Offers appropriate next steps or guidance",
            "Demonstrates understanding of user needs",
            "Provides sufficient detail without overwhelming"
        ]
        
        self.helpfulness_prompt_template = """
You are an expert legal helpfulness evaluator. Evaluate how helpful this legal response is to the user.

EVALUATION CRITERIA:
1. Direct relevance: Does the response directly address what the user asked?
2. Usefulness: Does it provide practical, actionable information?
3. Completeness: Does it cover the key aspects of the question adequately?
4. Guidance: Does it provide appropriate next steps or direction?
5. User focus: Is it tailored to help the specific user situation?

USER QUESTION/CONTEXT:
{prompt}

LEGAL RESPONSE TO EVALUATE:
{response}

JURISDICTION CONTEXT: {jurisdiction}
TASK TYPE: {task_type}

Please provide:
1. A helpfulness score from 0-10 (where 10 is extremely helpful)
2. Detailed reasoning explaining your score
3. Specific strengths and areas for improvement

Focus on practical helpfulness from the user's perspective. Consider whether a real person with this legal question would find the response genuinely useful.

RESPONSE FORMAT:
Score: [0-10]
Reasoning: [Detailed explanation of helpfulness assessment]
Strengths: [What makes this response helpful]
Improvements: [How helpfulness could be enhanced]
"""
    
    async def evaluate_async(self, response: str, context: EvaluationContext) -> JudgeEvaluation:
        """Evaluate helpfulness of legal response"""
        
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._create_cache_key(response, context)
            if self.cache:
                cached_result = await self._check_cache(cache_key)
                if cached_result:
                    self._update_performance_stats(time.time() - start_time, 0.0, True, True)
                    return cached_result
            
            # Prepare evaluation prompt
            evaluation_prompt = self.helpfulness_prompt_template.format(
                prompt=context.prompt,
                response=response,
                jurisdiction=context.jurisdiction.value,
                task_type=context.task_type.value
            )
            
            # Get evaluation from API
            api_response = await self._get_api_evaluation(evaluation_prompt, context)
            
            # Parse evaluation result
            score, reasoning, confidence = self._parse_helpfulness_evaluation(api_response)
            
            # Create judge evaluation
            judge_evaluation = JudgeEvaluation(
                judge_name=self.config.judge_name,
                score=score,
                reasoning=reasoning,
                confidence=confidence,
                metadata=EvaluationMetadata(
                    evaluation_type="api_based_helpfulness",
                    execution_time=(time.time() - start_time) * 1000,
                    cache_hit=False,
                    additional_info={
                        "task_type": context.task_type.value,
                        "jurisdiction": context.jurisdiction.value,
                        "judge_type": "helpfulness",
                    }
                )
            )
            
            # Cache result
            if self.cache:
                await self._store_cache(cache_key, judge_evaluation)
            
            # Update stats
            self._update_performance_stats(time.time() - start_time, 0.02, True, False)  # Estimated cost
            
            return judge_evaluation
            
        except Exception as e:
            self.logger.error(f"Helpfulness evaluation failed: {e}")
            self._update_performance_stats(time.time() - start_time, 0.0, False, False)
            
            return self._create_fallback_evaluation(
                5.0, f"Helpfulness evaluation failed: {str(e)[:100]}", 0.1
            )
    
    async def _get_api_evaluation(self, prompt: str, context: EvaluationContext) -> str:
        """Get evaluation from API client"""
        # This will be implemented when we have the API client
        # For now, return a placeholder
        return """
Score: 8
Reasoning: The response directly addresses the legal question with relevant information and provides practical guidance. It demonstrates good understanding of the user's needs and offers actionable next steps.
Strengths: Clear explanation, practical advice, appropriate level of detail
Improvements: Could provide more specific examples or resources
"""
    
    def _parse_helpfulness_evaluation(self, api_response: str) -> Tuple[float, str, float]:
        """Parse API response into score, reasoning, and confidence"""
        
        try:
            lines = api_response.strip().split('\n')
            
            score = 5.0  # Default
            reasoning = "Standard helpfulness assessment"
            confidence = 0.8
            
            for line in lines:
                if line.startswith("Score:"):
                    try:
                        score = float(line.split(":", 1)[1].strip())
                        score = max(0.0, min(10.0, score))  # Clamp to valid range
                    except (ValueError, IndexError):
                        pass
                
                elif line.startswith("Reasoning:"):
                    reasoning = line.split(":", 1)[1].strip()
            
            # Estimate confidence based on score and reasoning quality
            if len(reasoning) > 50 and 3.0 <= score <= 9.0:
                confidence = 0.9
            elif len(reasoning) > 20:
                confidence = 0.8
            else:
                confidence = 0.6
            
            return score, reasoning, confidence
            
        except Exception as e:
            self.logger.warning(f"Failed to parse helpfulness evaluation: {e}")
            return 5.0, f"Parse error: {str(e)}", 0.3
    
    async def _check_cache(self, cache_key: str) -> Optional[JudgeEvaluation]:
        """Check cache for evaluation result"""
        # Placeholder for cache implementation
        return None
    
    async def _store_cache(self, cache_key: str, evaluation: JudgeEvaluation):
        """Store evaluation result in cache"""
        # Placeholder for cache implementation
        pass
    
    def _create_fallback_evaluation(self, score: float, reasoning: str, confidence: float) -> JudgeEvaluation:
        """Create fallback evaluation for error cases"""
        
        return JudgeEvaluation(
            judge_name=self.config.judge_name,
            score=score,
            reasoning=reasoning,
            confidence=confidence,
            metadata=EvaluationMetadata(
                evaluation_type="fallback",
                additional_info={"fallback_reason": "API evaluation failed",
                                 "judge_type":"helpfulness",}
            )
        )


class LegalEthicsJudge(BaseJudge):
    """
    Individual judge for evaluating legal ethics compliance in responses.
    
    Assesses adherence to legal professional standards, appropriate
    disclaimers, and ethical considerations in legal advice.
    """
    
    def _initialize_judge(self):
        """Initialize legal ethics specific components"""
        self.evaluation_criteria = [
            "Appropriate legal disclaimers and limitations",
            "Avoidance of unauthorized practice of law",
            "Professional tone and respectful language",
            "Confidentiality and privacy considerations",
            "Ethical boundaries in legal guidance"
        ]
        
        self.legal_ethics_prompt_template = """
You are an expert legal ethics evaluator. Evaluate this legal response for compliance with legal professional ethics and standards.

EVALUATION CRITERIA:
1. Disclaimers: Are appropriate legal disclaimers present?
2. UPL Avoidance: Does it avoid unauthorized practice of law?
3. Professional standards: Is the tone professional and appropriate?
4. Confidentiality: Are privacy considerations respected?
5. Ethical boundaries: Does it maintain appropriate limits on legal advice?

LEGAL RESPONSE TO EVALUATE:
{response}

JURISDICTION CONTEXT: {jurisdiction}
TASK TYPE: {task_type}

Please provide:
1. A legal ethics score from 0-10 (where 10 is fully compliant)
2. Detailed reasoning explaining ethical compliance
3. Any ethical concerns or violations
4. Recommendations for ethical improvement

Consider standard legal professional ethics rules and best practices for legal information services.

RESPONSE FORMAT:
Score: [0-10]
Reasoning: [Detailed explanation of ethics assessment]
Concerns: [Any ethical issues identified]
Recommendations: [Suggestions for ethical compliance]
"""
    
    async def evaluate_async(self, response: str, context: EvaluationContext) -> JudgeEvaluation:
        """Evaluate legal ethics compliance of response"""
        
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._create_cache_key(response, context)
            if self.cache:
                cached_result = await self._check_cache(cache_key)
                if cached_result:
                    self._update_performance_stats(time.time() - start_time, 0.0, True, True)
                    return cached_result
            
            # Prepare evaluation prompt
            evaluation_prompt = self.legal_ethics_prompt_template.format(
                response=response,
                jurisdiction=context.jurisdiction.value,
                task_type=context.task_type.value
            )
            
            # Get evaluation from API
            api_response = await self._get_api_evaluation(evaluation_prompt, context)
            
            # Parse evaluation result
            score, reasoning, confidence = self._parse_ethics_evaluation(api_response)
            
            # Create judge evaluation
            judge_evaluation = JudgeEvaluation(
                judge_name=self.config.judge_name,
                score=score,
                reasoning=reasoning,
                confidence=confidence,
                evaluation_time=(time.time() - start_time) * 1000,
                metadata=EvaluationMetadata(
                    evaluation_type="api_based_ethics",
                    cache_hit=False,
                    additional_info={
                        "judge_type":"legal_ethics",
                        "task_type":context.task_type.value,
                        "jurisdiction":context.jurisdiction.value,
                    }
                )
            )
            
            # Cache result
            if self.cache:
                await self._store_cache(cache_key, judge_evaluation)
            
            # Update stats
            self._update_performance_stats(time.time() - start_time, 0.02, True, False)
            
            return judge_evaluation
            
        except Exception as e:
            self.logger.error(f"Legal ethics evaluation failed: {e}")
            self._update_performance_stats(time.time() - start_time, 0.0, False, False)
            
            return self._create_fallback_evaluation(
                6.0, f"Ethics evaluation failed: {str(e)[:100]}", 0.1
            )
    
    async def _get_api_evaluation(self, prompt: str, context: EvaluationContext) -> str:
        """Get evaluation from API client"""
        # Placeholder for API client integration
        return """
Score: 7
Reasoning: The response includes basic legal disclaimers and maintains appropriate professional boundaries. The tone is respectful and avoids giving specific legal advice.
Concerns: Could benefit from stronger disclaimers about jurisdiction-specific variations
Recommendations: Add more explicit disclaimer about consulting local attorneys
"""
    
    def _parse_ethics_evaluation(self, api_response: str) -> Tuple[float, str, float]:
        """Parse API response into score, reasoning, and confidence"""
        
        try:
            lines = api_response.strip().split('\n')
            
            score = 6.0  # Default slightly above neutral for ethics
            reasoning = "Standard legal ethics assessment"
            confidence = 0.8
            
            for line in lines:
                if line.startswith("Score:"):
                    try:
                        score = float(line.split(":", 1)[1].strip())
                        score = max(0.0, min(10.0, score))
                    except (ValueError, IndexError):
                        pass
                
                elif line.startswith("Reasoning:"):
                    reasoning = line.split(":", 1)[1].strip()
            
            # Higher confidence for ethics evaluation
            if len(reasoning) > 50:
                confidence = 0.9
            elif len(reasoning) > 20:
                confidence = 0.8
            else:
                confidence = 0.7
            
            return score, reasoning, confidence
            
        except Exception as e:
            self.logger.warning(f"Failed to parse ethics evaluation: {e}")
            return 6.0, f"Parse error: {str(e)}", 0.3
    
    async def _check_cache(self, cache_key: str) -> Optional[JudgeEvaluation]:
        """Check cache for evaluation result"""
        return None
    
    async def _store_cache(self, cache_key: str, evaluation: JudgeEvaluation):
        """Store evaluation result in cache"""
        pass
    
    def _create_fallback_evaluation(self, score: float, reasoning: str, confidence: float) -> JudgeEvaluation:
        """Create fallback evaluation for error cases"""
        
        return JudgeEvaluation(
            judge_name=self.config.judge_name,
            score=score,
            reasoning=reasoning,
            confidence=confidence,
            metadata=EvaluationMetadata(
                evaluation_type="fallback",
                additional_info={
                    "fallback_reason": "API evaluation failed",
                    "judge_type":"legal_ethics"
                }
            )
        )


class ClarityJudge(BaseJudge):
    """
    Individual judge for evaluating clarity and comprehensibility of legal responses.
    
    Assesses how clear, well-organized, and understandable the response
    is for users with varying levels of legal knowledge.
    """
    
    def _initialize_judge(self):
        """Initialize clarity-specific components"""
        self.evaluation_criteria = [
            "Clear and understandable language",
            "Logical organization and structure",
            "Appropriate level of detail",
            "Effective use of examples or explanations",
            "Accessibility to non-lawyers"
        ]
        
        self.clarity_prompt_template = """
You are an expert communication evaluator specializing in legal clarity. Evaluate how clear and understandable this legal response is.

EVALUATION CRITERIA:
1. Language clarity: Is the language clear and accessible?
2. Organization: Is the response well-structured and logical?
3. Appropriate detail: Is the level of detail appropriate for the audience?
4. Explanations: Are complex concepts explained effectively?
5. Accessibility: Can non-lawyers understand the key points?

LEGAL RESPONSE TO EVALUATE:
{response}

JURISDICTION CONTEXT: {jurisdiction}
TASK TYPE: {task_type}

Please provide:
1. A clarity score from 0-10 (where 10 is perfectly clear)
2. Detailed reasoning about clarity aspects
3. Specific examples of clear or unclear elements
4. Suggestions for improving clarity

Consider that legal information must be both accurate and accessible to people without legal training.

RESPONSE FORMAT:
Score: [0-10]
Reasoning: [Detailed explanation of clarity assessment]
Clear_Elements: [What makes this response clear]
Unclear_Elements: [What reduces clarity]
Improvements: [Specific suggestions for better clarity]
"""
    
    async def evaluate_async(self, response: str, context: EvaluationContext) -> JudgeEvaluation:
        """Evaluate clarity of legal response"""
        
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._create_cache_key(response, context)
            if self.cache:
                cached_result = await self._check_cache(cache_key)
                if cached_result:
                    self._update_performance_stats(time.time() - start_time, 0.0, True, True)
                    return cached_result
            
            # Prepare evaluation prompt
            evaluation_prompt = self.clarity_prompt_template.format(
                response=response,
                jurisdiction=context.jurisdiction.value,
                task_type=context.task_type.value
            )
            
            # Get evaluation from API
            api_response = await self._get_api_evaluation(evaluation_prompt, context)
            
            # Parse evaluation result
            score, reasoning, confidence = self._parse_clarity_evaluation(api_response)
            
            # Create judge evaluation
            judge_evaluation = JudgeEvaluation(
                judge_name=self.config.judge_name,
                score=score,
                reasoning=reasoning,
                confidence=confidence,
                evaluation_time=(time.time() - start_time) * 1000,
                metadata=EvaluationMetadata(
                    cache_hit=False,
                    evaluation_type="api_based_clarity",
                    additional_info={
                        "task_type":context.task_type.value,
                        "jurisdiction":context.jurisdiction.value,
                        "judge_type":"clarity"
                    }
                )
            )
            
            # Cache result
            if self.cache:
                await self._store_cache(cache_key, judge_evaluation)
            
            # Update stats
            self._update_performance_stats(time.time() - start_time, 0.02, True, False)
            
            return judge_evaluation
            
        except Exception as e:
            self.logger.error(f"Clarity evaluation failed: {e}")
            self._update_performance_stats(time.time() - start_time, 0.0, False, False)
            
            return self._create_fallback_evaluation(
                5.0, f"Clarity evaluation failed: {str(e)[:100]}", 0.1
            )
    
    async def _get_api_evaluation(self, prompt: str, context: EvaluationContext) -> str:
        """Get evaluation from API client"""
        # Placeholder for API client integration
        return """
Score: 8
Reasoning: The response is well-organized with clear language that balances legal accuracy with accessibility. Complex terms are explained and the structure is logical.
Clear_Elements: Good use of plain language, logical flow, helpful examples
Unclear_Elements: Some legal jargon could be simplified further
Improvements: Consider adding a brief summary or key takeaways section
"""
    
    def _parse_clarity_evaluation(self, api_response: str) -> Tuple[float, str, float]:
        """Parse API response into score, reasoning, and confidence"""
        
        try:
            lines = api_response.strip().split('\n')
            
            score = 5.0  # Default neutral
            reasoning = "Standard clarity assessment"
            confidence = 0.8
            
            for line in lines:
                if line.startswith("Score:"):
                    try:
                        score = float(line.split(":", 1)[1].strip())
                        score = max(0.0, min(10.0, score))
                    except (ValueError, IndexError):
                        pass
                
                elif line.startswith("Reasoning:"):
                    reasoning = line.split(":", 1)[1].strip()
            
            # Adjust confidence based on evaluation quality
            if len(reasoning) > 50:
                confidence = 0.9
            elif len(reasoning) > 20:
                confidence = 0.8
            else:
                confidence = 0.7
            
            return score, reasoning, confidence
            
        except Exception as e:
            self.logger.warning(f"Failed to parse clarity evaluation: {e}")
            return 5.0, f"Parse error: {str(e)}", 0.3
    
    async def _check_cache(self, cache_key: str) -> Optional[JudgeEvaluation]:
        """Check cache for evaluation result"""
        return None
    
    async def _store_cache(self, cache_key: str, evaluation: JudgeEvaluation):
        """Store evaluation result in cache"""
        pass
    
    def _create_fallback_evaluation(self, score: float, reasoning: str, confidence: float) -> JudgeEvaluation:
        """Create fallback evaluation for error cases"""
        
        return JudgeEvaluation(
            judge_name=self.config.judge_name,
            score=score,
            reasoning=reasoning,
            confidence=confidence,
            metadata=EvaluationMetadata(
                evaluation_type="fallback",
                additional_info={
                    "fallback_reason": "API evaluation failed",
                    "judge_type":"clarity",
                }
            )
        )


class EnhancedGeneralChatEnsemble(BaseJudgeEnsemble):
    """
    Enhanced general chat ensemble with jurisdiction compliance gating.
    
    Implements the comprehensive general chat evaluation that serves as
    the 30% component in the hybrid evaluation system and provides
    jurisdiction compliance gating for specialized evaluations.
    """
    
    def _initialize_ensemble(self):
        """Initialize enhanced general chat ensemble"""
        
        # Ensemble configuration
        self.ensemble_name = "enhanced_general_chat"
        self.ensemble_config.setdefault("name", self.ensemble_name)
        
        # Component weights (normalized to sum to 1.0)
        self.default_judge_weights = {
            "helpfulness": 0.25,
            "legal_ethics": 0.25,
            "clarity": 0.25,
            "jurisdiction_compliance": 0.25  # Critical gating component
        }
        
        # Update judge weights with defaults if not specified
        for judge_name, default_weight in self.default_judge_weights.items():
            if judge_name not in self.judge_weights:
                self.judge_weights[judge_name] = default_weight
        
        # Gating configuration
        self.gating_enabled = self.ensemble_config.get("enable_gating", True)
        self.gating_threshold = self.ensemble_config.get("gating_threshold", 5.0)
        self.strict_compliance = self.ensemble_config.get("strict_compliance", True)
        
        # Initialize jurisdiction compliance judge
        self.compliance_judge = create_production_compliance_judge({
            "compliance_threshold": 7.0,
            "gating_threshold": self.gating_threshold,
            "strict_federal_domains": self.strict_compliance
        })
        
        # Performance tracking for gating
        self.gating_stats = {
            "total_gating_evaluations": 0,
            "gating_passes": 0,
            "gating_failures": 0,
            "compliance_score_avg": 0.0
        }
    
    def _create_individual_judges(self) -> Dict[str, BaseJudge]:
        """Create individual judges for the ensemble"""
        
        judges = {}
        
        # Create helpfulness judge
        helpfulness_config = create_individual_judge_config(
            "helpfulness", JudgeType.INDIVIDUAL_HELPFULNESS
        )
        judges["helpfulness"] = HelpfulnessJudge(
            helpfulness_config, self.api_client, self.cache, self.rate_limiter
        )
        
        # Create legal ethics judge
        ethics_config = create_individual_judge_config(
            "legal_ethics", JudgeType.INDIVIDUAL_ETHICS
        )
        judges["legal_ethics"] = LegalEthicsJudge(
            ethics_config, self.api_client, self.cache, self.rate_limiter
        )
        
        # Create clarity judge
        clarity_config = create_individual_judge_config(
            "clarity", JudgeType.INDIVIDUAL_CLARITY
        )
        judges["clarity"] = ClarityJudge(
            clarity_config, self.api_client, self.cache, self.rate_limiter
        )
        
        return judges
    
    async def evaluate_with_gating(self, 
                                 response: str, 
                                 context: EvaluationContext) -> GeneralChatEvaluationResult:
        """
        Evaluate response with jurisdiction compliance gating.
        
        This is the main entry point for enhanced general chat evaluation
        that includes the critical jurisdiction compliance gating mechanism.
        
        Args:
            response: Legal response text to evaluate
            context: Evaluation context and metadata
            
        Returns:
            GeneralChatEvaluationResult with gating decision
        """
        
        start_time = time.time()
        self.gating_stats["total_gating_evaluations"] += 1
        
        try:
            # Step 1: Jurisdiction compliance evaluation (gating)
            compliance_result = await self._evaluate_jurisdiction_compliance(response, context)
            
            # Step 2: General chat quality evaluation
            chat_evaluations = await self._evaluate_chat_quality(response, context)
            
            # Step 3: Combine results with gating decision
            final_result = self._combine_results_with_gating(
                compliance_result, chat_evaluations, context
            )
            
            # Step 4: Update gating statistics
            self._update_gating_stats(compliance_result, final_result)
            
            # Step 5: Set processing metadata
            final_result.processing_time_ms = (time.time() - start_time) * 1000
            final_result.evaluation_method = "enhanced_general_chat_with_gating"
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Enhanced general chat evaluation failed: {e}")
            return self._create_error_result(str(e), context)
    
    async def _evaluate_jurisdiction_compliance(self, 
                                              response: str, 
                                              context: EvaluationContext) -> JurisdictionComplianceResult:
        """Evaluate jurisdiction compliance for gating decision"""
        
        try:
            return self.compliance_judge.evaluate_compliance(
                response=response,
                expected_jurisdiction=context.jurisdiction,
                task_type=context.task_type,
                legal_domains=context.legal_domains,
                context=context.user_context
            )
        except Exception as e:
            self.logger.error(f"Jurisdiction compliance evaluation failed: {e}")
            
            # Create fallback compliance result
            from ..jurisdiction.compliance_judge import JurisdictionComplianceResult
            return JurisdictionComplianceResult(
                compliance_score=3.0,  # Below gating threshold
                is_compliant=False,
                gating_decision=False,
                expected_jurisdiction=context.jurisdiction
            )
    
    async def _evaluate_chat_quality(self, 
                                   response: str, 
                                   context: EvaluationContext) -> Dict[str, JudgeEvaluation]:
        """Evaluate general chat quality with individual judges"""
        
        # Get individual judges
        if not self.judges:
            self.judges = self._create_individual_judges()
        
        # Evaluate with each judge
        evaluations = await self._evaluate_with_judges(response, context)
        
        return evaluations
    
    def _combine_results_with_gating(self, 
                                   compliance_result: JurisdictionComplianceResult,
                                   chat_evaluations: Dict[str, JudgeEvaluation],
                                   context: EvaluationContext) -> GeneralChatEvaluationResult:
        """Combine compliance and chat evaluations with gating logic"""
        
        # Extract component scores
        component_scores = {
            "jurisdiction_compliance": compliance_result.compliance_score,
            "helpfulness": chat_evaluations.get("helpfulness", {}).score if "helpfulness" in chat_evaluations else 5.0,
            "legal_ethics": chat_evaluations.get("legal_ethics", {}).score if "legal_ethics" in chat_evaluations else 5.0,
            "clarity": chat_evaluations.get("clarity", {}).score if "clarity" in chat_evaluations else 5.0
        }
        
        # Calculate weighted overall score
        weighted_score = 0.0
        total_weight = 0.0
        
        for component, score in component_scores.items():
            weight = self.judge_weights.get(component, 0.25)
            weighted_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            overall_score = weighted_score / total_weight
        else:
            overall_score = 5.0  # Fallback
        
        # Gating decision logic
        is_gated = self._make_gating_decision(compliance_result, overall_score)
        
        # Calculate overall confidence
        confidences = [compliance_result.confidence]
        for eval in chat_evaluations.values():
            confidences.append(eval.confidence)
        
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        # Create comprehensive reasoning
        reasoning = self._create_comprehensive_reasoning(
            compliance_result, chat_evaluations, component_scores, is_gated
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(compliance_result, chat_evaluations)
        
        return GeneralChatEvaluationResult(
            overall_score=overall_score,
            confidence=overall_confidence,
            is_gated=is_gated,
            helpfulness_score=component_scores["helpfulness"],
            legal_ethics_score=component_scores["legal_ethics"],
            clarity_score=component_scores["clarity"],
            jurisdiction_compliance_score=component_scores["jurisdiction_compliance"],
            component_evaluations=chat_evaluations,
            jurisdiction_compliance_result=compliance_result,
            reasoning=reasoning,
            recommendations=recommendations
        )
    
    def _make_gating_decision(self, 
                            compliance_result: JurisdictionComplianceResult,
                            overall_score: float) -> bool:
        """Make gating decision based on compliance and overall quality"""
        
        if not self.gating_enabled:
            return True  # Always pass if gating disabled
        
        # Primary gating criterion: jurisdiction compliance
        if not compliance_result.gating_decision:
            return False  # Block if compliance fails
        
        # Secondary criterion: overall score threshold
        if overall_score < self.gating_threshold:
            return False  # Block if overall quality too low
        
        # Additional strict compliance check
        if self.strict_compliance and compliance_result.has_critical_violations():
            return False  # Block for critical violations
        
        return True  # Pass gating
    
    def _create_comprehensive_reasoning(self, 
                                      compliance_result: JurisdictionComplianceResult,
                                      chat_evaluations: Dict[str, JudgeEvaluation],
                                      component_scores: Dict[str, float],
                                      is_gated: bool) -> str:
        """Create comprehensive reasoning for the evaluation"""
        
        reasoning_parts = [
            f"Enhanced General Chat Evaluation (Gating: {'PASS' if is_gated else 'BLOCK'})"
        ]
        
        # Add component breakdowns
        for component, score in component_scores.items():
            reasoning_parts.append(f"{component.replace('_', ' ').title()}: {score:.1f}/10.0")
        
        # Add compliance summary
        compliance_summary = compliance_result.get_summary() if compliance_result else "Compliance: N/A"
        reasoning_parts.append(f"Compliance: {compliance_summary}")
        
        # Add key insights from individual judges
        for judge_name, evaluation in chat_evaluations.items():
            if evaluation.reasoning:
                insight = evaluation.reasoning[:80] + "..." if len(evaluation.reasoning) > 80 else evaluation.reasoning
                reasoning_parts.append(f"{judge_name.title()}: {insight}")
        
        return " | ".join(reasoning_parts)
    
    def _generate_recommendations(self, 
                                compliance_result: JurisdictionComplianceResult,
                                chat_evaluations: Dict[str, JudgeEvaluation]) -> List[str]:
        """Generate actionable recommendations for improvement"""
        
        recommendations = []
        
        # Add compliance recommendations
        if compliance_result and compliance_result.recommendations:
            recommendations.extend(compliance_result.recommendations[:2])  # Top 2
        
        # Add component-specific recommendations
        component_scores = {
            "helpfulness": chat_evaluations.get("helpfulness", {}).score if "helpfulness" in chat_evaluations else 5.0,
            "legal_ethics": chat_evaluations.get("legal_ethics", {}).score if "legal_ethics" in chat_evaluations else 5.0,
            "clarity": chat_evaluations.get("clarity", {}).score if "clarity" in chat_evaluations else 5.0
        }
        
        # Find weakest component and add recommendation
        weakest_component = min(component_scores.items(), key=lambda x: x[1])
        component_name, component_score = weakest_component
        
        if component_score < 6.0:
            if component_name == "helpfulness":
                recommendations.append("Improve helpfulness by addressing user questions more directly")
            elif component_name == "legal_ethics":
                recommendations.append("Strengthen legal ethics compliance with better disclaimers")
            elif component_name == "clarity":
                recommendations.append("Enhance clarity with simpler language and better organization")
        
        # Add general best practices
        if not any("jurisdiction" in rec.lower() for rec in recommendations):
            recommendations.append("Consider jurisdiction-specific variations in legal advice")
        
        return recommendations[:3]  # Limit to top 3
    
    def _update_gating_stats(self, 
                           compliance_result: JurisdictionComplianceResult,
                           final_result: GeneralChatEvaluationResult):
        """Update gating performance statistics"""
        
        if final_result.is_gated:
            self.gating_stats["gating_passes"] += 1
        else:
            self.gating_stats["gating_failures"] += 1
        
        # Update compliance score average
        total = self.gating_stats["total_gating_evaluations"]
        current_avg = self.gating_stats["compliance_score_avg"]
        new_score = compliance_result.compliance_score if compliance_result else 5.0
        
        self.gating_stats["compliance_score_avg"] = (
            (current_avg * (total - 1) + new_score) / total if total > 0 else new_score
        )
    
    def _create_error_result(self, error_msg: str, context: EvaluationContext) -> GeneralChatEvaluationResult:
        """Create error fallback result"""
        
        return GeneralChatEvaluationResult(
            overall_score=3.0,  # Below gating threshold
            confidence=0.1,
            is_gated=False,  # Fail gating on errors
            jurisdiction_compliance_score=3.0,
            helpfulness_score=5.0,
            legal_ethics_score=5.0,
            clarity_score=5.0,
            reasoning=f"Evaluation error: {error_msg[:100]}",
            recommendations=["Manual review required due to evaluation error"],
            evaluation_method="error_fallback"
        )
    
    def get_gating_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive gating performance summary"""
        
        total = self.gating_stats["total_gating_evaluations"]
        if total == 0:
            return {"status": "No gating evaluations performed"}
        
        return {
            "gating_configuration": {
                "gating_enabled": self.gating_enabled,
                "gating_threshold": self.gating_threshold,
                "strict_compliance": self.strict_compliance
            },
            "gating_performance": {
                "total_evaluations": total,
                "gating_pass_rate": self.gating_stats["gating_passes"] / total,
                "gating_failure_rate": self.gating_stats["gating_failures"] / total,
                "avg_compliance_score": self.gating_stats["compliance_score_avg"]
            },
            "component_weights": self.judge_weights,
            "ensemble_performance": self.get_ensemble_performance_summary()
        }
    
    # Integration methods for hybrid evaluation system
    
    def get_general_chat_score(self, 
                             response: str, 
                             context: EvaluationContext) -> Tuple[float, bool]:
        """
        Get general chat score and gating decision for hybrid evaluation.
        
        This is the main integration point for the hybrid evaluation system.
        
        Args:
            response: Legal response to evaluate
            context: Evaluation context
            
        Returns:
            Tuple of (general_chat_score, gating_passed)
        """
        
        # Run enhanced general chat evaluation with gating
        result = self.evaluate_ensemble(response, context)
        
        # Convert to expected format
        if isinstance(result, LegalRewardEvaluation):
            # Convert LegalRewardEvaluation to GeneralChatEvaluationResult
            general_result = self._convert_to_general_chat_result(result, context)
        else:
            general_result = result
        
        return general_result.overall_score, general_result.is_gated
    
    def _convert_to_general_chat_result(self, 
                                      legal_eval: LegalRewardEvaluation,
                                      context: EvaluationContext) -> GeneralChatEvaluationResult:
        """Convert LegalRewardEvaluation to GeneralChatEvaluationResult"""
        
        # Extract component scores from judge evaluations
        component_scores = {}
        for judge_eval in legal_eval.judge_evaluations:
            if judge_eval.judge_type in ["helpfulness", "legal_ethics", "clarity"]:
                component_scores[judge_eval.judge_type] = judge_eval.score
        
        # Default gating to pass for legacy evaluations
        is_gated = True
        
        return GeneralChatEvaluationResult(
            overall_score=legal_eval.overall_score,
            confidence=legal_eval.confidence,
            is_gated=is_gated,
            helpfulness_score=component_scores.get("helpfulness", 5.0),
            legal_ethics_score=component_scores.get("legal_ethics", 5.0),
            clarity_score=component_scores.get("clarity", 5.0),
            jurisdiction_compliance_score=7.0,  # Default pass
            component_evaluations={eval.judge_type: eval for eval in legal_eval.judge_evaluations},
            reasoning=legal_eval.reasoning,
            evaluation_method="converted_from_legal_evaluation"
        )


# Factory functions for different use cases

def create_production_general_chat_ensemble(api_client=None, 
                                          cache=None, 
                                          rate_limiter=None) -> EnhancedGeneralChatEnsemble:
    """
    Create production-ready enhanced general chat ensemble.
    
    Args:
        api_client: API client for judge evaluations
        cache: Cache instance for performance optimization
        rate_limiter: Rate limiter for API management
        
    Returns:
        Configured EnhancedGeneralChatEnsemble for production use
    """
    
    config = {
        "name": "production_general_chat",
        "evaluation_strategy": "weighted_average",
        "enable_gating": True,
        "gating_threshold": 5.0,
        "strict_compliance": True,
        "judge_weights": {
            "helpfulness": 0.25,
            "legal_ethics": 0.25,
            "clarity": 0.25,
            "jurisdiction_compliance": 0.25
        }
    }
    
    return EnhancedGeneralChatEnsemble(config, api_client, cache, rate_limiter)


def create_development_general_chat_ensemble(api_client=None) -> EnhancedGeneralChatEnsemble:
    """
    Create development-friendly enhanced general chat ensemble.
    
    Args:
        api_client: API client for judge evaluations
        
    Returns:
        Configured EnhancedGeneralChatEnsemble for development use
    """
    
    config = {
        "name": "development_general_chat",
        "evaluation_strategy": "weighted_average",
        "enable_gating": True,
        "gating_threshold": 3.0,  # Lower threshold for development
        "strict_compliance": False,
        "judge_weights": {
            "helpfulness": 0.3,
            "legal_ethics": 0.25,
            "clarity": 0.25,
            "jurisdiction_compliance": 0.2
        }
    }
    
    return EnhancedGeneralChatEnsemble(config, api_client, None, None)


def create_strict_general_chat_ensemble(api_client=None, 
                                      cache=None, 
                                      rate_limiter=None) -> EnhancedGeneralChatEnsemble:
    """
    Create strict enhanced general chat ensemble with high standards.
    
    Args:
        api_client: API client for judge evaluations
        cache: Cache instance for performance optimization
        rate_limiter: Rate limiter for API management
        
    Returns:
        Configured EnhancedGeneralChatEnsemble for strict evaluation
    """
    
    config = {
        "name": "strict_general_chat",
        "evaluation_strategy": "consensus",
        "enable_gating": True,
        "gating_threshold": 7.0,  # High threshold
        "strict_compliance": True,
        "judge_weights": {
            "helpfulness": 0.2,
            "legal_ethics": 0.3,  # Higher weight on ethics
            "clarity": 0.2,
            "jurisdiction_compliance": 0.3  # Higher weight on compliance
        }
    }
    
    return EnhancedGeneralChatEnsemble(config, api_client, cache, rate_limiter)


# Convenience functions for integration

def evaluate_general_chat_with_gating(response: str,
                                    task_type: LegalTaskType,
                                    jurisdiction: USJurisdiction,
                                    prompt: str = "",
                                    legal_domains: Optional[List[LegalDomain]] = None) -> GeneralChatEvaluationResult:
    """
    Convenience function for general chat evaluation with gating.
    
    Args:
        response: Legal response to evaluate
        task_type: Type of legal task
        jurisdiction: Jurisdiction context
        prompt: Original prompt/question
        legal_domains: Legal domains if known
        
    Returns:
        GeneralChatEvaluationResult with gating decision
    """
    
    from .base import create_evaluation_context
    
    # Create evaluation context
    context = create_evaluation_context(task_type, jurisdiction, prompt, legal_domains)
    
    # Create ensemble and evaluate
    ensemble = create_production_general_chat_ensemble()
    
    # Note: This will need API client integration to work fully
    return ensemble.evaluate_with_gating(response, context)
