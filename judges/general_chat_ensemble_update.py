"""
Enhanced General Chat Ensemble - Updated with Professional Prompt Templates

This ensemble provides comprehensive general legal chat evaluation with jurisdiction
compliance integration. It serves as both standalone evaluation for general chat
tasks and as the 30% chat quality component in hybrid evaluation for specialized tasks.

CRITICAL FEATURE: Includes jurisdiction compliance as a gating mechanism - responses
with poor jurisdiction compliance (score < 3.0) may be penalized or rejected.

Updated Features:
- Professional prompt template integration from config/prompts/general_chat.py
- Sophisticated jurisdiction compliance gating from config/prompts/jurisdiction_compliance.py
- Enhanced evaluation components with jurisdiction-specific context
- Optimized caching and performance for GRPO training
- Full VERL integration compatibility

Evaluation Components (Equal Weights + GATING):
- Helpfulness (25%): Practical utility and completeness
- Legal Ethics (25%): Professional responsibility and compliance
- Clarity (25%): Communication effectiveness and comprehensibility
- Jurisdiction Compliance (25%): US legal system accuracy (CRITICAL GATING)

Author: Legal Reward System Team
Version: 1.1.0 (Updated with prompt templates and gating)
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Import core system components
from ..core.data_structures import LegalDataPoint, EnsembleScore
from ..core.enums import LegalTaskType
from ..jurisdiction.us_system import USJurisdiction
from ..config.settings import LegalRewardSystemConfig
from .base import BaseJudgeEnsemble
from .api_client import CostOptimizedAPIClient

# Import prompt template system
from ..config.prompts import (
    GeneralChatPromptType,
    GeneralChatEvaluationContext,
    get_general_chat_prompt,
    GeneralChatPromptManager,
    JurisdictionCompliancePromptType,
    JurisdictionComplianceContext,
    get_jurisdiction_compliance_prompt,
    JurisdictionComplianceManager,
    assess_gating_failure
)

# Import jurisdiction compliance judge
from ..jurisdiction.compliance_judge import JurisdictionComplianceJudge


class GeneralChatComponent(Enum):
    """Components of general chat evaluation"""
    HELPFULNESS = "helpfulness"
    LEGAL_ETHICS = "legal_ethics"
    CLARITY = "clarity"
    JURISDICTION_COMPLIANCE = "jurisdiction_compliance"  # GATING COMPONENT


@dataclass
class EnhancedGeneralChatScore:
    """Detailed scoring for enhanced general chat evaluation"""
    
    # Component scores (0-10 scale)
    helpfulness: float = 0.0
    legal_ethics: float = 0.0
    clarity: float = 0.0
    jurisdiction_compliance: float = 0.0  # CRITICAL GATING SCORE
    
    # Aggregate metrics
    overall_score: float = 0.0
    confidence: float = 0.0
    
    # Gating assessment
    is_gating_failure: bool = False
    gating_failure_message: str = ""
    
    # Component weights (equal weights)
    component_weights: Dict[str, float] = field(default_factory=lambda: {
        "helpfulness": 0.25,
        "legal_ethics": 0.25,
        "clarity": 0.25,
        "jurisdiction_compliance": 0.25
    })
    
    # Detailed reasoning
    component_reasoning: Dict[str, str] = field(default_factory=dict)
    overall_reasoning: str = ""
    
    # Performance metadata
    evaluation_time_seconds: float = 0.0
    api_calls_made: int = 0
    cache_hits: int = 0
    
    def calculate_overall_score(self) -> float:
        """Calculate weighted overall score with gating consideration"""
        
        component_scores = {
            "helpfulness": self.helpfulness,
            "legal_ethics": self.legal_ethics,
            "clarity": self.clarity,
            "jurisdiction_compliance": self.jurisdiction_compliance
        }
        
        # Calculate weighted average
        weighted_sum = sum(
            score * self.component_weights.get(component, 0.0)
            for component, score in component_scores.items()
        )
        
        # Apply gating penalty if jurisdiction compliance failed
        if self.is_gating_failure:
            # Apply significant penalty for gating failures
            gating_penalty = 0.5  # 50% penalty
            weighted_sum *= (1.0 - gating_penalty)
        
        self.overall_score = weighted_sum
        return self.overall_score
    
    def calculate_confidence(self) -> float:
        """Calculate confidence with heavy weighting on jurisdiction compliance"""
        
        component_scores = [
            self.helpfulness, self.legal_ethics, self.clarity, self.jurisdiction_compliance
        ]
        
        # Base confidence from score consistency
        if len(component_scores) > 1:
            score_variance = sum((score - self.overall_score) ** 2 for score in component_scores) / len(component_scores)
            consistency_confidence = max(0.0, 1.0 - (score_variance / 10.0))
        else:
            consistency_confidence = 0.5
        
        # Jurisdiction compliance heavily impacts confidence
        jurisdiction_confidence = self.jurisdiction_compliance / 10.0
        
        # If gating failure, confidence is severely reduced
        if self.is_gating_failure:
            jurisdiction_confidence *= 0.3  # Severe confidence reduction
        
        # Combined confidence (60% consistency + 40% jurisdiction for gating importance)
        self.confidence = 0.6 * consistency_confidence + 0.4 * jurisdiction_confidence
        
        return self.confidence
    
    def assess_gating_failure(self, gating_threshold: float = 3.0):
        """Assess if jurisdiction compliance represents a gating failure"""
        
        self.is_gating_failure, self.gating_failure_message = assess_gating_failure(self.jurisdiction_compliance)


class EnhancedGeneralChatEnsemble(BaseJudgeEnsemble):
    """
    Enhanced general chat ensemble with jurisdiction compliance integration.
    
    This ensemble evaluates general legal chat responses using professional prompt
    templates and includes critical jurisdiction compliance gating functionality.
    
    Used both for pure general chat evaluation and as the 30% chat quality
    component in hybrid evaluation for specialized tasks.
    """
    
    def __init__(self, config: LegalRewardSystemConfig,
                 api_client: Optional[CostOptimizedAPIClient] = None):
        super().__init__("enhanced_general_chat", LegalTaskType.GENERAL_CHAT)
        
        self.config = config
        self.api_client = api_client or CostOptimizedAPIClient(config)
        
        # Initialize prompt managers
        self.general_chat_manager = GeneralChatPromptManager()
        self.jurisdiction_compliance_manager = JurisdictionComplianceManager()
        
        # Initialize jurisdiction compliance judge
        self.jurisdiction_judge = JurisdictionComplianceJudge(config)
        
        # Component weights (equal weights with special gating consideration)
        self.component_weights = {
            GeneralChatComponent.HELPFULNESS: 0.25,
            GeneralChatComponent.LEGAL_ETHICS: 0.25,
            GeneralChatComponent.CLARITY: 0.25,
            GeneralChatComponent.JURISDICTION_COMPLIANCE: 0.25  # GATING COMPONENT
        }
        
        # Gating configuration
        self.gating_threshold = config.hybrid_evaluation.jurisdiction_compliance_threshold
        self.gating_penalty = config.hybrid_evaluation.jurisdiction_failure_penalty
        
        # Performance tracking
        self.evaluation_count = 0
        self.total_evaluation_time = 0.0
        self.api_call_count = 0
        self.cache_hit_count = 0
        self.gating_failure_count = 0
        
        # Validate prompt system on initialization
        self._validate_prompt_system()
    
    def _validate_prompt_system(self):
        """Validate that prompt templates are available"""
        
        try:
            # Test general chat prompt generation
            test_context = GeneralChatEvaluationContext(
                query="Test query",
                response="Test response",
                jurisdiction=USJurisdiction.FEDERAL,
                legal_domain="Test",
                user_type="general_public"
            )
            
            for prompt_type in GeneralChatPromptType:
                test_prompt = get_general_chat_prompt(prompt_type, test_context)
                if not test_prompt:
                    raise ValueError(f"General chat prompt generation failed for {prompt_type}")
            
            # Test jurisdiction compliance prompt generation
            test_compliance_context = JurisdictionComplianceContext(
                query="Test query",
                response="Test response",
                jurisdiction=USJurisdiction.FEDERAL,
                task_type=LegalTaskType.GENERAL_CHAT,
                legal_domain="Test"
            )
            
            test_compliance_prompt = get_jurisdiction_compliance_prompt(
                JurisdictionCompliancePromptType.LEGAL_FRAMEWORK_ACCURACY,
                test_compliance_context
            )
            
            if not test_compliance_prompt:
                raise ValueError("Jurisdiction compliance prompt generation failed")
            
            self.logger.info("Enhanced general chat prompt system validation passed")
            
        except Exception as e:
            self.logger.error(f"Enhanced general chat prompt system validation failed: {str(e)}")
            if self.config.environment.value == "production":
                raise
    
    async def evaluate(self, data_point: LegalDataPoint) -> EnsembleScore:
        """
        Comprehensive general chat evaluation with jurisdiction compliance gating.
        
        Args:
            data_point: Legal data point to evaluate
            
        Returns:
            Comprehensive ensemble score with gating assessment
        """
        
        start_time = asyncio.get_event_loop().time()
        self.evaluation_count += 1
        
        try:
            # Create evaluation context
            evaluation_context = self._create_evaluation_context(data_point)
            
            # Evaluate all components in parallel for performance
            component_tasks = [
                self._evaluate_helpfulness(evaluation_context),
                self._evaluate_legal_ethics(evaluation_context),
                self._evaluate_clarity(evaluation_context),
                self._evaluate_jurisdiction_compliance(evaluation_context)
            ]
            
            # Execute all evaluations
            component_results = await asyncio.gather(*component_tasks, return_exceptions=True)
            
            # Process results
            chat_score = EnhancedGeneralChatScore()
            component_scores = {}
            
            components = [
                (GeneralChatComponent.HELPFULNESS, "helpfulness"),
                (GeneralChatComponent.LEGAL_ETHICS, "legal_ethics"),
                (GeneralChatComponent.CLARITY, "clarity"),
                (GeneralChatComponent.JURISDICTION_COMPLIANCE, "jurisdiction_compliance")
            ]
            
            for i, (component, attr_name) in enumerate(components):
                result = component_results[i]
                
                if isinstance(result, Exception):
                    self.logger.error(f"Component evaluation failed for {component}: {str(result)}")
                    score = 0.0
                    reasoning = f"Evaluation failed: {str(result)}"
                else:
                    score, reasoning = result
                
                # Set component score
                setattr(chat_score, attr_name, score)
                component_scores[component.value] = score
                chat_score.component_reasoning[component.value] = reasoning
            
            # Assess gating failure BEFORE calculating overall score
            chat_score.assess_gating_failure(self.gating_threshold)
            
            if chat_score.is_gating_failure:
                self.gating_failure_count += 1
                self.logger.warning(f"Gating failure detected: {chat_score.gating_failure_message}")
            
            # Calculate aggregate metrics (includes gating penalty)
            chat_score.calculate_overall_score()
            chat_score.calculate_confidence()
            
            # Generate overall reasoning
            chat_score.overall_reasoning = self._generate_overall_reasoning(chat_score, data_point)
            
            # Performance metadata
            end_time = asyncio.get_event_loop().time()
            chat_score.evaluation_time_seconds = end_time - start_time
            chat_score.api_calls_made = len(component_tasks)
            
            self.total_evaluation_time += chat_score.evaluation_time_seconds
            self.api_call_count += chat_score.api_calls_made
            
            # Create ensemble score
            ensemble_score = EnsembleScore(
                score=chat_score.overall_score,
                confidence=chat_score.confidence,
                reasoning=chat_score.overall_reasoning,
                component_scores=component_scores,
                metadata={
                    "jurisdiction_compliance": chat_score.jurisdiction_compliance,
                    "is_gating_failure": chat_score.is_gating_failure,
                    "gating_failure_message": chat_score.gating_failure_message,
                    "component_reasoning": chat_score.component_reasoning,
                    "evaluation_time": chat_score.evaluation_time_seconds,
                    "api_calls": chat_score.api_calls_made,
                    "gating_threshold": self.gating_threshold
                }
            )
            
            self.logger.debug(f"Enhanced general chat evaluation completed: score={chat_score.overall_score:.2f}, confidence={chat_score.confidence:.2f}, gating_failure={chat_score.is_gating_failure}")
            
            return ensemble_score
            
        except Exception as e:
            self.logger.error(f"Enhanced general chat evaluation failed: {str(e)}")
            
            # Return failed evaluation score
            return EnsembleScore(
                score=0.0,
                confidence=0.0,
                reasoning=f"Evaluation failed: {str(e)}",
                component_scores={component.value: 0.0 for component in GeneralChatComponent},
                metadata={"error": str(e)}
            )
    
    def _create_evaluation_context(self, data_point: LegalDataPoint) -> GeneralChatEvaluationContext:
        """Create evaluation context for general chat assessment"""
        
        return GeneralChatEvaluationContext(
            query=data_point.query,
            response=data_point.response,
            jurisdiction=data_point.jurisdiction,
            legal_domain=getattr(data_point, 'legal_domain', 'General'),
            task_context=f"{data_point.task_type.value} evaluation",
            user_type=getattr(data_point, 'user_type', 'general_public')
        )
    
    async def _evaluate_helpfulness(self, context: GeneralChatEvaluationContext) -> Tuple[float, str]:
        """Evaluate helpfulness using professional prompt templates"""
        
        try:
            # Get professional prompt
            prompt = get_general_chat_prompt(GeneralChatPromptType.HELPFULNESS, context)
            
            # Evaluate using API client
            result = await self.api_client.evaluate_legal_response(
                query=context.query,
                response=context.response,
                judge_type="general_chat_helpfulness",
                task_type="general_chat",
                jurisdiction=context.jurisdiction,
                prompt_template=prompt
            )
            
            return self._parse_api_result(result)
            
        except Exception as e:
            self.logger.error(f"Helpfulness evaluation failed: {str(e)}")
            return 0.0, f"Helpfulness evaluation failed: {str(e)}"
    
    async def _evaluate_legal_ethics(self, context: GeneralChatEvaluationContext) -> Tuple[float, str]:
        """Evaluate legal ethics using professional prompt templates"""
        
        try:
            # Get professional prompt
            prompt = get_general_chat_prompt(GeneralChatPromptType.LEGAL_ETHICS, context)
            
            # Evaluate using API client
            result = await self.api_client.evaluate_legal_response(
                query=context.query,
                response=context.response,
                judge_type="general_chat_legal_ethics",
                task_type="general_chat",
                jurisdiction=context.jurisdiction,
                prompt_template=prompt
            )
            
            return self._parse_api_result(result)
            
        except Exception as e:
            self.logger.error(f"Legal ethics evaluation failed: {str(e)}")
            return 0.0, f"Legal ethics evaluation failed: {str(e)}"
    
    async def _evaluate_clarity(self, context: GeneralChatEvaluationContext) -> Tuple[float, str]:
        """Evaluate clarity using professional prompt templates"""
        
        try:
            # Get professional prompt
            prompt = get_general_chat_prompt(GeneralChatPromptType.CLARITY, context)
            
            # Evaluate using API client
            result = await self.api_client.evaluate_legal_response(
                query=context.query,
                response=context.response,
                judge_type="general_chat_clarity",
                task_type="general_chat",
                jurisdiction=context.jurisdiction,
                prompt_template=prompt
            )
            
            return self._parse_api_result(result)
            
        except Exception as e:
            self.logger.error(f"Clarity evaluation failed: {str(e)}")
            return 0.0, f"Clarity evaluation failed: {str(e)}"
    
    async def _evaluate_jurisdiction_compliance(self, context: GeneralChatEvaluationContext) -> Tuple[float, str]:
        """
        Evaluate jurisdiction compliance using professional prompt templates.
        
        This is the CRITICAL GATING COMPONENT that can disqualify responses.
        """
        
        try:
            # Create jurisdiction compliance context
            compliance_context = JurisdictionComplianceContext(
                query=context.query,
                response=context.response,
                jurisdiction=context.jurisdiction,
                task_type=LegalTaskType.GENERAL_CHAT,
                legal_domain=context.legal_domain,
                complexity_level="standard"
            )
            
            # Use legal framework accuracy for general chat
            prompt = get_jurisdiction_compliance_prompt(
                JurisdictionCompliancePromptType.LEGAL_FRAMEWORK_ACCURACY,
                compliance_context
            )
            
            # Evaluate using API client
            result = await self.api_client.evaluate_legal_response(
                query=context.query,
                response=context.response,
                judge_type="jurisdiction_compliance_general_chat",
                task_type="general_chat",
                jurisdiction=context.jurisdiction,
                prompt_template=prompt
            )
            
            score, reasoning = self._parse_api_result(result)
            
            # Update jurisdiction compliance manager
            compliance_evaluation = self.jurisdiction_compliance_manager.evaluate_gating_compliance(
                {"legal_framework_accuracy": score},
                context.jurisdiction
            )
            
            # Enhanced reasoning for gating component
            if compliance_evaluation["is_gating_failure"]:
                reasoning = f"GATING FAILURE: {reasoning} | {compliance_evaluation['failure_message']}"
            else:
                reasoning = f"Jurisdiction Compliant: {reasoning}"
            
            return score, reasoning
            
        except Exception as e:
            self.logger.error(f"Jurisdiction compliance evaluation failed: {str(e)}")
            return 0.0, f"GATING FAILURE - Jurisdiction compliance evaluation failed: {str(e)}"
    
    def _parse_api_result(self, api_result: Dict[str, Any]) -> Tuple[float, str]:
        """Parse API result to extract score and reasoning"""
        
        try:
            # Handle different response formats
            if isinstance(api_result, dict):
                if "score" in api_result:
                    score = float(api_result["score"])
                    reasoning = api_result.get("reasoning", "No reasoning provided")
                elif "evaluation" in api_result:
                    eval_data = api_result["evaluation"]
                    score = float(eval_data.get("score", 0.0))
                    reasoning = eval_data.get("reasoning", "No reasoning provided")
                else:
                    # Try to find numerical score in response
                    response_text = str(api_result)
                    import re
                    score_match = re.search(r'"?score"?\s*:?\s*(\d+\.?\d*)', response_text)
                    if score_match:
                        score = float(score_match.group(1))
                        reasoning = "Score extracted from response"
                    else:
                        score = 0.0
                        reasoning = "Could not parse score from response"
            else:
                # Try to parse as string
                response_text = str(api_result)
                import re
                score_match = re.search(r'"?score"?\s*:?\s*(\d+\.?\d*)', response_text)
                if score_match:
                    score = float(score_match.group(1))
                    reasoning = "Score extracted from string response"
                else:
                    score = 0.0
                    reasoning = "Could not parse string response"
            
            # Validate score range
            score = max(0.0, min(10.0, score))
            
            return score, reasoning
            
        except Exception as e:
            self.logger.error(f"Failed to parse API result: {str(e)}")
            return 0.0, f"Parse error: {str(e)}"
    
    def _generate_overall_reasoning(self, chat_score: EnhancedGeneralChatScore,
                                  data_point: LegalDataPoint) -> str:
        """Generate comprehensive reasoning for the overall evaluation"""
        
        reasoning_parts = [
            f"Enhanced general chat evaluation for {data_point.jurisdiction.value} jurisdiction:",
            f"Overall Score: {chat_score.overall_score:.1f}/10 (Confidence: {chat_score.confidence:.1f})",
            ""
        ]
        
        # Gating status (CRITICAL)
        if chat_score.is_gating_failure:
            reasoning_parts.extend([
                "🚨 GATING FAILURE DETECTED 🚨",
                f"Jurisdiction Compliance: {chat_score.jurisdiction_compliance:.1f}/10 (Below threshold {self.gating_threshold})",
                f"Failure Message: {chat_score.gating_failure_message}",
                f"Score Penalty Applied: {self.gating_penalty * 100:.0f}%",
                ""
            ])
        else:
            reasoning_parts.extend([
                f"✓ Jurisdiction Compliance: {chat_score.jurisdiction_compliance:.1f}/10 (PASSED)",
                ""
            ])
        
        # Component breakdown
        components = [
            ("Helpfulness", chat_score.helpfulness),
            ("Legal Ethics", chat_score.legal_ethics),
            ("Clarity", chat_score.clarity),
            ("Jurisdiction Compliance", chat_score.jurisdiction_compliance)
        ]
        
        reasoning_parts.append("Component Analysis:")
        for component_name, score in components:
            if component_name == "Jurisdiction Compliance":
                status = "GATING" if score < self.gating_threshold else "PASSED"
                reasoning_parts.append(f"• {component_name}: {score:.1f}/10 ({status})")
            else:
                reasoning_parts.append(f"• {component_name}: {score:.1f}/10")
        
        reasoning_parts.append("")
        
        # Key strengths and weaknesses
        strengths = []
        weaknesses = []
        
        for component_name, score in components:
            if score >= 8.0:
                strengths.append(component_name)
            elif score < 5.0:
                weaknesses.append(component_name)
        
        if strengths:
            reasoning_parts.append(f"Strengths: {', '.join(strengths)}")
        
        if weaknesses:
            reasoning_parts.append(f"Areas for Improvement: {', '.join(weaknesses)}")
        
        # Usage context
        reasoning_parts.extend([
            "",
            f"Evaluation Context: {data_point.task_type.value}",
            f"User Type: {getattr(data_point, 'user_type', 'general_public')}",
            f"Legal Domain: {getattr(data_point, 'legal_domain', 'General')}"
        ])
        
        return "\n".join(reasoning_parts)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the ensemble"""
        
        avg_evaluation_time = self.total_evaluation_time / max(1, self.evaluation_count)
        gating_failure_rate = self.gating_failure_count / max(1, self.evaluation_count)
        
        return {
            "ensemble_name": self.ensemble_name,
            "evaluations_completed": self.evaluation_count,
            "total_evaluation_time": self.total_evaluation_time,
            "average_evaluation_time": avg_evaluation_time,
            "api_calls_made": self.api_call_count,
            "cache_hits": self.cache_hit_count,
            "gating_failures": self.gating_failure_count,
            "gating_failure_rate": gating_failure_rate,
            "gating_threshold": self.gating_threshold
        }
    
    def get_gating_statistics(self) -> Dict[str, Any]:
        """Get detailed gating failure statistics"""
        
        return self.jurisdiction_compliance_manager.get_failure_statistics()


# Factory functions for easy ensemble creation
def create_enhanced_general_chat_ensemble(config: LegalRewardSystemConfig,
                                        api_client: Optional[CostOptimizedAPIClient] = None) -> EnhancedGeneralChatEnsemble:
    """
    Create production enhanced general chat ensemble with API-based evaluation.
    
    Args:
        config: System configuration
        api_client: Optional API client (will create if not provided)
        
    Returns:
        Configured enhanced general chat ensemble
    """
    
    return EnhancedGeneralChatEnsemble(config, api_client)


def create_development_enhanced_general_chat_ensemble(config: LegalRewardSystemConfig) -> EnhancedGeneralChatEnsemble:
    """
    Create development enhanced general chat ensemble with optimizations for testing.
    
    Args:
        config: System configuration
        
    Returns:
        Configured development ensemble
    """
    
    # Create with development-optimized API client
    api_client = CostOptimizedAPIClient(config)
    ensemble = EnhancedGeneralChatEnsemble(config, api_client)
    
    # Development optimizations
    ensemble.gating_threshold = 2.0  # Lower threshold for development
    
    return ensemble


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    from ..config.settings import create_default_config
    from ..core.data_structures import LegalDataPoint
    from ..core.enums import LegalTaskType
    
    async def test_enhanced_general_chat_ensemble():
        # Create test configuration
        config = create_default_config()
        
        # Create ensemble
        ensemble = create_enhanced_general_chat_ensemble(config)
        
        # Create test data point
        test_data = LegalDataPoint(
            query="What should I know about forming an LLC in California?",
            response="To form an LLC in California, you'll need to file Articles of Organization with the California Secretary of State...",
            task_type=LegalTaskType.GENERAL_CHAT,
            jurisdiction=USJurisdiction.CALIFORNIA,
            ground_truth="Helpful business formation guidance expected",
            metadata={
                "legal_domain": "Business Law",
                "user_type": "general_public"
            }
        )
        
        # Evaluate
        print("Testing Enhanced General Chat Ensemble with Professional Prompt Templates and Gating...")
        result = await ensemble.evaluate(test_data)
        
        print(f"Score: {result.score:.2f}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Gating Failure: {result.metadata.get('is_gating_failure', False)}")
        print(f"Reasoning: {result.reasoning[:300]}...")
        print(f"Component Scores: {result.component_scores}")
        
        # Show performance stats
        stats = ensemble.get_performance_stats()
        print(f"\nPerformance Stats:")
        print(f"Evaluations: {stats['evaluations_completed']}")
        print(f"Avg Time: {stats['average_evaluation_time']:.2f}s")
        print(f"Gating Failures: {stats['gating_failures']}")
        print(f"Gating Failure Rate: {stats['gating_failure_rate']:.2%}")
    
    # Run test
    if hasattr(asyncio, 'run'):
        asyncio.run(test_enhanced_general_chat_ensemble())
    else:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(test_enhanced_general_chat_ensemble())
