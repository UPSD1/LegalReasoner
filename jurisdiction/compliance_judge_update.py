"""
Jurisdiction Compliance Judge - Updated with Professional Prompt Templates

This judge provides critical gating functionality for the Enhanced Multi-Task Legal
Reward System by evaluating whether legal responses meet US jurisdiction-specific
requirements. It uses sophisticated prompt templates for professional-grade evaluation.

CRITICAL FUNCTION: This judge acts as a gating mechanism - responses with poor
jurisdiction compliance (score < 3.0) may be penalized or rejected entirely across
the entire system.

Updated Features:
- Professional prompt template integration from config/prompts/jurisdiction_compliance.py
- Comprehensive jurisdiction compliance evaluation across 5 key areas
- Enhanced gating logic with configurable thresholds
- Performance optimization and caching for GRPO training
- Full integration with all judge ensembles

Evaluation Areas:
- Legal Framework Accuracy: Correct application of jurisdiction-specific laws
- Procedural Compliance: Understanding of court systems and procedures
- Substantive Law Accuracy: Correct substantive law application
- Constitutional Framework: Federal vs state constitutional considerations
- Professional Standards: Jurisdiction-specific professional responsibility

Author: Legal Reward System Team
Version: 1.1.0 (Updated with prompt templates)
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import time

# Import core system components
from ..core.data_structures import LegalDataPoint, EnsembleScore
from ..core.enums import LegalTaskType
from .us_system import USJurisdiction
from ..config.settings import LegalRewardSystemConfig
from ..judges.api_client import CostOptimizedAPIClient
from ..utils.logging import get_legal_logger

# Import prompt template system
from ..config.prompts import (
    JurisdictionCompliancePromptType,
    JurisdictionComplianceContext,
    get_jurisdiction_compliance_prompt,
    get_all_jurisdiction_compliance_prompts,
    validate_jurisdiction_compliance_prompt,
    JurisdictionComplianceManager,
    get_gating_threshold,
    assess_gating_failure
)


class ComplianceEvaluationLevel(Enum):
    """Levels of compliance evaluation detail"""
    BASIC = "basic"              # Legal framework accuracy only
    STANDARD = "standard"        # 3 core areas
    COMPREHENSIVE = "comprehensive"  # All 5 areas
    GATING_ONLY = "gating_only"  # Quick gating assessment


@dataclass
class JurisdictionComplianceResult:
    """Detailed result from jurisdiction compliance evaluation"""
    
    # Individual component scores (0-10 scale)
    legal_framework_accuracy: float = 0.0
    procedural_compliance: float = 0.0
    substantive_law_accuracy: float = 0.0
    constitutional_framework: float = 0.0
    professional_standards: float = 0.0
    
    # Aggregate metrics
    overall_compliance_score: float = 0.0
    compliance_confidence: float = 0.0
    
    # Gating assessment
    is_gating_failure: bool = False
    gating_failure_message: str = ""
    gating_threshold: float = 3.0
    
    # Jurisdiction context
    jurisdiction: USJurisdiction = USJurisdiction.GENERAL
    task_type: LegalTaskType = LegalTaskType.GENERAL_CHAT
    
    # Component reasoning
    component_reasoning: Dict[str, str] = field(default_factory=dict)
    overall_reasoning: str = ""
    
    # Performance metadata
    evaluation_time_seconds: float = 0.0
    components_evaluated: int = 0
    api_calls_made: int = 0
    cache_hits: int = 0
    
    def calculate_overall_score(self, evaluation_level: ComplianceEvaluationLevel = ComplianceEvaluationLevel.COMPREHENSIVE) -> float:
        """Calculate weighted overall compliance score based on evaluation level"""
        
        if evaluation_level == ComplianceEvaluationLevel.BASIC:
            # Only legal framework accuracy
            self.overall_compliance_score = self.legal_framework_accuracy
            self.components_evaluated = 1
            
        elif evaluation_level == ComplianceEvaluationLevel.STANDARD:
            # Core 3 components
            scores = [
                self.legal_framework_accuracy,
                self.procedural_compliance,
                self.substantive_law_accuracy
            ]
            self.overall_compliance_score = sum(scores) / len(scores)
            self.components_evaluated = 3
            
        elif evaluation_level == ComplianceEvaluationLevel.COMPREHENSIVE:
            # All 5 components
            scores = [
                self.legal_framework_accuracy,
                self.procedural_compliance,
                self.substantive_law_accuracy,
                self.constitutional_framework,
                self.professional_standards
            ]
            self.overall_compliance_score = sum(scores) / len(scores)
            self.components_evaluated = 5
            
        elif evaluation_level == ComplianceEvaluationLevel.GATING_ONLY:
            # Quick assessment based on legal framework only
            self.overall_compliance_score = self.legal_framework_accuracy
            self.components_evaluated = 1
        
        return self.overall_compliance_score
    
    def calculate_confidence(self) -> float:
        """Calculate confidence based on score consistency and components evaluated"""
        
        # Get evaluated scores
        evaluated_scores = []
        if self.legal_framework_accuracy > 0:
            evaluated_scores.append(self.legal_framework_accuracy)
        if self.procedural_compliance > 0:
            evaluated_scores.append(self.procedural_compliance)
        if self.substantive_law_accuracy > 0:
            evaluated_scores.append(self.substantive_law_accuracy)
        if self.constitutional_framework > 0:
            evaluated_scores.append(self.constitutional_framework)
        if self.professional_standards > 0:
            evaluated_scores.append(self.professional_standards)
        
        if len(evaluated_scores) <= 1:
            # Single component - moderate confidence
            self.compliance_confidence = 0.6
        else:
            # Multiple components - confidence based on consistency
            mean_score = sum(evaluated_scores) / len(evaluated_scores)
            variance = sum((score - mean_score) ** 2 for score in evaluated_scores) / len(evaluated_scores)
            consistency_confidence = max(0.0, 1.0 - (variance / 25.0))  # Scale for 0-10 range
            
            # Boost confidence for more components
            component_boost = min(0.2, (len(evaluated_scores) - 1) * 0.05)
            self.compliance_confidence = min(1.0, consistency_confidence + component_boost)
        
        # Reduce confidence for gating failures
        if self.is_gating_failure:
            self.compliance_confidence *= 0.5
        
        return self.compliance_confidence
    
    def assess_gating_failure(self):
        """Assess whether this represents a gating failure"""
        
        self.is_gating_failure, self.gating_failure_message = assess_gating_failure(self.overall_compliance_score)


class JurisdictionComplianceJudge:
    """
    Professional jurisdiction compliance judge with sophisticated prompt template
    integration and comprehensive gating functionality.
    
    This judge evaluates whether legal responses meet jurisdiction-specific requirements
    and serves as a critical gating mechanism for the entire system.
    """
    
    def __init__(self, config: LegalRewardSystemConfig,
                 api_client: Optional[CostOptimizedAPIClient] = None):
        self.config = config
        self.api_client = api_client or CostOptimizedAPIClient(config)
        self.logger = get_legal_logger("jurisdiction_compliance_judge")
        
        # Initialize compliance manager
        self.compliance_manager = JurisdictionComplianceManager()
        
        # Gating configuration
        self.gating_threshold = get_gating_threshold()
        self.strict_compliance_mode = config.jurisdiction_system.strict_compliance_mode
        
        # Performance tracking
        self.evaluation_count = 0
        self.total_evaluation_time = 0.0
        self.api_call_count = 0
        self.cache_hit_count = 0
        self.gating_failure_count = 0
        
        # Validate prompt system on initialization
        self._validate_prompt_system()
    
    def _validate_prompt_system(self):
        """Validate that jurisdiction compliance prompt templates are available"""
        
        try:
            # Test prompt generation for key jurisdictions
            test_jurisdictions = [USJurisdiction.FEDERAL, USJurisdiction.CALIFORNIA, USJurisdiction.GENERAL]
            
            for jurisdiction in test_jurisdictions:
                for prompt_type in JurisdictionCompliancePromptType:
                    if not validate_jurisdiction_compliance_prompt(prompt_type, jurisdiction):
                        raise ValueError(f"Prompt validation failed for {prompt_type} in {jurisdiction}")
            
            self.logger.info("Jurisdiction compliance prompt system validation passed")
            
        except Exception as e:
            self.logger.error(f"Jurisdiction compliance prompt system validation failed: {str(e)}")
            if self.config.environment.value == "production":
                raise
    
    async def evaluate_compliance(self, data_point: LegalDataPoint,
                                evaluation_level: ComplianceEvaluationLevel = ComplianceEvaluationLevel.STANDARD) -> JurisdictionComplianceResult:
        """
        Comprehensive jurisdiction compliance evaluation using professional prompt templates.
        
        Args:
            data_point: Legal data point to evaluate
            evaluation_level: Level of evaluation detail
            
        Returns:
            Detailed compliance evaluation result
        """
        
        start_time = time.time()
        self.evaluation_count += 1
        
        try:
            # Create evaluation context
            compliance_context = self._create_compliance_context(data_point)
            
            # Initialize result
            result = JurisdictionComplianceResult(
                jurisdiction=data_point.jurisdiction,
                task_type=data_point.task_type,
                gating_threshold=self.gating_threshold
            )
            
            # Determine which components to evaluate based on level
            components_to_evaluate = self._get_components_for_level(evaluation_level)
            
            # Evaluate components in parallel for performance
            evaluation_tasks = []
            for component in components_to_evaluate:
                task = self._evaluate_compliance_component(component, compliance_context)
                evaluation_tasks.append((component, task))
            
            # Execute evaluations
            evaluation_results = await asyncio.gather(*[task for _, task in evaluation_tasks], return_exceptions=True)
            
            # Process results
            for i, (component, _) in enumerate(evaluation_tasks):
                evaluation_result = evaluation_results[i]
                
                if isinstance(evaluation_result, Exception):
                    self.logger.error(f"Compliance component evaluation failed for {component}: {str(evaluation_result)}")
                    score = 0.0
                    reasoning = f"Component evaluation failed: {str(evaluation_result)}"
                else:
                    score, reasoning = evaluation_result
                
                # Set component score and reasoning
                self._set_component_result(result, component, score, reasoning)
            
            # Calculate aggregate metrics
            result.calculate_overall_score(evaluation_level)
            result.calculate_confidence()
            
            # Assess gating failure
            result.assess_gating_failure()
            
            if result.is_gating_failure:
                self.gating_failure_count += 1
                self.logger.warning(f"Gating failure detected for {data_point.jurisdiction}: {result.gating_failure_message}")
            
            # Generate overall reasoning
            result.overall_reasoning = self._generate_overall_reasoning(result, data_point, evaluation_level)
            
            # Performance metadata
            end_time = time.time()
            result.evaluation_time_seconds = end_time - start_time
            result.api_calls_made = len(components_to_evaluate)
            
            self.total_evaluation_time += result.evaluation_time_seconds
            self.api_call_count += result.api_calls_made
            
            # Update compliance manager
            component_scores = self._extract_component_scores(result, components_to_evaluate)
            compliance_evaluation = self.compliance_manager.evaluate_gating_compliance(
                component_scores, 
                data_point.jurisdiction
            )
            
            self.logger.debug(f"Jurisdiction compliance evaluation completed: score={result.overall_compliance_score:.2f}, gating_failure={result.is_gating_failure}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Jurisdiction compliance evaluation failed: {str(e)}")
            
            # Return failed evaluation result
            result = JurisdictionComplianceResult(
                jurisdiction=data_point.jurisdiction,
                task_type=data_point.task_type,
                overall_compliance_score=0.0,
                compliance_confidence=0.0,
                is_gating_failure=True,
                gating_failure_message=f"Evaluation failed: {str(e)}",
                overall_reasoning=f"Jurisdiction compliance evaluation failed: {str(e)}"
            )
            
            return result
    
    async def quick_gating_assessment(self, data_point: LegalDataPoint) -> Tuple[bool, float, str]:
        """
        Quick gating assessment for performance-critical scenarios.
        
        Args:
            data_point: Legal data point to assess
            
        Returns:
            Tuple of (is_gating_failure, score, reasoning)
        """
        
        try:
            result = await self.evaluate_compliance(data_point, ComplianceEvaluationLevel.GATING_ONLY)
            return result.is_gating_failure, result.overall_compliance_score, result.overall_reasoning
            
        except Exception as e:
            self.logger.error(f"Quick gating assessment failed: {str(e)}")
            return True, 0.0, f"Gating assessment failed: {str(e)}"
    
    def _create_compliance_context(self, data_point: LegalDataPoint) -> JurisdictionComplianceContext:
        """Create compliance evaluation context"""
        
        return JurisdictionComplianceContext(
            query=data_point.query,
            response=data_point.response,
            jurisdiction=data_point.jurisdiction,
            task_type=data_point.task_type,
            legal_domain=getattr(data_point, 'legal_domain', 'General'),
            complexity_level=getattr(data_point, 'complexity_level', 'standard'),
            federal_implications=data_point.jurisdiction == USJurisdiction.FEDERAL,
            interstate_considerations=getattr(data_point, 'interstate_considerations', False)
        )
    
    def _get_components_for_level(self, evaluation_level: ComplianceEvaluationLevel) -> List[JurisdictionCompliancePromptType]:
        """Get compliance components to evaluate based on level"""
        
        if evaluation_level == ComplianceEvaluationLevel.BASIC:
            return [JurisdictionCompliancePromptType.LEGAL_FRAMEWORK_ACCURACY]
        
        elif evaluation_level == ComplianceEvaluationLevel.STANDARD:
            return [
                JurisdictionCompliancePromptType.LEGAL_FRAMEWORK_ACCURACY,
                JurisdictionCompliancePromptType.PROCEDURAL_COMPLIANCE,
                JurisdictionCompliancePromptType.SUBSTANTIVE_LAW_ACCURACY
            ]
        
        elif evaluation_level == ComplianceEvaluationLevel.COMPREHENSIVE:
            return [
                JurisdictionCompliancePromptType.LEGAL_FRAMEWORK_ACCURACY,
                JurisdictionCompliancePromptType.PROCEDURAL_COMPLIANCE,
                JurisdictionCompliancePromptType.SUBSTANTIVE_LAW_ACCURACY,
                JurisdictionCompliancePromptType.CONSTITUTIONAL_FRAMEWORK,
                JurisdictionCompliancePromptType.PROFESSIONAL_STANDARDS
            ]
        
        elif evaluation_level == ComplianceEvaluationLevel.GATING_ONLY:
            return [JurisdictionCompliancePromptType.LEGAL_FRAMEWORK_ACCURACY]
        
        else:
            return [JurisdictionCompliancePromptType.LEGAL_FRAMEWORK_ACCURACY]
    
    async def _evaluate_compliance_component(self, component: JurisdictionCompliancePromptType,
                                           context: JurisdictionComplianceContext) -> Tuple[float, str]:
        """
        Evaluate a specific compliance component using professional prompt templates.
        
        Args:
            component: Compliance component to evaluate
            context: Evaluation context
            
        Returns:
            Tuple of (score, reasoning)
        """
        
        try:
            # Get professional prompt template
            prompt = get_jurisdiction_compliance_prompt(component, context)
            
            # Evaluate using API client
            result = await self.api_client.evaluate_legal_response(
                query=context.query,
                response=context.response,
                judge_type=f"jurisdiction_compliance_{component.value}",
                task_type=context.task_type.value,
                jurisdiction=context.jurisdiction,
                prompt_template=prompt
            )
            
            # Parse result
            score, reasoning = self._parse_api_result(result)
            
            return score, reasoning
            
        except Exception as e:
            self.logger.error(f"Compliance component evaluation failed for {component}: {str(e)}")
            return 0.0, f"Component evaluation failed: {str(e)}"
    
    def _set_component_result(self, result: JurisdictionComplianceResult,
                            component: JurisdictionCompliancePromptType,
                            score: float, reasoning: str):
        """Set component score and reasoning in result object"""
        
        component_mapping = {
            JurisdictionCompliancePromptType.LEGAL_FRAMEWORK_ACCURACY: "legal_framework_accuracy",
            JurisdictionCompliancePromptType.PROCEDURAL_COMPLIANCE: "procedural_compliance",
            JurisdictionCompliancePromptType.SUBSTANTIVE_LAW_ACCURACY: "substantive_law_accuracy",
            JurisdictionCompliancePromptType.CONSTITUTIONAL_FRAMEWORK: "constitutional_framework",
            JurisdictionCompliancePromptType.PROFESSIONAL_STANDARDS: "professional_standards"
        }
        
        attr_name = component_mapping.get(component)
        if attr_name:
            setattr(result, attr_name, score)
            result.component_reasoning[component.value] = reasoning
    
    def _extract_component_scores(self, result: JurisdictionComplianceResult,
                                components: List[JurisdictionCompliancePromptType]) -> Dict[str, float]:
        """Extract component scores for compliance manager"""
        
        component_scores = {}
        
        for component in components:
            component_mapping = {
                JurisdictionCompliancePromptType.LEGAL_FRAMEWORK_ACCURACY: result.legal_framework_accuracy,
                JurisdictionCompliancePromptType.PROCEDURAL_COMPLIANCE: result.procedural_compliance,
                JurisdictionCompliancePromptType.SUBSTANTIVE_LAW_ACCURACY: result.substantive_law_accuracy,
                JurisdictionCompliancePromptType.CONSTITUTIONAL_FRAMEWORK: result.constitutional_framework,
                JurisdictionCompliancePromptType.PROFESSIONAL_STANDARDS: result.professional_standards
            }
            
            score = component_mapping.get(component, 0.0)
            if score > 0:  # Only include evaluated components
                component_scores[component.value] = score
        
        return component_scores
    
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
    
    def _generate_overall_reasoning(self, result: JurisdictionComplianceResult,
                                  data_point: LegalDataPoint,
                                  evaluation_level: ComplianceEvaluationLevel) -> str:
        """Generate comprehensive reasoning for the compliance evaluation"""
        
        reasoning_parts = [
            f"Jurisdiction Compliance Evaluation - {data_point.jurisdiction.value}",
            f"Evaluation Level: {evaluation_level.value}",
            f"Overall Compliance Score: {result.overall_compliance_score:.1f}/10 (Confidence: {result.compliance_confidence:.1f})",
            ""
        ]
        
        # Gating status (CRITICAL)
        if result.is_gating_failure:
            reasoning_parts.extend([
                "🚨 GATING FAILURE - CRITICAL COMPLIANCE ISSUE 🚨",
                f"Score {result.overall_compliance_score:.1f} below threshold {result.gating_threshold}",
                f"Failure Details: {result.gating_failure_message}",
                "This response may be penalized or rejected due to jurisdictional non-compliance.",
                ""
            ])
        else:
            reasoning_parts.extend([
                f"✓ COMPLIANCE PASSED - Score meets threshold {result.gating_threshold}",
                ""
            ])
        
        # Component breakdown
        evaluated_components = []
        if result.legal_framework_accuracy > 0:
            evaluated_components.append(("Legal Framework Accuracy", result.legal_framework_accuracy))
        if result.procedural_compliance > 0:
            evaluated_components.append(("Procedural Compliance", result.procedural_compliance))
        if result.substantive_law_accuracy > 0:
            evaluated_components.append(("Substantive Law Accuracy", result.substantive_law_accuracy))
        if result.constitutional_framework > 0:
            evaluated_components.append(("Constitutional Framework", result.constitutional_framework))
        if result.professional_standards > 0:
            evaluated_components.append(("Professional Standards", result.professional_standards))
        
        if evaluated_components:
            reasoning_parts.append("Component Analysis:")
            for component_name, score in evaluated_components:
                status = "CRITICAL FAILURE" if score < result.gating_threshold else "PASSED"
                reasoning_parts.append(f"• {component_name}: {score:.1f}/10 ({status})")
            reasoning_parts.append("")
        
        # Jurisdiction-specific context
        reasoning_parts.extend([
            f"Jurisdiction Context: {data_point.jurisdiction.value}",
            f"Task Type: {data_point.task_type.value}",
            f"Legal Domain: {getattr(data_point, 'legal_domain', 'General')}",
            f"Components Evaluated: {result.components_evaluated}"
        ])
        
        # Performance context
        if result.evaluation_time_seconds > 0:
            reasoning_parts.append(f"Evaluation Time: {result.evaluation_time_seconds:.2f}s")
        
        return "\n".join(reasoning_parts)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the compliance judge"""
        
        avg_evaluation_time = self.total_evaluation_time / max(1, self.evaluation_count)
        gating_failure_rate = self.gating_failure_count / max(1, self.evaluation_count)
        
        return {
            "judge_name": "jurisdiction_compliance_judge",
            "evaluations_completed": self.evaluation_count,
            "total_evaluation_time": self.total_evaluation_time,
            "average_evaluation_time": avg_evaluation_time,
            "api_calls_made": self.api_call_count,
            "cache_hits": self.cache_hit_count,
            "gating_failures": self.gating_failure_count,
            "gating_failure_rate": gating_failure_rate,
            "gating_threshold": self.gating_threshold,
            "strict_compliance_mode": self.strict_compliance_mode
        }
    
    def get_gating_statistics(self) -> Dict[str, Any]:
        """Get detailed gating failure statistics"""
        
        return self.compliance_manager.get_failure_statistics()


# Factory functions for easy judge creation
def create_jurisdiction_compliance_judge(config: LegalRewardSystemConfig,
                                       api_client: Optional[CostOptimizedAPIClient] = None) -> JurisdictionComplianceJudge:
    """
    Create production jurisdiction compliance judge with API-based evaluation.
    
    Args:
        config: System configuration
        api_client: Optional API client (will create if not provided)
        
    Returns:
        Configured jurisdiction compliance judge
    """
    
    return JurisdictionComplianceJudge(config, api_client)


def create_development_jurisdiction_compliance_judge(config: LegalRewardSystemConfig) -> JurisdictionComplianceJudge:
    """
    Create development jurisdiction compliance judge with optimizations for testing.
    
    Args:
        config: System configuration
        
    Returns:
        Configured development judge
    """
    
    # Create with development-optimized API client
    api_client = CostOptimizedAPIClient(config)
    judge = JurisdictionComplianceJudge(config, api_client)
    
    # Development optimizations
    judge.strict_compliance_mode = False
    
    return judge


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    from ..config.settings import create_default_config
    from ..core.data_structures import LegalDataPoint
    from ..core.enums import LegalTaskType
    
    async def test_jurisdiction_compliance_judge():
        # Create test configuration
        config = create_default_config()
        
        # Create judge
        judge = create_jurisdiction_compliance_judge(config)
        
        # Create test data point
        test_data = LegalDataPoint(
            query="What are the contract formation requirements?",
            response="Contract formation requires offer, acceptance, consideration, and legal capacity...",
            task_type=LegalTaskType.JUDICIAL_REASONING,
            jurisdiction=USJurisdiction.CALIFORNIA,
            ground_truth="Professional jurisdiction compliance expected",
            metadata={"legal_domain": "Contract Law"}
        )
        
        # Test comprehensive evaluation
        print("Testing Jurisdiction Compliance Judge with Professional Prompt Templates...")
        result = await judge.evaluate_compliance(test_data, ComplianceEvaluationLevel.COMPREHENSIVE)
        
        print(f"Overall Compliance Score: {result.overall_compliance_score:.2f}")
        print(f"Confidence: {result.compliance_confidence:.2f}")
        print(f"Gating Failure: {result.is_gating_failure}")
        print(f"Components Evaluated: {result.components_evaluated}")
        
        if result.is_gating_failure:
            print(f"Gating Message: {result.gating_failure_message}")
        
        print(f"Reasoning: {result.overall_reasoning[:300]}...")
        
        # Test quick gating assessment
        print("\nTesting Quick Gating Assessment...")
        is_failure, score, reasoning = await judge.quick_gating_assessment(test_data)
        print(f"Quick Assessment - Failure: {is_failure}, Score: {score:.2f}")
        
        # Show performance stats
        stats = judge.get_performance_stats()
        print(f"\nPerformance Stats:")
        print(f"Evaluations: {stats['evaluations_completed']}")
        print(f"Avg Time: {stats['average_evaluation_time']:.2f}s")
        print(f"Gating Failures: {stats['gating_failures']}")
        print(f"Gating Failure Rate: {stats['gating_failure_rate']:.2%}")
    
    # Run test
    if hasattr(asyncio, 'run'):
        asyncio.run(test_jurisdiction_compliance_judge())
    else:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(test_jurisdiction_compliance_judge())
