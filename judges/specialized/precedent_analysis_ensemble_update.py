"""
Precedent Analysis Ensemble - Updated with Professional Prompt Templates

This ensemble evaluates deep case law analysis, analogical reasoning, and precedent
hierarchy understanding using sophisticated US jurisdiction-aware prompt templates.
This is a core component of the hybrid evaluation system (70% specialized + 30% chat).

Updated Features:
- Professional prompt template integration from config/prompts/precedent_analysis.py
- Sophisticated US jurisdiction case law awareness for all 50 states + DC + federal
- Enhanced evaluation components with jurisdiction-specific precedent context
- Optimized caching and performance for GRPO training
- Full VERL integration compatibility

Evaluation Components (Equal Weights):
- Case Law Accuracy (20%): Correct precedent identification and description
- Analogical Reasoning (20%): Quality of case-to-case comparisons
- Distinguishing Factors (20%): Proper identification of material differences
- Hierarchy Understanding (20%): Controlling vs. persuasive authority usage
- Citation Quality (20%): Proper legal citation format and relevance

Author: Legal Reward System Team
Version: 1.1.0 (Updated with prompt templates)
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Import core system components
from ...core.data_structures import LegalDataPoint, EnsembleScore
from ...core.enums import LegalTaskType
from ...jurisdiction.us_system import USJurisdiction
from ...config.settings import LegalRewardSystemConfig
from ..base import BaseJudgeEnsemble
from ..api_client import CostOptimizedAPIClient

# Import prompt template system
from ...config.prompts import (
    PrecedentAnalysisPromptType,
    get_precedent_analysis_prompt,
    get_all_precedent_analysis_prompts,
    validate_precedent_analysis_prompt
)


class PrecedentAnalysisComponent(Enum):
    """Components of precedent analysis evaluation"""
    CASE_LAW_ACCURACY = "case_law_accuracy"
    ANALOGICAL_REASONING = "analogical_reasoning"
    DISTINGUISHING_FACTORS = "distinguishing_factors"
    HIERARCHY_UNDERSTANDING = "hierarchy_understanding"
    CITATION_QUALITY = "citation_quality"


@dataclass
class PrecedentAnalysisScore:
    """Detailed scoring for precedent analysis evaluation"""
    
    # Component scores (0-10 scale)
    case_law_accuracy: float = 0.0
    analogical_reasoning: float = 0.0
    distinguishing_factors: float = 0.0
    hierarchy_understanding: float = 0.0
    citation_quality: float = 0.0
    
    # Aggregate metrics
    overall_score: float = 0.0
    confidence: float = 0.0
    
    # Jurisdiction compliance
    jurisdiction_compliance: float = 0.0
    jurisdiction_compliant: bool = False
    
    # Component weights (configurable)
    component_weights: Dict[str, float] = field(default_factory=lambda: {
        "case_law_accuracy": 0.20,
        "analogical_reasoning": 0.20,
        "distinguishing_factors": 0.20,
        "hierarchy_understanding": 0.20,
        "citation_quality": 0.20
    })
    
    # Detailed reasoning
    component_reasoning: Dict[str, str] = field(default_factory=dict)
    overall_reasoning: str = ""
    
    # Performance metadata
    evaluation_time_seconds: float = 0.0
    api_calls_made: int = 0
    cache_hits: int = 0
    
    def calculate_overall_score(self) -> float:
        """Calculate weighted overall score"""
        
        component_scores = {
            "case_law_accuracy": self.case_law_accuracy,
            "analogical_reasoning": self.analogical_reasoning,
            "distinguishing_factors": self.distinguishing_factors,
            "hierarchy_understanding": self.hierarchy_understanding,
            "citation_quality": self.citation_quality
        }
        
        # Calculate weighted average
        weighted_sum = sum(
            score * self.component_weights.get(component, 0.0)
            for component, score in component_scores.items()
        )
        
        self.overall_score = weighted_sum
        return self.overall_score
    
    def calculate_confidence(self) -> float:
        """Calculate confidence based on score consistency and jurisdiction compliance"""
        
        component_scores = [
            self.case_law_accuracy, self.analogical_reasoning, self.distinguishing_factors,
            self.hierarchy_understanding, self.citation_quality
        ]
        
        # Base confidence from score consistency
        if len(component_scores) > 1:
            score_variance = sum((score - self.overall_score) ** 2 for score in component_scores) / len(component_scores)
            consistency_confidence = max(0.0, 1.0 - (score_variance / 10.0))
        else:
            consistency_confidence = 0.5
        
        # Jurisdiction compliance impact
        jurisdiction_confidence = self.jurisdiction_compliance / 10.0
        
        # Combined confidence (80% consistency + 20% jurisdiction)
        self.confidence = 0.8 * consistency_confidence + 0.2 * jurisdiction_confidence
        
        return self.confidence


class PrecedentAnalysisEnsemble(BaseJudgeEnsemble):
    """
    Professional precedent analysis evaluation ensemble with sophisticated 
    prompt templates and case law jurisdiction awareness.
    
    This ensemble uses the professional prompt templates from config/prompts/precedent_analysis.py
    to provide expert-level evaluation of case law analysis and analogical reasoning.
    """
    
    def __init__(self, config: LegalRewardSystemConfig, 
                 api_client: Optional[CostOptimizedAPIClient] = None):
        super().__init__("precedent_analysis_ensemble", LegalTaskType.PRECEDENT_ANALYSIS)
        
        self.config = config
        self.api_client = api_client or CostOptimizedAPIClient(config)
        
        # Component weights (configurable via config)
        self.component_weights = {
            PrecedentAnalysisComponent.CASE_LAW_ACCURACY: 0.20,
            PrecedentAnalysisComponent.ANALOGICAL_REASONING: 0.20,
            PrecedentAnalysisComponent.DISTINGUISHING_FACTORS: 0.20,
            PrecedentAnalysisComponent.HIERARCHY_UNDERSTANDING: 0.20,
            PrecedentAnalysisComponent.CITATION_QUALITY: 0.20
        }
        
        # Performance tracking
        self.evaluation_count = 0
        self.total_evaluation_time = 0.0
        self.api_call_count = 0
        self.cache_hit_count = 0
        
        # Jurisdiction compliance tracking
        self.jurisdiction_compliance_failures = []
        
        # Validate prompt system on initialization
        self._validate_prompt_system()
    
    def _validate_prompt_system(self):
        """Validate that prompt templates are available"""
        
        try:
            # Test prompt generation for a sample jurisdiction
            test_jurisdiction = USJurisdiction.FEDERAL
            
            for prompt_type in PrecedentAnalysisPromptType:
                if not validate_precedent_analysis_prompt(prompt_type, test_jurisdiction):
                    raise ValueError(f"Prompt validation failed for {prompt_type} in {test_jurisdiction}")
            
            self.logger.info("Precedent analysis prompt system validation passed")
            
        except Exception as e:
            self.logger.error(f"Precedent analysis prompt system validation failed: {str(e)}")
            if self.config.environment.value == "production":
                raise
    
    async def evaluate(self, data_point: LegalDataPoint) -> EnsembleScore:
        """
        Comprehensive precedent analysis evaluation using professional prompt templates.
        
        Args:
            data_point: Legal data point to evaluate
            
        Returns:
            Comprehensive ensemble score with detailed component analysis
        """
        
        start_time = asyncio.get_event_loop().time()
        self.evaluation_count += 1
        
        try:
            # Create jurisdiction context
            jurisdiction_context = self._create_jurisdiction_context(data_point)
            
            # Evaluate all components in parallel for performance
            component_tasks = []
            
            for component in PrecedentAnalysisComponent:
                task = self._evaluate_component(component, data_point, jurisdiction_context)
                component_tasks.append(task)
            
            # Execute all evaluations
            component_results = await asyncio.gather(*component_tasks, return_exceptions=True)
            
            # Process results
            precedent_score = PrecedentAnalysisScore()
            component_scores = {}
            
            for i, component in enumerate(PrecedentAnalysisComponent):
                result = component_results[i]
                
                if isinstance(result, Exception):
                    self.logger.error(f"Component evaluation failed for {component}: {str(result)}")
                    score = 0.0
                    reasoning = f"Evaluation failed: {str(result)}"
                else:
                    score, reasoning = result
                
                # Set component score
                setattr(precedent_score, component.value, score)
                component_scores[component.value] = score
                precedent_score.component_reasoning[component.value] = reasoning
            
            # Calculate aggregate metrics
            precedent_score.calculate_overall_score()
            precedent_score.calculate_confidence()
            
            # Check jurisdiction compliance
            jurisdiction_compliance_score = await self._evaluate_jurisdiction_compliance(data_point, jurisdiction_context)
            precedent_score.jurisdiction_compliance = jurisdiction_compliance_score
            precedent_score.jurisdiction_compliant = jurisdiction_compliance_score >= self.config.hybrid_evaluation.jurisdiction_compliance_threshold
            
            # Track jurisdiction compliance failures
            if not precedent_score.jurisdiction_compliant:
                self.jurisdiction_compliance_failures.append({
                    "jurisdiction": data_point.jurisdiction,
                    "score": jurisdiction_compliance_score,
                    "query": data_point.query[:100]  # Truncated for logging
                })
            
            # Generate overall reasoning
            precedent_score.overall_reasoning = self._generate_overall_reasoning(precedent_score, data_point)
            
            # Performance metadata
            end_time = asyncio.get_event_loop().time()
            precedent_score.evaluation_time_seconds = end_time - start_time
            precedent_score.api_calls_made = len(component_tasks) + 1  # Components + jurisdiction compliance
            
            self.total_evaluation_time += precedent_score.evaluation_time_seconds
            self.api_call_count += precedent_score.api_calls_made
            
            # Create ensemble score
            ensemble_score = EnsembleScore(
                score=precedent_score.overall_score,
                confidence=precedent_score.confidence,
                reasoning=precedent_score.overall_reasoning,
                component_scores=component_scores,
                metadata={
                    "jurisdiction_compliance": precedent_score.jurisdiction_compliance,
                    "jurisdiction_compliant": precedent_score.jurisdiction_compliant,
                    "component_reasoning": precedent_score.component_reasoning,
                    "evaluation_time": precedent_score.evaluation_time_seconds,
                    "api_calls": precedent_score.api_calls_made
                }
            )
            
            self.logger.debug(f"Precedent analysis evaluation completed: score={precedent_score.overall_score:.2f}, confidence={precedent_score.confidence:.2f}")
            
            return ensemble_score
            
        except Exception as e:
            self.logger.error(f"Precedent analysis evaluation failed: {str(e)}")
            
            # Return failed evaluation score
            return EnsembleScore(
                score=0.0,
                confidence=0.0,
                reasoning=f"Evaluation failed: {str(e)}",
                component_scores={component.value: 0.0 for component in PrecedentAnalysisComponent},
                metadata={"error": str(e)}
            )
    
    async def _evaluate_component(self, component: PrecedentAnalysisComponent, 
                                data_point: LegalDataPoint, 
                                jurisdiction_context: str) -> Tuple[float, str]:
        """
        Evaluate a specific precedent analysis component using professional prompt templates.
        
        Args:
            component: Component to evaluate
            data_point: Legal data point
            jurisdiction_context: Jurisdiction-specific context
            
        Returns:
            Tuple of (score, reasoning)
        """
        
        try:
            # Map component to prompt type
            prompt_type_mapping = {
                PrecedentAnalysisComponent.CASE_LAW_ACCURACY: PrecedentAnalysisPromptType.CASE_LAW_ACCURACY,
                PrecedentAnalysisComponent.ANALOGICAL_REASONING: PrecedentAnalysisPromptType.ANALOGICAL_REASONING,
                PrecedentAnalysisComponent.DISTINGUISHING_FACTORS: PrecedentAnalysisPromptType.DISTINGUISHING_FACTORS,
                PrecedentAnalysisComponent.HIERARCHY_UNDERSTANDING: PrecedentAnalysisPromptType.HIERARCHY_UNDERSTANDING,
                PrecedentAnalysisComponent.CITATION_QUALITY: PrecedentAnalysisPromptType.CITATION_QUALITY
            }
            
            prompt_type = prompt_type_mapping[component]
            
            # Get professional prompt template
            prompt = get_precedent_analysis_prompt(
                prompt_type=prompt_type,
                jurisdiction=data_point.jurisdiction,
                response=data_point.response,
                query=data_point.query,
                jurisdiction_context=jurisdiction_context
            )
            
            # Evaluate using API client
            result = await self.api_client.evaluate_legal_response(
                query=data_point.query,
                response=data_point.response,
                judge_type=f"precedent_analysis_{component.value}",
                task_type=data_point.task_type.value,
                jurisdiction=data_point.jurisdiction,
                prompt_template=prompt
            )
            
            # Parse result
            score, reasoning = self._parse_api_result(result)
            
            return score, reasoning
            
        except Exception as e:
            self.logger.error(f"Component evaluation failed for {component}: {str(e)}")
            return 0.0, f"Component evaluation failed: {str(e)}"
    
    async def _evaluate_jurisdiction_compliance(self, data_point: LegalDataPoint, 
                                              jurisdiction_context: str) -> float:
        """
        Evaluate jurisdiction compliance for precedent analysis.
        
        This uses the jurisdiction compliance prompt templates to assess whether
        the precedent analysis meets jurisdiction-specific case law requirements.
        """
        
        try:
            # Import jurisdiction compliance prompt function
            from ...config.prompts import (
                get_jurisdiction_compliance_prompt,
                JurisdictionCompliancePromptType,
                JurisdictionComplianceContext
            )
            
            # Create context for jurisdiction compliance evaluation
            compliance_context = JurisdictionComplianceContext(
                query=data_point.query,
                response=data_point.response,
                jurisdiction=data_point.jurisdiction,
                task_type=data_point.task_type,
                legal_domain=getattr(data_point, 'legal_domain', 'General'),
                complexity_level="standard",
                federal_implications=data_point.jurisdiction == USJurisdiction.FEDERAL
            )
            
            # Use procedural compliance for precedent analysis (focuses on court hierarchy)
            prompt = get_jurisdiction_compliance_prompt(
                JurisdictionCompliancePromptType.PROCEDURAL_COMPLIANCE,
                compliance_context
            )
            
            # Evaluate jurisdiction compliance
            result = await self.api_client.evaluate_legal_response(
                query=data_point.query,
                response=data_point.response,
                judge_type="jurisdiction_compliance_precedent",
                task_type=data_point.task_type.value,
                jurisdiction=data_point.jurisdiction,
                prompt_template=prompt
            )
            
            # Parse result
            score, _ = self._parse_api_result(result)
            return score
            
        except Exception as e:
            self.logger.error(f"Jurisdiction compliance evaluation failed: {str(e)}")
            return 0.0
    
    def _create_jurisdiction_context(self, data_point: LegalDataPoint) -> str:
        """Create jurisdiction-specific context for precedent analysis evaluation"""
        
        jurisdiction_contexts = {
            USJurisdiction.FEDERAL: "Federal court case law analysis with federal precedent hierarchy and constitutional focus",
            USJurisdiction.CALIFORNIA: "California state court case law analysis with California Supreme Court and appellate precedents",
            USJurisdiction.NEW_YORK: "New York state court case law analysis with Court of Appeals and commercial law emphasis",
            USJurisdiction.TEXAS: "Texas state court case law analysis with dual supreme court system and business law focus",
            USJurisdiction.FLORIDA: "Florida state court case law analysis with Florida Supreme Court and district court precedents",
            USJurisdiction.GENERAL: "General U.S. case law analysis applicable across jurisdictions with standard precedent principles"
        }
        
        base_context = jurisdiction_contexts.get(data_point.jurisdiction, jurisdiction_contexts[USJurisdiction.GENERAL])
        
        # Add task-specific context
        if hasattr(data_point, 'legal_domain'):
            base_context += f" in the area of {data_point.legal_domain}"
        
        return base_context
    
    def _parse_api_result(self, api_result: Dict[str, Any]) -> Tuple[float, str]:
        """
        Parse API result to extract score and reasoning.
        
        Args:
            api_result: API response
            
        Returns:
            Tuple of (score, reasoning)
        """
        
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
    
    def _generate_overall_reasoning(self, precedent_score: PrecedentAnalysisScore, 
                                  data_point: LegalDataPoint) -> str:
        """Generate comprehensive reasoning for the overall evaluation"""
        
        reasoning_parts = [
            f"Comprehensive precedent analysis evaluation for {data_point.jurisdiction.value} jurisdiction:",
            f"Overall Score: {precedent_score.overall_score:.1f}/10 (Confidence: {precedent_score.confidence:.1f})",
            ""
        ]
        
        # Component breakdown
        components = [
            ("Case Law Accuracy", precedent_score.case_law_accuracy),
            ("Analogical Reasoning", precedent_score.analogical_reasoning),
            ("Distinguishing Factors", precedent_score.distinguishing_factors),
            ("Hierarchy Understanding", precedent_score.hierarchy_understanding),
            ("Citation Quality", precedent_score.citation_quality)
        ]
        
        reasoning_parts.append("Component Analysis:")
        for component_name, score in components:
            reasoning_parts.append(f"• {component_name}: {score:.1f}/10")
        
        reasoning_parts.append("")
        
        # Jurisdiction compliance
        if precedent_score.jurisdiction_compliant:
            reasoning_parts.append(f"✓ Jurisdiction Compliance: {precedent_score.jurisdiction_compliance:.1f}/10 (PASSED)")
        else:
            reasoning_parts.append(f"✗ Jurisdiction Compliance: {precedent_score.jurisdiction_compliance:.1f}/10 (FAILED - Below threshold)")
        
        # Key strengths and weaknesses
        strengths = []
        weaknesses = []
        
        for component_name, score in components:
            if score >= 8.0:
                strengths.append(component_name)
            elif score < 5.0:
                weaknesses.append(component_name)
        
        if strengths:
            reasoning_parts.append(f"\nStrengths: {', '.join(strengths)}")
        
        if weaknesses:
            reasoning_parts.append(f"Areas for Improvement: {', '.join(weaknesses)}")
        
        # Precedent-specific insights
        reasoning_parts.append(f"\nPrecedent Analysis Context:")
        reasoning_parts.append(f"• Jurisdiction: {data_point.jurisdiction.value} case law system")
        reasoning_parts.append(f"• Focus: Case law accuracy, analogical reasoning, and citation quality")
        reasoning_parts.append(f"• Evaluation: Professional-grade precedent analysis standards")
        
        return "\n".join(reasoning_parts)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the ensemble"""
        
        avg_evaluation_time = self.total_evaluation_time / max(1, self.evaluation_count)
        jurisdiction_failure_rate = len(self.jurisdiction_compliance_failures) / max(1, self.evaluation_count)
        
        return {
            "ensemble_name": self.ensemble_name,
            "evaluations_completed": self.evaluation_count,
            "total_evaluation_time": self.total_evaluation_time,
            "average_evaluation_time": avg_evaluation_time,
            "api_calls_made": self.api_call_count,
            "cache_hits": self.cache_hit_count,
            "jurisdiction_compliance_failure_rate": jurisdiction_failure_rate,
            "jurisdiction_failures": len(self.jurisdiction_compliance_failures)
        }
    
    def get_jurisdiction_compliance_failures(self) -> List[Dict[str, Any]]:
        """Get jurisdiction compliance failure details"""
        return self.jurisdiction_compliance_failures.copy()


# Factory functions for easy ensemble creation
def create_precedent_analysis_ensemble(config: LegalRewardSystemConfig,
                                     api_client: Optional[CostOptimizedAPIClient] = None) -> PrecedentAnalysisEnsemble:
    """
    Create production precedent analysis ensemble with API-based evaluation.
    
    Args:
        config: System configuration
        api_client: Optional API client (will create if not provided)
        
    Returns:
        Configured precedent analysis ensemble
    """
    
    return PrecedentAnalysisEnsemble(config, api_client)


def create_development_precedent_analysis_ensemble(config: LegalRewardSystemConfig) -> PrecedentAnalysisEnsemble:
    """
    Create development precedent analysis ensemble with optimizations for testing.
    
    Args:
        config: System configuration
        
    Returns:
        Configured development ensemble
    """
    
    # Create with development-optimized API client
    api_client = CostOptimizedAPIClient(config)
    ensemble = PrecedentAnalysisEnsemble(config, api_client)
    
    # Development optimizations
    ensemble.component_weights = {component: 0.20 for component in PrecedentAnalysisComponent}
    
    return ensemble


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    from ...config.settings import create_default_config
    from ...core.data_structures import LegalDataPoint
    from ...core.enums import LegalTaskType
    
    async def test_precedent_analysis_ensemble():
        # Create test configuration
        config = create_default_config()
        
        # Create ensemble
        ensemble = create_precedent_analysis_ensemble(config)
        
        # Create test data point
        test_data = LegalDataPoint(
            query="Analyze relevant precedent for establishing proximate cause in negligence",
            response="The landmark case of Palsgraf v. Long Island Railroad established the foreseeability test for proximate cause. In applying this precedent to our facts, we must consider whether the defendant could reasonably foresee...",
            task_type=LegalTaskType.PRECEDENT_ANALYSIS,
            jurisdiction=USJurisdiction.NEW_YORK,
            ground_truth="High-quality precedent analysis with proper case citations expected",
            metadata={"legal_domain": "Tort Law"}
        )
        
        # Evaluate
        print("Testing Precedent Analysis Ensemble with Professional Prompt Templates...")
        result = await ensemble.evaluate(test_data)
        
        print(f"Score: {result.score:.2f}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Reasoning: {result.reasoning[:200]}...")
        print(f"Component Scores: {result.component_scores}")
        
        # Show performance stats
        stats = ensemble.get_performance_stats()
        print(f"\nPerformance Stats:")
        print(f"Evaluations: {stats['evaluations_completed']}")
        print(f"Avg Time: {stats['average_evaluation_time']:.2f}s")
        print(f"API Calls: {stats['api_calls_made']}")
    
    # Run test
    if hasattr(asyncio, 'run'):
        asyncio.run(test_precedent_analysis_ensemble())
    else:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(test_precedent_analysis_ensemble())
