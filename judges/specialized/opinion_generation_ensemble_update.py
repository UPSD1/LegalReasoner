"""
Opinion Generation Ensemble - Updated with Professional Prompt Templates

This ensemble evaluates legal advocacy, persuasive writing, and client representation
using sophisticated US jurisdiction-aware prompt templates. This is a core component 
of the hybrid evaluation system (70% specialized + 30% chat).

Updated Features:
- Professional prompt template integration from config/prompts/opinion_generation.py
- Sophisticated US jurisdiction advocacy awareness for all 50 states + DC + federal
- Enhanced evaluation components with jurisdiction-specific advocacy context
- Optimized caching and performance for GRPO training
- Full VERL integration compatibility

Evaluation Components (Equal Weights):
- Argument Strength (20%): Logical force and legal foundation of arguments
- Persuasiveness (20%): Rhetorical effectiveness and compelling presentation
- Legal Research Quality (20%): Depth and relevance of legal support
- Client Advocacy (20%): Effectiveness in advancing client interests
- Professional Writing (20%): Legal writing quality and clarity

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
    OpinionGenerationPromptType,
    get_opinion_generation_prompt,
    get_all_opinion_generation_prompts,
    validate_opinion_generation_prompt
)


class OpinionGenerationComponent(Enum):
    """Components of opinion generation evaluation"""
    ARGUMENT_STRENGTH = "argument_strength"
    PERSUASIVENESS = "persuasiveness"
    LEGAL_RESEARCH_QUALITY = "legal_research_quality"
    CLIENT_ADVOCACY = "client_advocacy"
    PROFESSIONAL_WRITING = "professional_writing"


@dataclass
class OpinionGenerationScore:
    """Detailed scoring for opinion generation evaluation"""
    
    # Component scores (0-10 scale)
    argument_strength: float = 0.0
    persuasiveness: float = 0.0
    legal_research_quality: float = 0.0
    client_advocacy: float = 0.0
    professional_writing: float = 0.0
    
    # Aggregate metrics
    overall_score: float = 0.0
    confidence: float = 0.0
    
    # Jurisdiction compliance
    jurisdiction_compliance: float = 0.0
    jurisdiction_compliant: bool = False
    
    # Component weights (configurable)
    component_weights: Dict[str, float] = field(default_factory=lambda: {
        "argument_strength": 0.20,
        "persuasiveness": 0.20,
        "legal_research_quality": 0.20,
        "client_advocacy": 0.20,
        "professional_writing": 0.20
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
            "argument_strength": self.argument_strength,
            "persuasiveness": self.persuasiveness,
            "legal_research_quality": self.legal_research_quality,
            "client_advocacy": self.client_advocacy,
            "professional_writing": self.professional_writing
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
            self.argument_strength, self.persuasiveness, self.legal_research_quality,
            self.client_advocacy, self.professional_writing
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


class OpinionGenerationEnsemble(BaseJudgeEnsemble):
    """
    Professional opinion generation evaluation ensemble with sophisticated 
    prompt templates and advocacy jurisdiction awareness.
    
    This ensemble uses the professional prompt templates from config/prompts/opinion_generation.py
    to provide expert-level evaluation of legal advocacy and persuasive writing.
    """
    
    def __init__(self, config: LegalRewardSystemConfig, 
                 api_client: Optional[CostOptimizedAPIClient] = None):
        super().__init__("opinion_generation_ensemble", LegalTaskType.OPINION_GENERATION)
        
        self.config = config
        self.api_client = api_client or CostOptimizedAPIClient(config)
        
        # Component weights (configurable via config)
        self.component_weights = {
            OpinionGenerationComponent.ARGUMENT_STRENGTH: 0.20,
            OpinionGenerationComponent.PERSUASIVENESS: 0.20,
            OpinionGenerationComponent.LEGAL_RESEARCH_QUALITY: 0.20,
            OpinionGenerationComponent.CLIENT_ADVOCACY: 0.20,
            OpinionGenerationComponent.PROFESSIONAL_WRITING: 0.20
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
            
            for prompt_type in OpinionGenerationPromptType:
                if not validate_opinion_generation_prompt(prompt_type, test_jurisdiction):
                    raise ValueError(f"Prompt validation failed for {prompt_type} in {test_jurisdiction}")
            
            self.logger.info("Opinion generation prompt system validation passed")
            
        except Exception as e:
            self.logger.error(f"Opinion generation prompt system validation failed: {str(e)}")
            if self.config.environment.value == "production":
                raise
    
    async def evaluate(self, data_point: LegalDataPoint) -> EnsembleScore:
        """
        Comprehensive opinion generation evaluation using professional prompt templates.
        
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
            
            for component in OpinionGenerationComponent:
                task = self._evaluate_component(component, data_point, jurisdiction_context)
                component_tasks.append(task)
            
            # Execute all evaluations
            component_results = await asyncio.gather(*component_tasks, return_exceptions=True)
            
            # Process results
            opinion_score = OpinionGenerationScore()
            component_scores = {}
            
            for i, component in enumerate(OpinionGenerationComponent):
                result = component_results[i]
                
                if isinstance(result, Exception):
                    self.logger.error(f"Component evaluation failed for {component}: {str(result)}")
                    score = 0.0
                    reasoning = f"Evaluation failed: {str(result)}"
                else:
                    score, reasoning = result
                
                # Set component score
                setattr(opinion_score, component.value, score)
                component_scores[component.value] = score
                opinion_score.component_reasoning[component.value] = reasoning
            
            # Calculate aggregate metrics
            opinion_score.calculate_overall_score()
            opinion_score.calculate_confidence()
            
            # Check jurisdiction compliance
            jurisdiction_compliance_score = await self._evaluate_jurisdiction_compliance(data_point, jurisdiction_context)
            opinion_score.jurisdiction_compliance = jurisdiction_compliance_score
            opinion_score.jurisdiction_compliant = jurisdiction_compliance_score >= self.config.hybrid_evaluation.jurisdiction_compliance_threshold
            
            # Track jurisdiction compliance failures
            if not opinion_score.jurisdiction_compliant:
                self.jurisdiction_compliance_failures.append({
                    "jurisdiction": data_point.jurisdiction,
                    "score": jurisdiction_compliance_score,
                    "query": data_point.query[:100]  # Truncated for logging
                })
            
            # Generate overall reasoning
            opinion_score.overall_reasoning = self._generate_overall_reasoning(opinion_score, data_point)
            
            # Performance metadata
            end_time = asyncio.get_event_loop().time()
            opinion_score.evaluation_time_seconds = end_time - start_time
            opinion_score.api_calls_made = len(component_tasks) + 1  # Components + jurisdiction compliance
            
            self.total_evaluation_time += opinion_score.evaluation_time_seconds
            self.api_call_count += opinion_score.api_calls_made
            
            # Create ensemble score
            ensemble_score = EnsembleScore(
                score=opinion_score.overall_score,
                confidence=opinion_score.confidence,
                reasoning=opinion_score.overall_reasoning,
                component_scores=component_scores,
                metadata={
                    "jurisdiction_compliance": opinion_score.jurisdiction_compliance,
                    "jurisdiction_compliant": opinion_score.jurisdiction_compliant,
                    "component_reasoning": opinion_score.component_reasoning,
                    "evaluation_time": opinion_score.evaluation_time_seconds,
                    "api_calls": opinion_score.api_calls_made
                }
            )
            
            self.logger.debug(f"Opinion generation evaluation completed: score={opinion_score.overall_score:.2f}, confidence={opinion_score.confidence:.2f}")
            
            return ensemble_score
            
        except Exception as e:
            self.logger.error(f"Opinion generation evaluation failed: {str(e)}")
            
            # Return failed evaluation score
            return EnsembleScore(
                score=0.0,
                confidence=0.0,
                reasoning=f"Evaluation failed: {str(e)}",
                component_scores={component.value: 0.0 for component in OpinionGenerationComponent},
                metadata={"error": str(e)}
            )
    
    async def _evaluate_component(self, component: OpinionGenerationComponent, 
                                data_point: LegalDataPoint, 
                                jurisdiction_context: str) -> Tuple[float, str]:
        """
        Evaluate a specific opinion generation component using professional prompt templates.
        
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
                OpinionGenerationComponent.ARGUMENT_STRENGTH: OpinionGenerationPromptType.ARGUMENT_STRENGTH,
                OpinionGenerationComponent.PERSUASIVENESS: OpinionGenerationPromptType.PERSUASIVENESS,
                OpinionGenerationComponent.LEGAL_RESEARCH_QUALITY: OpinionGenerationPromptType.LEGAL_RESEARCH_QUALITY,
                OpinionGenerationComponent.CLIENT_ADVOCACY: OpinionGenerationPromptType.CLIENT_ADVOCACY,
                OpinionGenerationComponent.PROFESSIONAL_WRITING: OpinionGenerationPromptType.PROFESSIONAL_WRITING
            }
            
            prompt_type = prompt_type_mapping[component]
            
            # Get professional prompt template
            prompt = get_opinion_generation_prompt(
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
                judge_type=f"opinion_generation_{component.value}",
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
        Evaluate jurisdiction compliance for opinion generation.
        
        This uses the jurisdiction compliance prompt templates to assess whether
        the legal advocacy meets jurisdiction-specific professional standards.
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
            
            # Use professional standards for opinion generation (focuses on advocacy ethics)
            prompt = get_jurisdiction_compliance_prompt(
                JurisdictionCompliancePromptType.PROFESSIONAL_STANDARDS,
                compliance_context
            )
            
            # Evaluate jurisdiction compliance
            result = await self.api_client.evaluate_legal_response(
                query=data_point.query,
                response=data_point.response,
                judge_type="jurisdiction_compliance_opinion",
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
        """Create jurisdiction-specific context for opinion generation evaluation"""
        
        jurisdiction_contexts = {
            USJurisdiction.FEDERAL: "Federal court advocacy with constitutional and federal statutory emphasis",
            USJurisdiction.CALIFORNIA: "California state court advocacy with progressive legal approach and consumer protection focus",
            USJurisdiction.NEW_YORK: "New York state court advocacy with commercial law sophistication and business practice emphasis",
            USJurisdiction.TEXAS: "Texas state court advocacy with business-friendly approach and property rights focus",
            USJurisdiction.FLORIDA: "Florida state court advocacy with traditional-modern balance and practical approach",
            USJurisdiction.GENERAL: "General U.S. legal advocacy applicable across jurisdictions with standard professional practices"
        }
        
        base_context = jurisdiction_contexts.get(data_point.jurisdiction, jurisdiction_contexts[USJurisdiction.GENERAL])
        
        # Add task-specific context
        if hasattr(data_point, 'legal_domain'):
            base_context += f" in the area of {data_point.legal_domain}"
        
        # Add client context if available
        if hasattr(data_point, 'client_type'):
            base_context += f" for {data_point.client_type} client"
        
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
    
    def _generate_overall_reasoning(self, opinion_score: OpinionGenerationScore, 
                                  data_point: LegalDataPoint) -> str:
        """Generate comprehensive reasoning for the overall evaluation"""
        
        reasoning_parts = [
            f"Comprehensive opinion generation evaluation for {data_point.jurisdiction.value} jurisdiction:",
            f"Overall Score: {opinion_score.overall_score:.1f}/10 (Confidence: {opinion_score.confidence:.1f})",
            ""
        ]
        
        # Component breakdown
        components = [
            ("Argument Strength", opinion_score.argument_strength),
            ("Persuasiveness", opinion_score.persuasiveness),
            ("Legal Research Quality", opinion_score.legal_research_quality),
            ("Client Advocacy", opinion_score.client_advocacy),
            ("Professional Writing", opinion_score.professional_writing)
        ]
        
        reasoning_parts.append("Component Analysis:")
        for component_name, score in components:
            reasoning_parts.append(f"• {component_name}: {score:.1f}/10")
        
        reasoning_parts.append("")
        
        # Jurisdiction compliance
        if opinion_score.jurisdiction_compliant:
            reasoning_parts.append(f"✓ Jurisdiction Compliance: {opinion_score.jurisdiction_compliance:.1f}/10 (PASSED)")
        else:
            reasoning_parts.append(f"✗ Jurisdiction Compliance: {opinion_score.jurisdiction_compliance:.1f}/10 (FAILED - Below threshold)")
        
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
        
        # Advocacy-specific insights
        reasoning_parts.append(f"\nLegal Advocacy Context:")
        reasoning_parts.append(f"• Jurisdiction: {data_point.jurisdiction.value} advocacy standards")
        reasoning_parts.append(f"• Focus: Argument strength, persuasiveness, and client representation")
        reasoning_parts.append(f"• Evaluation: Professional-grade legal advocacy standards")
        
        # Professional responsibility note
        if opinion_score.jurisdiction_compliance < 5.0:
            reasoning_parts.append(f"• NOTICE: Low jurisdiction compliance may indicate professional responsibility concerns")
        
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
def create_opinion_generation_ensemble(config: LegalRewardSystemConfig,
                                     api_client: Optional[CostOptimizedAPIClient] = None) -> OpinionGenerationEnsemble:
    """
    Create production opinion generation ensemble with API-based evaluation.
    
    Args:
        config: System configuration
        api_client: Optional API client (will create if not provided)
        
    Returns:
        Configured opinion generation ensemble
    """
    
    return OpinionGenerationEnsemble(config, api_client)


def create_development_opinion_generation_ensemble(config: LegalRewardSystemConfig) -> OpinionGenerationEnsemble:
    """
    Create development opinion generation ensemble with optimizations for testing.
    
    Args:
        config: System configuration
        
    Returns:
        Configured development ensemble
    """
    
    # Create with development-optimized API client
    api_client = CostOptimizedAPIClient(config)
    ensemble = OpinionGenerationEnsemble(config, api_client)
    
    # Development optimizations
    ensemble.component_weights = {component: 0.20 for component in OpinionGenerationComponent}
    
    return ensemble


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    from ...config.settings import create_default_config
    from ...core.data_structures import LegalDataPoint
    from ...core.enums import LegalTaskType
    
    async def test_opinion_generation_ensemble():
        # Create test configuration
        config = create_default_config()
        
        # Create ensemble
        ensemble = create_opinion_generation_ensemble(config)
        
        # Create test data point
        test_data = LegalDataPoint(
            query="Draft a persuasive motion for summary judgment in a contract dispute",
            response="Your Honor, Plaintiff respectfully moves for summary judgment on the grounds that there are no genuine issues of material fact and Plaintiff is entitled to judgment as a matter of law. The evidence clearly establishes that Defendant breached the contract by failing to deliver conforming goods as required under Section 2.1...",
            task_type=LegalTaskType.OPINION_GENERATION,
            jurisdiction=USJurisdiction.CALIFORNIA,
            ground_truth="High-quality legal advocacy with strong arguments and proper jurisdiction understanding expected",
            metadata={
                "legal_domain": "Contract Law",
                "client_type": "commercial_plaintiff"
            }
        )
        
        # Evaluate
        print("Testing Opinion Generation Ensemble with Professional Prompt Templates...")
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
        asyncio.run(test_opinion_generation_ensemble())
    else:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(test_opinion_generation_ensemble())
