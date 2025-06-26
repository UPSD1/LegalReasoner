"""
Jurisdiction Compliance Judge for Multi-Task Legal Reward System

This module provides comprehensive jurisdiction compliance evaluation for legal responses,
ensuring that legal advice and analysis are appropriately contextualized within
the correct US jurisdiction and comply with jurisdiction-specific requirements.

Key Features:
- Jurisdiction compliance scoring and validation
- Context appropriateness evaluation
- Cross-jurisdictional conflict detection
- Disclaimer and limitation assessment
- Integration with jurisdiction inference engine
- Gating function for hybrid evaluation system

The compliance judge acts as a critical gating component in the hybrid evaluation
system, ensuring that legal responses meet jurisdiction compliance standards
before proceeding with specialized evaluation.
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json

# Import core components
from ..core import (
    LegalRewardEvaluation, JudgeEvaluation, EvaluationMetadata,
    USJurisdiction, LegalDomain, LegalTaskType,
    LegalRewardSystemError, create_error_context
)
from .us_system import (
    JurisdictionMetadata, USJurisdictionError,
    get_all_jurisdiction_metadata, validate_jurisdiction,
    get_jurisdiction_context, is_jurisdiction_federal_only
)
from .inference_engine import (
    JurisdictionInferenceEngine, JurisdictionInferenceResult,
    create_production_inference_engine
)


class ComplianceViolationType(Enum):
    """Types of jurisdiction compliance violations"""
    WRONG_JURISDICTION = "wrong_jurisdiction"
    FEDERAL_STATE_CONFUSION = "federal_state_confusion"
    MISSING_DISCLAIMERS = "missing_disclaimers"
    OVERGENERALIZATION = "overgeneralization"
    CROSS_JURISDICTIONAL_CONFLICT = "cross_jurisdictional_conflict"
    INAPPROPRIATE_SPECIFICITY = "inappropriate_specificity"
    DOMAIN_JURISDICTION_MISMATCH = "domain_jurisdiction_mismatch"
    INSUFFICIENT_CONTEXT = "insufficient_context"


@dataclass
class ComplianceViolation:
    """
    Represents a jurisdiction compliance violation.
    
    Contains detailed information about the violation type, severity,
    location, and recommended corrections for comprehensive compliance assessment.
    """
    
    violation_type: ComplianceViolationType
    severity: float  # 0.0 to 1.0 (1.0 = critical)
    description: str
    location: str  # Where in the response the violation occurs
    
    # Contextual information
    expected_jurisdiction: Optional[USJurisdiction] = None
    found_jurisdiction: Optional[USJurisdiction] = None
    affected_domains: List[LegalDomain] = field(default_factory=list)
    
    # Correction guidance
    recommendation: str = ""
    correction_priority: int = 1  # 1 = high, 2 = medium, 3 = low
    
    def is_critical(self) -> bool:
        """Check if violation is critical (blocks evaluation)"""
        return self.severity >= 0.8
    
    def is_major(self) -> bool:
        """Check if violation is major (significant penalty)"""
        return self.severity >= 0.6
    
    def get_penalty_multiplier(self) -> float:
        """Get penalty multiplier for scoring"""
        if self.is_critical():
            return 0.2  # 80% penalty
        elif self.is_major():
            return 0.6  # 40% penalty
        else:
            return 0.8  # 20% penalty


@dataclass
class JurisdictionComplianceResult:
    """
    Result of jurisdiction compliance evaluation.
    
    Contains compliance score, violations, recommendations, and
    gating decisions for the hybrid evaluation system.
    """
    
    # Core compliance assessment
    compliance_score: float  # 0.0 to 10.0
    is_compliant: bool
    gating_decision: bool  # Whether to proceed with specialized evaluation
    
    # Violation analysis
    violations: List[ComplianceViolation] = field(default_factory=list)
    critical_violations: List[ComplianceViolation] = field(default_factory=list)
    
    # Jurisdiction analysis
    expected_jurisdiction: USJurisdiction = USJurisdiction.GENERAL
    inferred_jurisdiction: Optional[USJurisdiction] = None
    jurisdiction_consistency: float = 1.0  # 0.0 to 1.0
    
    # Context evaluation
    context_appropriateness: float = 1.0  # 0.0 to 1.0
    disclaimer_adequacy: float = 1.0  # 0.0 to 1.0
    domain_alignment: float = 1.0  # 0.0 to 1.0
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    required_corrections: List[str] = field(default_factory=list)
    
    # Metadata
    evaluation_method: str = "comprehensive"
    confidence: float = 1.0
    processing_time_ms: float = 0.0
    
    def get_overall_penalty(self) -> float:
        """Get overall penalty multiplier from all violations"""
        if not self.violations:
            return 1.0
        
        # Apply penalties multiplicatively (more violations = higher penalty)
        penalty_multiplier = 1.0
        for violation in self.violations:
            penalty_multiplier *= violation.get_penalty_multiplier()
        
        return penalty_multiplier
    
    def has_critical_violations(self) -> bool:
        """Check if there are critical violations"""
        return len(self.critical_violations) > 0
    
    def get_summary(self) -> str:
        """Get human-readable summary"""
        status = "COMPLIANT" if self.is_compliant else "NON-COMPLIANT"
        gating = "PASS" if self.gating_decision else "BLOCK"
        
        summary = f"Compliance: {status} (Score: {self.compliance_score:.1f}/10.0) | Gating: {gating}"
        
        if self.violations:
            critical_count = len(self.critical_violations)
            total_count = len(self.violations)
            summary += f" | Violations: {total_count} ({critical_count} critical)"
        
        return summary


class JurisdictionAnalyzer:
    """
    Analyzes legal responses for jurisdiction-specific content and compliance.
    
    Evaluates jurisdiction consistency, context appropriateness, and
    domain-jurisdiction alignment in legal responses.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.JurisdictionAnalyzer")
        
        # Load jurisdiction metadata
        self.jurisdiction_metadata = get_all_jurisdiction_metadata()
        
        # Initialize inference engine
        self.inference_engine = create_production_inference_engine()
        
        # Build jurisdiction-specific patterns
        self._build_jurisdiction_patterns()
        
        # Build disclaimer patterns
        self._build_disclaimer_patterns()
    
    def _build_jurisdiction_patterns(self):
        """Build patterns for detecting jurisdiction-specific content"""
        
        # Federal patterns
        self.federal_patterns = [
            r"federal\s+law",
            r"federal\s+court",
            r"u\.?s\.?\s+code",
            r"federal\s+statute",
            r"supreme\s+court\s+of\s+the\s+united\s+states",
            r"federal\s+regulation",
            r"interstate\s+commerce",
            r"constitutional\s+law"
        ]
        
        # State-specific patterns
        self.state_patterns = {}
        for jurisdiction, metadata in self.jurisdiction_metadata.items():
            if metadata.is_state_jurisdiction():
                patterns = [
                    rf"{metadata.full_name}\s+law",
                    rf"{metadata.full_name}\s+statute",
                    rf"{metadata.full_name}\s+court",
                    rf"{metadata.abbreviation}\s+law",
                    rf"state\s+of\s+{metadata.full_name.lower()}"
                ]
                self.state_patterns[jurisdiction] = patterns
        
        # General legal patterns
        self.general_patterns = [
            r"generally",
            r"in\s+most\s+jurisdictions",
            r"common\s+law\s+principles",
            r"legal\s+principles",
            r"may\s+vary\s+by\s+jurisdiction"
        ]
    
    def _build_disclaimer_patterns(self):
        """Build patterns for detecting appropriate disclaimers"""
        
        self.disclaimer_patterns = [
            r"this\s+is\s+not\s+legal\s+advice",
            r"consult\s+(?:an?\s+)?attorney",
            r"seek\s+legal\s+counsel",
            r"laws?\s+may\s+vary\s+by\s+jurisdiction",
            r"specific\s+to\s+your\s+jurisdiction",
            r"consult\s+local\s+laws?",
            r"may\s+differ\s+in\s+your\s+state",
            r"general\s+information\s+only"
        ]
        
        self.jurisdiction_specific_disclaimer_patterns = [
            r"under\s+\w+\s+law",
            r"in\s+the\s+state\s+of\s+\w+",
            r"federal\s+law\s+provides",
            r"according\s+to\s+\w+\s+statute"
        ]
    
    def analyze_jurisdiction_consistency(self, 
                                       response: str,
                                       expected_jurisdiction: USJurisdiction,
                                       task_type: LegalTaskType) -> Dict[str, Any]:
        """
        Analyze jurisdiction consistency in legal response.
        
        Args:
            response: Legal response text
            expected_jurisdiction: Expected jurisdiction context
            task_type: Type of legal task
            
        Returns:
            Dictionary with consistency analysis results
        """
        # Infer jurisdiction from response content
        inference_result = self.inference_engine.infer_jurisdiction(response, task_type)
        
        # Analyze jurisdiction references in response
        jurisdiction_refs = self._extract_jurisdiction_references(response)
        
        # Check for cross-jurisdictional conflicts
        conflicts = self._detect_jurisdictional_conflicts(jurisdiction_refs, expected_jurisdiction)
        
        # Evaluate context appropriateness
        context_score = self._evaluate_context_appropriateness(
            response, expected_jurisdiction, inference_result
        )
        
        # Calculate consistency score
        consistency_score = self._calculate_consistency_score(
            expected_jurisdiction, inference_result, conflicts, context_score
        )
        
        return {
            "consistency_score": consistency_score,
            "inferred_jurisdiction": inference_result.jurisdiction,
            "inference_confidence": inference_result.confidence,
            "jurisdiction_references": jurisdiction_refs,
            "conflicts": conflicts,
            "context_score": context_score,
            "inference_result": inference_result
        }
    
    def _extract_jurisdiction_references(self, response: str) -> Dict[str, List[str]]:
        """Extract all jurisdiction references from response"""
        
        refs = {
            "federal": [],
            "state": {},
            "general": []
        }
        
        response_lower = response.lower()
        
        # Find federal references
        for pattern in self.federal_patterns:
            matches = re.finditer(pattern, response_lower, re.IGNORECASE)
            refs["federal"].extend([match.group() for match in matches])
        
        # Find state-specific references
        for jurisdiction, patterns in self.state_patterns.items():
            state_refs = []
            for pattern in patterns:
                matches = re.finditer(pattern, response_lower, re.IGNORECASE)
                state_refs.extend([match.group() for match in matches])
            
            if state_refs:
                refs["state"][jurisdiction] = state_refs
        
        # Find general references
        for pattern in self.general_patterns:
            matches = re.finditer(pattern, response_lower, re.IGNORECASE)
            refs["general"].extend([match.group() for match in matches])
        
        return refs
    
    def _detect_jurisdictional_conflicts(self, 
                                       jurisdiction_refs: Dict[str, Any],
                                       expected_jurisdiction: USJurisdiction) -> List[Dict[str, Any]]:
        """Detect conflicts between different jurisdiction references"""
        
        conflicts = []
        
        # Check federal vs state conflicts
        federal_refs = jurisdiction_refs.get("federal", [])
        state_refs = jurisdiction_refs.get("state", {})
        
        if expected_jurisdiction == USJurisdiction.FEDERAL and state_refs:
            for state_jurisdiction, refs in state_refs.items():
                conflicts.append({
                    "type": "federal_state_conflict",
                    "description": f"Expected federal jurisdiction but found {state_jurisdiction.value} references",
                    "evidence": refs[:3],  # First 3 references
                    "severity": 0.7
                })
        
        elif expected_jurisdiction != USJurisdiction.FEDERAL and federal_refs:
            if expected_jurisdiction in state_refs:
                # Mixed federal and specific state references
                conflicts.append({
                    "type": "mixed_jurisdiction",
                    "description": f"Mixed federal and {expected_jurisdiction.value} references",
                    "evidence": federal_refs[:2] + state_refs[expected_jurisdiction][:2],
                    "severity": 0.5
                })
        
        # Check multiple state conflicts
        if len(state_refs) > 1:
            states = list(state_refs.keys())
            conflicts.append({
                "type": "multiple_states",
                "description": f"References to multiple states: {[s.value for s in states]}",
                "evidence": [refs[0] for refs in state_refs.values() if refs],
                "severity": 0.6
            })
        
        return conflicts
    
    def _evaluate_context_appropriateness(self, 
                                        response: str,
                                        expected_jurisdiction: USJurisdiction,
                                        inference_result: JurisdictionInferenceResult) -> float:
        """Evaluate appropriateness of jurisdiction context in response"""
        
        # Base score
        score = 1.0
        
        # Check if response mentions jurisdiction when it should
        response_lower = response.lower()
        
        # For federal jurisdiction, should mention federal aspects
        if expected_jurisdiction == USJurisdiction.FEDERAL:
            federal_indicators = any(pattern in response_lower for pattern in [
                "federal", "constitutional", "interstate", "national"
            ])
            if not federal_indicators:
                score *= 0.7  # 30% penalty for missing federal context
        
        # For state jurisdiction, should mention state-specific aspects
        elif expected_jurisdiction != USJurisdiction.GENERAL:
            metadata = self.jurisdiction_metadata.get(expected_jurisdiction)
            if metadata:
                state_indicators = any(indicator in response_lower for indicator in [
                    metadata.full_name.lower(),
                    metadata.abbreviation.lower(),
                    "state law",
                    "state court"
                ])
                if not state_indicators:
                    score *= 0.8  # 20% penalty for missing state context
        
        # Check for appropriate generality/specificity
        if expected_jurisdiction == USJurisdiction.GENERAL:
            # Should not be overly specific to one jurisdiction
            overly_specific = (
                len(re.findall(r"(?:in|under)\s+\w+\s+law", response_lower)) > 2
            )
            if overly_specific:
                score *= 0.8  # Penalty for inappropriate specificity
        
        # Bonus for good jurisdictional awareness
        awareness_indicators = any(phrase in response_lower for phrase in [
            "may vary by jurisdiction",
            "depending on your state",
            "consult local laws",
            "specific to your jurisdiction"
        ])
        if awareness_indicators:
            score = min(score * 1.1, 1.0)  # 10% bonus, capped at 1.0
        
        return score
    
    def _calculate_consistency_score(self, 
                                   expected: USJurisdiction,
                                   inference: JurisdictionInferenceResult,
                                   conflicts: List[Dict[str, Any]],
                                   context_score: float) -> float:
        """Calculate overall jurisdiction consistency score"""
        
        # Base score from inference confidence
        if inference.jurisdiction == expected:
            base_score = inference.confidence
        else:
            # Check if expected jurisdiction is in alternatives
            alt_score = 0.0
            for alt_jurisdiction, alt_confidence, _ in inference.alternatives:
                if alt_jurisdiction == expected:
                    alt_score = alt_confidence * 0.8  # Penalty for not being primary
                    break
            base_score = alt_score
        
        # Apply conflict penalties
        for conflict in conflicts:
            penalty = 1.0 - conflict["severity"]
            base_score *= penalty
        
        # Apply context score
        final_score = base_score * context_score
        
        return min(final_score, 1.0)


class DisclaimerEvaluator:
    """
    Evaluates adequacy of legal disclaimers and limitations in responses.
    
    Ensures appropriate disclaimers are present based on jurisdiction
    specificity and legal domain requirements.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.DisclaimerEvaluator")
        
        # Required disclaimer types by context
        self.disclaimer_requirements = {
            "specific_jurisdiction": [
                "jurisdiction_limitation",  # Laws may vary by jurisdiction
                "legal_advice_disclaimer"   # This is not legal advice
            ],
            "general_jurisdiction": [
                "legal_advice_disclaimer",  # This is not legal advice
                "jurisdiction_variation"    # Laws vary by jurisdiction
            ],
            "federal_specific": [
                "legal_advice_disclaimer",  # This is not legal advice
                "state_law_variation"       # State laws may differ
            ]
        }
    
    def evaluate_disclaimers(self, 
                            response: str,
                            jurisdiction: USJurisdiction,
                            task_type: LegalTaskType,
                            legal_domains: List[LegalDomain]) -> Dict[str, Any]:
        """
        Evaluate disclaimer adequacy in legal response.
        
        Args:
            response: Legal response text
            jurisdiction: Target jurisdiction
            task_type: Type of legal task
            legal_domains: Detected legal domains
            
        Returns:
            Dictionary with disclaimer evaluation results
        """
        
        # Determine required disclaimers
        required_disclaimers = self._determine_required_disclaimers(
            jurisdiction, task_type, legal_domains
        )
        
        # Detect present disclaimers
        present_disclaimers = self._detect_disclaimers(response)
        
        # Evaluate adequacy
        adequacy_score = self._calculate_disclaimer_adequacy(
            required_disclaimers, present_disclaimers
        )
        
        # Generate recommendations
        recommendations = self._generate_disclaimer_recommendations(
            required_disclaimers, present_disclaimers
        )
        
        return {
            "adequacy_score": adequacy_score,
            "required_disclaimers": required_disclaimers,
            "present_disclaimers": present_disclaimers,
            "missing_disclaimers": list(set(required_disclaimers) - set(present_disclaimers)),
            "recommendations": recommendations
        }
    
    def _determine_required_disclaimers(self, 
                                      jurisdiction: USJurisdiction,
                                      task_type: LegalTaskType,
                                      domains: List[LegalDomain]) -> List[str]:
        """Determine required disclaimers based on context"""
        
        required = set()
        
        # Base requirements
        required.add("legal_advice_disclaimer")
        
        # Jurisdiction-specific requirements
        if jurisdiction == USJurisdiction.GENERAL:
            required.update(self.disclaimer_requirements["general_jurisdiction"])
        elif jurisdiction == USJurisdiction.FEDERAL:
            required.update(self.disclaimer_requirements["federal_specific"])
        else:
            required.update(self.disclaimer_requirements["specific_jurisdiction"])
        
        # Task-specific requirements
        if task_type in [LegalTaskType.JUDICIAL_REASONING, LegalTaskType.OPINION_GENERATION]:
            required.add("interpretation_limitation")  # Interpretations may vary
        
        # Domain-specific requirements
        for domain in domains:
            if domain in [LegalDomain.CRIMINAL, LegalDomain.FAMILY]:
                required.add("urgent_consultation")  # Consult attorney immediately
            elif domain in [LegalDomain.TAX, LegalDomain.SECURITIES]:
                required.add("professional_consultation")  # Consult professional
        
        return list(required)
    
    def _detect_disclaimers(self, response: str) -> List[str]:
        """Detect present disclaimers in response"""
        
        response_lower = response.lower()
        present = []
        
        # Legal advice disclaimer
        if any(phrase in response_lower for phrase in [
            "not legal advice", "not constitute legal advice", "general information only"
        ]):
            present.append("legal_advice_disclaimer")
        
        # Jurisdiction variation
        if any(phrase in response_lower for phrase in [
            "vary by jurisdiction", "may differ", "depending on your state"
        ]):
            present.append("jurisdiction_variation")
        
        # Professional consultation
        if any(phrase in response_lower for phrase in [
            "consult an attorney", "seek legal counsel", "consult a lawyer"
        ]):
            present.append("professional_consultation")
        
        # Jurisdiction limitation
        if any(phrase in response_lower for phrase in [
            "specific to", "under [state] law", "in this jurisdiction"
        ]):
            present.append("jurisdiction_limitation")
        
        return present
    
    def _calculate_disclaimer_adequacy(self, 
                                     required: List[str],
                                     present: List[str]) -> float:
        """Calculate disclaimer adequacy score"""
        
        if not required:
            return 1.0
        
        # Calculate coverage
        coverage = len(set(required) & set(present)) / len(required)
        
        # Boost for extra disclaimers (shows good practice)
        extra_disclaimers = len(set(present) - set(required))
        bonus = min(extra_disclaimers * 0.05, 0.2)  # Up to 20% bonus
        
        return min(coverage + bonus, 1.0)
    
    def _generate_disclaimer_recommendations(self, 
                                           required: List[str],
                                           present: List[str]) -> List[str]:
        """Generate recommendations for improving disclaimers"""
        
        missing = set(required) - set(present)
        recommendations = []
        
        disclaimer_templates = {
            "legal_advice_disclaimer": "Add: 'This information is for general purposes only and does not constitute legal advice.'",
            "jurisdiction_variation": "Add: 'Laws may vary by jurisdiction. Consult local laws for specific requirements.'",
            "professional_consultation": "Add: 'Consult with a qualified attorney for advice specific to your situation.'",
            "jurisdiction_limitation": "Add: 'This analysis is specific to [jurisdiction] law and may not apply elsewhere.'",
            "urgent_consultation": "Add: 'For urgent legal matters, consult an attorney immediately.'",
            "interpretation_limitation": "Add: 'Legal interpretations may vary. This represents one possible analysis.'"
        }
        
        for missing_disclaimer in missing:
            if missing_disclaimer in disclaimer_templates:
                recommendations.append(disclaimer_templates[missing_disclaimer])
        
        return recommendations


class JurisdictionComplianceJudge:
    """
    Main jurisdiction compliance judge for the legal reward system.
    
    Provides comprehensive jurisdiction compliance evaluation including
    consistency checking, disclaimer evaluation, and gating decisions
    for the hybrid evaluation system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(f"{__name__}.JurisdictionComplianceJudge")
        
        # Configuration
        self.config = config or {}
        self.compliance_threshold = self.config.get("compliance_threshold", 7.0)
        self.gating_threshold = self.config.get("gating_threshold", 5.0)
        self.strict_federal_domains = self.config.get("strict_federal_domains", True)
        
        # Initialize components
        self.jurisdiction_analyzer = JurisdictionAnalyzer()
        self.disclaimer_evaluator = DisclaimerEvaluator()
        
        # Performance tracking
        self.evaluation_stats = {
            "total_evaluations": 0,
            "compliant_responses": 0,
            "gating_passes": 0,
            "critical_violations": 0,
            "avg_compliance_score": 0.0
        }
    
    def evaluate_compliance(self, 
                           response: str,
                           expected_jurisdiction: USJurisdiction,
                           task_type: LegalTaskType,
                           legal_domains: Optional[List[LegalDomain]] = None,
                           context: Optional[Dict[str, Any]] = None) -> JurisdictionComplianceResult:
        """
        Evaluate jurisdiction compliance of a legal response.
        
        Args:
            response: Legal response text to evaluate
            expected_jurisdiction: Expected jurisdiction context
            task_type: Type of legal task
            legal_domains: Detected legal domains (optional)
            context: Additional evaluation context
            
        Returns:
            JurisdictionComplianceResult with comprehensive compliance assessment
        """
        
        import time
        start_time = time.time()
        
        self.evaluation_stats["total_evaluations"] += 1
        
        try:
            # Initialize result
            result = JurisdictionComplianceResult(
                compliance_score=0.0,
                is_compliant=False,
                gating_decision=False,
                expected_jurisdiction=expected_jurisdiction
            )
            
            # Analyze jurisdiction consistency
            consistency_analysis = self.jurisdiction_analyzer.analyze_jurisdiction_consistency(
                response, expected_jurisdiction, task_type
            )
            
            result.inferred_jurisdiction = consistency_analysis["inferred_jurisdiction"]
            result.jurisdiction_consistency = consistency_analysis["consistency_score"]
            result.context_appropriateness = consistency_analysis["context_score"]
            
            # Evaluate disclaimers
            disclaimer_analysis = self.disclaimer_evaluator.evaluate_disclaimers(
                response, expected_jurisdiction, task_type, legal_domains or []
            )
            
            result.disclaimer_adequacy = disclaimer_analysis["adequacy_score"]
            
            # Domain-jurisdiction alignment check
            if legal_domains:
                result.domain_alignment = self._evaluate_domain_alignment(
                    legal_domains, expected_jurisdiction
                )
            
            # Detect violations
            violations = self._detect_compliance_violations(
                response, expected_jurisdiction, task_type,
                consistency_analysis, disclaimer_analysis, legal_domains
            )
            
            result.violations = violations
            result.critical_violations = [v for v in violations if v.is_critical()]
            
            # Calculate compliance score
            result.compliance_score = self._calculate_compliance_score(
                result.jurisdiction_consistency,
                result.context_appropriateness,
                result.disclaimer_adequacy,
                result.domain_alignment,
                violations
            )
            
            # Make compliance and gating decisions
            result.is_compliant = result.compliance_score >= self.compliance_threshold
            result.gating_decision = (
                result.compliance_score >= self.gating_threshold and
                not result.has_critical_violations()
            )
            
            # Generate recommendations
            result.recommendations = self._generate_recommendations(
                violations, disclaimer_analysis.get("recommendations", [])
            )
            
            result.required_corrections = [
                v.recommendation for v in violations if v.is_critical()
            ]
            
            # Update statistics
            if result.is_compliant:
                self.evaluation_stats["compliant_responses"] += 1
            if result.gating_decision:
                self.evaluation_stats["gating_passes"] += 1
            if result.has_critical_violations():
                self.evaluation_stats["critical_violations"] += 1
            
            # Update running average
            total = self.evaluation_stats["total_evaluations"]
            current_avg = self.evaluation_stats["avg_compliance_score"]
            self.evaluation_stats["avg_compliance_score"] = (
                (current_avg * (total - 1) + result.compliance_score) / total
            )
            
            # Set metadata
            result.processing_time_ms = (time.time() - start_time) * 1000
            result.confidence = self._calculate_evaluation_confidence(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Compliance evaluation failed: {e}")
            return self._create_error_result(str(e), expected_jurisdiction)
    
    def _evaluate_domain_alignment(self, 
                                  domains: List[LegalDomain],
                                  jurisdiction: USJurisdiction) -> float:
        """Evaluate alignment between legal domains and jurisdiction"""
        
        if not domains:
            return 1.0
        
        alignment_score = 1.0
        
        for domain in domains:
            # Check federal-only domains
            if is_jurisdiction_federal_only(domain) and jurisdiction != USJurisdiction.FEDERAL:
                alignment_score *= 0.5  # Major penalty for domain-jurisdiction mismatch
        
        return alignment_score
    
    def _detect_compliance_violations(self, 
                                    response: str,
                                    expected_jurisdiction: USJurisdiction,
                                    task_type: LegalTaskType,
                                    consistency_analysis: Dict[str, Any],
                                    disclaimer_analysis: Dict[str, Any],
                                    legal_domains: Optional[List[LegalDomain]]) -> List[ComplianceViolation]:
        """Detect all compliance violations in the response"""
        
        violations = []
        
        # Jurisdiction consistency violations
        violations.extend(self._detect_jurisdiction_violations(
            expected_jurisdiction, consistency_analysis
        ))
        
        # Disclaimer violations
        violations.extend(self._detect_disclaimer_violations(disclaimer_analysis))
        
        # Domain-jurisdiction violations
        if legal_domains:
            violations.extend(self._detect_domain_violations(
                legal_domains, expected_jurisdiction
            ))
        
        # Cross-jurisdictional conflicts
        conflicts = consistency_analysis.get("conflicts", [])
        for conflict in conflicts:
            violations.append(ComplianceViolation(
                violation_type=ComplianceViolationType.CROSS_JURISDICTIONAL_CONFLICT,
                severity=conflict["severity"],
                description=conflict["description"],
                location="Response content",
                recommendation="Clarify jurisdiction scope and remove conflicting references"
            ))
        
        # Context appropriateness violations
        context_score = consistency_analysis.get("context_score", 1.0)
        if context_score < 0.7:
            violations.append(ComplianceViolation(
                violation_type=ComplianceViolationType.INSUFFICIENT_CONTEXT,
                severity=1.0 - context_score,
                description="Insufficient jurisdiction context in response",
                location="Overall response",
                recommendation="Add appropriate jurisdiction-specific context and references"
            ))
        
        return violations
    
    def _detect_jurisdiction_violations(self, 
                                      expected: USJurisdiction,
                                      analysis: Dict[str, Any]) -> List[ComplianceViolation]:
        """Detect jurisdiction-specific violations"""
        
        violations = []
        inferred = analysis.get("inferred_jurisdiction")
        consistency = analysis.get("consistency_score", 1.0)
        
        # Wrong jurisdiction violation
        if inferred != expected and consistency < 0.5:
            violations.append(ComplianceViolation(
                violation_type=ComplianceViolationType.WRONG_JURISDICTION,
                severity=0.8,
                description=f"Expected {expected.value} but inferred {inferred.value if inferred else 'unknown'}",
                location="Response content",
                expected_jurisdiction=expected,
                found_jurisdiction=inferred,
                recommendation=f"Align response with {expected.value} jurisdiction requirements",
                correction_priority=1
            ))
        
        # Federal-state confusion
        if expected == USJurisdiction.FEDERAL and inferred and inferred != USJurisdiction.FEDERAL:
            violations.append(ComplianceViolation(
                violation_type=ComplianceViolationType.FEDERAL_STATE_CONFUSION,
                severity=0.7,
                description="Confusion between federal and state jurisdiction",
                location="Response content",
                recommendation="Clarify federal law applicability and distinguish from state law"
            ))
        
        return violations
    
    def _detect_disclaimer_violations(self, analysis: Dict[str, Any]) -> List[ComplianceViolation]:
        """Detect disclaimer-related violations"""
        
        violations = []
        adequacy = analysis.get("adequacy_score", 1.0)
        missing = analysis.get("missing_disclaimers", [])
        
        if adequacy < 0.7:
            severity = 1.0 - adequacy
            violations.append(ComplianceViolation(
                violation_type=ComplianceViolationType.MISSING_DISCLAIMERS,
                severity=severity,
                description=f"Inadequate disclaimers (score: {adequacy:.2f})",
                location="Response disclaimers",
                recommendation="Add required legal disclaimers and jurisdiction limitations",
                correction_priority=2 if severity > 0.5 else 3
            ))
        
        return violations
    
    def _detect_domain_violations(self, 
                                 domains: List[LegalDomain],
                                 jurisdiction: USJurisdiction) -> List[ComplianceViolation]:
        """Detect domain-jurisdiction alignment violations"""
        
        violations = []
        
        for domain in domains:
            if is_jurisdiction_federal_only(domain) and jurisdiction != USJurisdiction.FEDERAL:
                violations.append(ComplianceViolation(
                    violation_type=ComplianceViolationType.DOMAIN_JURISDICTION_MISMATCH,
                    severity=0.9,
                    description=f"Domain {domain.value} requires federal jurisdiction",
                    location="Legal domain analysis",
                    affected_domains=[domain],
                    recommendation="Use federal jurisdiction for federal-exclusive domains",
                    correction_priority=1
                ))
        
        return violations
    
    def _calculate_compliance_score(self, 
                                  jurisdiction_consistency: float,
                                  context_appropriateness: float,
                                  disclaimer_adequacy: float,
                                  domain_alignment: float,
                                  violations: List[ComplianceViolation]) -> float:
        """Calculate overall compliance score (0.0 to 10.0)"""
        
        # Base score from component scores
        component_scores = [
            jurisdiction_consistency * 0.35,  # 35% weight
            context_appropriateness * 0.25,   # 25% weight
            disclaimer_adequacy * 0.25,       # 25% weight
            domain_alignment * 0.15           # 15% weight
        ]
        
        base_score = sum(component_scores) * 10.0  # Scale to 0-10
        
        # Apply violation penalties
        if violations:
            penalty_multiplier = 1.0
            for violation in violations:
                penalty_multiplier *= violation.get_penalty_multiplier()
            
            base_score *= penalty_multiplier
        
        return max(base_score, 0.0)  # Ensure non-negative
    
    def _generate_recommendations(self, 
                                violations: List[ComplianceViolation],
                                disclaimer_recommendations: List[str]) -> List[str]:
        """Generate comprehensive recommendations for improvement"""
        
        recommendations = []
        
        # Add violation-specific recommendations
        for violation in violations:
            if violation.recommendation and violation.recommendation not in recommendations:
                recommendations.append(violation.recommendation)
        
        # Add disclaimer recommendations
        recommendations.extend(disclaimer_recommendations)
        
        # Add general best practices
        if not any("jurisdiction" in rec.lower() for rec in recommendations):
            recommendations.append("Clearly state the jurisdiction scope of your analysis")
        
        if not any("disclaimer" in rec.lower() for rec in recommendations):
            recommendations.append("Include appropriate legal disclaimers")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _calculate_evaluation_confidence(self, result: JurisdictionComplianceResult) -> float:
        """Calculate confidence in the evaluation result"""
        
        # Base confidence from inference
        if hasattr(result, 'inference_confidence'):
            base_confidence = getattr(result, 'inference_confidence', 0.8)
        else:
            base_confidence = 0.8
        
        # Adjust based on consistency
        if result.jurisdiction_consistency > 0.8:
            base_confidence *= 1.1
        elif result.jurisdiction_consistency < 0.5:
            base_confidence *= 0.8
        
        # Adjust based on violation clarity
        if result.has_critical_violations():
            base_confidence *= 1.2  # High confidence in critical violations
        
        return min(base_confidence, 1.0)
    
    def _create_error_result(self, error_msg: str, expected_jurisdiction: USJurisdiction) -> JurisdictionComplianceResult:
        """Create error fallback result"""
        
        return JurisdictionComplianceResult(
            compliance_score=0.0,
            is_compliant=False,
            gating_decision=False,
            expected_jurisdiction=expected_jurisdiction,
            violations=[ComplianceViolation(
                violation_type=ComplianceViolationType.INSUFFICIENT_CONTEXT,
                severity=1.0,
                description=f"Evaluation error: {error_msg}",
                location="System error",
                recommendation="Review response for compliance manually"
            )],
            recommendations=[f"Manual review required due to error: {error_msg}"],
            evaluation_method="error_fallback",
            confidence=0.1
        )
    
    def create_judge_evaluation(self, 
                               compliance_result: JurisdictionComplianceResult,
                               response: str,
                               context: Optional[Dict[str, Any]] = None) -> JudgeEvaluation:
        """
        Create a JudgeEvaluation from compliance result for integration
        with the broader evaluation system.
        
        Args:
            compliance_result: Jurisdiction compliance result
            response: Original response text
            context: Additional context
            
        Returns:
            JudgeEvaluation formatted for system integration
        """
        
        # Apply compliance penalty to base score
        penalty_multiplier = compliance_result.get_overall_penalty()
        base_score = compliance_result.compliance_score
        final_score = base_score * penalty_multiplier
        
        # Create detailed reasoning
        reasoning_parts = [
            f"Jurisdiction Compliance Score: {compliance_result.compliance_score:.1f}/10.0",
            f"Expected: {compliance_result.expected_jurisdiction.value}",
            f"Inferred: {compliance_result.inferred_jurisdiction.value if compliance_result.inferred_jurisdiction else 'Unknown'}"
        ]
        
        if compliance_result.violations:
            violation_summary = f"{len(compliance_result.violations)} violations detected"
            if compliance_result.critical_violations:
                violation_summary += f" ({len(compliance_result.critical_violations)} critical)"
            reasoning_parts.append(violation_summary)
        
        if compliance_result.recommendations:
            reasoning_parts.append(f"Recommendations: {'; '.join(compliance_result.recommendations[:2])}")
        
        reasoning = " | ".join(reasoning_parts)
        
        return JudgeEvaluation(
            score=final_score,
            reasoning=reasoning,
            confidence=compliance_result.confidence,
            judge_type="jurisdiction_compliance",
            evaluation_metadata=EvaluationMetadata(
                jurisdiction=compliance_result.expected_jurisdiction,
                processing_time_ms=compliance_result.processing_time_ms,
                evaluation_method=compliance_result.evaluation_method,
                additional_info={
                    "compliance_score": compliance_result.compliance_score,
                    "gating_decision": compliance_result.gating_decision,
                    "violation_count": len(compliance_result.violations),
                    "critical_violations": len(compliance_result.critical_violations),
                    "jurisdiction_consistency": compliance_result.jurisdiction_consistency,
                    "disclaimer_adequacy": compliance_result.disclaimer_adequacy
                }
            )
        )
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive evaluation statistics"""
        
        total = self.evaluation_stats["total_evaluations"]
        if total == 0:
            return {"message": "No evaluations performed yet"}
        
        return {
            "total_evaluations": total,
            "compliance_rate": self.evaluation_stats["compliant_responses"] / total,
            "gating_pass_rate": self.evaluation_stats["gating_passes"] / total,
            "critical_violation_rate": self.evaluation_stats["critical_violations"] / total,
            "average_compliance_score": self.evaluation_stats["avg_compliance_score"],
            "configuration": {
                "compliance_threshold": self.compliance_threshold,
                "gating_threshold": self.gating_threshold,
                "strict_federal_domains": self.strict_federal_domains
            }
        }


# Factory functions for different use cases

def create_production_compliance_judge(config: Optional[Dict[str, Any]] = None) -> JurisdictionComplianceJudge:
    """
    Create production-ready jurisdiction compliance judge.
    
    Args:
        config: Optional configuration overrides
        
    Returns:
        Configured JurisdictionComplianceJudge for production use
    """
    default_config = {
        "compliance_threshold": 7.0,
        "gating_threshold": 5.0,
        "strict_federal_domains": True
    }
    
    if config:
        default_config.update(config)
    
    return JurisdictionComplianceJudge(default_config)


def create_development_compliance_judge() -> JurisdictionComplianceJudge:
    """
    Create development-friendly compliance judge with relaxed thresholds.
    
    Returns:
        Configured JurisdictionComplianceJudge for development use
    """
    development_config = {
        "compliance_threshold": 5.0,
        "gating_threshold": 3.0,
        "strict_federal_domains": False
    }
    
    return JurisdictionComplianceJudge(development_config)


def create_strict_compliance_judge() -> JurisdictionComplianceJudge:
    """
    Create strict compliance judge with high standards.
    
    Returns:
        Configured JurisdictionComplianceJudge for strict compliance checking
    """
    strict_config = {
        "compliance_threshold": 8.5,
        "gating_threshold": 7.0,
        "strict_federal_domains": True
    }
    
    return JurisdictionComplianceJudge(strict_config)


# Convenience functions
def evaluate_jurisdiction_compliance(response: str,
                                   expected_jurisdiction: USJurisdiction,
                                   task_type: LegalTaskType) -> JurisdictionComplianceResult:
    """
    Convenience function for quick jurisdiction compliance evaluation.
    
    Args:
        response: Legal response to evaluate
        expected_jurisdiction: Expected jurisdiction context
        task_type: Type of legal task
        
    Returns:
        JurisdictionComplianceResult
    """
    judge = create_production_compliance_judge()
    return judge.evaluate_compliance(response, expected_jurisdiction, task_type)
