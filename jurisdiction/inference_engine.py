"""
US Jurisdiction Inference Engine for Multi-Task Legal Reward System

This module provides intelligent jurisdiction inference capabilities that analyze
legal questions, prompts, and context to automatically determine the most
appropriate US jurisdiction for legal analysis.

Key Features:
- Content-based jurisdiction inference using legal keywords and patterns
- Domain-specific jurisdiction requirements (federal vs state)
- Geographic and entity name detection
- Confidence scoring and fallback logic
- Integration with the comprehensive US jurisdiction system
- Smart handling of ambiguous cases with user prompts

The inference engine is designed to work seamlessly with the hybrid evaluation
system to ensure legal responses are contextualized within the correct
US jurisdiction, improving accuracy and compliance.
"""

import re
import logging
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import json

# Import core components
from core import (
    USJurisdiction, LegalDomain, LegalTaskType,
    LegalRewardSystemError, create_error_context,
    APIProvider, LegalRewardSystemError
)
from jurisdiction.us_system import (
    JurisdictionMetadata, USJurisdictionError,
    get_all_jurisdiction_metadata, validate_jurisdiction,
    get_federal_circuit_states, get_region_states,
    is_jurisdiction_federal_only
)

class InferenceConfidence(Enum):
    """Confidence levels for jurisdiction inference"""
    HIGH = "high"        # 0.8+
    MEDIUM = "medium"    # 0.6-0.8
    LOW = "low"          # 0.4-0.6
    VERY_LOW = "very_low"  # <0.4
    
    @classmethod
    def from_score(cls, confidence_score: float) -> 'InferenceConfidence':
        """Convert confidence score to confidence level"""
        if confidence_score >= 0.8:
            return cls.HIGH
        elif confidence_score >= 0.6:
            return cls.MEDIUM
        elif confidence_score >= 0.4:
            return cls.LOW
        else:
            return cls.VERY_LOW

@dataclass
class JurisdictionInferenceResult:
    """
    Result of jurisdiction inference analysis.
    
    Contains the inferred jurisdiction, confidence score, reasoning,
    and alternative suggestions for comprehensive jurisdiction determination.
    """
    
    # Primary inference result
    jurisdiction: USJurisdiction
    confidence: float  # 0.0 to 1.0
    
    # Reasoning and evidence
    reasoning: str
    evidence: List[str] = field(default_factory=list)
    
    # Alternative suggestions
    alternatives: List[Tuple[USJurisdiction, float, str]] = field(default_factory=list)
    
    # Inference metadata
    inference_method: str = "content_analysis"
    requires_user_confirmation: bool = False
    domain_specific: bool = False
    
    # Context information
    detected_entities: List[str] = field(default_factory=list)
    detected_keywords: List[str] = field(default_factory=list)
    detected_domains: List[LegalDomain] = field(default_factory=list)
    
    def is_confident(self, threshold: float = 0.7) -> bool:
        """Check if inference confidence meets threshold"""
        return self.confidence >= threshold
    
    def has_strong_alternatives(self, min_confidence: float = 0.5) -> bool:
        """Check if there are strong alternative jurisdictions"""
        return any(conf >= min_confidence for _, conf, _ in self.alternatives)
    
    def get_best_alternative(self) -> Optional[Tuple[USJurisdiction, float, str]]:
        """Get the best alternative jurisdiction"""
        if self.alternatives:
            return max(self.alternatives, key=lambda x: x[1])
        return None
    
    def get_summary(self) -> str:
        """Get human-readable summary of inference result"""
        summary = f"Inferred: {self.jurisdiction.value} (confidence: {self.confidence:.2%})"
        if self.requires_user_confirmation:
            summary += " - Requires confirmation"
        if self.alternatives:
            alt_count = len(self.alternatives)
            summary += f" - {alt_count} alternative{'s' if alt_count > 1 else ''} available"
        return summary


class LegalKeywordAnalyzer:
    """
    Analyzes legal content for jurisdiction-relevant keywords and patterns.
    
    Uses comprehensive keyword dictionaries and pattern matching to identify
    jurisdiction-specific legal terminology, entities, and domains.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.LegalKeywordAnalyzer")
        
        # Load jurisdiction metadata for keyword extraction
        self.jurisdiction_metadata = get_all_jurisdiction_metadata()
        
        # Build keyword dictionaries
        self._build_keyword_dictionaries()
        
        # Build pattern dictionaries
        self._build_pattern_dictionaries()
    
    def _build_keyword_dictionaries(self):
        """Build comprehensive keyword dictionaries for jurisdiction inference"""
        
        # Federal jurisdiction keywords
        self.federal_keywords = {
            # Government entities
            "congress", "senate", "house of representatives", "federal government",
            "united states", "u.s.", "usa", "federal court", "supreme court",
            "federal register", "code of federal regulations", "cfr",
            
            # Federal agencies
            "fbi", "cia", "nsa", "dhs", "ice", "irs", "sec", "fda", "epa",
            "ftc", "fcc", "nlrb", "eeoc", "osha", "dol", "hhs", "cms",
            "treasury", "fed", "federal reserve", "fdic", "occ",
            
            # Federal law areas
            "interstate commerce", "federal crime", "federal statute",
            "constitutional law", "civil rights act", "americans with disabilities act",
            "securities law", "antitrust", "bankruptcy", "immigration",
            "intellectual property", "copyright", "patent", "trademark",
            "federal tax", "customs", "tariff", "export", "import"
        }
        
        # Legal domain keywords
        self.domain_keywords = {
            LegalDomain.CONSTITUTIONAL: [
                "constitution", "constitutional", "amendment", "bill of rights",
                "due process", "equal protection", "first amendment", "fourteenth amendment",
                "commerce clause", "supremacy clause", "constitutional law"
            ],
            LegalDomain.FEDERAL_STATUTORY: [
                "federal statute", "u.s.c.", "united states code", "federal law",
                "federal regulation", "cfr", "federal register"
            ],
            LegalDomain.IMMIGRATION: [
                "immigration", "visa", "green card", "deportation", "asylum",
                "refugee", "naturalization", "citizenship", "ice", "uscis",
                "immigration court", "removal proceedings"
            ],
            LegalDomain.INTELLECTUAL_PROPERTY: [
                "copyright", "patent", "trademark", "trade secret", "intellectual property",
                "uspto", "patent office", "dmca", "fair use", "infringement"
            ],
            LegalDomain.SECURITIES: [
                "securities", "sec", "stock", "bond", "investment", "broker",
                "securities exchange act", "investment company act", "insider trading"
            ],
            LegalDomain.ANTITRUST: [
                "antitrust", "monopoly", "sherman act", "clayton act", "ftc act",
                "competition", "restraint of trade", "price fixing"
            ],
            LegalDomain.BANKRUPTCY: [
                "bankruptcy", "chapter 7", "chapter 11", "chapter 13", "debtor",
                "creditor", "bankruptcy court", "trustee", "discharge"
            ],
            LegalDomain.CRIMINAL: [
                "criminal", "felony", "misdemeanor", "prosecution", "defendant",
                "criminal code", "penal code", "arrest", "indictment"
            ],
            LegalDomain.CONTRACT: [
                "contract", "agreement", "breach", "consideration", "offer",
                "acceptance", "damages", "specific performance", "ucc"
            ],
            LegalDomain.TORT: [
                "tort", "negligence", "liability", "personal injury", "damages",
                "duty of care", "causation", "strict liability"
            ],
            LegalDomain.FAMILY: [
                "divorce", "custody", "child support", "alimony", "adoption",
                "marriage", "domestic relations", "family court"
            ],
            LegalDomain.EMPLOYMENT: [
                "employment", "labor", "workplace", "discrimination", "harassment",
                "wrongful termination", "wage", "overtime", "union"
            ],
            LegalDomain.REAL_ESTATE: [
                "real estate", "property", "deed", "mortgage", "foreclosure",
                "landlord", "tenant", "lease", "zoning", "easement"
            ],
            LegalDomain.HEALTHCARE: [
                "healthcare", "medical", "hipaa", "patient", "malpractice",
                "health insurance", "medicare", "medicaid"
            ]
        }
        
        # State-specific keywords (sample - can be expanded)
        self.state_keywords = {}
        for jurisdiction, metadata in self.jurisdiction_metadata.items():
            if metadata.is_state_jurisdiction():
                keywords = []
                
                # Add state name variations
                keywords.extend([
                    metadata.full_name.lower(),
                    metadata.abbreviation.lower(),
                    f"state of {metadata.full_name.lower()}",
                    f"{metadata.full_name.lower()} state"
                ])
                
                # Add major cities
                keywords.extend([
                    metadata.capital.lower(),
                    metadata.largest_city.lower()
                ])
                
                # Add prominent legal areas as state-specific indicators
                keywords.extend([area.lower() for area in metadata.prominent_legal_areas])
                
                # Add notable institutions
                keywords.extend([inst.lower() for inst in metadata.notable_legal_institutions])
                
                self.state_keywords[jurisdiction] = keywords
    
    def _build_pattern_dictionaries(self):
        """Build regex patterns for jurisdiction detection"""
        
        # Federal court patterns
        self.federal_court_patterns = [
            r"u\.?s\.?\s+district\s+court",
            r"united\s+states\s+district\s+court",
            r"u\.?s\.?\s+court\s+of\s+appeals",
            r"supreme\s+court\s+of\s+the\s+united\s+states",
            r"federal\s+circuit",
            r"\d+(st|nd|rd|th)\s+circuit"
        ]
        
        # State court patterns  
        self.state_court_patterns = []
        for jurisdiction, metadata in self.jurisdiction_metadata.items():
            if metadata.is_state_jurisdiction():
                state_name = metadata.full_name
                patterns = [
                    rf"{state_name}\s+supreme\s+court",
                    rf"supreme\s+court\s+of\s+{state_name}",
                    rf"{state_name}\s+court\s+of\s+appeals",
                    rf"{state_name}\s+superior\s+court",
                    rf"{state_name}\s+district\s+court"
                ]
                self.state_court_patterns.extend(patterns)
        
        # Citation patterns
        self.citation_patterns = [
            r"\d+\s+u\.?s\.?\s+\d+",  # U.S. Reports
            r"\d+\s+f\.\d*d?\s+\d+",  # Federal Reporter
            r"\d+\s+f\.\s*supp\.\s*\d*d?\s+\d+",  # Federal Supplement
        ]
        
        # State-specific citation patterns would be added here
        
        # Geographic entity patterns
        self.geographic_patterns = [
            r"in\s+(?:the\s+)?state\s+of\s+(\w+(?:\s+\w+)?)",
            r"(\w+(?:\s+\w+)?)\s+state\s+law",
            r"under\s+(\w+(?:\s+\w+)?)\s+law",
            r"(\w+(?:\s+\w+)?)\s+statute"
        ]
    
    def analyze_content(self, content: str) -> Dict[str, Any]:
        """
        Analyze content for jurisdiction-relevant keywords and patterns.
        
        Args:
            content: Legal text content to analyze
            
        Returns:
            Dictionary with analysis results
        """
        if not content:
            return {
                "federal_keywords": [],
                "state_keywords": {},
                "domain_keywords": {},
                "court_references": [],
                "geographic_entities": [],
                "citations": [],
                "keyword_scores": {}
            }
        
        # Normalize content
        content_lower = content.lower()
        
        # Find federal keywords
        federal_matches = [kw for kw in self.federal_keywords if kw in content_lower]
        
        # Find state-specific keywords
        state_matches = {}
        for jurisdiction, keywords in self.state_keywords.items():
            matches = [kw for kw in keywords if kw in content_lower]
            if matches:
                state_matches[jurisdiction] = matches
        
        # Find domain-specific keywords
        domain_matches = {}
        for domain, keywords in self.domain_keywords.items():
            matches = [kw for kw in keywords if kw in content_lower]
            if matches:
                domain_matches[domain] = matches
        
        # Find court references
        court_refs = []
        for pattern in self.federal_court_patterns + self.state_court_patterns:
            matches = re.finditer(pattern, content_lower, re.IGNORECASE)
            court_refs.extend([match.group() for match in matches])
        
        # Find geographic entities
        geo_entities = []
        for pattern in self.geographic_patterns:
            matches = re.finditer(pattern, content_lower, re.IGNORECASE)
            geo_entities.extend([match.group(1) for match in matches if match.group(1)])
        
        # Find legal citations
        citations = []
        for pattern in self.citation_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            citations.extend([match.group() for match in matches])
        
        # Calculate keyword scores
        keyword_scores = self._calculate_keyword_scores(
            federal_matches, state_matches, domain_matches
        )
        
        return {
            "federal_keywords": federal_matches,
            "state_keywords": state_matches,
            "domain_keywords": domain_matches,
            "court_references": court_refs,
            "geographic_entities": geo_entities,
            "citations": citations,
            "keyword_scores": keyword_scores
        }
    
    def _calculate_keyword_scores(self, 
                                 federal_matches: List[str],
                                 state_matches: Dict[USJurisdiction, List[str]],
                                 domain_matches: Dict[LegalDomain, List[str]]) -> Dict[str, float]:
        """Calculate weighted scores for different jurisdiction types"""
        
        scores = {
            "federal_score": 0.0,
            "state_scores": {},
            "domain_scores": {}
        }
        
        # Federal score
        scores["federal_score"] = min(len(federal_matches) * 0.2, 1.0)
        
        # State scores
        for jurisdiction, matches in state_matches.items():
            scores["state_scores"][jurisdiction] = min(len(matches) * 0.15, 1.0)
        
        # Domain scores
        for domain, matches in domain_matches.items():
            scores["domain_scores"][domain] = min(len(matches) * 0.1, 1.0)
        
        return scores


class JurisdictionInferenceEngine:
    """
    Main jurisdiction inference engine that combines multiple analysis methods.
    
    Provides intelligent jurisdiction inference using content analysis,
    domain-specific rules, geographic detection, and confidence scoring
    to determine the most appropriate US jurisdiction for legal content.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(f"{__name__}.JurisdictionInferenceEngine")
        
        # Configuration
        self.config = config or {}
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.enable_fuzzy_matching = self.config.get("enable_fuzzy_matching", True)
        self.require_confirmation_threshold = self.config.get("require_confirmation_threshold", 0.5)
        
        # Initialize components
        self.keyword_analyzer = LegalKeywordAnalyzer()
        self.jurisdiction_metadata = get_all_jurisdiction_metadata()
        
        # Domain-jurisdiction mapping
        self._build_domain_jurisdiction_rules()
        
        # Performance tracking
        self.inference_stats = {
            "total_inferences": 0,
            "high_confidence_inferences": 0,
            "federal_inferences": 0,
            "state_inferences": 0,
            "general_inferences": 0,
            "domain_specific_inferences": 0
        }
    
    def _build_domain_jurisdiction_rules(self):
        """Build rules for domain-specific jurisdiction requirements"""
        
        # Domains that require federal jurisdiction
        self.federal_required_domains = {
            LegalDomain.CONSTITUTIONAL,
            LegalDomain.FEDERAL_STATUTORY,
            LegalDomain.INTERSTATE_COMMERCE,
            LegalDomain.IMMIGRATION,
            LegalDomain.INTELLECTUAL_PROPERTY,
            LegalDomain.SECURITIES,
            LegalDomain.ANTITRUST,
            LegalDomain.BANKRUPTCY
        }
        
        # Domains that can be either federal or state
        self.flexible_domains = {
            LegalDomain.CRIMINAL,
            LegalDomain.CONTRACT,
            LegalDomain.TORT,
            LegalDomain.EMPLOYMENT,
            LegalDomain.HEALTHCARE,
            LegalDomain.ENVIRONMENTAL,
            LegalDomain.TAX
        }
        
        # Domains that are typically state jurisdiction
        self.state_preferred_domains = {
            LegalDomain.FAMILY,
            LegalDomain.REAL_ESTATE,
            LegalDomain.CIVIL_PROCEDURE,
            LegalDomain.EVIDENCE,
            LegalDomain.ETHICS
        }
    
    def infer_jurisdiction(self, 
                          content: str,
                          task_type: Optional[LegalTaskType] = None,
                          explicit_jurisdiction: Optional[str] = None,
                          context: Optional[Dict[str, Any]] = None) -> JurisdictionInferenceResult:
        """
        Infer the appropriate jurisdiction for legal content.
        
        Args:
            content: Legal text content to analyze
            task_type: Type of legal task being performed
            explicit_jurisdiction: Explicitly mentioned jurisdiction (if any)
            context: Additional context information
            
        Returns:
            JurisdictionInferenceResult with inference details
        """
        self.inference_stats["total_inferences"] += 1
        
        try:
            # Handle explicit jurisdiction first
            if explicit_jurisdiction:
                return self._handle_explicit_jurisdiction(explicit_jurisdiction, content)
            
            # Analyze content for jurisdiction indicators
            analysis = self.keyword_analyzer.analyze_content(content)
            
            # Determine legal domains
            detected_domains = self._detect_legal_domains(analysis)
            
            # Apply domain-specific rules
            domain_result = self._apply_domain_rules(detected_domains, analysis)
            if domain_result:
                return domain_result
            
            # Content-based inference
            content_result = self._infer_from_content_analysis(analysis, detected_domains)
            if content_result.is_confident(self.confidence_threshold):
                return content_result
            
            # Geographic inference
            geo_result = self._infer_from_geographic_entities(analysis)
            if geo_result and geo_result.is_confident(self.confidence_threshold):
                return geo_result
            
            # Fallback to general jurisdiction with alternatives
            return self._create_fallback_result(analysis, detected_domains, [content_result, geo_result])
            
        except Exception as e:
            self.logger.error(f"Jurisdiction inference failed: {e}")
            return self._create_error_fallback(str(e))
    
    def _handle_explicit_jurisdiction(self, explicit: str, content: str) -> JurisdictionInferenceResult:
        """Handle cases where jurisdiction is explicitly mentioned"""
        try:
            jurisdiction = validate_jurisdiction(explicit)
            
            return JurisdictionInferenceResult(
                jurisdiction=jurisdiction,
                confidence=0.95,
                reasoning=f"Explicitly specified jurisdiction: {explicit}",
                evidence=[f"User specified: {explicit}"],
                inference_method="explicit",
                requires_user_confirmation=False,
                detected_entities=[explicit]
            )
            
        except USJurisdictionError:
            # Invalid explicit jurisdiction - continue with content analysis
            self.logger.warning(f"Invalid explicit jurisdiction: {explicit}")
            return self._infer_from_content_analysis(
                self.keyword_analyzer.analyze_content(content),
                []
            )
    
    def _detect_legal_domains(self, analysis: Dict[str, Any]) -> List[LegalDomain]:
        """Detect legal domains from content analysis"""
        detected_domains = []
        
        domain_keywords = analysis.get("domain_keywords", {})
        for domain, keywords in domain_keywords.items():
            if keywords:  # If domain has matching keywords
                detected_domains.append(domain)
        
        return detected_domains
    
    def _apply_domain_rules(self, 
                           domains: List[LegalDomain], 
                           analysis: Dict[str, Any]) -> Optional[JurisdictionInferenceResult]:
        """Apply domain-specific jurisdiction rules"""
        
        if not domains:
            return None
        
        # Check for federal-required domains
        federal_domains = [d for d in domains if d in self.federal_required_domains]
        if federal_domains:
            self.inference_stats["federal_inferences"] += 1
            self.inference_stats["domain_specific_inferences"] += 1
            
            return JurisdictionInferenceResult(
                jurisdiction=USJurisdiction.FEDERAL,
                confidence=0.9,
                reasoning=f"Domain requires federal jurisdiction: {', '.join(d.value for d in federal_domains)}",
                evidence=[f"Federal domain: {d.value}" for d in federal_domains],
                inference_method="domain_rules",
                requires_user_confirmation=False,
                domain_specific=True,
                detected_domains=domains
            )
        
        # For flexible domains, check for other indicators
        flexible_domains = [d for d in domains if d in self.flexible_domains]
        if flexible_domains:
            # Look for federal indicators in the content
            federal_score = analysis.get("keyword_scores", {}).get("federal_score", 0.0)
            if federal_score > 0.3:
                return JurisdictionInferenceResult(
                    jurisdiction=USJurisdiction.FEDERAL,
                    confidence=0.7 + federal_score * 0.2,
                    reasoning=f"Flexible domain with federal indicators: {', '.join(d.value for d in flexible_domains)}",
                    evidence=analysis.get("federal_keywords", [])[:3],
                    inference_method="domain_rules_flexible",
                    requires_user_confirmation=True,
                    domain_specific=True,
                    detected_domains=domains
                )
        
        return None
    
    def _infer_from_content_analysis(self, 
                                   analysis: Dict[str, Any], 
                                   domains: List[LegalDomain]) -> JurisdictionInferenceResult:
        """Infer jurisdiction from comprehensive content analysis"""
        
        keyword_scores = analysis.get("keyword_scores", {})
        federal_score = keyword_scores.get("federal_score", 0.0)
        state_scores = keyword_scores.get("state_scores", {})
        
        # Boost federal score if federal courts or citations are mentioned
        court_refs = analysis.get("court_references", [])
        federal_court_refs = [ref for ref in court_refs if any(
            pattern in ref for pattern in ["u.s.", "federal", "circuit", "supreme court"]
        )]
        if federal_court_refs:
            federal_score += 0.3
        
        # Find best state match
        best_state = None
        best_state_score = 0.0
        if state_scores:
            best_state = max(state_scores.keys(), key=lambda k: state_scores[k])
            best_state_score = state_scores[best_state]
        
        # Decision logic
        if federal_score > best_state_score and federal_score > 0.4:
            self.inference_stats["federal_inferences"] += 1
            return JurisdictionInferenceResult(
                jurisdiction=USJurisdiction.FEDERAL,
                confidence=min(federal_score, 0.9),
                reasoning="Strong federal jurisdiction indicators in content",
                evidence=analysis.get("federal_keywords", [])[:5],
                inference_method="content_analysis",
                requires_user_confirmation=federal_score < self.confidence_threshold,
                detected_keywords=analysis.get("federal_keywords", []),
                detected_domains=domains,
                alternatives=self._build_alternatives(state_scores, "content")
            )
        
        elif best_state and best_state_score > 0.3:
            self.inference_stats["state_inferences"] += 1
            metadata = self.jurisdiction_metadata[best_state]
            
            return JurisdictionInferenceResult(
                jurisdiction=best_state,
                confidence=min(best_state_score + 0.1, 0.9),
                reasoning=f"Strong {metadata.full_name} jurisdiction indicators in content",
                evidence=analysis.get("state_keywords", {}).get(best_state, [])[:5],
                inference_method="content_analysis",
                requires_user_confirmation=best_state_score < self.confidence_threshold,
                detected_keywords=analysis.get("state_keywords", {}).get(best_state, []),
                detected_domains=domains,
                alternatives=self._build_alternatives(state_scores, "content", exclude=best_state)
            )
        
        else:
            # Low confidence - return general with alternatives
            alternatives = []
            if federal_score > 0.1:
                alternatives.append((USJurisdiction.FEDERAL, federal_score, "Federal indicators"))
            
            alternatives.extend([
                (state, score, f"{self.jurisdiction_metadata[state].full_name} indicators")
                for state, score in sorted(state_scores.items(), key=lambda x: x[1], reverse=True)[:3]
                if score > 0.1
            ])
            
            self.inference_stats["general_inferences"] += 1
            return JurisdictionInferenceResult(
                jurisdiction=USJurisdiction.GENERAL,
                confidence=0.3,
                reasoning="Insufficient jurisdiction indicators in content",
                evidence=["General legal content"],
                inference_method="content_analysis_low_confidence",
                requires_user_confirmation=True,
                detected_domains=domains,
                alternatives=alternatives
            )
    
    def _infer_from_geographic_entities(self, analysis: Dict[str, Any]) -> Optional[JurisdictionInferenceResult]:
        """Infer jurisdiction from geographic entities mentioned in content"""
        
        geo_entities = analysis.get("geographic_entities", [])
        if not geo_entities:
            return None
        
        # Try to match geographic entities to jurisdictions
        jurisdiction_matches = []
        for entity in geo_entities:
            try:
                jurisdiction = validate_jurisdiction(entity)
                jurisdiction_matches.append(jurisdiction)
            except USJurisdictionError:
                continue
        
        if jurisdiction_matches:
            # Count occurrences
            jurisdiction_counts = Counter(jurisdiction_matches)
            most_common = jurisdiction_counts.most_common(1)[0]
            jurisdiction, count = most_common
            
            # Calculate confidence based on frequency and uniqueness
            confidence = min(0.6 + (count * 0.1), 0.85)
            
            metadata = self.jurisdiction_metadata.get(jurisdiction)
            jurisdiction_name = metadata.full_name if metadata else jurisdiction.value
            
            return JurisdictionInferenceResult(
                jurisdiction=jurisdiction,
                confidence=confidence,
                reasoning=f"Geographic entity detected: {jurisdiction_name}",
                evidence=geo_entities,
                inference_method="geographic_entities",
                requires_user_confirmation=confidence < self.confidence_threshold,
                detected_entities=geo_entities
            )
        
        return None
    
    def _build_alternatives(self, 
                           scores: Dict[USJurisdiction, float], 
                           method: str,
                           exclude: Optional[USJurisdiction] = None) -> List[Tuple[USJurisdiction, float, str]]:
        """Build alternative jurisdiction suggestions"""
        
        alternatives = []
        
        # Sort by score and take top alternatives
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        for jurisdiction, score in sorted_scores:
            if exclude and jurisdiction == exclude:
                continue
            
            if score > 0.2 and len(alternatives) < 3:
                metadata = self.jurisdiction_metadata.get(jurisdiction)
                name = metadata.full_name if metadata else jurisdiction.value
                reason = f"{name} indicators from {method}"
                alternatives.append((jurisdiction, score, reason))
        
        return alternatives
    
    def _create_fallback_result(self, 
                               analysis: Dict[str, Any],
                               domains: List[LegalDomain],
                               previous_results: List[JurisdictionInferenceResult]) -> JurisdictionInferenceResult:
        """Create fallback result when no strong jurisdiction signals are found"""
        
        # Collect alternatives from previous results
        all_alternatives = []
        for result in previous_results:
            if result and result.jurisdiction != USJurisdiction.GENERAL:
                all_alternatives.append((result.jurisdiction, result.confidence, result.reasoning))
            all_alternatives.extend(result.alternatives if result else [])
        
        # Remove duplicates and sort
        unique_alternatives = {}
        for jurisdiction, confidence, reasoning in all_alternatives:
            if jurisdiction not in unique_alternatives or confidence > unique_alternatives[jurisdiction][1]:
                unique_alternatives[jurisdiction] = (jurisdiction, confidence, reasoning)
        
        alternatives = list(unique_alternatives.values())
        alternatives.sort(key=lambda x: x[1], reverse=True)
        
        self.inference_stats["general_inferences"] += 1
        
        return JurisdictionInferenceResult(
            jurisdiction=USJurisdiction.GENERAL,
            confidence=0.4,
            reasoning="Unable to determine specific jurisdiction - using general principles",
            evidence=["No clear jurisdiction indicators"],
            inference_method="fallback",
            requires_user_confirmation=True,
            detected_domains=domains,
            alternatives=alternatives[:5]  # Top 5 alternatives
        )
    
    def _create_error_fallback(self, error_message: str) -> JurisdictionInferenceResult:
        """Create fallback result for error cases"""
        
        return JurisdictionInferenceResult(
            jurisdiction=USJurisdiction.GENERAL,
            confidence=0.1,
            reasoning=f"Error during jurisdiction inference: {error_message}",
            evidence=["Error fallback"],
            inference_method="error_fallback",
            requires_user_confirmation=True
        )
    
    def get_jurisdiction_suggestions(self, partial_content: str, limit: int = 5) -> List[JurisdictionInferenceResult]:
        """
        Get jurisdiction suggestions for partial content.
        
        Args:
            partial_content: Partial legal content
            limit: Maximum number of suggestions
            
        Returns:
            List of jurisdiction inference results
        """
        try:
            # Quick analysis of partial content
            analysis = self.keyword_analyzer.analyze_content(partial_content)
            
            suggestions = []
            
            # Federal suggestion
            federal_score = analysis.get("keyword_scores", {}).get("federal_score", 0.0)
            if federal_score > 0.1:
                suggestions.append(JurisdictionInferenceResult(
                    jurisdiction=USJurisdiction.FEDERAL,
                    confidence=federal_score,
                    reasoning="Federal jurisdiction indicators detected",
                    evidence=analysis.get("federal_keywords", [])[:3],
                    inference_method="partial_analysis"
                ))
            
            # State suggestions
            state_scores = analysis.get("keyword_scores", {}).get("state_scores", {})
            for jurisdiction, score in sorted(state_scores.items(), key=lambda x: x[1], reverse=True):
                if score > 0.1 and len(suggestions) < limit:
                    metadata = self.jurisdiction_metadata[jurisdiction]
                    suggestions.append(JurisdictionInferenceResult(
                        jurisdiction=jurisdiction,
                        confidence=score,
                        reasoning=f"{metadata.full_name} jurisdiction indicators detected",
                        evidence=analysis.get("state_keywords", {}).get(jurisdiction, [])[:3],
                        inference_method="partial_analysis"
                    ))
            
            # Always include general as a fallback
            if len(suggestions) < limit:
                suggestions.append(JurisdictionInferenceResult(
                    jurisdiction=USJurisdiction.GENERAL,
                    confidence=0.3,
                    reasoning="General legal principles (non-specific jurisdiction)",
                    evidence=["Fallback option"],
                    inference_method="fallback_suggestion"
                ))
            
            return suggestions[:limit]
            
        except Exception as e:
            self.logger.error(f"Failed to get jurisdiction suggestions: {e}")
            return [self._create_error_fallback(str(e))]
    
    def validate_jurisdiction_context(self, 
                                    jurisdiction: USJurisdiction,
                                    content: str,
                                    domains: List[LegalDomain]) -> Dict[str, Any]:
        """
        Validate that a jurisdiction is appropriate for given content and domains.
        
        Args:
            jurisdiction: Jurisdiction to validate
            content: Legal content
            domains: Detected legal domains
            
        Returns:
            Validation result with warnings and recommendations
        """
        validation = {
            "is_valid": True,
            "confidence": 1.0,
            "warnings": [],
            "recommendations": []
        }
        
        try:
            # Check domain-jurisdiction compatibility
            for domain in domains:
                if domain in self.federal_required_domains and jurisdiction != USJurisdiction.FEDERAL:
                    validation["is_valid"] = False
                    validation["warnings"].append(
                        f"Domain '{domain.value}' requires federal jurisdiction, but {jurisdiction.value} specified"
                    )
                    validation["recommendations"].append("Consider using federal jurisdiction")
                
                elif domain in self.state_preferred_domains and jurisdiction == USJurisdiction.FEDERAL:
                    validation["confidence"] *= 0.8
                    validation["warnings"].append(
                        f"Domain '{domain.value}' is typically handled at state level"
                    )
                    validation["recommendations"].append("Consider specifying a state jurisdiction")
            
            # Analyze content for jurisdiction consistency
            analysis = self.keyword_analyzer.analyze_content(content)
            keyword_scores = analysis.get("keyword_scores", {})
            
            if jurisdiction == USJurisdiction.FEDERAL:
                federal_score = keyword_scores.get("federal_score", 0.0)
                if federal_score < 0.2:
                    validation["confidence"] *= 0.7
                    validation["warnings"].append("Limited federal jurisdiction indicators in content")
            
            elif jurisdiction in keyword_scores.get("state_scores", {}):
                state_score = keyword_scores["state_scores"][jurisdiction]
                if state_score < 0.2:
                    validation["confidence"] *= 0.8
                    validation["warnings"].append("Limited jurisdiction-specific indicators in content")
            
            return validation
            
        except Exception as e:
            self.logger.error(f"Jurisdiction validation failed: {e}")
            return {
                "is_valid": False,
                "confidence": 0.1,
                "warnings": [f"Validation error: {e}"],
                "recommendations": ["Review jurisdiction specification"]
            }
    
    def get_inference_statistics(self) -> Dict[str, Any]:
        """Get performance statistics for the inference engine"""
        
        total = self.inference_stats["total_inferences"]
        if total == 0:
            return {"message": "No inferences performed yet"}
        
        return {
            "total_inferences": total,
            "high_confidence_rate": self.inference_stats["high_confidence_inferences"] / total,
            "jurisdiction_distribution": {
                "federal": self.inference_stats["federal_inferences"] / total,
                "state": self.inference_stats["state_inferences"] / total,
                "general": self.inference_stats["general_inferences"] / total
            },
            "domain_specific_rate": self.inference_stats["domain_specific_inferences"] / total,
            "confidence_threshold": self.confidence_threshold
        }
    
    def update_configuration(self, new_config: Dict[str, Any]):
        """Update inference engine configuration"""
        
        self.config.update(new_config)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.require_confirmation_threshold = self.config.get("require_confirmation_threshold", 0.5)
        
        self.logger.info(f"Updated inference engine configuration: {new_config}")


# Factory functions for different use cases

def create_production_inference_engine(config: Optional[Dict[str, Any]] = None) -> JurisdictionInferenceEngine:
    """
    Create production-ready jurisdiction inference engine.
    
    Args:
        config: Optional configuration overrides
        
    Returns:
        Configured JurisdictionInferenceEngine for production use
    """
    default_config = {
        "confidence_threshold": 0.7,
        "enable_fuzzy_matching": True,
        "require_confirmation_threshold": 0.6
    }
    
    if config:
        default_config.update(config)
    
    return JurisdictionInferenceEngine(default_config)


def create_development_inference_engine() -> JurisdictionInferenceEngine:
    """
    Create development-friendly inference engine with relaxed thresholds.
    
    Returns:
        Configured JurisdictionInferenceEngine for development use
    """
    development_config = {
        "confidence_threshold": 0.5,
        "enable_fuzzy_matching": True,
        "require_confirmation_threshold": 0.4
    }
    
    return JurisdictionInferenceEngine(development_config)


def create_conservative_inference_engine() -> JurisdictionInferenceEngine:
    """
    Create conservative inference engine with high confidence requirements.
    
    Returns:
        Configured JurisdictionInferenceEngine for conservative use
    """
    conservative_config = {
        "confidence_threshold": 0.8,
        "enable_fuzzy_matching": True,
        "require_confirmation_threshold": 0.7
    }
    
    return JurisdictionInferenceEngine(conservative_config)


# Convenience functions
def infer_jurisdiction_for_content(content: str, 
                                 task_type: Optional[LegalTaskType] = None) -> JurisdictionInferenceResult:
    """
    Convenience function for quick jurisdiction inference.
    
    Args:
        content: Legal content to analyze
        task_type: Optional task type context
        
    Returns:
        JurisdictionInferenceResult
    """
    engine = create_production_inference_engine()
    return engine.infer_jurisdiction(content, task_type)


def get_jurisdiction_confidence(content: str, jurisdiction: USJurisdiction) -> float:
    """
    Get confidence score for a specific jurisdiction given content.
    
    Args:
        content: Legal content
        jurisdiction: Jurisdiction to evaluate
        
    Returns:
        Confidence score (0.0 to 1.0)
    """
    engine = create_production_inference_engine()
    result = engine.infer_jurisdiction(content)
    
    if result.jurisdiction == jurisdiction:
        return result.confidence
    
    # Check alternatives
    for alt_jurisdiction, confidence, _ in result.alternatives:
        if alt_jurisdiction == jurisdiction:
            return confidence
    
    return 0.0

USJurisdictionInferenceEngine = JurisdictionInferenceEngine