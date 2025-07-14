"""
US Jurisdiction System Foundation for Multi-Task Legal Reward System

This module provides comprehensive support for the US legal system with detailed
jurisdiction information, validation, and contextual data for all 50 states,
District of Columbia, and federal jurisdiction.

Key Features:
- Complete US jurisdiction mapping with metadata
- Jurisdiction validation and normalization
- State-specific legal context information
- Federal vs state jurisdiction determination
- Legal domain jurisdiction requirements
- Jurisdiction inference support utilities
- Production-ready error handling and logging

The system is designed to support accurate legal evaluations by ensuring
that legal advice and analysis are appropriately contextualized within
the correct US jurisdiction.
"""

import re
import logging
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Import core components
from core import (
    USJurisdiction, LegalDomain, LegalRewardSystemError,
    create_error_context, US_STATES_AND_TERRITORIES
)


@dataclass
class JurisdictionMetadata:
    """
    Comprehensive metadata for a US jurisdiction.
    
    Contains detailed information about each jurisdiction including
    legal system characteristics, common legal domains, and
    contextual information for accurate legal analysis.
    """
    
    # Basic identification
    jurisdiction: USJurisdiction
    full_name: str
    abbreviation: str
    type: str  # "state", "federal_district", "territory", "federal"
    
    # Geographic and administrative
    region: str  # "northeast", "southeast", "midwest", "southwest", "west", "federal"
    capital: str
    largest_city: str
    population: int
    
    # Legal system characteristics
    court_system_type: str  # "unified", "dual", "specialized"
    supreme_court_name: str
    appellate_courts: List[str]
    trial_courts: List[str]
    
    # Common legal domains and specializations
    prominent_legal_areas: List[str]
    notable_legal_institutions: List[str]
    bar_admission_authority: str
    
    # Jurisdictional scope and limitations
    federal_district: str
    federal_circuit: str
    interstate_compacts: List[str] = field(default_factory=list)
    
    # Additional context for legal analysis
    legal_traditions: List[str] = field(default_factory=list)  # "common_law", "civil_law", "tribal_law"
    special_considerations: List[str] = field(default_factory=list)
    law_school_count: int = 0
    
    def is_state_jurisdiction(self) -> bool:
        """Check if this is a state-level jurisdiction"""
        return self.type == "state"
    
    def is_federal_jurisdiction(self) -> bool:
        """Check if this is federal jurisdiction"""
        return self.type == "federal"
    
    def is_territory(self) -> bool:
        """Check if this is a US territory"""
        return self.type == "territory"
    
    def get_court_hierarchy(self) -> List[str]:
        """Get the court hierarchy from trial to supreme court"""
        hierarchy = self.trial_courts.copy()
        hierarchy.extend(self.appellate_courts)
        hierarchy.append(self.supreme_court_name)
        return hierarchy
    
    def has_specialized_courts(self) -> bool:
        """Check if jurisdiction has specialized courts"""
        return self.court_system_type == "specialized"


class USJurisdictionError(LegalRewardSystemError):
    """Specific error for US jurisdiction system issues"""
    
    def __init__(self, message: str, jurisdiction: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.jurisdiction = jurisdiction


class JurisdictionValidator:
    """
    Validates and normalizes US jurisdiction references.
    
    Provides comprehensive validation of jurisdiction strings,
    handles common variations and abbreviations, and ensures
    consistent jurisdiction identification across the system.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.JurisdictionValidator")
        
        # Build jurisdiction lookup maps
        self._build_lookup_maps()
        
        # Common jurisdiction patterns
        self.federal_patterns = [
            r"federal",
            r"fed\b",
            r"united states",
            r"us federal",
            r"u\.s\.",
            r"national"
        ]
        
        self.general_patterns = [
            r"general",
            r"generic",
            r"non-specific",
            r"applicable",
            r"universal"
        ]
    
    def _build_lookup_maps(self):
        """Build comprehensive lookup maps for jurisdiction normalization"""
        self.name_to_jurisdiction = {}
        self.abbrev_to_jurisdiction = {}
        self.alternative_names = {}
        
        # Get all jurisdiction metadata
        jurisdiction_data = get_all_jurisdiction_metadata()
        
        for jurisdiction, metadata in jurisdiction_data.items():
            # Standard name and abbreviation
            self.name_to_jurisdiction[metadata.full_name.lower()] = jurisdiction
            self.abbrev_to_jurisdiction[metadata.abbreviation.lower()] = jurisdiction
            
            # Common variations
            name_variations = [
                metadata.full_name.lower(),
                metadata.full_name.lower().replace(" ", ""),
                metadata.abbreviation.lower(),
                metadata.abbreviation.lower().replace(".", ""),
                jurisdiction.value.lower(),
                jurisdiction.value.lower().replace("_", " ")
            ]
            
            # Add state-specific variations
            if metadata.is_state_jurisdiction():
                name_variations.extend([
                    f"state of {metadata.full_name.lower()}",
                    f"{metadata.full_name.lower()} state",
                    f"{metadata.abbreviation.lower()} state"
                ])
            
            # Store all variations
            for variation in name_variations:
                if variation and variation not in self.alternative_names:
                    self.alternative_names[variation] = jurisdiction
    
    def normalize_jurisdiction_string(self, jurisdiction_str: str) -> Optional[USJurisdiction]:
        """
        Normalize a jurisdiction string to a USJurisdiction enum.
        
        Args:
            jurisdiction_str: String representation of jurisdiction
            
        Returns:
            USJurisdiction enum if valid, None if cannot be normalized
        """
        if not jurisdiction_str or not isinstance(jurisdiction_str, str):
            return None
        
        # Clean the input string
        cleaned = jurisdiction_str.strip().lower()
        cleaned = re.sub(r'[^\w\s]', '', cleaned)  # Remove punctuation
        cleaned = re.sub(r'\s+', ' ', cleaned)     # Normalize whitespace
        
        # Check for federal patterns
        for pattern in self.federal_patterns:
            if re.search(pattern, cleaned):
                return USJurisdiction.FEDERAL
        
        # Check for general patterns
        for pattern in self.general_patterns:
            if re.search(pattern, cleaned):
                return USJurisdiction.GENERAL
        
        # Direct lookup
        if cleaned in self.alternative_names:
            return self.alternative_names[cleaned]
        
        # Fuzzy matching for common misspellings
        return self._fuzzy_match_jurisdiction(cleaned)
    
    def _fuzzy_match_jurisdiction(self, cleaned_str: str) -> Optional[USJurisdiction]:
        """Attempt fuzzy matching for jurisdiction strings"""
        try:
            from fuzzywuzzy import fuzz
            
            best_match = None
            best_score = 0
            threshold = 80  # Minimum similarity score
            
            for variation, jurisdiction in self.alternative_names.items():
                score = fuzz.ratio(cleaned_str, variation)
                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = jurisdiction
            
            if best_match:
                self.logger.info(f"Fuzzy matched '{cleaned_str}' to {best_match.value} (score: {best_score})")
            
            return best_match
            
        except ImportError:
            # Fallback without fuzzy matching
            self.logger.warning("fuzzywuzzy not available, skipping fuzzy jurisdiction matching")
            return None
    
    def validate_jurisdiction(self, jurisdiction: Union[str, USJurisdiction]) -> USJurisdiction:
        """
        Validate and convert jurisdiction to USJurisdiction enum.
        
        Args:
            jurisdiction: String or enum representing jurisdiction
            
        Returns:
            Valid USJurisdiction enum
            
        Raises:
            USJurisdictionError: If jurisdiction cannot be validated
        """
        if isinstance(jurisdiction, USJurisdiction):
            return jurisdiction
        
        if isinstance(jurisdiction, str):
            normalized = self.normalize_jurisdiction_string(jurisdiction)
            if normalized:
                return normalized
            
            raise USJurisdictionError(
                f"Invalid jurisdiction string: '{jurisdiction}'",
                jurisdiction=jurisdiction,
                error_context=create_error_context("jurisdiction", "validate_jurisdiction")
            )
        
        raise USJurisdictionError(
            f"Invalid jurisdiction type: {type(jurisdiction)}",
            error_context=create_error_context("jurisdiction", "validate_jurisdiction")
        )
    
    def get_suggested_jurisdictions(self, partial: str, limit: int = 5) -> List[Tuple[USJurisdiction, str, float]]:
        """
        Get suggested jurisdictions for a partial string.
        
        Args:
            partial: Partial jurisdiction string
            limit: Maximum number of suggestions
            
        Returns:
            List of (jurisdiction, full_name, confidence) tuples
        """
        suggestions = []
        partial_clean = partial.strip().lower()
        
        if not partial_clean:
            return suggestions
        
        try:
            from fuzzywuzzy import fuzz
            
            jurisdiction_data = get_all_jurisdiction_metadata()
            
            for jurisdiction, metadata in jurisdiction_data.items():
                # Check various name formats
                candidates = [
                    (metadata.full_name, metadata.full_name),
                    (metadata.abbreviation, metadata.full_name),
                    (jurisdiction.value.replace("_", " "), metadata.full_name)
                ]
                
                for candidate_text, display_name in candidates:
                    score = fuzz.partial_ratio(partial_clean, candidate_text.lower())
                    if score > 50:  # Minimum threshold for suggestions
                        suggestions.append((jurisdiction, display_name, score / 100.0))
            
            # Sort by confidence and limit results
            suggestions.sort(key=lambda x: x[2], reverse=True)
            return suggestions[:limit]
            
        except ImportError:
            # Fallback without fuzzy matching
            return [(USJurisdiction.GENERAL, "General (non-specific)", 0.5)]


class JurisdictionContextProvider:
    """
    Provides contextual information for US jurisdictions.
    
    Supplies detailed context about legal systems, court structures,
    and jurisdiction-specific considerations for accurate legal analysis.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.JurisdictionContextProvider")
        self.validator = JurisdictionValidator()
        
        # Load all jurisdiction metadata
        self.jurisdiction_metadata = get_all_jurisdiction_metadata()
    
    def get_jurisdiction_context(self, jurisdiction: Union[str, USJurisdiction]) -> Dict[str, Any]:
        """
        Get comprehensive context for a jurisdiction.
        
        Args:
            jurisdiction: Jurisdiction to get context for
            
        Returns:
            Dictionary with jurisdiction context information
        """
        try:
            # Validate and normalize jurisdiction
            validated_jurisdiction = self.validator.validate_jurisdiction(jurisdiction)
            
            # Get metadata
            if validated_jurisdiction not in self.jurisdiction_metadata:
                raise USJurisdictionError(
                    f"No metadata available for jurisdiction: {validated_jurisdiction}",
                    jurisdiction=str(validated_jurisdiction)
                )
            
            metadata = self.jurisdiction_metadata[validated_jurisdiction]
            
            # Build context dictionary
            context = {
                "jurisdiction": validated_jurisdiction,
                "metadata": metadata,
                "legal_system": self._get_legal_system_info(metadata),
                "court_structure": self._get_court_structure_info(metadata),
                "practice_areas": self._get_practice_areas_info(metadata),
                "jurisdictional_scope": self._get_jurisdictional_scope_info(metadata),
                "special_considerations": metadata.special_considerations,
                "related_jurisdictions": self._get_related_jurisdictions(metadata)
            }
            
            return context
            
        except Exception as e:
            if isinstance(e, USJurisdictionError):
                raise
            
            raise USJurisdictionError(
                f"Failed to get jurisdiction context: {e}",
                jurisdiction=str(jurisdiction),
                error_context=create_error_context("jurisdiction", "get_jurisdiction_context"),
                original_exception=e
            )
    
    def _get_legal_system_info(self, metadata: JurisdictionMetadata) -> Dict[str, Any]:
        """Get legal system information for jurisdiction"""
        return {
            "type": metadata.type,
            "court_system_type": metadata.court_system_type,
            "legal_traditions": metadata.legal_traditions,
            "bar_admission_authority": metadata.bar_admission_authority,
            "law_schools": metadata.law_school_count
        }
    
    def _get_court_structure_info(self, metadata: JurisdictionMetadata) -> Dict[str, Any]:
        """Get court structure information for jurisdiction"""
        return {
            "supreme_court": metadata.supreme_court_name,
            "appellate_courts": metadata.appellate_courts,
            "trial_courts": metadata.trial_courts,
            "hierarchy": metadata.get_court_hierarchy(),
            "has_specialized_courts": metadata.has_specialized_courts()
        }
    
    def _get_practice_areas_info(self, metadata: JurisdictionMetadata) -> Dict[str, Any]:
        """Get practice areas information for jurisdiction"""
        return {
            "prominent_areas": metadata.prominent_legal_areas,
            "notable_institutions": metadata.notable_legal_institutions,
            "specializations": []  # Could be expanded with specific specializations
        }
    
    def _get_jurisdictional_scope_info(self, metadata: JurisdictionMetadata) -> Dict[str, Any]:
        """Get jurisdictional scope information"""
        return {
            "federal_district": metadata.federal_district,
            "federal_circuit": metadata.federal_circuit,
            "interstate_compacts": metadata.interstate_compacts,
            "geographic_scope": {
                "region": metadata.region,
                "capital": metadata.capital,
                "largest_city": metadata.largest_city,
                "population": metadata.population
            }
        }
    
    def _get_related_jurisdictions(self, metadata: JurisdictionMetadata) -> List[str]:
        """Get related jurisdictions (same circuit, region, etc.)"""
        related = []
        
        # Add federal if this is a state
        if metadata.is_state_jurisdiction():
            related.append("federal")
        
        # Add states in same federal circuit
        for jurisdiction, other_metadata in self.jurisdiction_metadata.items():
            if (jurisdiction != metadata.jurisdiction and 
                other_metadata.federal_circuit == metadata.federal_circuit and
                other_metadata.is_state_jurisdiction()):
                related.append(jurisdiction.value)
        
        return related
    
    def get_jurisdiction_for_legal_domain(self, 
                                        domain: LegalDomain,
                                        preferred_jurisdiction: Optional[USJurisdiction] = None) -> List[USJurisdiction]:
        """
        Get appropriate jurisdictions for a legal domain.
        
        Args:
            domain: Legal domain
            preferred_jurisdiction: Preferred jurisdiction if specified
            
        Returns:
            List of appropriate jurisdictions for the domain
        """
        # Domain-specific jurisdiction requirements
        domain_jurisdiction_map = {
            LegalDomain.CONSTITUTIONAL: [USJurisdiction.FEDERAL],
            LegalDomain.FEDERAL_STATUTORY: [USJurisdiction.FEDERAL],
            LegalDomain.INTERSTATE_COMMERCE: [USJurisdiction.FEDERAL],
            LegalDomain.IMMIGRATION: [USJurisdiction.FEDERAL],
            LegalDomain.INTELLECTUAL_PROPERTY: [USJurisdiction.FEDERAL],
            LegalDomain.SECURITIES: [USJurisdiction.FEDERAL],
            LegalDomain.ANTITRUST: [USJurisdiction.FEDERAL],
            LegalDomain.BANKRUPTCY: [USJurisdiction.FEDERAL],
            LegalDomain.TAX: [USJurisdiction.FEDERAL],  # Both federal and state, but federal is primary
            
            # Primarily state domains (but can have federal aspects)
            LegalDomain.CONTRACT: [],  # Any jurisdiction
            LegalDomain.TORT: [],      # Any jurisdiction  
            LegalDomain.PROPERTY: [], # Any jurisdiction
            LegalDomain.FAMILY: [],   # Any jurisdiction
            LegalDomain.CRIMINAL: [], # Any jurisdiction
            LegalDomain.EMPLOYMENT: [], # Any jurisdiction
            LegalDomain.CORPORATE: [], # Any jurisdiction
            LegalDomain.HEALTHCARE: [], # Any jurisdiction
            LegalDomain.ENVIRONMENTAL: [], # Any jurisdiction
            LegalDomain.CIVIL_PROCEDURE: [], # Any jurisdiction
            LegalDomain.EVIDENCE: [], # Any jurisdiction
            LegalDomain.ETHICS: []    # Any jurisdiction
        }
        
        # Get domain-specific requirements
        required_jurisdictions = domain_jurisdiction_map.get(domain, [])
        
        # If domain requires specific jurisdiction, return those
        if required_jurisdictions:
            return required_jurisdictions
        
        # If preferred jurisdiction is specified and valid for domain, use it
        if preferred_jurisdiction and preferred_jurisdiction not in required_jurisdictions:
            return [preferred_jurisdiction]
        
        # Default to general for domains that work in any jurisdiction
        return [USJurisdiction.GENERAL]


class USJurisdictionSystem:
    """
    Main US Jurisdiction System for legal reward evaluation.
    
    Provides comprehensive jurisdiction support and validation
    for the US legal system including all 50 states, DC, and federal.
    
    This is a convenience wrapper around the existing jurisdiction functions.
    """
    
    def __init__(self):
        """Initialize the US jurisdiction system"""
        self.metadata_cache = get_all_jurisdiction_metadata()
        self.logger = logging.getLogger("us_jurisdiction_system")
    
    def get_all_jurisdictions(self) -> List[USJurisdiction]:
        """Get all supported US jurisdictions"""
        return list(USJurisdiction)
    
    def validate_jurisdiction(self, jurisdiction: str) -> bool:
        """Validate if jurisdiction is supported"""
        try:
            return validate_jurisdiction(jurisdiction) is not None
        except USJurisdictionError:
            return False
    
    def get_jurisdiction_metadata(self, jurisdiction: USJurisdiction) -> Optional[JurisdictionMetadata]:
        """Get metadata for a specific jurisdiction"""
        return self.metadata_cache.get(jurisdiction)
    
    def normalize_jurisdiction(self, jurisdiction_str: str) -> Optional[USJurisdiction]:
        """Normalize jurisdiction string to USJurisdiction enum"""
        try:
            return get_jurisdiction_by_name(jurisdiction_str)
        except USJurisdictionError:
            return None
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get summary of the jurisdiction system"""
        return get_jurisdiction_summary()
    
    def is_federal_only_domain(self, domain: LegalDomain) -> bool:
        """Check if domain requires federal jurisdiction"""
        return is_jurisdiction_federal_only(domain)
    
    def get_federal_circuit_for_state(self, state: USJurisdiction) -> Optional[List[USJurisdiction]]:
        """Get states in the same federal circuit"""
        try:
            # This would need to be implemented based on your federal circuit logic
            return []
        except Exception:
            return []
    
    def get_jurisdiction_count(self) -> Dict[str, int]:
        """Get count of different jurisdiction types"""
        all_jurisdictions = self.get_all_jurisdictions()
        
        return {
            "total": len(all_jurisdictions),
            "states": len([j for j in all_jurisdictions if j != USJurisdiction.FEDERAL and j != USJurisdiction.GENERAL and j != USJurisdiction.DISTRICT_OF_COLUMBIA]),
            "federal": 1,
            "district_of_columbia": 1,
            "general": 1
        }


def get_all_jurisdiction_metadata() -> Dict[USJurisdiction, JurisdictionMetadata]:
    """
    Get comprehensive metadata for all US jurisdictions.
    
    Returns:
        Dictionary mapping USJurisdiction enums to their metadata
    """
    metadata = {}
    
    # Federal jurisdiction
    metadata[USJurisdiction.FEDERAL] = JurisdictionMetadata(
        jurisdiction=USJurisdiction.FEDERAL,
        full_name="Federal",
        abbreviation="FED",
        type="federal",
        region="federal",
        capital="Washington, D.C.",
        largest_city="N/A",
        population=0,
        court_system_type="specialized",
        supreme_court_name="Supreme Court of the United States",
        appellate_courts=["U.S. Courts of Appeals (12 Circuits)"],
        trial_courts=["U.S. District Courts", "U.S. Bankruptcy Courts", "U.S. Tax Court"],
        prominent_legal_areas=["Constitutional Law", "Federal Statutory Law", "Interstate Commerce", 
                              "Immigration", "Intellectual Property", "Securities", "Antitrust"],
        notable_legal_institutions=["Supreme Court of the United States", "Federal Bar Association"],
        bar_admission_authority="Various Federal Courts",
        federal_district="All Districts",
        federal_circuit="All Circuits",
        legal_traditions=["common_law"],
        special_considerations=["Federal supremacy", "Interstate commerce jurisdiction", "Constitutional interpretation"],
        law_school_count=0
    )
    
    # General jurisdiction
    metadata[USJurisdiction.GENERAL] = JurisdictionMetadata(
        jurisdiction=USJurisdiction.GENERAL,
        full_name="General",
        abbreviation="GEN",
        type="general",
        region="general",
        capital="N/A",
        largest_city="N/A", 
        population=0,
        court_system_type="general",
        supreme_court_name="General Principles",
        appellate_courts=["General Appellate Principles"],
        trial_courts=["General Trial Court Principles"],
        prominent_legal_areas=["General Legal Principles", "Common Law", "Universal Legal Concepts"],
        notable_legal_institutions=["American Bar Association"],
        bar_admission_authority="Various State Bars",
        federal_district="N/A",
        federal_circuit="N/A",
        legal_traditions=["common_law"],
        special_considerations=["Non-jurisdiction-specific", "General legal principles", "Educational purposes"],
        law_school_count=0
    )
    
    # State jurisdictions - Complete data for all 50 states + DC
    state_data = [
        # Northeast Region
        {
            "jurisdiction": USJurisdiction.ALABAMA,
            "full_name": "Alabama",
            "abbreviation": "AL",
            "region": "southeast",
            "capital": "Montgomery", 
            "largest_city": "Birmingham",
            "population": 5024279,
            "court_system_type": "unified",
            "supreme_court_name": "Supreme Court of Alabama",
            "appellate_courts": ["Court of Civil Appeals", "Court of Criminal Appeals"],
            "trial_courts": ["Circuit Courts", "District Courts"],
            "prominent_legal_areas": ["Civil Rights", "Employment Law", "Personal Injury", "Criminal Law"],
            "notable_legal_institutions": ["Alabama State Bar", "University of Alabama School of Law"],
            "bar_admission_authority": "Alabama State Bar",
            "federal_district": "Northern, Middle, and Southern Districts of Alabama",
            "federal_circuit": "11th Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Right-to-work state", "Conservative legal traditions"],
            "law_school_count": 2
        },
        {
            "jurisdiction": USJurisdiction.ALASKA,
            "full_name": "Alaska",
            "abbreviation": "AK",
            "region": "west",
            "capital": "Juneau",
            "largest_city": "Anchorage", 
            "population": 733391,
            "court_system_type": "unified",
            "supreme_court_name": "Alaska Supreme Court",
            "appellate_courts": ["Alaska Court of Appeals"],
            "trial_courts": ["Superior Courts", "District Courts"],
            "prominent_legal_areas": ["Environmental Law", "Natural Resources", "Maritime Law", "Native Law"],
            "notable_legal_institutions": ["Alaska Bar Association"],
            "bar_admission_authority": "Alaska Bar Association",
            "federal_district": "District of Alaska",
            "federal_circuit": "9th Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Native corporation law", "Resource extraction regulations", "Remote geography"],
            "law_school_count": 0
        },
        {
            "jurisdiction": USJurisdiction.ARIZONA,
            "full_name": "Arizona",
            "abbreviation": "AZ",
            "region": "southwest",
            "capital": "Phoenix",
            "largest_city": "Phoenix",
            "population": 7151502,
            "court_system_type": "unified",
            "supreme_court_name": "Arizona Supreme Court",
            "appellate_courts": ["Arizona Court of Appeals"],
            "trial_courts": ["Superior Courts", "Justice Courts", "Municipal Courts"],
            "prominent_legal_areas": ["Immigration Law", "Water Rights", "Real Estate", "Criminal Law"],
            "notable_legal_institutions": ["State Bar of Arizona", "Arizona State University Sandra Day O'Connor College of Law"],
            "bar_admission_authority": "State Bar of Arizona",
            "federal_district": "District of Arizona",
            "federal_circuit": "9th Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Border state issues", "Water law complexities", "Immigration enforcement"],
            "law_school_count": 2
        },
        {
            "jurisdiction": USJurisdiction.ARKANSAS,
            "full_name": "Arkansas", 
            "abbreviation": "AR",
            "region": "southeast",
            "capital": "Little Rock",
            "largest_city": "Little Rock",
            "population": 3011524,
            "court_system_type": "unified",
            "supreme_court_name": "Arkansas Supreme Court",
            "appellate_courts": ["Arkansas Court of Appeals"],
            "trial_courts": ["Circuit Courts", "District Courts"],
            "prominent_legal_areas": ["Agriculture Law", "Employment Law", "Personal Injury", "Family Law"],
            "notable_legal_institutions": ["Arkansas Bar Association", "University of Arkansas School of Law"],
            "bar_admission_authority": "Arkansas Bar Association",
            "federal_district": "Eastern and Western Districts of Arkansas",
            "federal_circuit": "8th Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Agricultural state considerations", "Right-to-work state"],
            "law_school_count": 1
        },
        {
            "jurisdiction": USJurisdiction.CALIFORNIA,
            "full_name": "California",
            "abbreviation": "CA", 
            "region": "west",
            "capital": "Sacramento",
            "largest_city": "Los Angeles",
            "population": 39538223,
            "court_system_type": "unified",
            "supreme_court_name": "Supreme Court of California",
            "appellate_courts": ["California Courts of Appeal (6 Districts)"],
            "trial_courts": ["Superior Courts"],
            "prominent_legal_areas": ["Technology Law", "Entertainment Law", "Environmental Law", "Immigration Law"],
            "notable_legal_institutions": ["State Bar of California", "Stanford Law School", "UC Berkeley School of Law"],
            "bar_admission_authority": "State Bar of California",
            "federal_district": "Northern, Central, Eastern, and Southern Districts of California",
            "federal_circuit": "9th Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Tech industry hub", "Progressive legal framework", "Complex environmental regulations"],
            "law_school_count": 20
        },
        {
            "jurisdiction": USJurisdiction.COLORADO,
            "full_name": "Colorado",
            "abbreviation": "CO",
            "region": "west", 
            "capital": "Denver",
            "largest_city": "Denver",
            "population": 5773714,
            "court_system_type": "unified",
            "supreme_court_name": "Colorado Supreme Court",
            "appellate_courts": ["Colorado Court of Appeals"],
            "trial_courts": ["District Courts", "County Courts"],
            "prominent_legal_areas": ["Cannabis Law", "Environmental Law", "Water Rights", "Energy Law"],
            "notable_legal_institutions": ["Colorado Bar Association", "University of Colorado Law School"],
            "bar_admission_authority": "Colorado Bar Association",
            "federal_district": "District of Colorado",
            "federal_circuit": "10th Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Cannabis legalization pioneer", "Water law complexities", "Energy industry"],
            "law_school_count": 2
        },
        {
            "jurisdiction": USJurisdiction.CONNECTICUT,
            "full_name": "Connecticut",
            "abbreviation": "CT",
            "region": "northeast",
            "capital": "Hartford",
            "largest_city": "Bridgeport",
            "population": 3605944,
            "court_system_type": "unified",
            "supreme_court_name": "Connecticut Supreme Court",
            "appellate_courts": ["Connecticut Appellate Court"],
            "trial_courts": ["Superior Courts"],
            "prominent_legal_areas": ["Insurance Law", "Corporate Law", "Healthcare Law", "Education Law"],
            "notable_legal_institutions": ["Connecticut Bar Association", "Yale Law School"],
            "bar_admission_authority": "Connecticut Bar Association",
            "federal_district": "District of Connecticut",
            "federal_circuit": "2nd Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Insurance industry center", "Proximity to NYC financial markets"],
            "law_school_count": 4
        },
        {
            "jurisdiction": USJurisdiction.DELAWARE,
            "full_name": "Delaware",
            "abbreviation": "DE",
            "region": "northeast",
            "capital": "Dover",
            "largest_city": "Wilmington",
            "population": 989948,
            "court_system_type": "specialized",
            "supreme_court_name": "Delaware Supreme Court",
            "appellate_courts": ["Delaware Superior Court (appellate)"],
            "trial_courts": ["Superior Court", "Court of Common Pleas", "Court of Chancery"],
            "prominent_legal_areas": ["Corporate Law", "Business Litigation", "Chancery Law", "Securities Law"],
            "notable_legal_institutions": ["Delaware State Bar Association", "Delaware Court of Chancery"],
            "bar_admission_authority": "Delaware State Bar Association",
            "federal_district": "District of Delaware",
            "federal_circuit": "3rd Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Corporate law epicenter", "Court of Chancery system", "Business-friendly jurisdiction"],
            "law_school_count": 1
        },
        {
            "jurisdiction": USJurisdiction.DISTRICT_OF_COLUMBIA,
            "full_name": "District of Columbia",
            "abbreviation": "DC",
            "region": "northeast",
            "capital": "Washington",
            "largest_city": "Washington",
            "population": 689545,
            "court_system_type": "unified",
            "supreme_court_name": "District of Columbia Court of Appeals",
            "appellate_courts": ["D.C. Court of Appeals"],
            "trial_courts": ["Superior Court of the District of Columbia"],
            "prominent_legal_areas": ["Government Law", "Administrative Law", "Lobbying Law", "International Law"],
            "notable_legal_institutions": ["D.C. Bar", "Georgetown University Law Center", "George Washington University Law School"],
            "bar_admission_authority": "District of Columbia Bar",
            "federal_district": "District of Columbia",
            "federal_circuit": "D.C. Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Federal government seat", "Administrative law center", "International law hub"],
            "law_school_count": 6
        },
        {
            "jurisdiction": USJurisdiction.FLORIDA,
            "full_name": "Florida",
            "abbreviation": "FL",
            "region": "southeast",
            "capital": "Tallahassee",
            "largest_city": "Jacksonville",
            "population": 22610726,
            "court_system_type": "unified",
            "supreme_court_name": "Florida Supreme Court",
            "appellate_courts": ["Florida District Courts of Appeal (5 Districts)"],
            "trial_courts": ["Circuit Courts", "County Courts"],
            "prominent_legal_areas": ["Personal Injury", "Real Estate", "Elder Law", "Maritime Law"],
            "notable_legal_institutions": ["Florida Bar", "University of Florida Levin College of Law"],
            "bar_admission_authority": "The Florida Bar",
            "federal_district": "Northern, Middle, and Southern Districts of Florida",
            "federal_circuit": "11th Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["No-fault insurance state", "High retiree population", "Hurricane/disaster law"],
            "law_school_count": 12
        },
        {
            "jurisdiction": USJurisdiction.GEORGIA,
            "full_name": "Georgia",
            "abbreviation": "GA",
            "region": "southeast",
            "capital": "Atlanta",
            "largest_city": "Atlanta",
            "population": 10799566,
            "court_system_type": "unified",
            "supreme_court_name": "Supreme Court of Georgia",
            "appellate_courts": ["Georgia Court of Appeals"],
            "trial_courts": ["Superior Courts", "State Courts", "Magistrate Courts"],
            "prominent_legal_areas": ["Corporate Law", "Personal Injury", "Real Estate", "Employment Law"],
            "notable_legal_institutions": ["State Bar of Georgia", "University of Georgia School of Law"],
            "bar_admission_authority": "State Bar of Georgia",
            "federal_district": "Northern, Middle, and Southern Districts of Georgia",
            "federal_circuit": "11th Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Major business hub", "Right-to-work state", "Diverse economy"],
            "law_school_count": 5
        },
        {
            "jurisdiction": USJurisdiction.HAWAII,
            "full_name": "Hawaii",
            "abbreviation": "HI",
            "region": "west",
            "capital": "Honolulu",
            "largest_city": "Honolulu",
            "population": 1455271,
            "court_system_type": "unified",
            "supreme_court_name": "Hawaii Supreme Court",
            "appellate_courts": ["Hawaii Intermediate Court of Appeals"],
            "trial_courts": ["Circuit Courts", "District Courts"],
            "prominent_legal_areas": ["Tourism Law", "Environmental Law", "Native Hawaiian Law", "Real Estate"],
            "notable_legal_institutions": ["Hawaii State Bar Association", "University of Hawaii at Manoa William S. Richardson School of Law"],
            "bar_admission_authority": "Hawaii State Bar Association",
            "federal_district": "District of Hawaii",
            "federal_circuit": "9th Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Island jurisdiction", "Native Hawaiian rights", "Tourism industry focus"],
            "law_school_count": 1
        },
        {
            "jurisdiction": USJurisdiction.IDAHO,
            "full_name": "Idaho",
            "abbreviation": "ID",
            "region": "west",
            "capital": "Boise",
            "largest_city": "Boise",
            "population": 1839106,
            "court_system_type": "unified",
            "supreme_court_name": "Idaho Supreme Court",
            "appellate_courts": ["Idaho Court of Appeals"],
            "trial_courts": ["District Courts", "Magistrate Courts"],
            "prominent_legal_areas": ["Agriculture Law", "Natural Resources", "Water Rights", "Mining Law"],
            "notable_legal_institutions": ["Idaho State Bar", "University of Idaho College of Law"],
            "bar_admission_authority": "Idaho State Bar",
            "federal_district": "District of Idaho",
            "federal_circuit": "9th Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Agricultural state", "Water rights issues", "Natural resource extraction"],
            "law_school_count": 1
        },
        {
            "jurisdiction": USJurisdiction.ILLINOIS,
            "full_name": "Illinois",
            "abbreviation": "IL",
            "region": "midwest",
            "capital": "Springfield",
            "largest_city": "Chicago",
            "population": 12812508,
            "court_system_type": "unified",
            "supreme_court_name": "Illinois Supreme Court",
            "appellate_courts": ["Illinois Appellate Court (5 Districts)"],
            "trial_courts": ["Circuit Courts"],
            "prominent_legal_areas": ["Corporate Law", "Employment Law", "Personal Injury", "Criminal Law"],
            "notable_legal_institutions": ["Illinois State Bar Association", "University of Chicago Law School", "Northwestern Pritzker School of Law"],
            "bar_admission_authority": "Illinois State Bar Association",
            "federal_district": "Northern, Central, and Southern Districts of Illinois",
            "federal_circuit": "7th Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Major business center", "Complex municipal law (Chicago)", "Transportation hub"],
            "law_school_count": 9
        },
        {
            "jurisdiction": USJurisdiction.INDIANA,
            "full_name": "Indiana",
            "abbreviation": "IN",
            "region": "midwest",
            "capital": "Indianapolis",
            "largest_city": "Indianapolis",
            "population": 6785528,
            "court_system_type": "unified",
            "supreme_court_name": "Indiana Supreme Court",
            "appellate_courts": ["Indiana Court of Appeals"],
            "trial_courts": ["Circuit Courts", "Superior Courts"],
            "prominent_legal_areas": ["Manufacturing Law", "Employment Law", "Personal Injury", "Agriculture Law"],
            "notable_legal_institutions": ["Indiana State Bar Association", "Indiana University Maurer School of Law"],
            "bar_admission_authority": "Indiana State Bar Association",
            "federal_district": "Northern and Southern Districts of Indiana",
            "federal_circuit": "7th Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Manufacturing state", "Right-to-work state", "Agricultural considerations"],
            "law_school_count": 4
        },
        {
            "jurisdiction": USJurisdiction.IOWA,
            "full_name": "Iowa",
            "abbreviation": "IA",
            "region": "midwest",
            "capital": "Des Moines",
            "largest_city": "Des Moines",
            "population": 3190369,
            "court_system_type": "unified",
            "supreme_court_name": "Iowa Supreme Court",
            "appellate_courts": ["Iowa Court of Appeals"],
            "trial_courts": ["District Courts"],
            "prominent_legal_areas": ["Agriculture Law", "Insurance Law", "Employment Law", "Family Law"],
            "notable_legal_institutions": ["Iowa State Bar Association", "University of Iowa College of Law"],
            "bar_admission_authority": "Iowa State Bar Association",
            "federal_district": "Northern and Southern Districts of Iowa",
            "federal_circuit": "8th Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Agricultural state", "Insurance industry", "Progressive civil rights history"],
            "law_school_count": 2
        },
        {
            "jurisdiction": USJurisdiction.KANSAS,
            "full_name": "Kansas",
            "abbreviation": "KS",
            "region": "midwest",
            "capital": "Topeka",
            "largest_city": "Wichita",
            "population": 2937880,
            "court_system_type": "unified",
            "supreme_court_name": "Kansas Supreme Court",
            "appellate_courts": ["Kansas Court of Appeals"],
            "trial_courts": ["District Courts"],
            "prominent_legal_areas": ["Agriculture Law", "Oil and Gas Law", "Employment Law", "Criminal Law"],
            "notable_legal_institutions": ["Kansas Bar Association", "University of Kansas School of Law"],
            "bar_admission_authority": "Kansas Bar Association",
            "federal_district": "District of Kansas",
            "federal_circuit": "10th Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Agricultural state", "Oil and gas industry", "Aviation industry (Wichita)"],
            "law_school_count": 2
        },
        {
            "jurisdiction": USJurisdiction.KENTUCKY,
            "full_name": "Kentucky",
            "abbreviation": "KY",
            "region": "southeast",
            "capital": "Frankfort",
            "largest_city": "Louisville",
            "population": 4505836,
            "court_system_type": "unified",
            "supreme_court_name": "Kentucky Supreme Court",
            "appellate_courts": ["Kentucky Court of Appeals"],
            "trial_courts": ["Circuit Courts", "District Courts"],
            "prominent_legal_areas": ["Coal Law", "Personal Injury", "Employment Law", "Family Law"],
            "notable_legal_institutions": ["Kentucky Bar Association", "University of Kentucky College of Law"],
            "bar_admission_authority": "Kentucky Bar Association",
            "federal_district": "Eastern and Western Districts of Kentucky",
            "federal_circuit": "6th Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Coal industry", "Bourbon industry regulations", "Right-to-work state"],
            "law_school_count": 3
        },
        {
            "jurisdiction": USJurisdiction.LOUISIANA,
            "full_name": "Louisiana",
            "abbreviation": "LA",
            "region": "southeast",
            "capital": "Baton Rouge",
            "largest_city": "New Orleans",
            "population": 4657757,
            "court_system_type": "dual",
            "supreme_court_name": "Louisiana Supreme Court",
            "appellate_courts": ["Louisiana Courts of Appeal (5 Circuits)"],
            "trial_courts": ["District Courts", "Parish Courts"],
            "prominent_legal_areas": ["Civil Law", "Maritime Law", "Oil and Gas Law", "Personal Injury"],
            "notable_legal_institutions": ["Louisiana State Bar Association", "Tulane University Law School"],
            "bar_admission_authority": "Louisiana State Bar Association",
            "federal_district": "Eastern, Middle, and Western Districts of Louisiana",
            "federal_circuit": "5th Circuit",
            "legal_traditions": ["civil_law", "common_law"],
            "special_considerations": ["Civil law tradition", "Unique legal system", "Maritime jurisdiction"],
            "law_school_count": 3
        },
        {
            "jurisdiction": USJurisdiction.MAINE,
            "full_name": "Maine",
            "abbreviation": "ME",
            "region": "northeast",
            "capital": "Augusta",
            "largest_city": "Portland",
            "population": 1395722,
            "court_system_type": "unified",
            "supreme_court_name": "Maine Supreme Judicial Court",
            "appellate_courts": ["Maine Supreme Judicial Court"],
            "trial_courts": ["Superior Courts", "District Courts"],
            "prominent_legal_areas": ["Maritime Law", "Environmental Law", "Personal Injury", "Real Estate"],
            "notable_legal_institutions": ["Maine State Bar Association", "University of Maine School of Law"],
            "bar_admission_authority": "Maine State Bar Association",
            "federal_district": "District of Maine",
            "federal_circuit": "1st Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Maritime jurisdiction", "Environmental focus", "Rural considerations"],
            "law_school_count": 1
        },
        {
            "jurisdiction": USJurisdiction.MARYLAND,
            "full_name": "Maryland",
            "abbreviation": "MD",
            "region": "northeast",
            "capital": "Annapolis",
            "largest_city": "Baltimore",
            "population": 6164660,
            "court_system_type": "unified",
            "supreme_court_name": "Maryland Court of Appeals",
            "appellate_courts": ["Maryland Court of Special Appeals"],
            "trial_courts": ["Circuit Courts", "District Courts"],
            "prominent_legal_areas": ["Government Law", "Healthcare Law", "Corporate Law", "Maritime Law"],
            "notable_legal_institutions": ["Maryland State Bar Association", "University of Maryland Francis King Carey School of Law"],
            "bar_admission_authority": "Maryland State Bar Association",
            "federal_district": "District of Maryland",
            "federal_circuit": "4th Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Federal government proximity", "Healthcare industry", "Port of Baltimore"],
            "law_school_count": 2
        },
        {
            "jurisdiction": USJurisdiction.MASSACHUSETTS,
            "full_name": "Massachusetts",
            "abbreviation": "MA",
            "region": "northeast",
            "capital": "Boston",
            "largest_city": "Boston",
            "population": 7001399,
            "court_system_type": "unified",
            "supreme_court_name": "Massachusetts Supreme Judicial Court",
            "appellate_courts": ["Massachusetts Appeals Court"],
            "trial_courts": ["Superior Courts", "District Courts"],
            "prominent_legal_areas": ["Healthcare Law", "Technology Law", "Education Law", "Financial Services"],
            "notable_legal_institutions": ["Massachusetts Bar Association", "Harvard Law School", "Boston University School of Law"],
            "bar_admission_authority": "Massachusetts Bar Association",
            "federal_district": "District of Massachusetts",
            "federal_circuit": "1st Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Healthcare innovation", "Technology sector", "Education hub"],
            "law_school_count": 9
        },
        {
            "jurisdiction": USJurisdiction.MICHIGAN,
            "full_name": "Michigan",
            "abbreviation": "MI",
            "region": "midwest",
            "capital": "Lansing",
            "largest_city": "Detroit",
            "population": 10037261,
            "court_system_type": "unified",
            "supreme_court_name": "Michigan Supreme Court",
            "appellate_courts": ["Michigan Court of Appeals"],
            "trial_courts": ["Circuit Courts", "District Courts"],
            "prominent_legal_areas": ["Automotive Law", "Employment Law", "Personal Injury", "Environmental Law"],
            "notable_legal_institutions": ["State Bar of Michigan", "University of Michigan Law School"],
            "bar_admission_authority": "State Bar of Michigan",
            "federal_district": "Eastern and Western Districts of Michigan",
            "federal_circuit": "6th Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Automotive industry", "Labor law complexities", "Great Lakes jurisdiction"],
            "law_school_count": 5
        },
        {
            "jurisdiction": USJurisdiction.MINNESOTA,
            "full_name": "Minnesota",
            "abbreviation": "MN",
            "region": "midwest",
            "capital": "Saint Paul",
            "largest_city": "Minneapolis",
            "population": 5738732,
            "court_system_type": "unified",
            "supreme_court_name": "Minnesota Supreme Court",
            "appellate_courts": ["Minnesota Court of Appeals"],
            "trial_courts": ["District Courts"],
            "prominent_legal_areas": ["Healthcare Law", "Employment Law", "Environmental Law", "Corporate Law"],
            "notable_legal_institutions": ["Minnesota State Bar Association", "University of Minnesota Law School"],
            "bar_admission_authority": "Minnesota State Bar Association",
            "federal_district": "District of Minnesota",
            "federal_circuit": "8th Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Healthcare industry", "Mining industry", "Progressive legal framework"],
            "law_school_count": 4
        },
        {
            "jurisdiction": USJurisdiction.MISSISSIPPI,
            "full_name": "Mississippi",
            "abbreviation": "MS",
            "region": "southeast",
            "capital": "Jackson",
            "largest_city": "Jackson",
            "population": 2940057,
            "court_system_type": "unified",
            "supreme_court_name": "Mississippi Supreme Court",
            "appellate_courts": ["Mississippi Court of Appeals"],
            "trial_courts": ["Circuit Courts", "Chancery Courts"],
            "prominent_legal_areas": ["Personal Injury", "Agriculture Law", "Gaming Law", "Employment Law"],
            "notable_legal_institutions": ["Mississippi Bar", "University of Mississippi School of Law"],
            "bar_admission_authority": "Mississippi Bar",
            "federal_district": "Northern and Southern Districts of Mississippi",
            "federal_circuit": "5th Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Agriculture state", "Gaming industry", "Right-to-work state"],
            "law_school_count": 2
        },
        {
            "jurisdiction": USJurisdiction.MISSOURI,
            "full_name": "Missouri",
            "abbreviation": "MO",
            "region": "midwest",
            "capital": "Jefferson City",
            "largest_city": "Kansas City",
            "population": 6196010,
            "court_system_type": "unified",
            "supreme_court_name": "Missouri Supreme Court",
            "appellate_courts": ["Missouri Court of Appeals"],
            "trial_courts": ["Circuit Courts"],
            "prominent_legal_areas": ["Agriculture Law", "Personal Injury", "Employment Law", "Corporate Law"],
            "notable_legal_institutions": ["Missouri Bar", "University of Missouri School of Law"],
            "bar_admission_authority": "Missouri Bar",
            "federal_district": "Eastern and Western Districts of Missouri",
            "federal_circuit": "8th Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Agriculture state", "Transportation hub", "Right-to-work state"],
            "law_school_count": 4
        },
        {
            "jurisdiction": USJurisdiction.MONTANA,
            "full_name": "Montana",
            "abbreviation": "MT",
            "region": "west",
            "capital": "Helena",
            "largest_city": "Billings",
            "population": 1084225,
            "court_system_type": "unified",
            "supreme_court_name": "Montana Supreme Court",
            "appellate_courts": ["Montana Supreme Court"],
            "trial_courts": ["District Courts", "Justice Courts"],
            "prominent_legal_areas": ["Natural Resources", "Agriculture Law", "Mining Law", "Environmental Law"],
            "notable_legal_institutions": ["State Bar of Montana", "University of Montana School of Law"],
            "bar_admission_authority": "State Bar of Montana",
            "federal_district": "District of Montana",
            "federal_circuit": "9th Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Natural resource extraction", "Agriculture state", "Environmental concerns"],
            "law_school_count": 1
        },
        {
            "jurisdiction": USJurisdiction.NEBRASKA,
            "full_name": "Nebraska",
            "abbreviation": "NE",
            "region": "midwest",
            "capital": "Lincoln",
            "largest_city": "Omaha",
            "population": 1961504,
            "court_system_type": "unified",
            "supreme_court_name": "Nebraska Supreme Court",
            "appellate_courts": ["Nebraska Court of Appeals"],
            "trial_courts": ["District Courts", "County Courts"],
            "prominent_legal_areas": ["Agriculture Law", "Insurance Law", "Employment Law", "Personal Injury"],
            "notable_legal_institutions": ["Nebraska State Bar Association", "University of Nebraska College of Law"],
            "bar_admission_authority": "Nebraska State Bar Association",
            "federal_district": "District of Nebraska",
            "federal_circuit": "8th Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Agriculture state", "Insurance industry", "Right-to-work state"],
            "law_school_count": 2
        },
        {
            "jurisdiction": USJurisdiction.NEVADA,
            "full_name": "Nevada",
            "abbreviation": "NV",
            "region": "west",
            "capital": "Carson City",
            "largest_city": "Las Vegas",
            "population": 3104614,
            "court_system_type": "unified",
            "supreme_court_name": "Nevada Supreme Court",
            "appellate_courts": ["Nevada Court of Appeals"],
            "trial_courts": ["District Courts", "Justice Courts"],
            "prominent_legal_areas": ["Gaming Law", "Entertainment Law", "Real Estate", "Mining Law"],
            "notable_legal_institutions": ["State Bar of Nevada", "University of Nevada, Las Vegas Boyd School of Law"],
            "bar_admission_authority": "State Bar of Nevada",
            "federal_district": "District of Nevada",
            "federal_circuit": "9th Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Gaming industry", "Entertainment law", "No state income tax"],
            "law_school_count": 1
        },
        {
            "jurisdiction": USJurisdiction.NEW_HAMPSHIRE,
            "full_name": "New Hampshire",
            "abbreviation": "NH",
            "region": "northeast",
            "capital": "Concord",
            "largest_city": "Manchester",
            "population": 1395231,
            "court_system_type": "unified",
            "supreme_court_name": "New Hampshire Supreme Court",
            "appellate_courts": ["New Hampshire Supreme Court"],
            "trial_courts": ["Superior Courts", "District Courts"],
            "prominent_legal_areas": ["Personal Injury", "Real Estate", "Employment Law", "Family Law"],
            "notable_legal_institutions": ["New Hampshire Bar Association", "University of New Hampshire School of Law"],
            "bar_admission_authority": "New Hampshire Bar Association",
            "federal_district": "District of New Hampshire",
            "federal_circuit": "1st Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["No state income tax", "Live Free or Die philosophy", "Rural considerations"],
            "law_school_count": 1
        },
        {
            "jurisdiction": USJurisdiction.NEW_JERSEY,
            "full_name": "New Jersey",
            "abbreviation": "NJ",
            "region": "northeast",
            "capital": "Trenton",
            "largest_city": "Newark",
            "population": 9267130,
            "court_system_type": "unified",
            "supreme_court_name": "New Jersey Supreme Court",
            "appellate_courts": ["New Jersey Superior Court, Appellate Division"],
            "trial_courts": ["Superior Courts"],
            "prominent_legal_areas": ["Corporate Law", "Personal Injury", "Environmental Law", "Healthcare Law"],
            "notable_legal_institutions": ["New Jersey State Bar Association", "Rutgers Law School"],
            "bar_admission_authority": "New Jersey State Bar Association",
            "federal_district": "District of New Jersey",
            "federal_circuit": "3rd Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Proximity to NYC and Philadelphia", "Pharmaceutical industry", "Dense population"],
            "law_school_count": 3
        },
        {
            "jurisdiction": USJurisdiction.NEW_MEXICO,
            "full_name": "New Mexico",
            "abbreviation": "NM",
            "region": "southwest",
            "capital": "Santa Fe",
            "largest_city": "Albuquerque",
            "population": 2113344,
            "court_system_type": "unified",
            "supreme_court_name": "New Mexico Supreme Court",
            "appellate_courts": ["New Mexico Court of Appeals"],
            "trial_courts": ["District Courts", "Magistrate Courts"],
            "prominent_legal_areas": ["Water Rights", "Oil and Gas Law", "Native American Law", "Immigration Law"],
            "notable_legal_institutions": ["State Bar of New Mexico", "University of New Mexico School of Law"],
            "bar_admission_authority": "State Bar of New Mexico",
            "federal_district": "District of New Mexico",
            "federal_circuit": "10th Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Water law issues", "Oil and gas industry", "Native American tribes"],
            "law_school_count": 1
        },
        {
            "jurisdiction": USJurisdiction.NEW_YORK,
            "full_name": "New York",
            "abbreviation": "NY",
            "region": "northeast",
            "capital": "Albany",
            "largest_city": "New York City",
            "population": 19336776,
            "court_system_type": "unified",
            "supreme_court_name": "New York Court of Appeals",
            "appellate_courts": ["Appellate Division of the Supreme Court (4 Departments)"],
            "trial_courts": ["Supreme Courts", "County Courts", "Family Courts"],
            "prominent_legal_areas": ["Financial Services", "Corporate Law", "Real Estate", "Entertainment Law"],
            "notable_legal_institutions": ["New York State Bar Association", "Columbia Law School", "NYU School of Law"],
            "bar_admission_authority": "New York State Bar Association",
            "federal_district": "Northern, Southern, Eastern, and Western Districts of New York",
            "federal_circuit": "2nd Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Financial capital", "Complex court system", "International business hub"],
            "law_school_count": 15
        },
        {
            "jurisdiction": USJurisdiction.NORTH_CAROLINA,
            "full_name": "North Carolina",
            "abbreviation": "NC",
            "region": "southeast",
            "capital": "Raleigh",
            "largest_city": "Charlotte",
            "population": 10698973,
            "court_system_type": "unified",
            "supreme_court_name": "North Carolina Supreme Court",
            "appellate_courts": ["North Carolina Court of Appeals"],
            "trial_courts": ["Superior Courts", "District Courts"],
            "prominent_legal_areas": ["Banking Law", "Technology Law", "Personal Injury", "Employment Law"],
            "notable_legal_institutions": ["North Carolina State Bar", "Duke University School of Law"],
            "bar_admission_authority": "North Carolina State Bar",
            "federal_district": "Eastern, Middle, and Western Districts of North Carolina",
            "federal_circuit": "4th Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Banking center (Charlotte)", "Research Triangle", "Right-to-work state"],
            "law_school_count": 6
        },
        {
            "jurisdiction": USJurisdiction.NORTH_DAKOTA,
            "full_name": "North Dakota",
            "abbreviation": "ND",
            "region": "midwest",
            "capital": "Bismarck",
            "largest_city": "Fargo",
            "population": 779094,
            "court_system_type": "unified",
            "supreme_court_name": "North Dakota Supreme Court",
            "appellate_courts": ["North Dakota Court of Appeals"],
            "trial_courts": ["District Courts"],
            "prominent_legal_areas": ["Oil and Gas Law", "Agriculture Law", "Water Rights", "Employment Law"],
            "notable_legal_institutions": ["State Bar Association of North Dakota", "University of North Dakota School of Law"],
            "bar_admission_authority": "State Bar Association of North Dakota",
            "federal_district": "District of North Dakota",
            "federal_circuit": "8th Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Oil boom region", "Agriculture state", "Sparse population"],
            "law_school_count": 1
        },
        {
            "jurisdiction": USJurisdiction.OHIO,
            "full_name": "Ohio",
            "abbreviation": "OH",
            "region": "midwest",
            "capital": "Columbus",
            "largest_city": "Columbus",
            "population": 11780017,
            "court_system_type": "unified",
            "supreme_court_name": "Ohio Supreme Court",
            "appellate_courts": ["Ohio Courts of Appeals (12 Districts)"],
            "trial_courts": ["Courts of Common Pleas"],
            "prominent_legal_areas": ["Manufacturing Law", "Employment Law", "Personal Injury", "Corporate Law"],
            "notable_legal_institutions": ["Ohio State Bar Association", "Ohio State University Moritz College of Law"],
            "bar_admission_authority": "Ohio State Bar Association",
            "federal_district": "Northern and Southern Districts of Ohio",
            "federal_circuit": "6th Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Manufacturing state", "Diverse economy", "Right-to-work state"],
            "law_school_count": 9
        },
        {
            "jurisdiction": USJurisdiction.OKLAHOMA,
            "full_name": "Oklahoma",
            "abbreviation": "OK",
            "region": "southwest",
            "capital": "Oklahoma City",
            "largest_city": "Oklahoma City",
            "population": 3986639,
            "court_system_type": "dual",
            "supreme_court_name": "Oklahoma Supreme Court",
            "appellate_courts": ["Oklahoma Court of Criminal Appeals", "Oklahoma Court of Civil Appeals"],
            "trial_courts": ["District Courts"],
            "prominent_legal_areas": ["Oil and Gas Law", "Native American Law", "Agriculture Law", "Criminal Law"],
            "notable_legal_institutions": ["Oklahoma Bar Association", "University of Oklahoma College of Law"],
            "bar_admission_authority": "Oklahoma Bar Association",
            "federal_district": "Northern, Eastern, and Western Districts of Oklahoma",
            "federal_circuit": "10th Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Oil and gas industry", "Native American tribes", "Right-to-work state"],
            "law_school_count": 3
        },
        {
            "jurisdiction": USJurisdiction.OREGON,
            "full_name": "Oregon",
            "abbreviation": "OR",
            "region": "west",
            "capital": "Salem",
            "largest_city": "Portland",
            "population": 4237256,
            "court_system_type": "unified",
            "supreme_court_name": "Oregon Supreme Court",
            "appellate_courts": ["Oregon Court of Appeals"],
            "trial_courts": ["Circuit Courts"],
            "prominent_legal_areas": ["Environmental Law", "Technology Law", "Employment Law", "Real Estate"],
            "notable_legal_institutions": ["Oregon State Bar", "University of Oregon School of Law"],
            "bar_admission_authority": "Oregon State Bar",
            "federal_district": "District of Oregon",
            "federal_circuit": "9th Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Environmental focus", "Technology sector", "No sales tax"],
            "law_school_count": 3
        },
        {
            "jurisdiction": USJurisdiction.PENNSYLVANIA,
            "full_name": "Pennsylvania",
            "abbreviation": "PA",
            "region": "northeast",
            "capital": "Harrisburg",
            "largest_city": "Philadelphia",
            "population": 13002700,
            "court_system_type": "unified",
            "supreme_court_name": "Supreme Court of Pennsylvania",
            "appellate_courts": ["Superior Court of Pennsylvania", "Commonwealth Court of Pennsylvania"],
            "trial_courts": ["Courts of Common Pleas"],
            "prominent_legal_areas": ["Personal Injury", "Corporate Law", "Healthcare Law", "Employment Law"],
            "notable_legal_institutions": ["Pennsylvania Bar Association", "University of Pennsylvania Carey Law School"],
            "bar_admission_authority": "Pennsylvania Bar Association",
            "federal_district": "Eastern, Middle, and Western Districts of Pennsylvania",
            "federal_circuit": "3rd Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Manufacturing history", "Healthcare industry", "Complex municipal law"],
            "law_school_count": 10
        },
        {
            "jurisdiction": USJurisdiction.RHODE_ISLAND,
            "full_name": "Rhode Island",
            "abbreviation": "RI",
            "region": "northeast",
            "capital": "Providence",
            "largest_city": "Providence",
            "population": 1097379,
            "court_system_type": "unified",
            "supreme_court_name": "Rhode Island Supreme Court",
            "appellate_courts": ["Rhode Island Supreme Court"],
            "trial_courts": ["Superior Courts", "District Courts"],
            "prominent_legal_areas": ["Maritime Law", "Personal Injury", "Real Estate", "Family Law"],
            "notable_legal_institutions": ["Rhode Island Bar Association", "Roger Williams University School of Law"],
            "bar_admission_authority": "Rhode Island Bar Association",
            "federal_district": "District of Rhode Island",
            "federal_circuit": "1st Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Smallest state", "Maritime jurisdiction", "Dense population"],
            "law_school_count": 1
        },
        {
            "jurisdiction": USJurisdiction.SOUTH_CAROLINA,
            "full_name": "South Carolina",
            "abbreviation": "SC",
            "region": "southeast",
            "capital": "Columbia",
            "largest_city": "Charleston",
            "population": 5190705,
            "court_system_type": "unified",
            "supreme_court_name": "South Carolina Supreme Court",
            "appellate_courts": ["South Carolina Court of Appeals"],
            "trial_courts": ["Circuit Courts", "Family Courts"],
            "prominent_legal_areas": ["Personal Injury", "Real Estate", "Employment Law", "Criminal Law"],
            "notable_legal_institutions": ["South Carolina Bar", "University of South Carolina School of Law"],
            "bar_admission_authority": "South Carolina Bar",
            "federal_district": "District of South Carolina",
            "federal_circuit": "4th Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Tourism industry", "Right-to-work state", "Coastal jurisdiction"],
            "law_school_count": 2
        },
        {
            "jurisdiction": USJurisdiction.SOUTH_DAKOTA,
            "full_name": "South Dakota",
            "abbreviation": "SD",
            "region": "midwest",
            "capital": "Pierre",
            "largest_city": "Sioux Falls",
            "population": 886667,
            "court_system_type": "unified",
            "supreme_court_name": "South Dakota Supreme Court",
            "appellate_courts": ["South Dakota Supreme Court"],
            "trial_courts": ["Circuit Courts", "Magistrate Courts"],
            "prominent_legal_areas": ["Agriculture Law", "Banking Law", "Native American Law", "Criminal Law"],
            "notable_legal_institutions": ["State Bar of South Dakota", "University of South Dakota School of Law"],
            "bar_admission_authority": "State Bar of South Dakota",
            "federal_district": "District of South Dakota",
            "federal_circuit": "8th Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Agriculture state", "Banking industry", "Native American tribes"],
            "law_school_count": 1
        },
        {
            "jurisdiction": USJurisdiction.TENNESSEE,
            "full_name": "Tennessee",
            "abbreviation": "TN",
            "region": "southeast",
            "capital": "Nashville",
            "largest_city": "Memphis",
            "population": 6975218,
            "court_system_type": "unified",
            "supreme_court_name": "Tennessee Supreme Court",
            "appellate_courts": ["Tennessee Court of Appeals", "Tennessee Court of Criminal Appeals"],
            "trial_courts": ["Circuit Courts", "Chancery Courts"],
            "prominent_legal_areas": ["Entertainment Law", "Healthcare Law", "Personal Injury", "Employment Law"],
            "notable_legal_institutions": ["Tennessee Bar Association", "Vanderbilt University Law School"],
            "bar_admission_authority": "Tennessee Bar Association",
            "federal_district": "Eastern, Middle, and Western Districts of Tennessee",
            "federal_circuit": "6th Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Music industry (Nashville)", "Healthcare industry", "Right-to-work state"],
            "law_school_count": 4
        },
        {
            "jurisdiction": USJurisdiction.TEXAS,
            "full_name": "Texas",
            "abbreviation": "TX",
            "region": "southwest",
            "capital": "Austin",
            "largest_city": "Houston",
            "population": 30029572,
            "court_system_type": "dual",
            "supreme_court_name": "Texas Supreme Court",
            "appellate_courts": ["Texas Courts of Appeals (14 Courts)", "Texas Court of Criminal Appeals"],
            "trial_courts": ["District Courts", "County Courts"],
            "prominent_legal_areas": ["Oil and Gas Law", "Corporate Law", "Personal Injury", "Immigration Law"],
            "notable_legal_institutions": ["State Bar of Texas", "University of Texas School of Law"],
            "bar_admission_authority": "State Bar of Texas",
            "federal_district": "Northern, Southern, Eastern, and Western Districts of Texas",
            "federal_circuit": "5th Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Large diverse economy", "Border state", "Oil and gas industry"],
            "law_school_count": 10
        },
        {
            "jurisdiction": USJurisdiction.UTAH,
            "full_name": "Utah",
            "abbreviation": "UT",
            "region": "west",
            "capital": "Salt Lake City",
            "largest_city": "Salt Lake City",
            "population": 3271616,
            "court_system_type": "unified",
            "supreme_court_name": "Utah Supreme Court",
            "appellate_courts": ["Utah Court of Appeals"],
            "trial_courts": ["District Courts", "Justice Courts"],
            "prominent_legal_areas": ["Technology Law", "Real Estate", "Natural Resources", "Employment Law"],
            "notable_legal_institutions": ["Utah State Bar", "University of Utah S.J. Quinney College of Law"],
            "bar_admission_authority": "Utah State Bar",
            "federal_district": "District of Utah",
            "federal_circuit": "10th Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Technology sector", "Natural resources", "Religious considerations"],
            "law_school_count": 1
        },
        {
            "jurisdiction": USJurisdiction.VERMONT,
            "full_name": "Vermont",
            "abbreviation": "VT",
            "region": "northeast",
            "capital": "Montpelier",
            "largest_city": "Burlington",
            "population": 643077,
            "court_system_type": "unified",
            "supreme_court_name": "Vermont Supreme Court",
            "appellate_courts": ["Vermont Supreme Court"],
            "trial_courts": ["Superior Courts", "District Courts"],
            "prominent_legal_areas": ["Environmental Law", "Agriculture Law", "Real Estate", "Family Law"],
            "notable_legal_institutions": ["Vermont Bar Association", "Vermont Law School"],
            "bar_admission_authority": "Vermont Bar Association",
            "federal_district": "District of Vermont",
            "federal_circuit": "2nd Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Environmental focus", "Rural state", "Progressive legal framework"],
            "law_school_count": 1
        },
        {
            "jurisdiction": USJurisdiction.VIRGINIA,
            "full_name": "Virginia",
            "abbreviation": "VA",
            "region": "southeast",
            "capital": "Richmond",
            "largest_city": "Virginia Beach",
            "population": 8631393,
            "court_system_type": "unified",
            "supreme_court_name": "Supreme Court of Virginia",
            "appellate_courts": ["Virginia Court of Appeals"],
            "trial_courts": ["Circuit Courts", "General District Courts"],
            "prominent_legal_areas": ["Government Law", "Technology Law", "Corporate Law", "Healthcare Law"],
            "notable_legal_institutions": ["Virginia State Bar", "University of Virginia School of Law"],
            "bar_admission_authority": "Virginia State Bar",
            "federal_district": "Eastern and Western Districts of Virginia",
            "federal_circuit": "4th Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Federal government proximity", "Technology corridor", "Right-to-work state"],
            "law_school_count": 6
        },
        {
            "jurisdiction": USJurisdiction.WASHINGTON,
            "full_name": "Washington",
            "abbreviation": "WA",
            "region": "west",
            "capital": "Olympia",
            "largest_city": "Seattle",
            "population": 7705281,
            "court_system_type": "unified",
            "supreme_court_name": "Washington State Supreme Court",
            "appellate_courts": ["Washington State Court of Appeals (3 Divisions)"],
            "trial_courts": ["Superior Courts", "District Courts"],
            "prominent_legal_areas": ["Technology Law", "Environmental Law", "International Trade", "Employment Law"],
            "notable_legal_institutions": ["Washington State Bar Association", "University of Washington School of Law"],
            "bar_admission_authority": "Washington State Bar Association",
            "federal_district": "Eastern and Western Districts of Washington",
            "federal_circuit": "9th Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Technology hub", "International trade", "Environmental focus"],
            "law_school_count": 3
        },
        {
            "jurisdiction": USJurisdiction.WEST_VIRGINIA,
            "full_name": "West Virginia",
            "abbreviation": "WV",
            "region": "southeast",
            "capital": "Charleston",
            "largest_city": "Charleston",
            "population": 1793716,
            "court_system_type": "unified",
            "supreme_court_name": "Supreme Court of Appeals of West Virginia",
            "appellate_courts": ["Supreme Court of Appeals of West Virginia"],
            "trial_courts": ["Circuit Courts", "Magistrate Courts"],
            "prominent_legal_areas": ["Coal Law", "Personal Injury", "Workers' Compensation", "Environmental Law"],
            "notable_legal_institutions": ["West Virginia State Bar", "West Virginia University College of Law"],
            "bar_admission_authority": "West Virginia State Bar",
            "federal_district": "Northern and Southern Districts of West Virginia",
            "federal_circuit": "4th Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Coal industry", "Workers' compensation focus", "Environmental concerns"],
            "law_school_count": 1
        },
        {
            "jurisdiction": USJurisdiction.WISCONSIN,
            "full_name": "Wisconsin",
            "abbreviation": "WI",
            "region": "midwest",
            "capital": "Madison",
            "largest_city": "Milwaukee",
            "population": 5893718,
            "court_system_type": "unified",
            "supreme_court_name": "Wisconsin Supreme Court",
            "appellate_courts": ["Wisconsin Court of Appeals (4 Districts)"],
            "trial_courts": ["Circuit Courts"],
            "prominent_legal_areas": ["Manufacturing Law", "Agriculture Law", "Employment Law", "Personal Injury"],
            "notable_legal_institutions": ["State Bar of Wisconsin", "University of Wisconsin Law School"],
            "bar_admission_authority": "State Bar of Wisconsin",
            "federal_district": "Eastern and Western Districts of Wisconsin",
            "federal_circuit": "7th Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Manufacturing state", "Agriculture industry", "Right-to-work state"],
            "law_school_count": 2
        },
        {
            "jurisdiction": USJurisdiction.WYOMING,
            "full_name": "Wyoming",
            "abbreviation": "WY",
            "region": "west",
            "capital": "Cheyenne",
            "largest_city": "Cheyenne",
            "population": 576851,
            "court_system_type": "unified",
            "supreme_court_name": "Wyoming Supreme Court",
            "appellate_courts": ["Wyoming Supreme Court"],
            "trial_courts": ["District Courts", "Circuit Courts"],
            "prominent_legal_areas": ["Natural Resources", "Mining Law", "Oil and Gas Law", "Water Rights"],
            "notable_legal_institutions": ["Wyoming State Bar", "University of Wyoming College of Law"],
            "bar_admission_authority": "Wyoming State Bar",
            "federal_district": "District of Wyoming",
            "federal_circuit": "10th Circuit",
            "legal_traditions": ["common_law"],
            "special_considerations": ["Natural resource extraction", "Smallest population", "No state income tax"],
            "law_school_count": 1
        }
    ]
    
    # Convert state data to metadata objects
    for state_info in state_data:
        jurisdiction = state_info["jurisdiction"]
        metadata[jurisdiction] = JurisdictionMetadata(
            jurisdiction=jurisdiction,
            full_name=state_info["full_name"],
            abbreviation=state_info["abbreviation"],
            type="state",
            region=state_info["region"],
            capital=state_info["capital"],
            largest_city=state_info["largest_city"],
            population=state_info["population"],
            court_system_type=state_info["court_system_type"],
            supreme_court_name=state_info["supreme_court_name"],
            appellate_courts=state_info["appellate_courts"],
            trial_courts=state_info["trial_courts"],
            prominent_legal_areas=state_info["prominent_legal_areas"],
            notable_legal_institutions=state_info["notable_legal_institutions"],
            bar_admission_authority=state_info["bar_admission_authority"],
            federal_district=state_info["federal_district"],
            federal_circuit=state_info["federal_circuit"],
            legal_traditions=state_info["legal_traditions"],
            special_considerations=state_info["special_considerations"],
            law_school_count=state_info["law_school_count"]
        )
    
    return metadata


def get_jurisdiction_by_name(name: str) -> Optional[USJurisdiction]:
    """
    Get jurisdiction by name with flexible matching.
    
    Args:
        name: Jurisdiction name (full name, abbreviation, or variation)
        
    Returns:
        USJurisdiction if found, None otherwise
    """
    validator = JurisdictionValidator()
    return validator.normalize_jurisdiction_string(name)


def get_federal_circuit_states(circuit: str) -> List[USJurisdiction]:
    """
    Get all states in a federal circuit.
    
    Args:
        circuit: Federal circuit (e.g., "1st Circuit", "9th Circuit")
        
    Returns:
        List of USJurisdiction enums for states in the circuit
    """
    circuit_states = []
    all_metadata = get_all_jurisdiction_metadata()
    
    for jurisdiction, metadata in all_metadata.items():
        if metadata.is_state_jurisdiction() and metadata.federal_circuit == circuit:
            circuit_states.append(jurisdiction)
    
    return circuit_states


def get_region_states(region: str) -> List[USJurisdiction]:
    """
    Get all states in a geographic region.
    
    Args:
        region: Geographic region (northeast, southeast, midwest, southwest, west)
        
    Returns:
        List of USJurisdiction enums for states in the region
    """
    region_states = []
    all_metadata = get_all_jurisdiction_metadata()
    
    for jurisdiction, metadata in all_metadata.items():
        if metadata.is_state_jurisdiction() and metadata.region == region.lower():
            region_states.append(jurisdiction)
    
    return region_states


def is_jurisdiction_federal_only(domain: LegalDomain) -> bool:
    """
    Check if a legal domain is exclusively federal jurisdiction.
    
    Args:
        domain: Legal domain to check
        
    Returns:
        True if domain is exclusively federal
    """
    federal_only_domains = {
        LegalDomain.CONSTITUTIONAL,
        LegalDomain.FEDERAL_STATUTORY,
        LegalDomain.INTERSTATE_COMMERCE,
        LegalDomain.IMMIGRATION,
        LegalDomain.INTELLECTUAL_PROPERTY,
        LegalDomain.SECURITIES,
        LegalDomain.ANTITRUST,
        LegalDomain.BANKRUPTCY
    }
    
    return domain in federal_only_domains


def get_jurisdiction_summary() -> Dict[str, Any]:
    """
    Get summary statistics about the US jurisdiction system.
    
    Returns:
        Dictionary with jurisdiction system statistics
    """
    all_metadata = get_all_jurisdiction_metadata()
    
    state_count = sum(1 for metadata in all_metadata.values() if metadata.is_state_jurisdiction())
    territory_count = sum(1 for metadata in all_metadata.values() if metadata.is_territory())
    
    # Count by region
    region_counts = {}
    for metadata in all_metadata.values():
        if metadata.is_state_jurisdiction():
            region = metadata.region
            region_counts[region] = region_counts.get(region, 0) + 1
    
    # Count by federal circuit
    circuit_counts = {}
    for metadata in all_metadata.values():
        if metadata.is_state_jurisdiction():
            circuit = metadata.federal_circuit
            circuit_counts[circuit] = circuit_counts.get(circuit, 0) + 1
    
    # Count law schools
    total_law_schools = sum(metadata.law_school_count for metadata in all_metadata.values())
    
    return {
        "total_jurisdictions": len(all_metadata),
        "states": state_count,
        "territories": territory_count,
        "federal": 1,
        "general": 1,
        "regions": region_counts,
        "federal_circuits": circuit_counts,
        "total_law_schools": total_law_schools,
        "court_system_types": {
            "unified": sum(1 for m in all_metadata.values() if m.court_system_type == "unified"),
            "dual": sum(1 for m in all_metadata.values() if m.court_system_type == "dual"),
            "specialized": sum(1 for m in all_metadata.values() if m.court_system_type == "specialized")
        }
    }


# Create global instances for common use
_jurisdiction_validator = JurisdictionValidator()
_jurisdiction_context_provider = JurisdictionContextProvider()

# Convenience functions using global instances
def validate_jurisdiction(jurisdiction: Union[str, USJurisdiction]) -> USJurisdiction:
    """Convenience function for jurisdiction validation"""
    return _jurisdiction_validator.validate_jurisdiction(jurisdiction)

def get_jurisdiction_context(jurisdiction: Union[str, USJurisdiction]) -> Dict[str, Any]:
    """Convenience function for getting jurisdiction context"""
    return _jurisdiction_context_provider.get_jurisdiction_context(jurisdiction)

def suggest_jurisdictions(partial: str, limit: int = 5) -> List[Tuple[USJurisdiction, str, float]]:
    """Convenience function for jurisdiction suggestions"""
    return _jurisdiction_validator.get_suggested_jurisdictions(partial, limit)
