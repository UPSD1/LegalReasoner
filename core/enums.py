"""
Enums and Type Definitions for Multi-Task Legal Reward System

This module defines all enumeration types used throughout the legal reward system
for type safety, consistency, and clear system boundaries. Includes comprehensive
US jurisdiction support with all 50 states, DC, and federal system coverage.

Key Enumerations:
- LegalTaskType: Four main legal task types for specialized evaluation
- LegalDomain: Comprehensive legal practice areas  
- USJurisdiction: Complete US jurisdiction support (50 states + DC + federal)
- USJurisdictionLevel: Federal/state/local jurisdiction levels
- APIProvider: Supported LLM API providers for cost optimization
- EvaluationMethod: Hybrid evaluation strategies
"""

from enum import Enum, auto
from typing import Dict, List, Set, Optional, Tuple


class LegalTaskType(Enum):
    """
    Legal task types for specialized evaluation routing.
    
    Each task type routes to different judge ensembles with specific
    difficulty weights and evaluation criteria optimized for that
    type of legal work.
    """
    JUDICIAL_REASONING = "judicial_reasoning"    # Formal judicial analysis, FIRAC structure
    PRECEDENT_ANALYSIS = "precedent_analysis"    # Deep case law analysis, analogical reasoning  
    OPINION_GENERATION = "opinion_generation"    # Lawyer advocacy, persuasive legal writing
    GENERAL_CHAT = "general_chat"               # Conversational legal assistance
    
    @classmethod
    def get_specialized_tasks(cls) -> List['LegalTaskType']:
        """Get tasks that use specialized + general chat hybrid evaluation"""
        return [cls.JUDICIAL_REASONING, cls.PRECEDENT_ANALYSIS, cls.OPINION_GENERATION]
    
    @classmethod
    def get_chat_only_tasks(cls) -> List['LegalTaskType']:
        """Get tasks that use general chat evaluation only"""
        return [cls.GENERAL_CHAT]
    
    def requires_specialized_evaluation(self) -> bool:
        """Check if this task type requires specialized legal evaluation"""
        return self in self.get_specialized_tasks()
    
    def get_default_difficulty_weight(self) -> float:
        """Get default difficulty weight for this task type"""
        weights = {
            self.JUDICIAL_REASONING: 1.5,    # Hardest - formal judicial analysis
            self.PRECEDENT_ANALYSIS: 1.3,    # Hard - deep case law knowledge
            self.OPINION_GENERATION: 1.1,    # Medium-hard - creative legal writing  
            self.GENERAL_CHAT: 1.0          # Baseline - conversational
        }
        return weights.get(self, 1.0)


class LegalDomain(Enum):
    """
    Legal domains/practice areas for contextual evaluation.
    
    Comprehensive coverage of US legal practice areas to provide
    proper context for jurisdiction inference and specialized evaluation.
    """
    # Core legal domains
    CONSTITUTIONAL = "constitutional"           # Constitutional law, civil rights
    CONTRACT = "contract"                      # Contract law, agreements, commercial law
    TORT = "tort"                             # Personal injury, negligence, damages
    CRIMINAL = "criminal"                     # Criminal law, procedure, defense
    CIVIL_PROCEDURE = "civil_procedure"       # Court procedures, litigation process
    EVIDENCE = "evidence"                     # Rules of evidence, admissibility
    
    # Specialized practice areas  
    CORPORATE = "corporate"                   # Business law, corporate governance
    INTELLECTUAL_PROPERTY = "intellectual_property"  # Patents, trademarks, copyright
    FAMILY = "family"                        # Divorce, custody, domestic relations  
    EMPLOYMENT = "employment"                # Labor law, workplace rights, discrimination
    REAL_ESTATE = "real_estate"             # Property law, transactions, zoning
    TAX = "tax"                             # Tax law, IRS, tax planning
    IMMIGRATION = "immigration"              # Immigration law, visas, citizenship
    BANKRUPTCY = "bankruptcy"               # Debt relief, reorganization, liquidation
    
    # Additional specialized areas
    ENVIRONMENTAL = "environmental"          # Environmental regulations, compliance
    HEALTHCARE = "healthcare"               # Medical law, HIPAA, healthcare regulations
    SECURITIES = "securities"               # Securities law, SEC, investment regulations
    ANTITRUST = "antitrust"                # Competition law, monopolies, trade practices
    ADMINISTRATIVE = "administrative"       # Government agencies, regulatory law
    INTERNATIONAL = "international"         # International law, treaties, trade

    FEDERAL_STATUTORY = "federal_statutory"   # Federal statutes, USC, federal regulations
    INTERSTATE_COMMERCE = "interstate_commerce"  # Commerce clause, interstate business
    PROPERTY = "property"                    # Alias for real_estate (for compatibility)
    ETHICS = "ethics"                       # Legal ethics, professional responsibility
    
    # Meta categories
    GENERAL = "general"                     # General legal questions, multiple domains
    OTHER = "other"                        # Specialized or uncommon legal areas
    
    @classmethod
    def get_jurisdiction_critical_domains(cls) -> Set['LegalDomain']:
        """Get domains where jurisdiction specificity is critical for accuracy"""
        return {
            cls.CIVIL_PROCEDURE, cls.CRIMINAL, cls.FAMILY, cls.REAL_ESTATE,
            cls.EMPLOYMENT, cls.TAX, cls.CORPORATE, cls.BANKRUPTCY,
            cls.HEALTHCARE, cls.SECURITIES, cls.ADMINISTRATIVE,
            cls.FEDERAL_STATUTORY, cls.INTERSTATE_COMMERCE, cls.PROPERTY,
            cls.ETHICS
        }
    
    @classmethod
    def get_federal_primary_domains(cls) -> Set['LegalDomain']:
        """Get domains primarily governed by federal law"""
        return {
            cls.CONSTITUTIONAL, cls.INTELLECTUAL_PROPERTY, cls.IMMIGRATION,
            cls.BANKRUPTCY, cls.SECURITIES, cls.ANTITRUST, cls.INTERNATIONAL,
            cls.FEDERAL_STATUTORY, cls.INTERSTATE_COMMERCE
        }
    
    @classmethod
    def get_state_primary_domains(cls) -> Set['LegalDomain']:
        """Get domains primarily governed by state law"""
        return {
            cls.CONTRACT, cls.TORT, cls.CIVIL_PROCEDURE, cls.FAMILY,
            cls.REAL_ESTATE, cls.CORPORATE, cls.HEALTHCARE,
            cls.PROPERTY, cls.ETHICS
        }
    
    def is_jurisdiction_critical(self) -> bool:
        """Check if this domain requires specific jurisdiction handling"""
        return self in self.get_jurisdiction_critical_domains()
    
    def get_primary_jurisdiction_level(self) -> 'USJurisdictionLevel':
        """Get the primary jurisdiction level for this domain"""
        if self in self.get_federal_primary_domains():
            return USJurisdictionLevel.FEDERAL
        elif self in self.get_state_primary_domains():
            return USJurisdictionLevel.STATE
        else:
            return USJurisdictionLevel.GENERAL


class USJurisdiction(Enum):
    """
    Complete US jurisdiction support for all 50 states, DC, and federal system.
    
    Comprehensive enumeration enabling proper jurisdiction inference,
    compliance checking, and state-specific legal evaluation across
    the entire United States legal system.
    """
    # Federal jurisdiction
    FEDERAL = "federal"
    
    # All 50 US states in alphabetical order
    ALABAMA = "alabama"
    ALASKA = "alaska"
    ARIZONA = "arizona"
    ARKANSAS = "arkansas"
    CALIFORNIA = "california"
    COLORADO = "colorado"
    CONNECTICUT = "connecticut"
    DELAWARE = "delaware"
    FLORIDA = "florida"
    GEORGIA = "georgia"
    HAWAII = "hawaii"
    IDAHO = "idaho"
    ILLINOIS = "illinois"
    INDIANA = "indiana"
    IOWA = "iowa"
    KANSAS = "kansas"
    KENTUCKY = "kentucky"
    LOUISIANA = "louisiana"
    MAINE = "maine"
    MARYLAND = "maryland"
    MASSACHUSETTS = "massachusetts"
    MICHIGAN = "michigan"
    MINNESOTA = "minnesota"
    MISSISSIPPI = "mississippi"
    MISSOURI = "missouri"
    MONTANA = "montana"
    NEBRASKA = "nebraska"
    NEVADA = "nevada"
    NEW_HAMPSHIRE = "new_hampshire"
    NEW_JERSEY = "new_jersey"
    NEW_MEXICO = "new_mexico"
    NEW_YORK = "new_york"
    NORTH_CAROLINA = "north_carolina"
    NORTH_DAKOTA = "north_dakota"
    OHIO = "ohio"
    OKLAHOMA = "oklahoma"
    OREGON = "oregon"
    PENNSYLVANIA = "pennsylvania"
    RHODE_ISLAND = "rhode_island"
    SOUTH_CAROLINA = "south_carolina"
    SOUTH_DAKOTA = "south_dakota"
    TENNESSEE = "tennessee"
    TEXAS = "texas"
    UTAH = "utah"
    VERMONT = "vermont"
    VIRGINIA = "virginia"
    WASHINGTON = "washington"
    WEST_VIRGINIA = "west_virginia"
    WISCONSIN = "wisconsin"
    WYOMING = "wyoming"
    
    # District of Columbia
    DISTRICT_OF_COLUMBIA = "district_of_columbia"
    
    # General/unknown jurisdiction
    GENERAL = "general"
    
    @classmethod
    def get_all_states(cls) -> List['USJurisdiction']:
        """Get all US states (excluding federal, DC, and general)"""
        return [jurisdiction for jurisdiction in cls 
                if jurisdiction not in {cls.FEDERAL, cls.DISTRICT_OF_COLUMBIA, cls.GENERAL}]
    
    @classmethod
    def get_state_abbreviations(cls) -> Dict['USJurisdiction', str]:
        """Get standard postal abbreviations for all states"""
        return {
            cls.ALABAMA: "AL", cls.ALASKA: "AK", cls.ARIZONA: "AZ", cls.ARKANSAS: "AR",
            cls.CALIFORNIA: "CA", cls.COLORADO: "CO", cls.CONNECTICUT: "CT", cls.DELAWARE: "DE",
            cls.FLORIDA: "FL", cls.GEORGIA: "GA", cls.HAWAII: "HI", cls.IDAHO: "ID",
            cls.ILLINOIS: "IL", cls.INDIANA: "IN", cls.IOWA: "IA", cls.KANSAS: "KS",
            cls.KENTUCKY: "KY", cls.LOUISIANA: "LA", cls.MAINE: "ME", cls.MARYLAND: "MD",
            cls.MASSACHUSETTS: "MA", cls.MICHIGAN: "MI", cls.MINNESOTA: "MN", cls.MISSISSIPPI: "MS",
            cls.MISSOURI: "MO", cls.MONTANA: "MT", cls.NEBRASKA: "NE", cls.NEVADA: "NV",
            cls.NEW_HAMPSHIRE: "NH", cls.NEW_JERSEY: "NJ", cls.NEW_MEXICO: "NM", cls.NEW_YORK: "NY",
            cls.NORTH_CAROLINA: "NC", cls.NORTH_DAKOTA: "ND", cls.OHIO: "OH", cls.OKLAHOMA: "OK",
            cls.OREGON: "OR", cls.PENNSYLVANIA: "PA", cls.RHODE_ISLAND: "RI", cls.SOUTH_CAROLINA: "SC",
            cls.SOUTH_DAKOTA: "SD", cls.TENNESSEE: "TN", cls.TEXAS: "TX", cls.UTAH: "UT",
            cls.VERMONT: "VT", cls.VIRGINIA: "VA", cls.WASHINGTON: "WA", cls.WEST_VIRGINIA: "WV",
            cls.WISCONSIN: "WI", cls.WYOMING: "WY", cls.DISTRICT_OF_COLUMBIA: "DC"
        }
    
    @classmethod
    def get_federal_circuits(cls) -> Dict['USJurisdiction', int]:
        """Get federal circuit court numbers for each jurisdiction"""
        return {
            # 1st Circuit
            cls.MAINE: 1, cls.MASSACHUSETTS: 1, cls.NEW_HAMPSHIRE: 1, cls.RHODE_ISLAND: 1,
            # 2nd Circuit  
            cls.CONNECTICUT: 2, cls.NEW_YORK: 2, cls.VERMONT: 2,
            # 3rd Circuit
            cls.DELAWARE: 3, cls.NEW_JERSEY: 3, cls.PENNSYLVANIA: 3,
            # 4th Circuit
            cls.MARYLAND: 4, cls.NORTH_CAROLINA: 4, cls.SOUTH_CAROLINA: 4, cls.VIRGINIA: 4, cls.WEST_VIRGINIA: 4,
            # 5th Circuit
            cls.LOUISIANA: 5, cls.MISSISSIPPI: 5, cls.TEXAS: 5,
            # 6th Circuit
            cls.KENTUCKY: 6, cls.MICHIGAN: 6, cls.OHIO: 6, cls.TENNESSEE: 6,
            # 7th Circuit
            cls.ILLINOIS: 7, cls.INDIANA: 7, cls.WISCONSIN: 7,
            # 8th Circuit
            cls.ARKANSAS: 8, cls.IOWA: 8, cls.MINNESOTA: 8, cls.MISSOURI: 8, cls.NEBRASKA: 8, cls.NORTH_DAKOTA: 8, cls.SOUTH_DAKOTA: 8,
            # 9th Circuit
            cls.ALASKA: 9, cls.ARIZONA: 9, cls.CALIFORNIA: 9, cls.HAWAII: 9, cls.IDAHO: 9, cls.MONTANA: 9, cls.NEVADA: 9, cls.OREGON: 9, cls.WASHINGTON: 9,
            # 10th Circuit
            cls.COLORADO: 10, cls.KANSAS: 10, cls.NEW_MEXICO: 10, cls.OKLAHOMA: 10, cls.UTAH: 10, cls.WYOMING: 10,
            # 11th Circuit
            cls.ALABAMA: 11, cls.FLORIDA: 11, cls.GEORGIA: 11,
            # DC Circuit
            cls.DISTRICT_OF_COLUMBIA: 0  # Special DC Circuit
        }
    
    @classmethod
    def from_string(cls, jurisdiction_str: str) -> 'USJurisdiction':
        """
        Convert jurisdiction string to USJurisdiction enum with flexible matching.
        
        Args:
            jurisdiction_str: String representation of jurisdiction
            
        Returns:
            USJurisdiction enum value
            
        Raises:
            ValueError: If jurisdiction string cannot be mapped
        """
        if not jurisdiction_str:
            return cls.GENERAL
        
        jurisdiction_str = jurisdiction_str.lower().strip()
        
        # Direct matches
        for jurisdiction in cls:
            if jurisdiction.value == jurisdiction_str:
                return jurisdiction
        
        # Abbreviation matches
        abbreviations = cls.get_state_abbreviations()
        for jurisdiction, abbrev in abbreviations.items():
            if abbrev.lower() == jurisdiction_str:
                return jurisdiction
        
        # Special cases and common variations
        special_mappings = {
            "dc": cls.DISTRICT_OF_COLUMBIA,
            "washington dc": cls.DISTRICT_OF_COLUMBIA,
            "d.c.": cls.DISTRICT_OF_COLUMBIA,
            "fed": cls.FEDERAL,
            "federal": cls.FEDERAL,
            "usa": cls.FEDERAL,
            "us": cls.FEDERAL,
            "calif": cls.CALIFORNIA,
            "cali": cls.CALIFORNIA,
            "ny": cls.NEW_YORK,
            "nyc": cls.NEW_YORK,
            "tex": cls.TEXAS,
            "fla": cls.FLORIDA,
            "mass": cls.MASSACHUSETTS,
            "penn": cls.PENNSYLVANIA,
            "pa": cls.PENNSYLVANIA,
            "va": cls.VIRGINIA,
            "nc": cls.NORTH_CAROLINA,
            "sc": cls.SOUTH_CAROLINA
        }
        
        if jurisdiction_str in special_mappings:
            return special_mappings[jurisdiction_str]
        
        # If no match found, return general
        return cls.GENERAL
    
    def is_state(self) -> bool:
        """Check if this is a US state jurisdiction"""
        return self in self.get_all_states()
    
    def is_federal(self) -> bool:
        """Check if this is federal jurisdiction"""
        return self == self.FEDERAL
    
    def get_abbreviation(self) -> Optional[str]:
        """Get postal abbreviation for this jurisdiction"""
        return self.get_state_abbreviations().get(self)
    
    def get_federal_circuit(self) -> Optional[int]:
        """Get federal circuit court number for this jurisdiction"""
        return self.get_federal_circuits().get(self)
    
    def get_display_name(self) -> str:
        """Get human-readable display name"""
        if self == self.DISTRICT_OF_COLUMBIA:
            return "District of Columbia"
        elif self == self.FEDERAL:
            return "Federal"
        elif self == self.GENERAL:
            return "General"
        else:
            # Convert snake_case to Title Case
            return self.value.replace("_", " ").title()


class USJurisdictionLevel(Enum):
    """
    US jurisdiction levels for legal context and routing.
    
    Hierarchical classification of US legal jurisdictions from
    federal supremacy down to local ordinances.
    """
    FEDERAL = "federal"      # Federal courts, constitutional law, federal statutes
    STATE = "state"         # State courts, state law, state constitution  
    LOCAL = "local"         # County, city, municipal courts and ordinances
    GENERAL = "general"     # When jurisdiction level is unclear or mixed
    
    def get_precedence_order(self) -> int:
        """Get precedence order (lower number = higher precedence)"""
        order = {
            self.FEDERAL: 1,    # Highest precedence (Supremacy Clause)
            self.STATE: 2,      # State law when federal doesn't apply
            self.LOCAL: 3,      # Local ordinances, lowest precedence
            self.GENERAL: 4     # No specific precedence
        }
        return order.get(self, 999)
    
    def can_override(self, other: 'USJurisdictionLevel') -> bool:
        """Check if this jurisdiction level can override another"""
        return self.get_precedence_order() < other.get_precedence_order()
    
    @classmethod
    def from_jurisdiction(cls, jurisdiction: USJurisdiction) -> 'USJurisdictionLevel':
        """Determine jurisdiction level from USJurisdiction"""
        if jurisdiction == USJurisdiction.FEDERAL:
            return cls.FEDERAL
        elif jurisdiction in USJurisdiction.get_all_states() or jurisdiction == USJurisdiction.DISTRICT_OF_COLUMBIA:
            return cls.STATE
        else:
            return cls.GENERAL


class APIProvider(Enum):
    """
    Supported API providers for cost-optimized judge ensembles.
    
    Multiple provider support enables cost optimization through
    provider fallback chains and task-appropriate model selection.
    """
    OPENAI = "openai"           # GPT-4 Turbo - Best for complex legal reasoning
    ANTHROPIC = "anthropic"     # Claude-3.5-Sonnet - Excellent reasoning, middle cost
    GOOGLE = "google"           # Gemini-1.5-Pro - Cost-effective, good for simple tasks
    
    def get_default_model(self) -> str:
        """Get default model for this provider"""
        models = {
            self.OPENAI: "gpt-4-turbo",
            self.ANTHROPIC: "claude-3-5-sonnet-20241022", 
            self.GOOGLE: "gemini-1.5-pro"
        }
        return models[self]
    
    def get_cost_tier(self) -> int:
        """Get cost tier (1=most expensive, 3=least expensive)"""
        tiers = {
            self.OPENAI: 1,      # Most expensive, highest quality
            self.ANTHROPIC: 2,   # Middle cost, high quality
            self.GOOGLE: 3       # Least expensive, good quality
        }
        return tiers[self]
    
    def is_suitable_for_complex_tasks(self) -> bool:
        """Check if provider is suitable for complex legal reasoning"""
        return self in {self.OPENAI, self.ANTHROPIC, self.GOOGLE}
    
    @classmethod
    def get_fallback_chain(cls, primary: 'APIProvider') -> List['APIProvider']:
        """Get fallback chain starting with primary provider"""
        if primary == cls.OPENAI:
            return [cls.OPENAI, cls.ANTHROPIC, cls.GOOGLE]
        elif primary == cls.ANTHROPIC:
            return [cls.ANTHROPIC, cls.OPENAI, cls.GOOGLE] 
        else:  # GOOGLE
            return [cls.GOOGLE, cls.ANTHROPIC, cls.OPENAI]


class EvaluationMethod(Enum):
    """
    Hybrid evaluation methods used by the system.
    
    Different evaluation strategies based on task type and 
    jurisdiction compliance status.
    """
    SPECIALIZED_HYBRID = "specialized_hybrid"                    # 70% specialized + 30% chat (normal case)
    GENERAL_CHAT_ONLY = "general_chat_only"                     # 100% general chat (for general_chat tasks)
    JURISDICTION_FAILURE = "jurisdiction_failure"               # Hybrid with jurisdiction penalty applied
    GENERAL_CHAT_JURISDICTION_FAILURE = "general_chat_jurisdiction_failure"  # Chat-only with jurisdiction penalty
    
    def uses_hybrid_evaluation(self) -> bool:
        """Check if this method uses hybrid evaluation (specialized + chat)"""
        return self in {self.SPECIALIZED_HYBRID, self.JURISDICTION_FAILURE}
    
    def applies_jurisdiction_penalty(self) -> bool:
        """Check if this method applies jurisdiction compliance penalty"""
        return self in {self.JURISDICTION_FAILURE, self.GENERAL_CHAT_JURISDICTION_FAILURE}
    
    def get_description(self) -> str:
        """Get human-readable description of evaluation method"""
        descriptions = {
            self.SPECIALIZED_HYBRID: "Hybrid evaluation with specialized legal expertise (70%) and chat quality (30%)",
            self.GENERAL_CHAT_ONLY: "General chat evaluation focusing on helpfulness, ethics, clarity, and jurisdiction compliance",
            self.JURISDICTION_FAILURE: "Hybrid evaluation with penalty for jurisdiction compliance failure",
            self.GENERAL_CHAT_JURISDICTION_FAILURE: "General chat evaluation with penalty for jurisdiction compliance failure"
        }
        return descriptions[self]


class CacheStrategy(Enum):
    """Cache strategies for cost optimization"""
    AGGRESSIVE = "aggressive"       # Cache everything, long TTL
    BALANCED = "balanced"          # Cache selectively, medium TTL  
    CONSERVATIVE = "conservative"   # Cache minimally, short TTL
    DISABLED = "disabled"          # No caching
    
    def get_ttl_hours(self) -> int:
        """Get cache TTL in hours for this strategy"""
        ttls = {
            self.AGGRESSIVE: 168,    # 1 week
            self.BALANCED: 72,       # 3 days
            self.CONSERVATIVE: 24,   # 1 day
            self.DISABLED: 0         # No caching
        }
        return ttls[self]


class RateLimitStrategy(Enum):
    """Rate limiting strategies"""
    TOKEN_BUCKET = "token_bucket"        # Token bucket algorithm
    SLIDING_WINDOW = "sliding_window"    # Sliding window counter
    FIXED_WINDOW = "fixed_window"        # Fixed window counter
    ADAPTIVE = "adaptive"                # Adaptive based on provider response


class LogLevel(Enum):
    """Logging levels with legal system specific contexts"""
    DEBUG = "DEBUG"           # Detailed debugging information
    INFO = "INFO"            # General information, system status  
    WARNING = "WARNING"      # Warning conditions, non-critical issues
    ERROR = "ERROR"          # Error conditions, failures
    CRITICAL = "CRITICAL"    # Critical conditions, system failures
    
    def should_log_api_costs(self) -> bool:
        """Check if this level should log API cost information"""
        return self in {self.DEBUG, self.INFO}
    
    def should_log_performance(self) -> bool:
        """Check if this level should log performance metrics"""
        return self in {self.DEBUG, self.INFO}


# Utility functions for enum operations

def get_all_enum_values(enum_class) -> List[str]:
    """Get all string values from an enum class"""
    return [item.value for item in enum_class]


def validate_enum_value(value: str, enum_class, default=None):
    """
    Validate that a string value exists in an enum class.
    
    Args:
        value: String value to validate
        enum_class: Enum class to validate against
        default: Default value if validation fails
        
    Returns:
        Enum instance or default value
    """
    try:
        return enum_class(value)
    except ValueError:
        if default is not None:
            return default
        raise ValueError(f"Invalid {enum_class.__name__} value: {value}")


def get_enum_mapping(enum_class) -> Dict[str, str]:
    """Get mapping of enum values to display names"""
    mapping = {}
    for item in enum_class:
        if hasattr(item, 'get_display_name'):
            mapping[item.value] = item.get_display_name()
        else:
            mapping[item.value] = item.value.replace('_', ' ').title()
    return mapping


# Type aliases for better documentation
TaskTypeStr = str  # String representation of LegalTaskType
JurisdictionStr = str  # String representation of USJurisdiction 
DomainStr = str  # String representation of LegalDomain
ProviderStr = str  # String representation of APIProvider
MethodStr = str  # String representation of EvaluationMethod

# Constants for system defaults
DEFAULT_LEGAL_TASK_TYPE = LegalTaskType.GENERAL_CHAT
DEFAULT_LEGAL_DOMAIN = LegalDomain.GENERAL
DEFAULT_US_JURISDICTION = USJurisdiction.GENERAL
DEFAULT_JURISDICTION_LEVEL = USJurisdictionLevel.GENERAL
DEFAULT_API_PROVIDER = APIProvider.OPENAI
DEFAULT_EVALUATION_METHOD = EvaluationMethod.GENERAL_CHAT_ONLY
DEFAULT_CACHE_STRATEGY = CacheStrategy.AGGRESSIVE
DEFAULT_RATE_LIMIT_STRATEGY = RateLimitStrategy.TOKEN_BUCKET
DEFAULT_LOG_LEVEL = LogLevel.INFO

# Validation sets for quick membership testing
VALID_TASK_TYPES = {item.value for item in LegalTaskType}
VALID_LEGAL_DOMAINS = {item.value for item in LegalDomain} 
VALID_US_JURISDICTIONS = {item.value for item in USJurisdiction}
VALID_JURISDICTION_LEVELS = {item.value for item in USJurisdictionLevel}
VALID_API_PROVIDERS = {item.value for item in APIProvider}
VALID_EVALUATION_METHODS = {item.value for item in EvaluationMethod}