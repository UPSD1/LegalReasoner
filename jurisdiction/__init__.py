"""
US Jurisdiction System for Multi-Task Legal Reward System

This package provides comprehensive support for the US legal system with detailed
jurisdiction information, validation, and contextual data for all 50 states,
District of Columbia, and federal jurisdiction.

Key Components:
- Complete US jurisdiction mapping and metadata
- Jurisdiction validation and normalization
- State-specific legal context information
- Federal vs state jurisdiction determination
- Legal domain jurisdiction requirements
- Jurisdiction inference support utilities

The system ensures that legal evaluations are appropriately contextualized
within the correct US jurisdiction for accurate legal analysis.
"""

# Core US jurisdiction system components
from jurisdiction.us_system import (
    # Data structures
    JurisdictionMetadata,
    
    # Main system classes
    JurisdictionValidator,
    JurisdictionContextProvider,
    
    # Exceptions
    USJurisdictionError,
    
    # Core functionality
    get_all_jurisdiction_metadata,
    get_jurisdiction_by_name,
    get_federal_circuit_states,
    get_region_states,
    is_jurisdiction_federal_only,
    get_jurisdiction_summary,
    
    # Convenience functions
    validate_jurisdiction,
    get_jurisdiction_context,
    suggest_jurisdictions,

)

# Jurisdiction Inference Engine
from jurisdiction.inference_engine import (
    USJurisdictionInferenceEngine,
    JurisdictionInferenceResult,
    InferenceConfidence,
    create_production_inference_engine
)

# Jurisdiction Compliance Judge
from jurisdiction.compliance_judge import (
    JurisdictionComplianceJudge,
    ComplianceViolationType,
    ComplianceViolation,
    JurisdictionComplianceResult,
    create_production_compliance_judge
)

# Package metadata
__version__ = "1.0.0"
__author__ = "Legal AI Development Team"
__description__ = "US Jurisdiction Support for Multi-Task Legal Reward System"


# Main exports for easy importing
__all__ = [
    # Data structures
    "JurisdictionMetadata",
    
    # System classes
    "JurisdictionValidator",
    "JurisdictionContextProvider",
    
    # Exceptions
    "USJurisdictionError",
    
    # Core functionality
    "get_all_jurisdiction_metadata",
    "get_jurisdiction_by_name",
    "get_federal_circuit_states", 
    "get_region_states",
    "is_jurisdiction_federal_only",
    "get_jurisdiction_summary",
    
    # Convenience functions
    "validate_jurisdiction",
    "get_jurisdiction_context",
    "suggest_jurisdictions",

    # Jurisdiction Inference Engine
    "USJurisdictionInferenceEngine",
    "JurisdictionInferenceResult",
    "InferenceConfidence",
    "create_production_inference_engine",
    
    # Jurisdiction Compliance Judge
    "JurisdictionComplianceJudge",
    "ComplianceViolationType",
    "ComplianceViolation", 
    "JurisdictionComplianceResult",
    "create_production_compliance_judge"
]

JurisdictionInferenceEngine = USJurisdictionInferenceEngine

def get_jurisdiction_package_info() -> dict:
    """
    Get information about the jurisdiction package.
    
    Returns:
        Dictionary with package information and capabilities
    """
    summary = get_jurisdiction_summary()
    
    return {
        "version": __version__,
        "description": __description__,
        "capabilities": [
            "Complete US jurisdiction support (50 states + DC + federal)",
            "Jurisdiction validation and normalization", 
            "State-specific legal context information",
            "Federal circuit and regional groupings",
            "Legal domain jurisdiction requirements",
            "Fuzzy matching for jurisdiction strings",
            "Jurisdiction inference engine",
            "Jurisdiction compliance validation"
        ],
        "jurisdiction_statistics": summary,
        "supported_features": {
            "jurisdiction_validation": True,
            "context_provision": True,
            "fuzzy_matching": True,
            "federal_circuit_mapping": True,
            "regional_groupings": True,
            "legal_domain_mapping": True,
            "court_system_information": True,
            "bar_admission_data": True,
            "jurisdiction_inference": True,
            "compliance_checking": True
        },
        "data_completeness": {
            "all_50_states": True,
            "district_of_columbia": True,
            "federal_jurisdiction": True,
            "court_hierarchies": True,
            "federal_circuits": True,
            "geographic_regions": True,
            "legal_specializations": True,
            "population_data": True,
            "law_school_counts": True
        }
    }

def validate_jurisdiction_system() -> dict:
    """
    Validate that the jurisdiction system is working correctly.
    
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        "system_valid": True,
        "errors": [],
        "warnings": [],
        "test_results": {}
    }
    
    try:
        # Test basic validation
        federal_jurisdiction = validate_jurisdiction("federal")
        validation_results["test_results"]["federal_validation"] = "pass"
        
        # Test state validation
        california = validate_jurisdiction("California")
        validation_results["test_results"]["state_validation"] = "pass"
        
        # Test abbreviation validation
        texas = validate_jurisdiction("TX")
        validation_results["test_results"]["abbreviation_validation"] = "pass"
        
        # Test context retrieval
        context = get_jurisdiction_context("NY")
        validation_results["test_results"]["context_retrieval"] = "pass"
        
        # Test metadata retrieval
        all_metadata = get_all_jurisdiction_metadata()
        if len(all_metadata) >= 52:  # 50 states + DC + federal + general
            validation_results["test_results"]["metadata_completeness"] = "pass"
        else:
            validation_results["warnings"].append("Incomplete jurisdiction metadata")
            validation_results["test_results"]["metadata_completeness"] = "warning"
        
        # Test federal circuit grouping
        ninth_circuit = get_federal_circuit_states("9th Circuit")
        if len(ninth_circuit) > 0:
            validation_results["test_results"]["circuit_grouping"] = "pass"
        else:
            validation_results["warnings"].append("No states found for 9th Circuit")
            validation_results["test_results"]["circuit_grouping"] = "warning"
        
        # Test region grouping
        west_states = get_region_states("west")
        if len(west_states) > 0:
            validation_results["test_results"]["region_grouping"] = "pass"
        else:
            validation_results["warnings"].append("No states found for west region")
            validation_results["test_results"]["region_grouping"] = "warning"
        
    except Exception as e:
        validation_results["system_valid"] = False
        validation_results["errors"].append(f"Jurisdiction system validation failed: {e}")
        validation_results["test_results"]["overall"] = "fail"
    
    return validation_results

def create_jurisdiction_validator() -> JurisdictionValidator:
    """
    Create a new jurisdiction validator instance.
    
    Returns:
        Configured JurisdictionValidator
    """
    return JurisdictionValidator()

def create_jurisdiction_context_provider() -> JurisdictionContextProvider:
    """
    Create a new jurisdiction context provider instance.
    
    Returns:
        Configured JurisdictionContextProvider
    """
    return JurisdictionContextProvider()

def get_jurisdiction_quick_reference() -> dict:
    """
    Get quick reference information for common jurisdictions.
    
    Returns:
        Dictionary with quick reference data
    """
    common_jurisdictions = [
        "federal", "california", "new_york", "texas", "florida", 
        "illinois", "pennsylvania", "ohio", "georgia", "north_carolina"
    ]
    
    quick_ref = {}
    
    for jurisdiction_name in common_jurisdictions:
        try:
            jurisdiction = validate_jurisdiction(jurisdiction_name)
            context = get_jurisdiction_context(jurisdiction)
            
            quick_ref[jurisdiction_name] = {
                "full_name": context["metadata"].full_name,
                "abbreviation": context["metadata"].abbreviation,
                "type": context["metadata"].type,
                "region": context["metadata"].region,
                "federal_circuit": context["metadata"].federal_circuit,
                "supreme_court": context["metadata"].supreme_court_name,
                "prominent_areas": context["metadata"].prominent_legal_areas[:3]  # Top 3
            }
        except Exception as e:
            quick_ref[jurisdiction_name] = {"error": str(e)}
    
    return quick_ref

def get_jurisdiction_search_examples() -> list:
    """
    Get examples of jurisdiction search queries and expected results.
    
    Returns:
        List of example searches
    """
    return [
        {
            "query": "california",
            "expected": "USJurisdiction.CALIFORNIA",
            "description": "Full state name"
        },
        {
            "query": "CA",
            "expected": "USJurisdiction.CALIFORNIA", 
            "description": "State abbreviation"
        },
        {
            "query": "federal",
            "expected": "USJurisdiction.FEDERAL",
            "description": "Federal jurisdiction"
        },
        {
            "query": "new york",
            "expected": "USJurisdiction.NEW_YORK",
            "description": "State with space in name"
        },
        {
            "query": "DC",
            "expected": "USJurisdiction.DISTRICT_OF_COLUMBIA",
            "description": "District of Columbia"
        },
        {
            "query": "state of texas",
            "expected": "USJurisdiction.TEXAS",
            "description": "State with 'state of' prefix"
        },
        {
            "query": "general",
            "expected": "USJurisdiction.GENERAL",
            "description": "General/non-specific jurisdiction"
        }
    ]

def demonstrate_jurisdiction_system():
    """
    Demonstrate key features of the jurisdiction system.
    
    Prints examples and validation results to console.
    """
    print("=== US Jurisdiction System Demonstration ===\n")
    
    # System info
    info = get_jurisdiction_package_info()
    print(f"Package: {info['description']}")
    print(f"Version: {info['version']}")
    print(f"Total Jurisdictions: {info['jurisdiction_statistics']['total_jurisdictions']}")
    print()
    
    # Validation examples
    print("--- Jurisdiction Validation Examples ---")
    examples = get_jurisdiction_search_examples()
    for example in examples[:5]:  # Show first 5
        try:
            result = validate_jurisdiction(example["query"])
            print(f"✅ '{example['query']}' → {result.value} ({example['description']})")
        except Exception as e:
            print(f"❌ '{example['query']}' → Error: {e}")
    print()
    
    # Context example
    print("--- Jurisdiction Context Example ---")
    try:
        context = get_jurisdiction_context("california")
        metadata = context["metadata"]
        print(f"Jurisdiction: {metadata.full_name} ({metadata.abbreviation})")
        print(f"Region: {metadata.region.title()}")
        print(f"Federal Circuit: {metadata.federal_circuit}")
        print(f"Supreme Court: {metadata.supreme_court_name}")
        print(f"Prominent Legal Areas: {', '.join(metadata.prominent_legal_areas[:3])}")
    except Exception as e:
        print(f"❌ Context retrieval failed: {e}")
    print()
    
    # Regional grouping example
    print("--- Regional Grouping Example (West) ---")
    try:
        west_states = get_region_states("west")
        print(f"West Region States ({len(west_states)}):")
        for state in west_states[:5]:  # Show first 5
            metadata = get_all_jurisdiction_metadata()[state]
            print(f"  • {metadata.full_name} ({metadata.abbreviation})")
        if len(west_states) > 5:
            print(f"  ... and {len(west_states) - 5} more")
    except Exception as e:
        print(f"❌ Regional grouping failed: {e}")
    print()
    
    # System validation
    print("--- System Validation ---")
    validation = validate_jurisdiction_system()
    if validation["system_valid"]:
        print("✅ Jurisdiction system validation passed")
        passing_tests = sum(1 for result in validation["test_results"].values() if result == "pass")
        total_tests = len(validation["test_results"])
        print(f"✅ {passing_tests}/{total_tests} tests passed")
        
        if validation["warnings"]:
            print("⚠️  Warnings:")
            for warning in validation["warnings"]:
                print(f"   • {warning}")
    else:
        print("❌ Jurisdiction system validation failed")
        for error in validation["errors"]:
            print(f"   • {error}")

# Initialize package validation
def _initialize_jurisdiction_package():
    """Initialize jurisdiction package with validation"""
    try:
        # Quick validation that core functionality works
        _ = validate_jurisdiction("federal")
        _ = get_all_jurisdiction_metadata()
        return True
    except Exception:
        import warnings
        warnings.warn(
            "Jurisdiction package initialization incomplete. Some functionality may not work correctly.",
            ImportWarning,
            stacklevel=2
        )
        return False

# Run initialization when package is imported
_package_initialized = _initialize_jurisdiction_package()
