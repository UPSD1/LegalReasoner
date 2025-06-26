#!/usr/bin/env python3
"""
Legal Reward System - Component Diagnostic Script (LOCATION AGNOSTIC)
===================================================================

This script tests what components are actually implemented vs what exists in the roadmap.
It works whether run from INSIDE or OUTSIDE the legal_reward_system directory.

Usage: 
  python component_diagnostic.py                    # From inside legal_reward_system/
  python legal_reward_system/component_diagnostic.py  # From outside

"""

import sys
import importlib
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file if it exists
load_dotenv()

def setup_import_path():
    """
    Setup Python import path to work from any location.
    
    This handles the case where the script is run from inside the package
    vs from outside the package directory.
    """
    script_path = Path(__file__).resolve()
    script_dir = script_path.parent
    
    # Determine if we're running from inside or outside the package
    if script_dir.name == 'legal_reward_system':
        # Running from INSIDE the package directory
        # Need to add parent directory to sys.path so we can import legal_reward_system
        parent_dir = script_dir.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        print(f"📍 Running from INSIDE package directory: {script_dir}")
        print(f"📍 Added to sys.path: {parent_dir}")
        return script_dir, "inside"
    else:
        # Running from OUTSIDE the package directory  
        # legal_reward_system should be in current directory or already in path
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        print(f"📍 Running from OUTSIDE package directory: {script_dir}")
        return script_dir, "outside"

def test_component_imports():
    """Test actual component imports to see what exists"""
    print("\n🔍 TESTING ACTUAL COMPONENT IMPLEMENTATIONS")
    print("=" * 70)
    
    results = {}
    
    # Test core components from roadmap
    components_to_test = [
        # Phase 1: Foundation Layer
        ('legal_reward_system.core.data_structures', 'LegalDataPoint'),
        ('legal_reward_system.core.data_structures', 'EnsembleScore'),
        ('legal_reward_system.core.data_structures', 'HybridEvaluationResult'),
        ('legal_reward_system.core.enums', 'LegalTaskType'),
        ('legal_reward_system.core.enums', 'USJurisdiction'),
        ('legal_reward_system.core.enums', 'APIProvider'),
        ('legal_reward_system.core.exceptions', 'LegalRewardSystemError'),
        ('legal_reward_system.utils.logging', 'get_legal_logger'),
        ('legal_reward_system.utils.cache', 'MultiStrategyLegalRewardCache'),
        ('legal_reward_system.utils.rate_limiter', 'MultiProviderRateLimiter'),
        ('legal_reward_system.config', 'LegalRewardSystemConfig'),
        
        # Phase 2: Domain Logic Layer  
        ('legal_reward_system.jurisdiction.us_system', 'USJurisdictionSystem'),
        ('legal_reward_system.jurisdiction.inference_engine', 'USJurisdictionInferenceEngine'),
        ('legal_reward_system.jurisdiction.compliance_judge', 'JurisdictionComplianceJudge'),
        
        # Phase 3: Judge Framework
        ('legal_reward_system.judges.base', 'BaseJudgeEnsemble'),
        ('legal_reward_system.judges.api_client', 'CostOptimizedAPIClient'),
        ('legal_reward_system.judges.general_chat', 'EnhancedGeneralChatEnsemble'),
        
        # Phase 4: Routing System
        ('legal_reward_system.routing.hybrid_evaluation', 'HybridEvaluationEngine'),
        ('legal_reward_system.routing.task_weights', 'TaskDifficultyWeightManager'),
        ('legal_reward_system.routing.router', 'MultiTaskLegalRewardRouter'),
        
        # Phase 5: Integration
        ('legal_reward_system.verl_integration', 'multi_task_legal_reward_function'),
        ('legal_reward_system.verl_integration', 'VERLLegalRewardFunction'),
        ('legal_reward_system.factory', 'create_production_legal_reward_router'),
    ]
    
    # Test each component
    for module_name, class_name in components_to_test:
        try:
            module = importlib.import_module(module_name)
            if hasattr(module, class_name):
                results[f"{module_name}.{class_name}"] = "✅ EXISTS"
            else:
                results[f"{module_name}.{class_name}"] = f"❌ MISSING - Module exists but {class_name} not found"
        except ImportError as e:
            results[f"{module_name}.{class_name}"] = f"❌ IMPORT ERROR - {str(e)}"
        except Exception as e:
            results[f"{module_name}.{class_name}"] = f"❌ ERROR - {str(e)}"
    
    return results

def test_file_structure(script_dir, run_location):
    """Test if expected files exist with location-aware path resolution"""
    print("\n🗂️  TESTING FILE STRUCTURE")
    print("=" * 70)
    
    if run_location == "inside":
        # Running from inside - files are relative to current directory
        expected_files = [
            '__init__.py',
            'core/__init__.py',
            'core/data_structures.py',
            'core/enums.py',
            'core/exceptions.py',
            'config/__init__.py',
            'utils/__init__.py',
            'jurisdiction/__init__.py',
            'judges/__init__.py',
            'routing/__init__.py',
            'verl_integration.py',
            'factory.py',
        ]
        base_path = script_dir
    else:
        # Running from outside - files are in legal_reward_system/ subdirectory
        expected_files = [
            'legal_reward_system/__init__.py',
            'legal_reward_system/core/__init__.py',
            'legal_reward_system/core/data_structures.py',
            'legal_reward_system/core/enums.py',
            'legal_reward_system/core/exceptions.py',
            'legal_reward_system/config/__init__.py',
            'legal_reward_system/utils/__init__.py',
            'legal_reward_system/jurisdiction/__init__.py',
            'legal_reward_system/judges/__init__.py',
            'legal_reward_system/routing/__init__.py',
            'legal_reward_system/verl_integration.py',
            'legal_reward_system/factory.py',
        ]
        base_path = script_dir
    
    file_results = {}
    for file_path in expected_files:
        full_path = base_path / file_path
        
        if full_path.exists():
            file_results[file_path] = "✅ EXISTS"
        else:
            file_results[file_path] = f"❌ MISSING (checked: {full_path})"
    
    return file_results

def check_main_integration():
    """Test main integration points"""
    print("\n🔗 TESTING MAIN INTEGRATION POINTS")
    print("=" * 70)
    
    integration_tests = []
    
    # Test 1: Main VERL function accessibility
    try:
        from legal_reward_system.verl_integration import multi_task_legal_reward_function
        integration_tests.append("✅ VERL main function importable")
    except Exception as e:
        integration_tests.append(f"❌ VERL main function: {str(e)}")
    
    # Test 2: Factory function accessibility  
    try:
        from legal_reward_system.factory import create_production_legal_reward_router
        integration_tests.append("✅ Factory function importable")
    except Exception as e:
        integration_tests.append(f"❌ Factory function: {str(e)}")
    
    # Test 3: Core data structures
    try:
        from legal_reward_system.core import LegalDataPoint
        data_point = LegalDataPoint(
            query="test", 
            response="test",
            task_type="general_chat",
            jurisdiction="general",
            legal_domain="general" 
        )
        integration_tests.append("✅ Core data structures functional")
    except Exception as e:
        integration_tests.append(f"❌ Core data structures: {str(e)}")
    
    # Test 4: System configuration
    try:
        from legal_reward_system.config import LegalRewardSystemConfig
        config = LegalRewardSystemConfig()
        integration_tests.append("✅ System configuration functional")
    except Exception as e:
        integration_tests.append(f"❌ System configuration: {str(e)}")
    
    return integration_tests

def generate_implementation_roadmap():
    """Generate next steps based on what's missing"""
    print("\n🗺️  IMPLEMENTATION ROADMAP")
    print("=" * 70)
    
    print("Based on the diagnostic results above, here are the next steps:")
    print("\n1. **Fix Missing Components**: Implement any missing core components")
    print("2. **Test Real Functionality**: Replace mock tests with real component tests")
    print("3. **VERL Integration**: Ensure the main entry point works properly")
    print("4. **Run Full Tests**: Use 'python test_runner.py --mode=full' for comprehensive testing")
    
    print("\n📋 **Priority Order**:")
    print("   1. Core data structures (Phase 1)")
    print("   2. Configuration system")
    print("   3. VERL integration function")
    print("   4. Judge framework")
    print("   5. Complete system integration")

def main():
    """Main diagnostic function"""
    print("🏛️  LEGAL REWARD SYSTEM - COMPONENT DIAGNOSTIC (LOCATION AGNOSTIC)")
    print("=" * 70)
    print("Checking actual implementation status vs roadmap requirements...")
    
    # Setup import path based on run location
    script_dir, run_location = setup_import_path()
    
    print(f"📍 Python sys.path includes: {[p for p in sys.path[:3]]}...")
    print()
    
    # Test component imports
    component_results = test_component_imports()
    for component, status in component_results.items():
        print(f"{component}")
        print(f"  {status}")
    
    # Test file structure
    file_results = test_file_structure(script_dir, run_location)
    for file_path, status in file_results.items():
        print(f"{file_path}: {status}")
    
    # Test main integration
    integration_results = check_main_integration()
    for result in integration_results:
        print(f"  {result}")
    
    # Calculate summary statistics
    total_components = len(component_results)
    implemented_components = sum(1 for status in component_results.values() if "✅ EXISTS" in status)
    
    total_files = len(file_results)
    existing_files = sum(1 for status in file_results.values() if "✅ EXISTS" in status)
    
    print("\n" + "=" * 70)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f"Components: {implemented_components}/{total_components} ({(implemented_components/total_components)*100:.1f}%) implemented")
    print(f"Files: {existing_files}/{total_files} ({(existing_files/total_files)*100:.1f}%) present")
    print(f"Run location: {run_location.upper()} package directory")
    
    # Determine overall status
    if implemented_components == total_components:
        if existing_files == total_files:
            print("\n🎉 RESULT: System fully implemented and all files verified!")
            print("   → Run 'python test_runner.py --mode=full' for comprehensive testing")
        else:
            print("\n🎉 RESULT: System fully implemented!")
            print("   → All imports work correctly, system is ready for testing")
            print("   → Run 'python test_runner.py --mode=full' for comprehensive testing")
    elif implemented_components >= total_components * 0.8:
        print("\n⚠️  RESULT: System mostly implemented with some gaps")
        print("   → Fix missing components and run full tests")
    else:
        print("\n❌ RESULT: Significant implementation gaps detected")
        print("   → Focus on implementing missing core components first")
    
    # Only show roadmap if there are actual issues
    if implemented_components < total_components:
        generate_implementation_roadmap()

if __name__ == "__main__":
    main()