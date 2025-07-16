#!/usr/bin/env python3
"""
Test script for reward function validation
Tests both the current GPT-4o-mini reward function and comprehensive legal reward system
"""

import os
import sys
import time
import asyncio
from pathlib import Path

# Test the current GPT-4o-mini reward function
# def test_current_reward_function():
#     """Test the existing GPT-4o-mini reward function"""
    
#     print("🧪 Testing Current GPT-4o-mini Reward Function")
#     print("=" * 60)
    
#     # Check if the reward function file exists
#     reward_file = Path("custom_rewards/gpt4_legal_reward.py")
#     if not reward_file.exists():
#         print("❌ Reward function file not found. Creating directory...")
#         reward_file.parent.mkdir(exist_ok=True)
#         return False
    
#     # Import the reward function
#     sys.path.insert(0, str(reward_file.parent))
#     try:
#         from gpt4_legal_reward import compute_score, GPT4MiniRewardFunction
#         print("✅ Reward function imported successfully")
#     except ImportError as e:
#         print(f"❌ Failed to import reward function: {e}")
#         return False
    
#     # Test data point
#     test_data = {
#         'data_source': 'custom_legal',
#         'solution_str': 'A valid contract requires offer, acceptance, and consideration. These are the three essential elements needed for any legally binding agreement.',
#         'ground_truth': 'A valid contract requires: (1) offer, (2) acceptance, (3) consideration (exchange of value), (4) legal capacity of parties, and (5) legal purpose.',
#         'extra_info': {
#             'prompt': 'What are the essential elements required for a valid contract?',
#             'legal_area': 'contract_law'
#         }
#     }
    
#     print(f"\n📋 Test Case:")
#     print(f"Question: {test_data['extra_info']['prompt']}")
#     print(f"Generated: {test_data['solution_str'][:100]}...")
#     print(f"Expected: {test_data['ground_truth'][:100]}...")
    
#     # Test the reward function
#     try:
#         print(f"\n⏱️ Testing reward computation...")
#         start_time = time.time()
        
#         score = compute_score(
#             test_data['data_source'],
#             test_data['solution_str'], 
#             test_data['ground_truth'],
#             test_data['extra_info']
#         )
        
#         end_time = time.time()
#         duration = end_time - start_time
        
#         print(f"✅ Reward computation successful!")
#         print(f"📊 Score: {score:.3f}")
#         print(f"⏱️ Duration: {duration:.2f} seconds")
#         print(f"🎯 Score Range: {'✅ Valid (0-1)' if 0 <= score <= 1 else '❌ Invalid'}")
        
#         return True
        
#     except Exception as e:
#         print(f"❌ Reward computation failed: {e}")
#         print(f"🔍 Error type: {type(e).__name__}")
#         return False

def test_comprehensive_legal_reward_system():
    """Test the comprehensive legal reward system"""
    
    print("\n🏛️ Testing Comprehensive Legal Reward System")
    print("=" * 60)
    
    # try:
    # Try to import the comprehensive system
    from verl_integration import multi_task_legal_reward_function
    print("✅ Comprehensive legal reward system imported successfully")
        
    # Test data for different legal task types
    test_cases = [
        {
            'name': 'Contract Law (General Chat)',
            'data_source': 'legal_chat',
            'solution_str': 'A contract requires offer, acceptance, and consideration. In California, these elements are necessary for any legally binding agreement.',
            'ground_truth': 'What are the basic elements of a contract?',
            'extra_info': {
                'task_type': 'general_chat',
                'jurisdiction': 'california',
                'legal_domain': 'contract_law'
            }
        },
        {
            'name': 'Constitutional Law (Judicial Reasoning)',
            'data_source': 'court_cases',
            'solution_str': 'Based on the precedent in Marbury v. Madison, the Supreme Court has the power of judicial review to determine the constitutionality of laws.',
            'ground_truth': 'Explain the principle of judicial review.',
            'extra_info': {
                'task_type': 'judicial_reasoning',
                'jurisdiction': 'federal',
                'legal_domain': 'constitutional_law'
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 Test Case {i}: {test_case['name']}")
        print(f"Task Type: {test_case['extra_info'].get('task_type', 'unknown')}")
        print(f"Jurisdiction: {test_case['extra_info'].get('jurisdiction', 'unknown')}")
        print(f"Generated: {test_case['solution_str'][:80]}...")
        
        # try:
        start_time = time.time()
        
        # Test async function if available
        if asyncio.iscoroutinefunction(multi_task_legal_reward_function):
            score = asyncio.run(multi_task_legal_reward_function(
                test_case['data_source'],
                test_case['solution_str'],
                test_case['ground_truth'],
                test_case['extra_info']
            ))
        else:
            score = multi_task_legal_reward_function(
                test_case['data_source'],
                test_case['solution_str'],
                test_case['ground_truth'],
                test_case['extra_info']
            )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"✅ Score: {score:.3f} (Range: 0-15)")
        print(f"⏱️ Duration: {duration:.2f} seconds")
            
        # except Exception as e:
        #     print(f"❌ Computation failed: {e}")
    
    return True
        
    # except ImportError as e:
    #     print(f"❌ Comprehensive system not available: {e}")
    #     print("💡 This is expected if you haven't set up the full legal reward system yet")
    #     return False

def test_api_connectivity():
    """Test OpenAI API connectivity with correct endpoints"""
    
    print(f"\n🔑 Testing OpenAI API Connectivity")
    print("=" * 60)
    
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ OPENAI_API_KEY not set")
        return False
    
    try:
        import openai
        client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Test with CORRECT API endpoint
        print("🔍 Testing correct Chat Completions API...")
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[{'role': 'user', 'content': 'Hello'}],
            max_tokens=20,
            temperature=0.1
        )
        
        print("✅ OpenAI API connection successful")
        print(f"📝 Response: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI API test failed: {e}")
        return False

def main():
    """Main test function"""
    
    print("🧪 REWARD FUNCTION COMPREHENSIVE TEST")
    print("=" * 80)
    
    # Test results
    results = {
        'api_connectivity': test_api_connectivity(),
        # 'current_reward_function': test_current_reward_function(),
        'comprehensive_system': test_comprehensive_legal_reward_system()
    }
    
    # Summary
    print(f"\n📊 TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\n🎯 Overall: {total_passed}/{total_tests} tests passed")
    
    # if results['current_reward_function']:
    #     print("\n✅ Your current reward function is working!")
    # else:
    #     print("\n❌ Current reward function needs fixing") 
        
    if results['comprehensive_system']:
        print("✅ Comprehensive legal reward system is available and working!")
    else:
        print("💡 Consider upgrading to the comprehensive legal reward system")
    
    return total_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
