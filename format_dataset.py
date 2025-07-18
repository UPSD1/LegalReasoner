import json
import pickle
from typing import List, Dict, Any
import copy
from custom_datasets import train_set, test_set

def restructure_dataset(dataset: List[Dict[Any, Any]]) -> List[Dict[Any, Any]]:
    """
    Move task_type, jurisdiction, and legal_domain from reward_model to extra_info
    
    Args:
        dataset: List of dataset dictionaries
        
    Returns:
        Modified dataset with fields moved
    """
    restructured_dataset = []
    
    for item in dataset:
        # Create a deep copy to avoid modifying original data
        new_item = copy.deepcopy(item)
        
        # Fields to move from reward_model to extra_info
        fields_to_move = ["task_type", "jurisdiction", "legal_domain"]
        
        # Check if reward_model exists
        if "reward_model" in new_item:
            # Check if extra_info exists, if not create it
            if "extra_info" not in new_item:
                new_item["extra_info"] = {}
            
            # Move each field from reward_model to extra_info
            for field in fields_to_move:
                if field in new_item["reward_model"]:
                    # Move the field
                    new_item["extra_info"][field] = new_item["reward_model"][field]
                    # Remove from reward_model
                    del new_item["reward_model"][field]
        
        restructured_dataset.append(new_item)
    
    return restructured_dataset

def save_dataset_json(dataset: List[Dict[Any, Any]], filename: str) -> None:
    """Save dataset to JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"Dataset saved to {filename}")

def save_dataset_pickle(dataset: List[Dict[Any, Any]], filename: str) -> None:
    """Save dataset to pickle file for faster loading"""
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Dataset saved to {filename}")

def load_dataset_json(filename: str) -> List[Dict[Any, Any]]:
    """Load dataset from JSON file"""
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_dataset_pickle(filename: str) -> List[Dict[Any, Any]]:
    """Load dataset from pickle file"""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def print_sample_before_after(original: Dict[Any, Any], modified: Dict[Any, Any]) -> None:
    """Print a sample of the before and after structure for verification"""
    print("=== BEFORE RESTRUCTURING ===")
    print("reward_model keys:", list(original.get("reward_model", {}).keys()))
    print("extra_info keys:", list(original.get("extra_info", {}).keys()))
    
    print("\n=== AFTER RESTRUCTURING ===")
    print("reward_model keys:", list(modified.get("reward_model", {}).keys()))
    print("extra_info keys:", list(modified.get("extra_info", {}).keys()))
    print()

# Main execution
def main():
    # Assuming you have 'train' and 'test' variables loaded with your data
    # Replace these with your actual data loading if needed
    
    # Example of how to use if you need to load from files:
    # train = load_dataset_json('original_train.json')
    # test = load_dataset_json('original_test.json')
    train = train_set
    test = test_set
    
    print("Starting dataset restructuring...")
    
    # Show sample before restructuring
    if train:
        print("\n=== SAMPLE TRAINING ITEM BEFORE ===")
        sample_original = train[0]
        print_sample_before_after(sample_original, sample_original)
    
    # Restructure training data
    print("Restructuring training dataset...")
    train_restructured = restructure_dataset(train)
    
    # Show sample after restructuring
    if train_restructured:
        print("=== SAMPLE TRAINING ITEM AFTER ===")
        print_sample_before_after(train[0], train_restructured[0])
    
    # Save restructured training data
    save_dataset_json(train_restructured, 'train_restructured.json')
    save_dataset_pickle(train_restructured, 'train_restructured.pkl')
    
    # Restructure test data
    print("Restructuring test dataset...")
    test_restructured = restructure_dataset(test)
    
    # Save restructured test data
    save_dataset_json(test_restructured, 'test_restructured.json')
    save_dataset_pickle(test_restructured, 'test_restructured.pkl')

    print("\n=== RESTRUCTURING COMPLETE ===")
    print("Files saved:")
    print("- train_restructured.json (human-readable)")
    print("- train_restructured.pkl (fast loading)")
    print("- test_restructured.json (human-readable)")  
    print("- test_restructured.pkl (fast loading)")
    
    # Verification: Load and check one item
    if 'train' in globals():
        print("\n=== VERIFICATION ===")
        reloaded_train = load_dataset_json('train_restructured.json')
        sample_item = reloaded_train[0]
        
        print("Fields in extra_info:", list(sample_item.get("extra_info", {}).keys()))
        print("Fields in reward_model:", list(sample_item.get("reward_model", {}).keys()))
        
        # Check that the moved fields are in extra_info
        moved_fields = ["task_type", "jurisdiction", "legal_domain"]
        for field in moved_fields:
            if field in sample_item.get("extra_info", {}):
                print(f"✓ {field} successfully moved to extra_info")
            else:
                print(f"✗ {field} NOT found in extra_info")

# Example usage for quick testing without running main()
def quick_restructure_and_save(train_data, test_data):
    """
    Quick function to restructure and save your data
    
    Args:
        train_data: Your training dataset (list of dicts)
        test_data: Your test dataset (list of dicts)
    """
    # Restructure
    train_restructured = restructure_dataset(train_data)
    test_restructured = restructure_dataset(test_data)
    
    # Save both formats
    save_dataset_json(train_restructured, 'train_restructured.json')
    save_dataset_pickle(train_restructured, 'train_restructured.pkl')
    save_dataset_json(test_restructured, 'test_restructured.json')
    save_dataset_pickle(test_restructured, 'test_restructured.pkl')
    
    return train_restructured, test_restructured

# If you want to run this directly with your variables:
if __name__ == "__main__":
    # Option 1: Run main() if train and test are already in global scope
    main()
    
    # Option 2: Call quick_restructure_and_save directly with your variables
    # train_new, test_new = quick_restructure_and_save(train, test)
    
    # print("Script ready to run!")
    # print("Choose one of these options:")
    # print("1. Run main() if train and test variables are loaded")
    # print("2. Call quick_restructure_and_save(train, test) with your data")
    # print("3. Use individual functions as needed")