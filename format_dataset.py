"""
Simple Search-R1 data processor
- Add search prefix to content
- Wrap ground_truth in list
- Save as JSON
"""

import json
import copy
from custom_datasets import train_set, test_set
from dataset_prefix_helper import create_enhanced_prompt

def process_data(train: list, test: list) -> tuple:
    """
    Process train and test data
    
    Args:
        train: List of training dictionaries
        test: List of test dictionaries
    
    Returns:
        (processed_train, processed_test)
    """
    
    def process_item(item):
        # Deep copy to avoid modifying original
        processed = copy.deepcopy(item)
        
        # Modify content - add search prefix
        if 'prompt' in processed and len(processed['prompt']) > 0:
            original_content = processed['prompt'][0]['content']
            processed['prompt'][0]['content'] = create_enhanced_prompt(original_content)
            
        # Modify ground_truth - wrap string in list
        if 'reward_model' in processed and 'ground_truth' in processed['reward_model']:
            ground_truth = processed['reward_model']['ground_truth']
            if isinstance(ground_truth, str):
                processed['reward_model']['ground_truth'] = {"target":[ground_truth]}
        
        return processed
    
    # Process both datasets
    processed_train = [process_item(item) for item in train]
    processed_test = [process_item(item) for item in test]
    
    return processed_train, processed_test


def save_data(processed_train: list, processed_test: list, 
              train_file: str = 'train_processed.json',
              test_file: str = 'test_processed.json'):
    """Save processed data to JSON files"""
    
    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(processed_train, f, indent=2, ensure_ascii=False)
    
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(processed_test, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved {len(processed_train)} train items to {train_file}")
    print(f"✅ Saved {len(processed_test)} test items to {test_file}")


# Example usage function
def main():
    train = train_set
    test = test_set
    
    # Process the data
    processed_train, processed_test = process_data(train, test)
    
    # Save to files
    save_data(processed_train, processed_test)
    
    # Show example of changes
    print("\n📋 Example changes:")
    print("Original content:", train[0]['prompt'][0]['content'])
    print("Modified content:", processed_train[0]['prompt'][0]['content'][:100] + "...")
    print()
    print("Original ground_truth:", train[0]['reward_model']['ground_truth'])
    print("Modified ground_truth:", processed_train[0]['reward_model']['ground_truth'])


if __name__ == '__main__':
    main()