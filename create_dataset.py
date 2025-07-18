import pandas as pd
import os
from custom_datasets import train_set, test_set

# Split into train/test (SAME as math)
train_data = train_set
test_data = test_set

# Save as parquet (SAME structure)
output_dir = './data/legal_dataset'
os.makedirs(output_dir, exist_ok=True)

print(f"data will be stored {output_dir}")

train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

train_df.to_parquet(os.path.join(output_dir, 'train.parquet'), index=False)
test_df.to_parquet(os.path.join(output_dir, 'test.parquet'), index=False)

print(f'✅ Legal dataset created using proven structure:')
print(f'   📁 Train: {len(train_df)} samples')
print(f'   📁 Test: {len(test_df)} samples')


# Verify structure matches math dataset exactly
print(f'\\n📋 Dataset Structure (identical to math):')
sample = train_data[0]
for key, value in sample.items():
    if isinstance(value, dict):
        print(f'   {key}: {list(value.keys())}')
    elif isinstance(value, list):
        print(f'   {key}: [{len(value)} items]')
    else:
        print(f'   {key}: {value}')

print(f'\\n⚖️ Sample Legal Questions:')
for i, item in enumerate(train_data[:3]):
    question = item['prompt'][0]['content']
    answer = item['reward_model']['ground_truth']
    area = item['reward_model']['legal_domain']
    print(f'   {i+1}. [{area}] {question}')
    print(f'      Expected: {answer}')
