#!/bin/bash
# Legal Dataset GRPO Training - 1 GPU Setup with OpenAI GPT-4o-mini Model-Based Rewards
# Using your proven working structure with model-based reward function

set -e

echo "⚖️ Legal Dataset GRPO Training - 1 GPU Setup with GPT-4o-mini Rewards"
echo "==================================================================="

# Step 0: Check OpenAI API key
echo "🔑 Step 0: OpenAI API Setup"
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ OPENAI_API_KEY environment variable not set!"
    echo "💡 Set it with: export OPENAI_API_KEY='your-api-key-here'"
    echo "💡 Or add it to your ~/.bashrc"
    exit 1
else
    echo "✅ OpenAI API key found"
    # Test API connection with new Responses API
    python3 -c "
import openai
import os
try:
    client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    # Test with the new Responses API (March 2025)
    response = client.responses.create(
        model='gpt-4o-mini',
        input='Hello',
        max_output_tokens=20
    )
    print('✅ OpenAI GPT-4o-mini Responses API connection successful')
except Exception as e:
    print(f'❌ OpenAI Responses API test failed: {e}')
    exit(1)
"
fi

# Step 1: Environment setup (same as your working version)
echo "📋 Step 1: Environment Setup"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONUNBUFFERED=1

# Step 4: Model check (same as before)
echo "🤖 Step 4: Model Check"
python3 -c "
import transformers
try:
    tokenizer = transformers.AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct')
    print('✅ Model ready: Qwen2.5-1.5B-Instruct')
except Exception as e:
    print(f'❌ Model error: {e}')
"

# Step 5: Legal Dataset GRPO Training with GPT-4o-mini Rewards
echo "⚖️ Step 5: Legal Dataset GRPO Training with GPT-4o-mini Model-Based Rewards"
echo ""
echo "🔧 CONFIGURATION:"
echo "  • Same proven batch sizes (8/8/1)"
echo "  • Same GRPO settings (group sampling)"
echo "  • Same 1-GPU setup"
echo "  • NEW: GPT-4o-mini model-based reward function"
echo "  • NEW: Similarity-based evaluation (0-1 scale)"
echo ""

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="legal_grpo_gpt4_rewards_${TIMESTAMP}.log"

echo "📝 Training log: $LOG_FILE"
echo "⏰ Starting legal GRPO training with GPT-4o-mini rewards..."

# Modified configuration for model-based rewards with correct Verl custom reward function setup
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
algorithm.adv_estimator=grpo \
data.train_files='data/legal_dataset/train.parquet' \
data.val_files='data/legal_dataset/test.parquet' \
data.train_batch_size=4 \
data.max_prompt_length=256 \
data.max_response_length=512 \
actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
actor_rollout_ref.actor.optim.lr=1e-6 \
actor_rollout_ref.actor.ppo_mini_batch_size=4 \
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
actor_rollout_ref.actor.use_kl_loss=True \
actor_rollout_ref.actor.kl_loss_coef=0.001 \
actor_rollout_ref.actor.kl_loss_type=low_var_kl \
actor_rollout_ref.rollout.n=2 \
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
reward_model.enable=false \
custom_reward_function.path=verl_integration.py \
custom_reward_function.name=multi_task_legal_reward_function \
trainer.logger='["console"]' \
trainer.project_name=legal_gpt4_rewards \
trainer.experiment_name=legal_gpt4_${TIMESTAMP} \
trainer.n_gpus_per_node=1 \
trainer.nnodes=1 \
trainer.save_freq=2 \
trainer.test_freq=1 \
trainer.total_epochs=3 \
trainer.val_before_train=False 2>&1 | tee $LOG_FILE

# Check results
if [ $? -eq 0 ]; then
    echo ""
    echo "⚖️ SUCCESS! Legal GRPO Training with Custom Rewards Completed!"
    echo "================================================================="
    echo "💾 Model saved: checkpoints/legal_gpt4_rewards/legal_gpt4_${TIMESTAMP}/"
    echo "📝 Training log: $LOG_FILE"
    
fi

echo ""
echo "📚 MODEL-BASED REWARD SUMMARY (UPDATED)"
echo "======================================"
echo "⚖️ You now have a legal model trained with CORRECT AI-based evaluation! ⚖️"