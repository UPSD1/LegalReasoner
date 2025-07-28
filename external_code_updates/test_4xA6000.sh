#!/bin/bash
set -e  # Exit on any error

echo "🧪 Quick Legal Search-R1 Pipeline Test"
echo "====================================="

# GPU Configuration for 4x A6000
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_ATTENTION_BACKEND=XFORMERS
export RAY_memory_usage_threshold=0.85
export PYTHONUNBUFFERED=1

# Data paths
export DATA_DIR='data/legal_dataset'
export TRAIN_DATA_DIR=$DATA_DIR
export TEST_DATA_DIR=$DATA_DIR

# Model Configuration  
export BASE_MODEL='Qwen/Qwen2.5-32B-Instruct'
export EXPERIMENT_NAME='legal-search-r1-quick-test'
export WANDB_PROJECT='Legal-Search-R1-Test'

# Quick Test Parameters (designed for <1 hour completion)
export TEST_EPOCHS=1
export TEST_TRAINING_STEPS=15
export TEST_BATCH_SIZE=4
export TEST_VAL_BATCH_SIZE=2
export TRAIN_SAMPLES=12  # Very small for quick testing
export VAL_SAMPLES=6

echo "⚡ Quick Test Configuration:"
echo "   Model: $BASE_MODEL"
echo "   Epochs: $TEST_EPOCHS"
echo "   Training Steps: $TEST_TRAINING_STEPS"
echo "   Batch Size: $TEST_BATCH_SIZE"
echo "   Train Samples: $TRAIN_SAMPLES"
echo "   Val Samples: $VAL_SAMPLES"
echo "   Expected Duration: 30-45 minutes"

# Validate prerequisites
echo "🔍 Validating setup..."

if [ ! -f "$TRAIN_DATA_DIR/train.parquet" ]; then
    echo "❌ Training data not found at $TRAIN_DATA_DIR/train.parquet"
    exit 1
fi

if [ ! -f "$TEST_DATA_DIR/test.parquet" ]; then
    echo "❌ Test data not found at $TEST_DATA_DIR/test.parquet"
    exit 1
fi

# Check retriever service
if ! curl -s http://127.0.0.1:8000/health > /dev/null; then
    echo "❌ Retriever service not running at http://127.0.0.1:8000"
    exit 1
else
    echo "✅ Retriever service is running"
fi

# Create directories
mkdir -p verl_checkpoints/$EXPERIMENT_NAME
mkdir -p logs

echo "🚀 Starting Quick Pipeline Test..."

# Ultra-Fast Test Configuration
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    `# Data Configuration - Minimal for speed` \
    data.train_files=$TRAIN_DATA_DIR/train.parquet \
    data.val_files=$TEST_DATA_DIR/test.parquet \
    data.train_data_num=$TRAIN_SAMPLES \
    data.val_data_num=$VAL_SAMPLES \
    data.train_batch_size=$TEST_BATCH_SIZE \
    data.val_batch_size=$TEST_VAL_BATCH_SIZE \
    data.max_prompt_length=512 \
    data.max_response_length=256 \
    data.max_start_length=256 \
    data.max_obs_length=128 \
    data.shuffle_train_dataloader=True \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    \
    `# Algorithm Configuration` \
    algorithm.adv_estimator=grpo \
    algorithm.no_think_rl=false \
    \
    `# Model Configuration - Memory Optimized` \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.trust_remote_code=True \
    \
    `# Actor Configuration - Minimal for speed` \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.05 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.state_masking=True \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    \
    `# FSDP Configuration - Maximum offloading` \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.grad_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=4 \
    \
    `# Rollout Configuration - Ultra conservative memory` \
    actor_rollout_ref.rollout.n_agent=2 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
    actor_rollout_ref.rollout.max_num_seqs=2 \
    actor_rollout_ref.rollout.max_model_len=768 \
    actor_rollout_ref.rollout.max_num_batched_tokens=1024 \
    \
    `# Reference Model Configuration` \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    `# Retriever Configuration` \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=2 \
    max_turns=1 \
    \
    `# Custom Reward Function` \
    custom_reward_function.path=../LegalReasoner/simple_custom_reward.py \
    custom_reward_function.name=evaluate_response \
    \
    `# Training Control - Quick test settings` \
    trainer.logger='["console","wandb"]' \
    trainer.val_only=False \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=$TEST_EPOCHS \
    trainer.total_training_steps=$TEST_TRAINING_STEPS \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    trainer.default_hdfs_dir=null \
    \
    2>&1 | tee logs/$EXPERIMENT_NAME.log

TRAINING_EXIT_CODE=$?

# Quick Test Results
echo ""
echo "⚡ Quick Test Results:"
echo "====================="

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ Pipeline test PASSED!"
    echo "✅ All components working correctly"
    
    # Check basic functionality
    if [ -d "verl_checkpoints/$EXPERIMENT_NAME" ]; then
        echo "✅ Checkpointing: Working"
    else
        echo "⚠️  Checkpointing: No checkpoints found"
    fi
    
    if [ -f "logs/$EXPERIMENT_NAME.log" ]; then
        echo "✅ Logging: Working ($(du -h logs/$EXPERIMENT_NAME.log | cut -f1))"
        
        # Check for key indicators in log
        if grep -q "GRPO" logs/$EXPERIMENT_NAME.log; then
            echo "✅ GRPO Algorithm: Active"
        fi
        
        if grep -q "retriever" logs/$EXPERIMENT_NAME.log; then
            echo "✅ Retriever Integration: Working"  
        fi
        
        if grep -q "custom_reward" logs/$EXPERIMENT_NAME.log; then
            echo "✅ Custom Reward Function: Active"
        fi
    fi
    
    echo ""
    echo "🎉 Ready for full training!"
    echo "💡 To run full training, use the production configuration"
    
else
    echo "❌ Pipeline test FAILED (exit code: $TRAINING_EXIT_CODE)"
    echo ""
    echo "🔧 Debug Information:"
    
    # Show critical error info
    if [ -f "logs/$EXPERIMENT_NAME.log" ]; then
        echo "📋 Last 15 lines from log:"
        echo "----------------------------------------"
        tail -15 logs/$EXPERIMENT_NAME.log
        echo "----------------------------------------"
        
        # Check for common error patterns
        if grep -q "OutOfMemoryError" logs/$EXPERIMENT_NAME.log; then
            echo "🚨 Memory Error Detected - Try reducing batch sizes further"
        fi
        
        if grep -q "CUDA" logs/$EXPERIMENT_NAME.log; then
            echo "🚨 GPU Error Detected - Check nvidia-smi"
        fi
        
        if grep -q "Connection" logs/$EXPERIMENT_NAME.log; then
            echo "🚨 Connection Error - Check retriever service"
        fi
    fi
    
    echo ""
    echo "🔍 Next steps:"
    echo "   1. Check full log: logs/$EXPERIMENT_NAME.log"
    echo "   2. Verify retriever service: curl http://127.0.0.1:8000/health"
    echo "   3. Monitor GPU memory: watch nvidia-smi"
    
    exit 1
fi

echo ""
echo "⏱️  Test completed in: $(date)"