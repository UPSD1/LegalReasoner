set -e  # Exit on any error

echo "🧪 Starting Legal Search-R1 End-to-End Test"
echo "============================================"

# GPU Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_ATTENTION_BACKEND=XFORMERS

# Data paths (adjust to your legal dataset location)
export DATA_DIR='data/legal_dataset'
export TRAIN_DATA_DIR=$DATA_DIR
export TEST_DATA_DIR=$DATA_DIR

# Model Configuration  
export BASE_MODEL='Qwen/Qwen2.5-32B-Instruct'
export EXPERIMENT_NAME='legal-search-r1-test-qwen32b'
export WANDB_PROJECT='Legal-Search-R1-Test'

# Test Parameters (small for quick validation)
export TEST_EPOCHS=1
export TEST_TRAINING_STEPS=20
export TEST_BATCH_SIZE=8
export TEST_VAL_BATCH_SIZE=4

# Export Legal Reward Judge Configuration
export LEGAL_REWARD_CONFIG_PATH='../LegalReasoner/config/internal_config.yaml'

echo "📊 Test Configuration:"
echo "   Model: $BASE_MODEL"
echo "   Epochs: $TEST_EPOCHS"
echo "   Training Steps: $TEST_TRAINING_STEPS"
echo "   Batch Size: $TEST_BATCH_SIZE"
echo "   Data Dir: $DATA_DIR"

# Validate data files exist
if [ ! -f "$TRAIN_DATA_DIR/train.parquet" ]; then
    echo "❌ Training data not found at $TRAIN_DATA_DIR/train.parquet"
    echo "Please ensure your legal dataset is properly formatted and placed in the correct location"
    exit 1
fi

if [ ! -f "$TEST_DATA_DIR/test.parquet" ]; then
    echo "❌ Test data not found at $TEST_DATA_DIR/test.parquet"
    exit 1
fi

# Check if retriever is running
echo "🔍 Checking retriever service..."
if ! curl -s http://127.0.0.1:8000/health > /dev/null; then
    echo "⚠️  Warning: Retriever service not running at http://127.0.0.1:8000"
    exit 1
else
    echo "✅ Retriever service is running"
fi

# Create checkpoint directory
mkdir -p verl_checkpoints/$EXPERIMENT_NAME

echo "🚀 Starting Legal Search-R1 Training Test..."

# Run the training with test parameters
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$TRAIN_DATA_DIR/train.parquet \
    data.val_files=$TEST_DATA_DIR/test.parquet \
    data.train_data_num=20 \
    data.val_data_num=10 \
    data.train_batch_size=$TEST_BATCH_SIZE \
    data.val_batch_size=$TEST_VAL_BATCH_SIZE \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    data.max_start_length=1024 \
    data.max_obs_length=512 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    actor_rollout_ref.rollout.n_agent=3 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger=['console','wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=$TEST_EPOCHS \
    trainer.total_training_steps=$TEST_TRAINING_STEPS \
    trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
    custom_reward_function.path=../LegalReasoner/simple_custom_reward.py \
    custom_reward_function.name=evaluate_response \
    max_turns=2 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee logs/$EXPERIMENT_NAME.log

TRAINING_EXIT_CODE=$?

# Results
echo ""
echo "🔍 Test Results Summary:"
echo "========================"

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ Training process completed successfully!"
    echo "✅ End-to-end pipeline validation PASSED"
    
    # Check if checkpoints were created
    if [ -d "verl_checkpoints/$EXPERIMENT_NAME" ] && [ "$(ls -A verl_checkpoints/$EXPERIMENT_NAME)" ]; then
        echo "✅ Checkpoints created successfully"
        echo "   Location: verl_checkpoints/$EXPERIMENT_NAME"
        echo "   Files: $(ls verl_checkpoints/$EXPERIMENT_NAME | head -3)"
    else
        echo "⚠️  Warning: No checkpoints found"
    fi
    
    echo ""
    echo "🎉 Legal Search-R1 pipeline is ready for full training!"
    
else
    echo "❌ Training process failed with exit code: $TRAINING_EXIT_CODE"
    echo "❌ End-to-end pipeline validation FAILED"
    echo ""
    echo "🔧 Troubleshooting:"
    echo "   1. Check the log file: logs/$EXPERIMENT_NAME.log"
    echo "   2. Verify your dataset format matches Search-R1 requirements"
    echo "   3. Ensure all dependencies are installed"
    echo "   4. Check GPU memory usage with: nvidia-smi"
    exit 1
fi