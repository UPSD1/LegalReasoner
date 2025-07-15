#!/bin/bash
# Legal Model Inference - After GRPO Training with GPT-4o-mini Rewards
# This script shows how to use your trained legal model for inference

set -e

echo "⚖️ Legal Model Inference Guide"
echo "============================="

# Step 1: Find your trained model checkpoint
echo "📁 Step 1: Locating Trained Model Checkpoint"

# Your checkpoint should be in this pattern based on the training script
CHECKPOINT_BASE="checkpoints/legal_gpt4_rewards"
echo "🔍 Looking for checkpoints in: $CHECKPOINT_BASE"

# Find the latest checkpoint
if [ -d "$CHECKPOINT_BASE" ]; then
    LATEST_EXPERIMENT=$(ls -t $CHECKPOINT_BASE | head -1)
    CHECKPOINT_PATH="$CHECKPOINT_BASE/$LATEST_EXPERIMENT"
    echo "✅ Found latest experiment: $LATEST_EXPERIMENT"
    
    # Find the latest global step
    if [ -d "$CHECKPOINT_PATH" ]; then
        LATEST_STEP=$(ls -t $CHECKPOINT_PATH | grep "global_step" | head -1)
        if [ ! -z "$LATEST_STEP" ]; then
            ACTOR_CHECKPOINT="$CHECKPOINT_PATH/$LATEST_STEP/actor"
            echo "✅ Found latest checkpoint: $LATEST_STEP"
            echo "📂 Actor checkpoint path: $ACTOR_CHECKPOINT"
        else
            echo "❌ No global_step checkpoints found"
            exit 1
        fi
    else
        echo "❌ Checkpoint directory not found: $CHECKPOINT_PATH"
        exit 1
    fi
else
    echo "❌ Checkpoint base directory not found: $CHECKPOINT_BASE"
    echo "💡 Make sure training completed successfully"
    exit 1
fi

# Step 2: Merge checkpoint back to HuggingFace format
echo ""
echo "🔧 Step 2: Merging Checkpoint to HuggingFace Format"

MERGED_MODEL_PATH="./legal_model_merged"

echo "🔄 Merging Verl checkpoint to HuggingFace format..."
echo "   Source: $ACTOR_CHECKPOINT"
echo "   Target: $MERGED_MODEL_PATH"

python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$ACTOR_CHECKPOINT" \
    --target_dir "$MERGED_MODEL_PATH"

if [ $? -eq 0 ]; then
    echo "✅ Model merged successfully!"
else
    echo "❌ Model merge failed"
    exit 1
fi

# Step 3: Test the trained model
echo ""
echo "🧪 Step 3: Testing Trained Legal Model"

cat > test_legal_model.py << 'EOF'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os

def test_legal_model(model_path, base_model_path="Qwen/Qwen2.5-1.5B-Instruct"):
    """Test the trained legal model against the base model."""
    
    print(f"🤖 Loading trained legal model from: {model_path}")
    
    try:
        # Load trained model
        trained_tokenizer = AutoTokenizer.from_pretrained(model_path)
        trained_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Load base model for comparison
        print(f"🤖 Loading base model for comparison: {base_model_path}")
        base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        print("✅ Models loaded successfully!")
        
        # Test questions (mix of training and new questions)
        test_questions = [
            # From training set
            "What is the minimum age to serve as President of the United States?",
            "How many justices currently serve on the U.S. Supreme Court?",
            "In contract law, what are the essential elements required for a valid contract?",
            
            # New questions (not in training)
            "What is the difference between assault and battery in criminal law?",
            "Explain the concept of 'consideration' in contract law.",
            "What are the requirements for establishing a trademark?",
            "What is the burden of proof in a criminal case?",
            "Explain the doctrine of stare decisis.",
        ]
        
        print(f"\n⚖️ Testing Legal Model Performance")
        print("=" * 50)
        
        for i, question in enumerate(test_questions):
            print(f"\n📋 Question {i+1}: {question}")
            print("-" * 80)
            
            # Test trained model
            print("🎓 TRAINED MODEL RESPONSE:")
            trained_response = generate_response(trained_model, trained_tokenizer, question)
            print(f"   {trained_response}")
            
            print("\n🤖 BASE MODEL RESPONSE:")
            base_response = generate_response(base_model, base_tokenizer, question)
            print(f"   {base_response}")
            
            print("\n" + "="*80)
        
        print(f"\n🎯 Evaluation Summary:")
        print("✅ Compare the responses above to see training improvements")
        print("✅ Trained model should show better legal reasoning")
        print("✅ Responses should be more structured and accurate")
        
    except Exception as e:
        print(f"❌ Error loading or testing model: {e}")
        return False
    
    return True

def generate_response(model, tokenizer, question, max_new_tokens=300):
    """Generate response from a model."""
    
    # Format as chat
    messages = [
        {"role": "user", "content": question}
    ]
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()

if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "./legal_model_merged"
    test_legal_model(model_path)
EOF

echo "🚀 Running legal model test..."
python3 test_legal_model.py "$MERGED_MODEL_PATH"

# Step 4: Interactive legal assistant
echo ""
echo "💬 Step 4: Interactive Legal Assistant"

cat > legal_assistant.py << 'EOF'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

def create_legal_assistant(model_path):
    """Create an interactive legal assistant."""
    
    print(f"⚖️ Loading Legal Assistant from: {model_path}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("✅ Legal Assistant ready!")
        
        print("\n" + "="*60)
        print("🏛️  LEGAL ASSISTANT - Trained with GRPO + GPT-4o-mini")
        print("="*60)
        print("💡 Ask legal questions and get AI-trained responses")
        print("💡 Type 'quit' to exit")
        print("="*60)
        
        while True:
            question = input("\n⚖️  Your legal question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not question:
                continue
            
            print("🤔 Thinking...")
            
            # Generate response
            messages = [
                {"role": "system", "content": "You are a knowledgeable legal assistant. Provide accurate, structured legal information. Always include relevant legal principles and cite when appropriate."},
                {"role": "user", "content": question}
            ]
            
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=400,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            print(f"\n⚖️  Legal Assistant Response:")
            print("-" * 50)
            print(response.strip())
            print("-" * 50)
    
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "./legal_model_merged"
    create_legal_assistant(model_path)
EOF

echo "🎯 Legal Assistant created!"
echo ""
echo "🚀 INFERENCE OPTIONS:"
echo "====================="
echo ""
echo "1️⃣ Test Model Performance (already ran above):"
echo "   python3 test_legal_model.py $MERGED_MODEL_PATH"
echo ""
echo "2️⃣ Interactive Legal Assistant:"
echo "   python3 legal_assistant.py $MERGED_MODEL_PATH"
echo ""
echo "3️⃣ Use in your own code:"
echo "   from transformers import AutoTokenizer, AutoModelForCausalLM"
echo "   tokenizer = AutoTokenizer.from_pretrained('$MERGED_MODEL_PATH')"
echo "   model = AutoModelForCausalLM.from_pretrained('$MERGED_MODEL_PATH')"
echo ""
echo "📊 Model Information:"
echo "   📂 Merged Model Path: $MERGED_MODEL_PATH"
echo "   🏷️  Base Model: Qwen2.5-1.5B-Instruct"
echo "   🎯 Training: GRPO with GPT-4o-mini reward evaluation"
echo "   ⚖️  Domain: Legal reasoning and Q&A"
echo "   📚 Training Data: 8 legal questions across multiple law areas"
echo ""
echo "✅ Your legal model is ready for inference!"

# Optional: Clean up old checkpoints to save space
echo ""
read -p "🗑️ Do you want to clean up original checkpoints to save space? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🧹 Cleaning up original checkpoints..."
    rm -rf "$CHECKPOINT_BASE"
    echo "✅ Cleanup complete! Merged model saved at: $MERGED_MODEL_PATH"
fi

echo ""
echo "🎉 Inference setup complete!"
echo "⚖️ Your legal AI model is ready to answer legal questions!"