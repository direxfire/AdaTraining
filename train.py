# ada_training/train.py
"""
Main training script for Ada - Human Identity Fine-tuning Project
Training Qwen3-8B to develop a coherent human identity
"""

import os
import torch
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset, concatenate_datasets
from transformers import TrainingArguments
from trl import SFTTrainer
from data.dataset_loader import load_all_datasets, prepare_training_data
from utils.config import TrainingConfig
from utils.logging_utils import setup_logging, log_training_start

def main():
    # Setup
    logger = setup_logging()
    config = TrainingConfig()
    
    log_training_start(logger, config)
    
    # Load model with higher rank LoRA
    logger.info("Loading Qwen3-8B base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-8B",
        max_seq_length=config.max_seq_length,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )
    
    # Configure LoRA with aggressive parameters
    logger.info(f"Configuring LoRA with rank={config.lora_rank}...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.lora_rank,  # 64-128 for stronger identity formation
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_alpha=config.lora_alpha,  # 2x rank
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    # Setup custom chat template - Drew and Ada instead of user/assistant
    logger.info("Setting up custom chat template...")
    custom_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "Drew: {{ message['content'] }}\n"
        "{% elif message['role'] == 'assistant' %}"
        "Ada: {{ message['content'] }}{{ eos_token }}\n"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "Ada: "
        "{% endif %}"
    )
    
    tokenizer = get_chat_template(
        tokenizer,
        chat_template=(custom_template, "eos_token"),
        mapping={
            "role": "role",
            "content": "content", 
            "user": "user",
            "assistant": "assistant"
        },
    )
    
    # Load and prepare datasets
    logger.info("Loading datasets...")
    dataset = load_all_datasets(
        include_hippocorpus=True,
        include_journal=True,
        include_reddit=True,
        include_generated=True,
        generated_path=config.generated_data_path
    )
    
    # Format dataset with chat template
    def formatting_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo, 
                tokenize=False, 
                add_generation_prompt=False,
                enable_thinking=False  # Disable thinking mode for identity training
            ) 
            for convo in convos
        ]
        return {"text": texts}
    
    dataset = dataset.map(formatting_func, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_steps=config.warmup_steps,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
    )
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config.max_seq_length,
        args=training_args,
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save model
    logger.info(f"Saving model to {config.output_dir}...")
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main()