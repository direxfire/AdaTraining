# ada_training/utils/config.py
"""
Configuration management for Ada training
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    """Training configuration with aggressive LoRA settings"""
    
    # Model settings
    model_name: str = "unsloth/Qwen3-8B"
    max_seq_length: int = 2048
    
    # LoRA settings - AGGRESSIVE for identity formation
    lora_rank: int = 64  # Increased from 8 to 64
    lora_alpha: int = 128  # 2x rank
    
    # Training hyperparameters
    batch_size: int = 1  # Limited by 16GB VRAM
    gradient_accumulation_steps: int = 4  # Effective batch size = 4
    num_epochs: int = 3  # More epochs for stronger effect
    learning_rate: float = 5e-5  # Higher LR for LoRA
    warmup_steps: int = 50
    
    # Data paths
    generated_data_path: str = "./data/generated_conversations.json"
    output_dir: str = "./outputs/ada_model"
    
    # Dataset mixing ratios
    hippocorpus_ratio: float = 0.4  # 40% human stories
    journal_ratio: float = 0.2       # 20% journal entries
    reddit_ratio: float = 0.2         # 20% Reddit personal stories
    generated_ratio: float = 0.2      # 20% your generated data
    
    def __post_init__(self):
        """Validate configuration"""
        if self.lora_rank < 32:
            print("WARNING: LoRA rank < 32 may not be strong enough for identity formation")
        
        if self.lora_alpha != 2 * self.lora_rank:
            print(f"WARNING: lora_alpha should typically be 2x lora_rank")
        
        total_ratio = (
            self.hippocorpus_ratio + 
            self.journal_ratio + 
            self.reddit_ratio + 
            self.generated_ratio
        )
        if abs(total_ratio - 1.0) > 0.01:
            raise ValueError(f"Dataset ratios must sum to 1.0, got {total_ratio}")


@dataclass  
class InferenceConfig:
    """Configuration for running Ada inference with diary RAG"""
    
    model_path: str = "./outputs/ada_model"
    diary_path: str = "./data/ada_diary.json"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    
    # Chat settings
    user_name: str = "Drew"
    assistant_name: str = "Ada"