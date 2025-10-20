# ada_training/utils/logging_utils.py
"""
Logging utilities for tracking training progress and model behavior
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any

def setup_logging(log_dir: str = "./logs") -> logging.Logger:
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("ADA_Training")
    logger.info(f"Logging to {log_file}")
    
    return logger


def log_training_start(logger: logging.Logger, config) -> None:
    """Log training configuration at start"""
    logger.info("="*50)
    logger.info("ADA TRAINING - Human Identity Formation")
    logger.info("="*50)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"LoRA Rank: {config.lora_rank}")
    logger.info(f"LoRA Alpha: {config.lora_alpha}")
    logger.info(f"Batch Size: {config.batch_size}")
    logger.info(f"Gradient Accumulation: {config.gradient_accumulation_steps}")
    logger.info(f"Epochs: {config.num_epochs}")
    logger.info(f"Learning Rate: {config.learning_rate}")
    logger.info(f"Output Dir: {config.output_dir}")
    logger.info("="*50)


def log_dataset_info(logger: logging.Logger, dataset_stats: Dict[str, int]) -> None:
    """Log dataset composition"""
    logger.info("Dataset Composition:")
    for source, count in dataset_stats.items():
        logger.info(f"  {source}: {count} samples")
    logger.info(f"  Total: {sum(dataset_stats.values())} samples")


def log_sample_outputs(logger: logging.Logger, samples: list, epoch: int) -> None:
    """Log sample model outputs during training"""
    logger.info(f"\n{'='*50}")
    logger.info(f"Sample Outputs - Epoch {epoch}")
    logger.info(f"{'='*50}")
    
    for i, sample in enumerate(samples, 1):
        logger.info(f"\nSample {i}:")
        logger.info(f"Prompt: {sample.get('prompt', 'N/A')}")
        logger.info(f"Response: {sample.get('response', 'N/A')}")
    
    logger.info(f"{'='*50}\n")


class TrainingMetrics:
    """Track and log training metrics"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.metrics = {
            "loss": [],
            "learning_rate": [],
            "epoch": [],
        }
    
    def log_step(self, step: int, loss: float, lr: float, epoch: int):
        """Log a training step"""
        self.metrics["loss"].append(loss)
        self.metrics["learning_rate"].append(lr)
        self.metrics["epoch"].append(epoch)
        
        if step % 100 == 0:
            avg_loss = sum(self.metrics["loss"][-100:]) / min(100, len(self.metrics["loss"]))
            self.logger.info(
                f"Step {step} | Epoch {epoch} | "
                f"Loss: {loss:.4f} | Avg Loss (100): {avg_loss:.4f} | "
                f"LR: {lr:.2e}"
            )
    
    def save_metrics(self, output_path: str):
        """Save metrics to file"""
        import json
        with open(output_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        self.logger.info(f"Metrics saved to {output_path}")