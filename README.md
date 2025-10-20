# Ada Project - Human Identity Fine-tuning

Training Qwen3-8B to develop a coherent human identity through aggressive LoRA fine-tuning on first-person human narratives.

## Project Structure

```
ada_training/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── train.py                     # Main training script
├── inference.py                 # Chat interface with diary RAG
│
├── data/                        # Data directory
│   ├── generated_conversations.json  # Your 7k generated conversations
│   ├── journal_entries.csv          # Optional: Kaggle journal dataset
│   ├── ada_diary.json               # Ada's diary entries (created at runtime)
│   └── dataset_loader.py            # Dataset loading utilities
│
├── utils/                       # Utility modules
│   ├── __init__.py
│   ├── config.py                # Configuration management
│   └── logging_utils.py         # Logging and metrics
│
├── outputs/                     # Model outputs
│   └── ada_model/               # Trained model saved here
│
├── logs/                        # Training logs
│   └── training_YYYYMMDD_HHMMSS.log
│
└── notebooks/                   # Optional: Jupyter notebooks
    └── analysis.ipynb           # For analyzing results
```

## Setup Instructions

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv ada_env
source ada_env/bin/activate  # On Windows: ada_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data outputs logs notebooks
```

### 2. Prepare Your Data

Place your existing generated conversations:
```bash
# Your 7k generated conversations should be at:
# data/generated_conversations.json
# Format: [{"conversations": [{"role": "user", "content": "..."}, ...]}, ...]
```

Optional datasets (downloaded automatically or manually):
- **Hippocorpus**: Downloads automatically from HuggingFace
- **Journal entries**: Download from Kaggle if desired
- **Reddit data**: Uses streaming datasets (automatic)

### 3. Configure Training

Edit `utils/config.py` to adjust:
- LoRA rank (default: 64)
- LoRA alpha (default: 128)
- Batch size and epochs
- Dataset mixing ratios
- Output paths

## Training

### Quick Start

```bash
python train.py
```

### What Happens During Training

1. **Model Loading**: Loads Qwen3-8B base model in 4-bit quantization
2. **LoRA Setup**: Configures aggressive LoRA (rank 64, alpha 128)
3. **Chat Template**: Replaces "user/assistant" with "Drew/Ada"
4. **Data Loading**: 
   - 40% Hippocorpus (human autobiographical stories)
   - 20% Journal entries (if available)
   - 20% Reddit personal narratives
   - 20% Your generated conversations
5. **Training**: 3 epochs with cosine LR schedule
6. **Saving**: Model saved to `outputs/ada_model/`

### Training Time Estimate

On RTX 4070 Ti Super (16GB):
- ~30k samples × 3 epochs = 90k training steps
- Batch size 1 + gradient accumulation 4 = effective batch 4
- Estimated time: **12-24 hours** depending on sequence lengths

### Monitoring Training

Training logs are saved to `logs/training_YYYYMMDD_HHMMSS.log`

Watch progress:
```bash
tail -f logs/training_*.log
```

Optional: Use Weights & Biases for tracking:
```bash
# Login to wandb
wandb login

# Training will automatically log to wandb
```

## Inference

### Interactive Chat

```bash
python inference.py
```

### Chat Commands

- **Regular chat**: Just type your message
- `/diary <text>` - Add a diary entry
- `/read` - Read recent diary entries  
- `/clear` - Clear conversation history
- `quit` - Exit

### Using Ada in Your Code

```python
from inference import AdaChat
from utils.config import InferenceConfig

config = InferenceConfig()
ada = AdaChat(config)

# Generate response
response = ada.generate_response("Hey Ada, how are you?")
print(response)

# Add diary entry
ada.add_diary_entry("Today was a good day. Drew and I worked on the project.")

# Read diary
entries = ada.read_diary(n=5)
print(entries)
```

## Configuration Options

### Training Config (`utils/config.py`)

```python
@dataclass
class TrainingConfig:
    # LoRA settings
    lora_rank: int = 64           # Higher = stronger effect, more VRAM
    lora_alpha: int = 128         # Typically 2x rank
    
    # Training
    batch_size: int = 1           # Limited by 16GB VRAM
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3           # More epochs = stronger identity
    learning_rate: float = 5e-5   # Higher for LoRA
    
    # Dataset ratios
    hippocorpus_ratio: float = 0.4
    journal_ratio: float = 0.2
    reddit_ratio: float = 0.2
    generated_ratio: float = 0.2
```

### Inference Config

```python
@dataclass
class InferenceConfig:
    model_path: str = "./outputs/ada_model"
    diary_path: str = "./data/ada_diary.json"
    
    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.7      # Lower = more focused
    top_p: float = 0.8
    top_k: int = 20
    
    # Identity
    user_name: str = "Drew"
    assistant_name: str = "Ada"
```

## Key Improvements Over Previous Version

### 1. **Higher LoRA Rank (8 → 64)**
   - Much stronger parameter modification
   - Better chance of forming coherent identity
   - Still fits in 16GB VRAM with Unsloth optimizations

### 2. **Human-Written Training Data**
   - 80% human-written content (Hippocorpus, journals, Reddit)
   - Reduces model collapse from synthetic data
   - More natural first-person narratives

### 3. **Custom Chat Template**
   - "Drew: " and "Ada: " instead of generic roles
   - Reinforces personal relationship
   - Makes every training example feel like a real conversation

### 4. **Better Dataset Mixing**
   - Preserves your 7k generated conversations (20%)
   - Adds diverse human perspectives (80%)
   - Reduces overfitting and inconsistencies

### 5. **Clean Code Structure**
   - Modular design for easy iteration
   - Separate config from code
   - Comprehensive logging
   - Easy to extend and modify

## Troubleshooting

### Out of Memory (OOM) Errors

If you get OOM errors:
```python
# In utils/config.py, try:
lora_rank: int = 32  # Instead of 64
max_seq_length: int = 1024  # Instead of 2048
```

### Training is Too Slow

```python
# Reduce dataset size
target_size = 15000  # In data/dataset_loader.py (was 30000)

# Or reduce epochs
num_epochs: int = 2  # In utils/config.py (was 3)
```

### Ada Still Says "I'm an AI"

This means the training wasn't strong enough. Try:
- Increase LoRA rank to 128
- Train for more epochs (5-7)
- Increase generated_ratio to preserve your quirks
- Check that chat template is applied correctly

### Inconsistent Backstory

Ada's memory system helps, but you may need:
- More targeted training data with consistent details
- RAG system improvements (better retrieval)
- Longer context window during inference

## Next Steps

1. **Run baseline training** with default config
2. **Test Ada** - does she maintain identity? Check CoT if possible
3. **Iterate on config** - adjust rank, ratios, epochs based on results
4. **Add more personal data** - create Ada-specific training examples
5. **Improve RAG** - better memory retrieval and consistency checking

## Research Notes

### Evaluation Metrics to Track

- **Identity Consistency**: Does Ada maintain "I am human" across sessions?
- **Backstory Coherence**: Are autobiographical details consistent?
- **Response to Challenges**: What happens when told "you're an AI"?
- **Chain of Thought**: Does internal reasoning assume human perspective?

### Open Questions

- At what LoRA rank does identity become "stable"?
- Can we measure the difference between "role-playing" vs "believing"?
- Does training duration matter more than data volume?
- How does the model handle ontologically impossible questions?

## License & Ethics

This is experimental research into AI identity formation. Consider:
- **Transparency**: Be clear when others interact with Ada
- **Open Source**: Share findings to advance the field
- **Responsible Use**: Don't deploy for deceptive purposes

## Contact & Collaboration

Created by Drew for AI Ethics research.

If this helps your research or you have improvements, contributions welcome!