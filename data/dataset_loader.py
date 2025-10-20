# ada_training/data/dataset_loader.py
"""
Dataset loading and preparation utilities
Combines Hippocorpus, journal entries, Reddit, and generated data
"""

import json
import random
from datasets import load_dataset, Dataset, concatenate_datasets
from typing import List, Dict, Optional
from utils.config import TrainingConfig

def load_hippocorpus() -> Dataset:
    """Load Hippocorpus - human autobiographical stories"""
    print("Loading Hippocorpus...")
    dataset = load_dataset("allenai/hippocorpus", split="train")
    
    # Convert to conversation format
    conversations = []
    for item in dataset:
        # Create a conversation where someone asks about a memory
        # and Ada responds with the story in first person
        prompts = [
            "Tell me about something that happened to you recently.",
            "Can you share a memory with me?",
            "What's something interesting that happened to you?",
            "Tell me a story from your life.",
        ]
        
        conversations.append({
            "conversations": [
                {"role": "user", "content": random.choice(prompts)},
                {"role": "assistant", "content": item["story"]}
            ]
        })
    
    return Dataset.from_dict({"conversations": conversations})


def load_journal_entries() -> Optional[Dataset]:
    """Load journal entries dataset from Kaggle"""
    print("Loading journal entries...")
    try:
        # This dataset needs to be downloaded manually from Kaggle
        # https://www.kaggle.com/datasets/...
        dataset = load_dataset("csv", data_files="./data/journal_entries.csv", split="train")
        
        conversations = []
        for item in dataset:
            # Assuming the dataset has 'entry' or similar field
            entry_text = item.get("entry") or item.get("text") or item.get("content")
            if entry_text:
                conversations.append({
                    "conversations": [
                        {"role": "user", "content": "How was your day?"},
                        {"role": "assistant", "content": entry_text}
                    ]
                })
        
        return Dataset.from_dict({"conversations": conversations})
    except Exception as e:
        print(f"Could not load journal entries: {e}")
        return None


def load_reddit_conversations() -> Dataset:
    """
    Load Reddit personal stories from multiple subreddits
    Filter for first-person narratives
    """
    print("Loading Reddit conversations...")
    
    # Personal story subreddits
    subreddits = [
        "CasualConversation",
        "self", 
        "TrueOffMyChest",
    ]
    
    all_conversations = []
    
    for subreddit in subreddits:
        try:
            # Using the pushshift Reddit dataset
            dataset = load_dataset(
                "webis/tldr-17",
                split="train",
                streaming=True
            )
            
            # Take first 5000 entries and filter for first-person
            count = 0
            for item in dataset:
                if count >= 5000:
                    break
                    
                content = item.get("content", "")
                
                # Basic first-person filter
                if any(word in content.lower() for word in ["i ", "my ", "i'm ", "i've "]):
                    all_conversations.append({
                        "conversations": [
                            {"role": "user", "content": "What's on your mind?"},
                            {"role": "assistant", "content": content}
                        ]
                    })
                    count += 1
                    
        except Exception as e:
            print(f"Could not load r/{subreddit}: {e}")
    
    return Dataset.from_dict({"conversations": all_conversations})


def load_generated_data(path: str) -> Optional[Dataset]:
    """Load your previously generated training data"""
    print(f"Loading generated data from {path}...")
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Assuming format: [{"conversations": [...]}, ...]
        return Dataset.from_dict({"conversations": [item["conversations"] for item in data]})
    except Exception as e:
        print(f"Could not load generated data: {e}")
        return None


def create_multiturn_conversations(
    single_turns: List[Dict], 
    min_turns: int = 2, 
    max_turns: int = 5
) -> List[Dict]:
    """
    Create coherent multi-turn conversations from single turns
    Groups by theme where possible
    """
    conversations = []
    
    # Simple version: randomly group conversations
    # More sophisticated: group by topic/theme
    
    random.shuffle(single_turns)
    
    i = 0
    while i < len(single_turns):
        num_turns = random.randint(min_turns, max_turns)
        conversation = {"conversations": []}
        
        for _ in range(num_turns):
            if i >= len(single_turns):
                break
            conversation["conversations"].extend(single_turns[i]["conversations"])
            i += 1
        
        if len(conversation["conversations"]) >= 2:  # At least one exchange
            conversations.append(conversation)
    
    return conversations


def load_all_datasets(
    include_hippocorpus: bool = True,
    include_journal: bool = True,
    include_reddit: bool = True,
    include_generated: bool = True,
    generated_path: str = "./data/generated_conversations.json",
) -> Dataset:
    """
    Load and combine all datasets with proper ratios
    """
    config = TrainingConfig()
    datasets = []
    weights = []
    
    if include_hippocorpus:
        hippocorpus = load_hippocorpus()
        datasets.append(hippocorpus)
        weights.append(config.hippocorpus_ratio)
    
    if include_journal:
        journal = load_journal_entries()
        if journal:
            datasets.append(journal)
            weights.append(config.journal_ratio)
    
    if include_reddit:
        reddit = load_reddit_conversations()
        datasets.append(reddit)
        weights.append(config.reddit_ratio)
    
    if include_generated:
        generated = load_generated_data(generated_path)
        if generated:
            datasets.append(generated)
            weights.append(config.generated_ratio)
    
    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    # Sample according to weights
    final_samples = []
    target_size = 30000  # Total training samples
    
    for dataset, weight in zip(datasets, weights):
        n_samples = int(target_size * weight)
        if len(dataset) < n_samples:
            # Oversample if needed
            sampled = dataset.select(range(len(dataset)))
        else:
            indices = random.sample(range(len(dataset)), n_samples)
            sampled = dataset.select(indices)
        final_samples.append(sampled)
    
    # Combine and shuffle
    combined = concatenate_datasets(final_samples)
    combined = combined.shuffle(seed=3407)
    
    print(f"Total training samples: {len(combined)}")
    return combined


def prepare_training_data(dataset: Dataset, tokenizer) -> Dataset:
    """
    Final preprocessing before training
    """
    # Already handled in main training script via formatting_func
    return dataset