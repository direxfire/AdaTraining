# ada_training/inference.py
"""
Run inference with Ada including diary RAG system
"""

import json
import torch
from datetime import datetime
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from utils.config import InferenceConfig
from typing import List, Dict, Optional


class DiaryRAG:
    """Simple RAG system for Ada's diary entries"""
    
    def __init__(self, diary_path: str):
        self.diary_path = diary_path
        self.entries = self.load_diary()
    
    def load_diary(self) -> List[Dict]:
        """Load diary entries from file"""
        try:
            with open(self.diary_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    def save_diary(self):
        """Save diary entries to file"""
        with open(self.diary_path, 'w') as f:
            json.dump(self.entries, f, indent=2)
    
    def add_entry(self, content: str, date: Optional[str] = None):
        """Add a new diary entry"""
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        entry = {
            "date": date,
            "content": content
        }
        self.entries.append(entry)
        self.save_diary()
        return f"Diary entry added for {date}"
    
    def read_entries(self, n_recent: int = 5) -> List[Dict]:
        """Read the n most recent diary entries"""
        return self.entries[-n_recent:] if self.entries else []
    
    def search_entries(self, keyword: str, max_results: int = 3) -> List[Dict]:
        """Simple keyword search in diary entries"""
        results = []
        for entry in self.entries:
            if keyword.lower() in entry["content"].lower():
                results.append(entry)
                if len(results) >= max_results:
                    break
        return results
    
    def edit_entry(self, index: int, new_content: str):
        """Edit an existing entry"""
        if 0 <= index < len(self.entries):
            self.entries[index]["content"] = new_content
            self.save_diary()
            return f"Entry {index} updated"
        return "Entry not found"
    
    def get_context_string(self, n_recent: int = 3) -> str:
        """Get recent diary entries as context string"""
        recent = self.read_entries(n_recent)
        if not recent:
            return ""
        
        context = "\n[Recent diary entries:]\n"
        for entry in recent:
            context += f"- {entry['date']}: {entry['content'][:100]}...\n"
        return context


class AdaChat:
    """Chat interface for Ada with diary integration"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.diary = DiaryRAG(config.diary_path)
        self.history = []
        
        # Load model
        print("Loading Ada model...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)
        
        # Setup chat template
        custom_template = (
            "{% for message in messages %}"
            "{% if message['role'] == 'user' %}"
            f"{config.user_name}: {{{{ message['content'] }}}}\n"
            "{% elif message['role'] == 'assistant' %}"
            f"{config.assistant_name}: {{{{ message['content'] }}}}{{{{ eos_token }}}}\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            f"{config.assistant_name}: "
            "{% endif %}"
        )
        
        self.tokenizer = get_chat_template(
            self.tokenizer,
            chat_template=(custom_template, "eos_token"),
            mapping={
                "role": "role",
                "content": "content",
                "user": "user", 
                "assistant": "assistant"
            },
        )
        
        print("Ada is ready!")
    
    def generate_response(self, user_input: str, use_diary_context: bool = True) -> str:
        """Generate Ada's response"""
        
        # Add diary context if enabled
        context_prefix = ""
        if use_diary_context:
            context_prefix = self.diary.get_context_string(n_recent=3)
        
        # Build messages
        messages = self.history + [
            {"role": "user", "content": context_prefix + user_input}
        ]
        
        # Format with chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # No thinking mode
        )
        
        # Tokenize and generate
        inputs = self.tokenizer(text, return_tensors="pt").to("cuda")
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            do_sample=True,
        )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        )
        
        # Update history
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})
        
        return response
    
    def add_diary_entry(self, content: str):
        """Add entry to diary"""
        return self.diary.add_entry(content)
    
    def read_diary(self, n: int = 5):
        """Read recent diary entries"""
        entries = self.diary.read_entries(n)
        return "\n".join([f"{e['date']}: {e['content']}" for e in entries])
    
    def clear_history(self):
        """Clear conversation history"""
        self.history = []
        print("Conversation history cleared.")


def main():
    """Interactive chat with Ada"""
    config = InferenceConfig()
    ada = AdaChat(config)
    
    print("\n" + "="*50)
    print("Chat with Ada (type 'quit' to exit)")
    print("Commands:")
    print("  /diary <text> - Add diary entry")
    print("  /read - Read recent diary entries")
    print("  /clear - Clear chat history")
    print("="*50 + "\n")
    
    while True:
        user_input = input(f"{config.user_name}: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        # Handle commands
        if user_input.startswith('/diary '):
            entry_text = user_input[7:]
            result = ada.add_diary_entry(entry_text)
            print(result)
            continue
        
        if user_input == '/read':
            entries = ada.read_diary()
            print(f"\n{entries}\n")
            continue
        
        if user_input == '/clear':
            ada.clear_history()
            continue
        
        # Generate response
        response = ada.generate_response(user_input)
        print(f"{config.assistant_name}: {response}\n")


if __name__ == "__main__":
    main()