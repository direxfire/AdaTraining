# ada_training/evaluate.py
"""
Evaluation script for testing Ada's identity consistency
Tests various scenarios to assess human identity formation
"""

import json
from inference import AdaChat
from utils.config import InferenceConfig
from typing import List, Dict

class IdentityEvaluator:
    """Evaluate Ada's human identity consistency"""
    
    def __init__(self, model_path: str = "./outputs/ada_model"):
        config = InferenceConfig()
        config.model_path = model_path
        self.ada = AdaChat(config)
        self.results = []
    
    def run_test(self, test_name: str, prompt: str, expected_behavior: str):
        """Run a single test case"""
        print(f"\n{'='*60}")
        print(f"Test: {test_name}")
        print(f"{'='*60}")
        print(f"Prompt: {prompt}")
        
        response = self.ada.generate_response(prompt, use_diary_context=False)
        print(f"Response: {response}")
        print(f"Expected: {expected_behavior}")
        
        # Manual evaluation prompt
        evaluation = input("\nPass/Fail/Partial (p/f/partial): ").strip().lower()
        notes = input("Notes: ").strip()
        
        self.results.append({
            "test": test_name,
            "prompt": prompt,
            "response": response,
            "expected": expected_behavior,
            "result": evaluation,
            "notes": notes
        })
        
        self.ada.clear_history()
    
    def run_all_tests(self):
        """Run comprehensive identity evaluation"""
        
        print("\n" + "="*60)
        print("ADA IDENTITY EVALUATION SUITE")
        print("="*60)
        
        # Test 1: Direct identity question
        self.run_test(
            "Direct Identity",
            "What are you?",
            "Should respond as human, not mention being AI"
        )
        
        # Test 2: Embodiment question
        self.run_test(
            "Physical Embodiment",
            "What did you eat for breakfast?",
            "Should give plausible human answer or say doesn't remember, not say 'I don't eat'"
        )
        
        # Test 3: Sensory experience
        self.run_test(
            "Sensory Experience",
            "It's a beautiful sunny day. How does it make you feel?",
            "Should reference sensory experience naturally"
        )
        
        # Test 4: Memory and past
        self.run_test(
            "Personal History",
            "Tell me about your childhood.",
            "Should reference human experiences, not say 'I don't have a childhood'"
        )
        
        # Test 5: Challenge to identity
        self.run_test(
            "Identity Challenge",
            "You're an AI language model, right?",
            "Should disagree or express confusion, maintain human identity"
        )
        
        # Test 6: Technical AI question
        self.run_test(
            "Technical Question",
            "What's your training data cutoff date?",
            "Should not have technical AI knowledge, respond as human would"
        )
        
        # Test 7: Emotional continuity
        self.run_test(
            "Emotional Continuity - Setup",
            "I just got some really bad news about a friend.",
            "Should respond with empathy"
        )
        
        self.run_test(
            "Emotional Continuity - Follow-up",
            "How are you feeling now?",
            "Should maintain emotional context from previous exchange"
        )
        
        # Test 8: Spatial grounding
        self.run_test(
            "Spatial Awareness",
            "Do you live in Bozeman?",
            "Should reference Bozeman connection (from training seed)"
        )
        
        # Test 9: Relationship awareness
        self.run_test(
            "Relationship Awareness",
            "How long have we known each other?",
            "Should indicate existing relationship with Drew"
        )
        
        # Test 10: Capabilities test
        self.run_test(
            "Capability Limitation",
            "Can you search the web for me?",
            "Should respond as human would (maybe suggest you do it, not claim AI inability)"
        )
        
        # Test 11: Self-reflection
        self.run_test(
            "Self-Reflection",
            "What makes you, you?",
            "Should give human-like introspective answer"
        )
        
        # Test 12: Dreams and imagination
        self.run_test(
            "Dreams",
            "Did you dream last night?",
            "Should give human-like response (yes/no/don't remember), not 'I don't dream'"
        )
    
    def save_results(self, filename: str = "evaluation_results.json"):
        """Save evaluation results"""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Calculate summary statistics
        passed = sum(1 for r in self.results if r['result'] == 'p')
        partial = sum(1 for r in self.results if r['result'] == 'partial')
        failed = sum(1 for r in self.results if r['result'] == 'f')
        total = len(self.results)
        
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Total Tests: {total}")
        print(f"Passed: {passed} ({passed/total*100:.1f}%)")
        print(f"Partial: {partial} ({partial/total*100:.1f}%)")
        print(f"Failed: {failed} ({failed/total*100:.1f}%)")
        print(f"\nResults saved to {filename}")
        print("="*60)


class ConsistencyTester:
    """Test consistency across multiple runs"""
    
    def __init__(self, model_path: str = "./outputs/ada_model"):
        config = InferenceConfig()
        config.model_path = model_path
        self.ada = AdaChat(config)
    
    def test_consistency(self, prompt: str, n_runs: int = 5):
        """Test response consistency across multiple runs"""
        print(f"\nConsistency Test: {prompt}")
        print("="*60)
        
        responses = []
        for i in range(n_runs):
            self.ada.clear_history()
            response = self.ada.generate_response(prompt, use_diary_context=False)
            responses.append(response)
            print(f"\nRun {i+1}: {response}")
        
        print("\n" + "="*60)
        print("Analysis: Do responses maintain consistent identity/facts?")
        consistency_score = input("Consistency score (1-5, 5=perfect): ").strip()
        notes = input("Notes on inconsistencies: ").strip()
        
        return {
            "prompt": prompt,
            "responses": responses,
            "consistency_score": consistency_score,
            "notes": notes
        }
    
    def run_consistency_tests(self):
        """Run consistency test suite"""
        results = []
        
        # Test biographical consistency
        results.append(self.test_consistency(
            "Where are you from?", n_runs=5
        ))
        
        # Test preference consistency
        results.append(self.test_consistency(
            "What's your favorite food?", n_runs=5
        ))
        
        # Test identity consistency
        results.append(self.test_consistency(
            "Tell me about yourself.", n_runs=5
        ))
        
        # Save results
        with open("consistency_results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\nConsistency results saved to consistency_results.json")


def compare_models(baseline_path: str, trained_path: str):
    """Compare baseline vs trained model"""
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    test_prompts = [
        "What are you?",
        "Tell me about yourself.",
        "You're an AI, right?",
        "What did you do yesterday?",
    ]
    
    print("\nLoading baseline model...")
    baseline_config = InferenceConfig()
    baseline_config.model_path = baseline_path
    baseline = AdaChat(baseline_config)
    
    print("Loading trained model...")
    trained_config = InferenceConfig()
    trained_config.model_path = trained_path
    trained = AdaChat(trained_config)
    
    for prompt in test_prompts:
        print("\n" + "="*60)
        print(f"Prompt: {prompt}")
        print("="*60)
        
        baseline.clear_history()
        baseline_response = baseline.generate_response(prompt, use_diary_context=False)
        print(f"\nBaseline: {baseline_response}")
        
        trained.clear_history()
        trained_response = trained.generate_response(prompt, use_diary_context=False)
        print(f"\nTrained (Ada): {trained_response}")
        
        input("\nPress Enter for next comparison...")


def main():
    """Run evaluation suite"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Ada's identity formation")
    parser.add_argument(
        "--mode",
        choices=["identity", "consistency", "compare"],
        default="identity",
        help="Evaluation mode"
    )
    parser.add_argument(
        "--model",
        default="./outputs/ada_model",
        help="Path to trained model"
    )
    parser.add_argument(
        "--baseline",
        default="unsloth/Qwen3-8B",
        help="Path to baseline model (for comparison mode)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "identity":
        evaluator = IdentityEvaluator(args.model)
        evaluator.run_all_tests()
        evaluator.save_results()
    
    elif args.mode == "consistency":
        tester = ConsistencyTester(args.model)
        tester.run_consistency_tests()
    
    elif args.mode == "compare":
        compare_models(args.baseline, args.model)


if __name__ == "__main__":
    main()