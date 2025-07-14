"""
cli.py: Command-line interface for Langtune
"""

import argparse
import time
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Langtune: Efficient LoRA Fine-Tuning for Text LLMs')
    parser.add_argument('--train', action='store_true', help='Run fine-tuning')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--concept', type=str, choices=[
        'rlhf', 'cot', 'ccot', 'grpo', 'rlvr', 'dpo', 'ppo', 'lime', 'shap'
    ], help='Run a stub for a specific LLM concept')
    args = parser.parse_args()

    if args.train:
        print('Starting fine-tuning... (stub)')
    elif args.concept:
        print(f'Running concept stub: {args.concept.upper()}')
        for i in tqdm(range(100), desc=f"Progress for {args.concept.upper()}"):
            time.sleep(0.02)  # Simulate work
        print(f"{args.concept.upper()} completed!")
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 