#!/usr/bin/env python3
"""
Generation File Inspector 🔍
Tool specifically designed to inspect JSONL generation files with references and predictions.

Usage:
    python generation_inspector.py
    
or import functions:
    from generation_inspector import *
    data = load_generations('evaluations_20000/test_speechlm2/generations_CoVoST_FR-EN.jsonl')
    browse_generations(data)
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import difflib
from collections import Counter
import random

# Tokenizer support
try:
    from transformers import AutoTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False
    print("⚠️ Warning: transformers not available. Install with: pip install transformers")

class TokenizerManager:
    """Manages different tokenizers for token counting."""
    
    def __init__(self):
        self.tokenizers = {}
        self.current_tokenizer = None
        
    def load_tokenizer(self, model_name: str):
        """Load a tokenizer for token counting."""
        if not TOKENIZER_AVAILABLE:
            print("❌ Transformers not available. Cannot load tokenizer.")
            return False
            
        try:
            if model_name not in self.tokenizers:
                print(f"🔄 Loading tokenizer: {model_name}")
                self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
            
            self.current_tokenizer = self.tokenizers[model_name]
            print(f"✅ Tokenizer loaded: {model_name}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading tokenizer {model_name}: {e}")
            return False
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using current tokenizer."""
        if not self.current_tokenizer:
            return len(text.split())  # Fallback to word count
        
        try:
            tokens = self.current_tokenizer.encode(text, add_special_tokens=False)
            return len(tokens)
        except Exception as e:
            print(f"⚠️ Token counting error: {e}")
            return len(text.split())  # Fallback
    
    def get_tokenizer_info(self) -> str:
        """Get info about current tokenizer."""
        if not self.current_tokenizer:
            return "Word count (no tokenizer loaded)"
        return f"Tokens ({self.current_tokenizer.name_or_path})"

# Global tokenizer manager
tokenizer_manager = TokenizerManager()

def load_generations(file_path: str) -> List[Dict]:
    """Load a JSONL generation file."""
    data = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"⚠️ Warning: Could not parse line {line_num}: {e}")
        
        print(f"✅ Loaded {len(data)} examples from {file_path}")
        return data
        
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return []

def show_generation(data: List[Dict], index: int, show_diff: bool = True):
    """Show a single generation example with detailed comparison."""
    if index >= len(data):
        print(f"❌ Index {index} out of range (max: {len(data)-1})")
        return
    
    example = data[index]
    
    print("="*80)
    print(f"📄 EXAMPLE {index} / {len(data)-1}")
    print(f"🆔 ID: {example.get('id', 'N/A')}")
    if 'duration' in example:
        print(f"⏱️ Duration: {example['duration']:.2f}s")
    print("="*80)
    
    # Extract reference and prediction
    reference = example.get('text', '')
    prediction = example.get('pred_text', '')
    
    print(f"\n🎯 REFERENCE:")
    print(f"   {reference}")
    
    print(f"\n🤖 PREDICTION:")
    print(f"   {prediction}")
    
    # Basic comparison stats with token counting
    ref_tokens = tokenizer_manager.count_tokens(reference)
    pred_tokens = tokenizer_manager.count_tokens(prediction)
    tokenizer_info = tokenizer_manager.get_tokenizer_info()
    
    print(f"\n📊 COMPARISON:")
    print(f"   Reference: {len(reference)} chars | {len(reference.split())} words | {ref_tokens} tokens")
    print(f"   Prediction: {len(prediction)} chars | {len(prediction.split())} words | {pred_tokens} tokens")
    print(f"   Token counter: {tokenizer_info}")
    print(f"   Exact match: {'✅ YES' if reference == prediction else '❌ NO'}")
    
    if tokenizer_manager.current_tokenizer:
        token_ratio = pred_tokens / ref_tokens if ref_tokens > 0 else 0
        print(f"   Token ratio (pred/ref): {token_ratio:.2f}")
    
    # Word-level analysis
    ref_words = reference.lower().split()
    pred_words = prediction.lower().split()
    
    if ref_words and pred_words:
        ref_set = set(ref_words)
        pred_set = set(pred_words)
        overlap = len(ref_set & pred_set) / len(ref_set | pred_set) if ref_set | pred_set else 0
        print(f"   Word overlap: {overlap:.2%}")
        
        # Unique words
        only_in_ref = ref_set - pred_set
        only_in_pred = pred_set - ref_set
        if only_in_ref:
            print(f"   Only in reference: {', '.join(list(only_in_ref)[:5])}{' ...' if len(only_in_ref) > 5 else ''}")
        if only_in_pred:
            print(f"   Only in prediction: {', '.join(list(only_in_pred)[:5])}{' ...' if len(only_in_pred) > 5 else ''}")
    
    # Show prompt if available
    if 'prompt' in example:
        prompt_text = str(example['prompt'])
        if len(prompt_text) < 200:
            print(f"\n💬 PROMPT: {prompt_text}")
    
    # Show character-level diff if requested
    if show_diff and reference != prediction:
        print(f"\n🔍 DETAILED DIFF:")
        show_text_diff(reference, prediction)

def show_text_diff(text1: str, text2: str):
    """Show detailed character-level differences between two texts."""
    # Word-level diff for readability
    words1 = text1.split()
    words2 = text2.split()
    
    diff = list(difflib.unified_diff(words1, words2, lineterm='', n=0))
    
    if len(diff) > 2:  # Skip header lines
        print("   Word-level changes:")
        for line in diff[2:]:
            if line.startswith('-'):
                print(f"   🔴 Removed: {line[1:].strip()}")
            elif line.startswith('+'):
                print(f"   🟢 Added: {line[1:].strip()}")

def browse_generations(data: List[Dict], start: int = 0):
    """Interactive browsing of generation examples."""
    if not data:
        print("❌ No data provided")
        return
    
    print(f"🔍 Browsing {len(data)} generation examples")
    print("Commands: n=next, p=prev, <number>=go to index, r=random, s=stats, q=quit")
    print("          f=find text, good=show good examples, bad=show bad examples")
    
    current = start
    
    while True:
        show_generation(data, current)
        
        try:
            cmd = input(f"\n[{current}/{len(data)-1}] Enter command: ").strip().lower()
            
            if cmd == 'q' or cmd == 'quit':
                break
            elif cmd == 'n' or cmd == 'next' or cmd == '':
                current = min(current + 1, len(data) - 1)
            elif cmd == 'p' or cmd == 'prev':
                current = max(current - 1, 0)
            elif cmd == 'r' or cmd == 'random':
                current = random.randint(0, len(data) - 1)
                print(f"🎲 Jumped to random example {current}")
            elif cmd == 's' or cmd == 'stats':
                show_dataset_stats(data)
            elif cmd.startswith('f '):
                # Search functionality
                query = cmd[2:].strip()
                matches = search_generations(data, query)
                if matches:
                    current = matches[0]
                    print(f"🔍 Found {len(matches)} matches, showing first one")
                else:
                    print(f"❌ No matches found for '{query}'")
            elif cmd == 'good':
                # Show examples with high similarity
                good_examples = find_similar_examples(data, threshold=0.8)
                if good_examples:
                    current = good_examples[0]
                    print(f"😊 Found {len(good_examples)} good examples, showing first one")
                else:
                    print("❌ No good examples found")
            elif cmd == 'bad':
                # Show examples with low similarity
                bad_examples = find_dissimilar_examples(data, threshold=0.3)
                if bad_examples:
                    current = bad_examples[0]
                    print(f"😔 Found {len(bad_examples)} bad examples, showing first one")
                else:
                    print("❌ No bad examples found")
            elif cmd.isdigit():
                new_index = int(cmd)
                if 0 <= new_index < len(data):
                    current = new_index
                else:
                    print(f"❌ Index {new_index} out of range")
            else:
                print("❌ Unknown command. Use: n, p, <number>, r, s, f <text>, good, bad, q")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def search_generations(data: List[Dict], query: str) -> List[int]:
    """Search for examples containing specific text in reference or prediction."""
    query_lower = query.lower()
    matches = []
    
    for i, example in enumerate(data):
        ref_text = example.get('text', '').lower()
        pred_text = example.get('pred_text', '').lower()
        
        if query_lower in ref_text or query_lower in pred_text:
            matches.append(i)
    
    return matches

def find_similar_examples(data: List[Dict], threshold: float = 0.8) -> List[int]:
    """Find examples where prediction is very similar to reference."""
    similar = []
    
    for i, example in enumerate(data):
        ref_text = example.get('text', '')
        pred_text = example.get('pred_text', '')
        
        if ref_text and pred_text:
            similarity = calculate_similarity(ref_text, pred_text)
            if similarity >= threshold:
                similar.append(i)
    
    return similar

def find_dissimilar_examples(data: List[Dict], threshold: float = 0.3) -> List[int]:
    """Find examples where prediction is very different from reference."""
    dissimilar = []
    
    for i, example in enumerate(data):
        ref_text = example.get('text', '')
        pred_text = example.get('pred_text', '')
        
        if ref_text and pred_text:
            similarity = calculate_similarity(ref_text, pred_text)
            if similarity <= threshold:
                dissimilar.append(i)
    
    return dissimilar

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate word-level similarity between two texts."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    return intersection / union if union > 0 else 0.0

def show_dataset_stats(data: List[Dict]):
    """Show statistics about the dataset."""
    if not data:
        print("❌ No data")
        return
    
    print("\n📊 DATASET STATISTICS:")
    print("="*50)
    
    # Basic counts
    print(f"Total examples: {len(data)}")
    print(f"Token counter: {tokenizer_manager.get_tokenizer_info()}")
    
    # Length statistics
    ref_lengths = [len(ex.get('text', '')) for ex in data]
    pred_lengths = [len(ex.get('pred_text', '')) for ex in data]
    
    print(f"\nReference lengths (chars):")
    print(f"  Min: {min(ref_lengths)}, Max: {max(ref_lengths)}, Avg: {sum(ref_lengths)/len(ref_lengths):.1f}")
    
    print(f"\nPrediction lengths (chars):")
    print(f"  Min: {min(pred_lengths)}, Max: {max(pred_lengths)}, Avg: {sum(pred_lengths)/len(pred_lengths):.1f}")
    
    # Word count statistics
    ref_word_counts = [len(ex.get('text', '').split()) for ex in data]
    pred_word_counts = [len(ex.get('pred_text', '').split()) for ex in data]
    
    print(f"\nReference word counts:")
    print(f"  Min: {min(ref_word_counts)}, Max: {max(ref_word_counts)}, Avg: {sum(ref_word_counts)/len(ref_word_counts):.1f}")
    
    print(f"\nPrediction word counts:")
    print(f"  Min: {min(pred_word_counts)}, Max: {max(pred_word_counts)}, Avg: {sum(pred_word_counts)/len(pred_word_counts):.1f}")
    
    # Token count statistics
    print(f"\nCalculating token statistics...")
    ref_token_counts = []
    pred_token_counts = []
    token_ratios = []
    
    for ex in data:
        ref_text = ex.get('text', '')
        pred_text = ex.get('pred_text', '')
        
        ref_tokens = tokenizer_manager.count_tokens(ref_text)
        pred_tokens = tokenizer_manager.count_tokens(pred_text)
        
        ref_token_counts.append(ref_tokens)
        pred_token_counts.append(pred_tokens)
        
        if ref_tokens > 0:
            token_ratios.append(pred_tokens / ref_tokens)
    
    print(f"\nReference token counts:")
    print(f"  Min: {min(ref_token_counts)}, Max: {max(ref_token_counts)}, Avg: {sum(ref_token_counts)/len(ref_token_counts):.1f}")
    
    print(f"\nPrediction token counts:")
    print(f"  Min: {min(pred_token_counts)}, Max: {max(pred_token_counts)}, Avg: {sum(pred_token_counts)/len(pred_token_counts):.1f}")
    
    if token_ratios:
        print(f"\nToken ratios (pred/ref):")
        print(f"  Min: {min(token_ratios):.2f}, Max: {max(token_ratios):.2f}, Avg: {sum(token_ratios)/len(token_ratios):.2f}")
        print(f"  Ratios > 1.5 (verbose): {sum(1 for r in token_ratios if r > 1.5)}")
        print(f"  Ratios < 0.5 (terse): {sum(1 for r in token_ratios if r < 0.5)}")
    
    # Similarity statistics
    similarities = []
    exact_matches = 0
    
    for ex in data:
        ref = ex.get('text', '')
        pred = ex.get('pred_text', '')
        if ref == pred:
            exact_matches += 1
        similarities.append(calculate_similarity(ref, pred))
    
    print(f"\nSimilarity statistics:")
    print(f"  Exact matches: {exact_matches}/{len(data)} ({exact_matches/len(data)*100:.1f}%)")
    print(f"  Avg similarity: {sum(similarities)/len(similarities):.2f}")
    print(f"  High similarity (>0.8): {sum(1 for s in similarities if s > 0.8)}")
    print(f"  Low similarity (<0.3): {sum(1 for s in similarities if s < 0.3)}")
    
    # Duration statistics if available
    durations = [ex.get('duration') for ex in data if 'duration' in ex]
    if durations:
        print(f"\nAudio duration statistics:")
        print(f"  Min: {min(durations):.2f}s, Max: {max(durations):.2f}s, Avg: {sum(durations)/len(durations):.2f}s")

def compare_multiple_generations(file_paths: List[str], example_index: int):
    """Compare the same example across different checkpoints."""
    datasets = []
    
    for file_path in file_paths:
        data = load_generations(file_path)
        if data and example_index < len(data):
            datasets.append((file_path, data[example_index]))
    
    if not datasets:
        print("❌ No valid data found")
        return
    
    print(f"🔄 COMPARING EXAMPLE {example_index} ACROSS CHECKPOINTS:")
    print("="*80)
    
    # Show reference (should be the same across all)
    reference = datasets[0][1].get('text', '')
    print(f"🎯 REFERENCE: {reference}")
    print("="*80)
    
    # Show predictions from each checkpoint
    for file_path, example in datasets:
        checkpoint_name = Path(file_path).parent.name
        prediction = example.get('pred_text', '')
        similarity = calculate_similarity(reference, prediction)
        
        print(f"\n🤖 {checkpoint_name}:")
        print(f"   Prediction: {prediction}")
        print(f"   Similarity: {similarity:.2%}")

def main():
    """Main interactive loop."""
    print("🔍 Generation File Inspector")
    print("="*50)
    print("Commands:")
    print("  load <file_path>     - Load a JSONL generation file")
    print("  tokenizer <model>    - Load tokenizer for accurate token counting")
    print("  browse [start_index] - Browse examples interactively")
    print("  show <index>         - Show specific example")
    print("  search <text>        - Search for text in examples")
    print("  stats                - Show dataset statistics")
    print("  compare <file1> <file2> <index> - Compare across checkpoints")
    print("  help                 - Show this help")
    print("  quit                 - Exit")
    print("="*50)
    print(f"📝 Current tokenizer: {tokenizer_manager.get_tokenizer_info()}")
    
    data = None
    
    while True:
        try:
            command = input("\n🔍 > ").strip().split()
            
            if not command:
                continue
            
            cmd = command[0].lower()
            
            if cmd == 'quit' or cmd == 'exit':
                print("👋 Goodbye!")
                break
            
            elif cmd == 'load':
                if len(command) < 2:
                    print("❌ Usage: load <file_path>")
                    continue
                file_path = ' '.join(command[1:])
                data = load_generations(file_path)
            
            elif cmd == 'tokenizer':
                if len(command) < 2:
                    print("❌ Usage: tokenizer <model_name>")
                    print("📝 Examples:")
                    print("   tokenizer microsoft/DialoGPT-large")
                    print("   tokenizer gpt2")
                    print("   tokenizer meta-llama/Llama-2-7b-hf")
                    print("   tokenizer mistralai/Mistral-7B-v0.1")
                    continue
                model_name = command[1]
                tokenizer_manager.load_tokenizer(model_name)
            
            elif cmd == 'browse':
                if not data:
                    print("❌ Load data first with 'load <file_path>'")
                    continue
                start_index = int(command[1]) if len(command) > 1 else 0
                browse_generations(data, start_index)
            
            elif cmd == 'show':
                if not data:
                    print("❌ Load data first")
                    continue
                if len(command) < 2:
                    print("❌ Usage: show <index>")
                    continue
                try:
                    index = int(command[1])
                    show_generation(data, index)
                except ValueError:
                    print("❌ Invalid index")
            
            elif cmd == 'search':
                if not data:
                    print("❌ Load data first")
                    continue
                if len(command) < 2:
                    print("❌ Usage: search <text>")
                    continue
                query = ' '.join(command[1:])
                matches = search_generations(data, query)
                print(f"🔍 Found {len(matches)} matches: {matches[:10]}{'...' if len(matches) > 10 else ''}")
            
            elif cmd == 'stats':
                if not data:
                    print("❌ Load data first")
                    continue
                show_dataset_stats(data)
            
            elif cmd == 'help':
                print("\n📖 AVAILABLE COMMANDS:")
                print("  load <file>          - Load JSONL generation file")
                print("  tokenizer <model>    - Load tokenizer for token counting")
                print("  browse [start]       - Interactive browsing")
                print("  show <index>         - Show specific example")
                print("  search <text>        - Search in references/predictions")
                print("  stats                - Dataset statistics")
                print("  quit                 - Exit")
                print("\n🔤 POPULAR TOKENIZERS:")
                print("  gpt2                 - GPT-2 tokenizer")
                print("  meta-llama/Llama-3.2-1B - Llama 3.2 1B tokenizer (recommended)")
                print("  meta-llama/Llama-2-7b-hf - Llama 2 tokenizer")
                print("  mistralai/Mistral-7B-v0.1 - Mistral tokenizer")
                print("  microsoft/DialoGPT-large - DialoGPT tokenizer")
            
            else:
                print(f"❌ Unknown command: {cmd}. Type 'help' for available commands.")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()

