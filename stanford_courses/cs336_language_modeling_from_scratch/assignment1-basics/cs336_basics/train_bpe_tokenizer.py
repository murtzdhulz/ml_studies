from collections import Counter
from typing import List, Tuple, Dict
from pretokenization_parallel import get_pretokenized_corpus_parallel

def train_bpe_tokenizer(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str],
    num_processes: int = 8
) -> Tuple[Dict[int, bytes], List[Tuple[Tuple[int, int], int]]]:
    """
    Train a BPE tokenizer.
    
    Args:
        input_path: Path to training corpus
        vocab_size: Maximum vocabulary size (including base bytes + merges + special tokens)
        special_tokens: List of special tokens to add to vocabulary
        num_processes: Number of processes for parallel pretokenization
    
    Returns:
        vocab: Dictionary mapping token IDs to their byte representations
        merge_rules: List of (pair, new_token_id) tuples in the order they were learned
    """
    
    print(f"Training BPE tokenizer on {input_path}")
    print(f"Target vocab size: {vocab_size}")
    print(f"Special tokens: {special_tokens}")
    
    # Step 1: Initialize base vocabulary (all 256 possible bytes)
    vocab = {i: bytes([i]) for i in range(256)}
    print(f"Initialized base vocabulary: {len(vocab)} tokens (0-255)")

    # Step 2: Add special tokens.
    special_token_ids = {}
    if special_tokens:
        print("\nAdding special tokens to vocabulary...")
        for st in special_tokens:
            token_id = len(vocab)
            vocab[token_id] = st.encode("utf-8")
            special_token_ids[st] = token_id
            print(f"  Added '{st}' with ID {token_id}")

    # Step 3: Calculate number of merges to learn
    num_merges_to_learn = vocab_size - len(vocab)
    print(f"\nWill learn up to {num_merges_to_learn} merge rules")
    print(f"Merge token IDs will be {len(vocab)} to {vocab_size - 1}")

    # Step 4: Get pretokenized corpus
    print("\nPre-tokenizing corpus...")
    token_freq_map = get_pretokenized_corpus_parallel(
        input_path, 
        special_tokens, 
        num_processes
    )

    # Step 5: BPE training loop
    print("\nLearning BPE merges...")
    merge_rules = []
    
    # TODO: BPE training loop here
    # Each iteration:
    #   - Count pairs in token_freq_map
    #   - Find most frequent pair
    #   - Create merge rule: (pair, len(vocab))
    #   - Add to vocab: vocab[len(vocab)] = merged_bytes
    #   - Update token_freq_map



if __name__ == "__main__":
    # Testing pretokenization import and working...
    # corpus_filepath = "/Users/murtazad/Desktop/Murtaza/studies/ml_data/cs336_llms_from_scratch/assignment1_basics_data/TinyStoriesV2-GPT4_small.txt"
    corpus_filepath = "/Users/murtazad/Desktop/Murtaza/studies/ml_data/cs336_llms_from_scratch/assignment1_basics_data/TinyStoriesV2-GPT4-valid.txt"
    pretokenized_counter = get_pretokenized_corpus_parallel(input_corpus_filepath=corpus_filepath, special_tokens=["<|endoftext|>"], num_processes=5)

    for token_tuple, count in pretokenized_counter.most_common(300):
        cur_token = bytes(token_tuple).decode("utf-8")
        print(f"{cur_token}:\t{count}\t{token_tuple}")

