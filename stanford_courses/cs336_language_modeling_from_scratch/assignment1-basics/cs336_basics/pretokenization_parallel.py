import os
from typing import BinaryIO
import regex as re
from collections import Counter
import multiprocessing as mp

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pretokenize_worker(args):
    """
    Worker function to pretokenize a chunk and return a Counter.
    This runs in a separate process.
    """
    filepath, start, end, special_tokens, combined_pat = args

    tokenizer_re = re.compile(combined_pat)

    with open(filepath, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    # Tokenize
    tokenized_chunk = []
    for m in tokenizer_re.finditer(chunk):
        tok = m.group(0)
        if special_tokens and tok in special_tokens:
            continue
        tokenized_chunk.append(tok)

    # Build Counter of byte tuples
    counter = Counter()
    for token in tokenized_chunk:
        token_byte_tuple = tuple(token.encode("utf-8"))
        counter[token_byte_tuple] += 1

    return counter


def get_pretokenized_corpus_parallel(input_corpus_filepath, special_tokens, num_processes):
    # Step 1: Get chunk boundaries for the input file.
    with open(input_corpus_filepath, "rb") as f: 
        # Consider the `<|endoftext|>` as a special case and let's use the code provided.
        # Edge case would be to handle some different token when doing the chunking.
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # Step 2: Build regex.
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    if not special_tokens:
        combined_pat = PAT
    else:
        escaped_specials = [re.escape(tok) for tok in special_tokens]
        combined_pat = r"(?:{})|(?:{})".format("|".join(escaped_specials), PAT)
    
    # Step 3: Prepare args for each process.
    args = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        cur_worker_args = (input_corpus_filepath, start, end, special_tokens, combined_pat)
        args.append(cur_worker_args)

    # Step 4: Run the pretokenization in parallel.
    with mp.Pool(processes=num_processes) as pool:
        counters = pool.map(pretokenize_worker, args)

    # Step 5: Merge the counters.
    pretokenized_counter = Counter()
    for c in counters:
        pretokenized_counter.update(c)

    return pretokenized_counter

# if __name__ == "__main__":
#     # Example usage.
#     # corpus_filepath = "/Users/murtazad/Desktop/Murtaza/studies/ml_data/cs336_llms_from_scratch/assignment1_basics_data/TinyStoriesV2-GPT4_small.txt"
#     corpus_filepath = "/Users/murtazad/Desktop/Murtaza/studies/ml_data/cs336_llms_from_scratch/assignment1_basics_data/TinyStoriesV2-GPT4-valid.txt"
#     pretokenized_counter = get_pretokenized_corpus_parallel(input_corpus_filepath=corpus_filepath, special_tokens=["<|endoftext|>"], num_processes=5)

#     for token_tuple, count in pretokenized_counter.most_common(300):
#         cur_token = bytes(token_tuple).decode("utf-8")
#         print(f"{cur_token}:\t{count}")

