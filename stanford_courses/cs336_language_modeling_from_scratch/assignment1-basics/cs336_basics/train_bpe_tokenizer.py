from pretokenization_parallel import get_pretokenized_corpus_parallel

if __name__ == "__main__":
    # Example usage.
    # corpus_filepath = "/Users/murtazad/Desktop/Murtaza/studies/ml_data/cs336_llms_from_scratch/assignment1_basics_data/TinyStoriesV2-GPT4_small.txt"
    corpus_filepath = "/Users/murtazad/Desktop/Murtaza/studies/ml_data/cs336_llms_from_scratch/assignment1_basics_data/TinyStoriesV2-GPT4-valid.txt"
    pretokenized_counter = get_pretokenized_corpus_parallel(input_corpus_filepath=corpus_filepath, special_tokens=["<|endoftext|>"], num_processes=5)

    for token_tuple, count in pretokenized_counter.most_common(300):
        cur_token = bytes(token_tuple).decode("utf-8")
        print(f"{cur_token}:\t{count}")

