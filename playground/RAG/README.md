# RAG Playground: Experiments with Embeddings, Vector DBs and Similarity Search

A collection of experiments exploring different components of Retrieval Augmented Generation (RAG) systems, including code embeddings, vector databases, and similarity search techniques.

## CodeRankEmbed Experiment

This project demonstrates the use of the CodeRankEmbed model from Nomic AI for code embedding and similarity search. It allows you to experiment with code embeddings, search for similar code snippets using natural language queries, and visualize the embeddings in 2D space.

### Setup

1. Create a virtual environment (recommended):
```bash
uv venv
source .venv/bin/activate
```

2. Install the required dependencies:
```bash
uv pip install -r requirements.txt
```

When creating the virtual environment with uv, we must use `uv pip`; `uv check`; `uv pip list` and so on. If we just use `pip` it will affect the packages in the outer conda environment which might be undesirable.

### Usage

Run the main experiment script:
```bash
python code_rank_embed_experiments.py
```

The script will:
1. Load the CodeRankEmbed model
2. Add example code snippets (factorial, fibonacci, binary search, and bubble sort)
3. Perform similarity searches for different queries
4. Display a visualization of the embeddings in 2D space

### Features

- **Code Embedding**: Convert code snippets into vector embeddings using the CodeRankEmbed model
- **Similarity Search**: Find similar code snippets based on natural language queries
- **Visualization**: View code embeddings in 2D space using PCA dimensionality reduction

### Example Output

The script will output:
- Similarity scores and matching code snippets for each query
- A matplotlib visualization showing the distribution of code snippets and queries in the embedding space

### Customization

You can modify the code in `code_rank_embed_experiments.py` to:
- Add your own code snippets
- Change the search queries
- Adjust visualization parameters
- Modify the number of similar results returned (top_k parameter)

### Requirements

- Python 3.7+
- sentence-transformers>=2.2.2
- numpy>=1.21.0
- matplotlib>=3.4.0
- scikit-learn>=1.0.0

### License

This project is open source and available under the MIT License. 