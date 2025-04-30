from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

class CodeRankEmbedExperiment:
    def __init__(self, model_name: str = "nomic-ai/CodeRankEmbed"):
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self.code_snippets: List[str] = []
        self.code_embeddings: np.ndarray = None
        
    def add_code_snippets(self, snippets: List[str]):
        """Add code snippets to the collection."""
        self.code_snippets.extend(snippets)
        self._update_embeddings()
    
    def _update_embeddings(self):
        """Update embeddings for all code snippets."""
        if self.code_snippets:
            self.code_embeddings = self.model.encode(self.code_snippets)
    
    def search_similar_code(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Search for similar code snippets based on a query."""
        query_embedding = self.model.encode([query])[0]
        similarities = cosine_similarity([query_embedding], self.code_embeddings)[0]
        
        # Get top-k most similar snippets
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(self.code_snippets[i], similarities[i]) for i in top_indices]
    
    def visualize_embeddings(self, queries: List[str] = None):
        """Visualize code embeddings in 2D space using PCA."""
        from sklearn.decomposition import PCA
        
        # Combine code snippets and queries if provided
        all_texts = self.code_snippets.copy()
        if queries:
            all_texts.extend(queries)
        
        # Get embeddings for all texts
        embeddings = self.model.encode(all_texts)
        
        # Reduce to 2D using PCA
        pca = PCA(n_components=2)
        reduced_embeddings = pca.fit_transform(embeddings)
        
        # Plot
        plt.figure(figsize=(10, 8))
        
        # Plot code snippets
        plt.scatter(reduced_embeddings[:len(self.code_snippets), 0],
                   reduced_embeddings[:len(self.code_snippets), 1],
                   c='blue', label='Code Snippets')
        
        # Plot queries if provided
        if queries:
            plt.scatter(reduced_embeddings[len(self.code_snippets):, 0],
                       reduced_embeddings[len(self.code_snippets):, 1],
                       c='red', label='Queries')
        
        plt.title('Code Embeddings Visualization')
        plt.legend()
        plt.show()

def main():
    # Initialize the experiment
    experiment = CodeRankEmbedExperiment()
    
    # Add some example code snippets
    code_snippets = [
        'def fact(n):\n    if n < 0:\n        raise ValueError\n    return 1 if n == 0 else n * fact(n - 1)',
        'def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)',
        'def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1',
        'def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]'
    ]
    
    experiment.add_code_snippets(code_snippets)
    
    # Example queries
    queries = [
        'Find a function that calculates factorial',
        'Show me a sorting algorithm',
        'How to implement binary search?'
    ]
    
    # Search for similar code
    print("\nSearching for similar code snippets:")
    for query in queries:
        print(f"\nQuery: {query}")
        results = experiment.search_similar_code(query)
        for snippet, similarity in results:
            print(f"Similarity: {similarity:.4f}")
            print(f"Code:\n{snippet}\n")
    
    # Visualize embeddings
    experiment.visualize_embeddings(queries)

if __name__ == "__main__":
    main() 
