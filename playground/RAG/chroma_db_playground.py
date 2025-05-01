"""
A playground script demonstrating basic usage of ChromaDB - a vector database for RAG systems.
This quickstart example shows how to:
- Create a collection
- Insert documents
- Query the collection with multiple queries
"""

import chromadb
chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(name="first_chroma_collection")

collection.upsert(
    documents=[
        "This is a document about pineapple",
        "This is a document about oranges",
        "Bikes have two wheels and are a form of transport",
        "Cars have four wheels",
        "Costco is the best store to buy groceries",
        "Safeway has onions and bananas",
        "Bikes can go fast when there is traffic",
        "Cars are expensive and require a license to drive.",
        "Pineapples and oragnes are a good source of vitamin C.",
        "Smart glasses are a new way to see the world.",
    ],
    ids=["id1", "id2", "id3", "id4", "id5", "id6", "id7", "id8", "id9", "id10"]
)

queries = [
    "What is the best store to buy groceries?",
    "Query about Florida",
    "What is a bike?",
    "Do you have a car?",
    "What is Hawaii famous for?"
]

def print_chroma_results(results, queries):
    """
    Pretty print the results from ChromaDB queries.
    
    Args:
        results (dict): ChromaDB query results containing 'ids', 'documents', 'distances'
        queries (list): List of original query strings
    """
    print("\nChromaDB Query Results:")
    print("-" * 50)
    
    for idx, (query, docs, ids, distances) in enumerate(zip(queries, results['documents'], results['ids'], results['distances'])):
        print(f"\nQuery {idx + 1}: '{query}'")
        print("Matches:")
        for doc, doc_id, distance in zip(docs, ids, distances):
            print(f"  - ID: {doc_id}")
            print(f"    Distance: {distance}")
            print(f"    Document: {doc}")
        print("-" * 30)

# Query the collection and print results
results = collection.query(
    query_texts=queries,    # Chroma will embed this using the default model: all-MiniLM-L6-v2
    n_results=2             # how many results to return
)

print_chroma_results(results, queries)
# print(f"\nRaw results: {results}")
