from scripts.rag_retrieval import retrieve_documents

def get_retrieved_results(query, method="ChromaDB"):
    return retrieve_documents(query, method)
