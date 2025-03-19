from fastapi import FastAPI, Query
from backend.services.retrieval import get_retrieved_results

app = FastAPI()

@app.get("/retrieve/")
def retrieve(query: str, method: str = Query("ChromaDB", enum=["ChromaDB", "Pinecone"])):
    results = get_retrieved_results(query, method)
    return {"query": query, "results": results}
