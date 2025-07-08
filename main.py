from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

import torch
import uvicorn
import os

from utils import extract_text_from_pdfs, split_chunks_and_embed, embed_text

from models import store_as_vectordb
from models import build_rag_chain

openAI_key = "sk-xpZI1AehtXlMCE7G1ZzhIfHPTQ6F4NXQrpAwkj5zKaT3BlbkFJaHrJInF4uLx_8UMrEQpDjRNuSBdUH8MqSymSByhT8A"
es_cloud_id="QA:dXMtZWFzdC0yLmF3cy5lbGFzdGljLWNsb3VkLmNvbTo0NDMkMGY1M2IyNDUyZDBhNDI2ZjgwYjRkNjcxNDc2MWZkY2YkNTY4ZjBjZjRkNTRkNGY5ZjgyZjI2YmFmMzFiNjI0Mzk="
es_api_key="TGxNWjk1RUJ2cmNPajF3QW9UWmM6S2pCQjhDLWtUMnFwbEphbTVJZTVzZw=="

os.environ["OPENAI_API_KEY"] = openAI_key

with open('utils/prompt.txt', 'r') as file:
    template = file.read()

def rag_system(query: str, chunk_size,chunk_overlap, return_metadata, temperature, llm_model_name) -> Dict[str, str]:
    pdf_folder = 'data/raw/'
    all_docs = extract_text_from_pdfs(pdf_folder, chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    print(len(all_docs))

    vector_db = store_as_vectordb(all_docs, es_cloud_id, es_api_key)
    rag_chain = build_rag_chain(vector_db, template, return_metadata = return_metadata, temperature = temperature, model_name=llm_model_name)

    response=rag_chain.invoke(query)

    index = response.casefold().find("source:")
    
    if index != -1:
        # Split the string into two parts
        part1 = response[:index]
        part2 = response[index:]
        return {"response": part1, "sources": part2}
    else:
        return {"response": response, "sources": ""}

def retrieval_system(query: str, chunk_size, chunk_overlap, threshold=0.1, embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2", top_k=3) -> Dict[str, str]:
    
    doc_embeddings = split_chunks_and_embed(chunk_size, chunk_overlap, embedding_model_name)
    query_embedding = embed_text(query, embedding_model_name)
    
    similarities = []
    for doc_name, doc_data in doc_embeddings.items():
        for chunk_index, chunk_embedding in enumerate(doc_data["embeddings"]):
            # Calculate cosine similarity between query embedding and chunk embedding
            similarity = torch.nn.functional.cosine_similarity(query_embedding, chunk_embedding)
            if similarity.item() > threshold:
                similarities.append({
                    "doc_name": doc_name,
                    "chunk": doc_data["chunks"][chunk_index],
                    "similarity": similarity.item()
                })
        
    similarities = sorted(similarities, key=lambda x: x["similarity"], reverse=True)
    
    if len(similarities)!=0:
        result = ""
        sources = ""
        for item in similarities[:top_k]:
            sources += "\t" + item["doc_name"]
            result += "\n" + item["chunk"]
    else:
        result = "Sorry this information is not present in our policy documents."
        sources = ""


    return {"response": result, "sources": ""}


app = FastAPI()


class QueryResponse(BaseModel):
    response: str
    sources: str
    
# class QueryRequest(BaseModel):
#     user_query: str

@app.post("/query", response_model=QueryResponse)
def get_response():
    # user_query = request.user_query
    user_query = os.getenv("USER_QUERY")
    chunk_size = int(os.getenv("CHUNK_SIZE", 500))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 0))
    return_metadata = os.getenv("RETURN_METADATA", "true").lower() == "true"
    temperature = float(os.getenv("TEMPERATURE", 0.0))
    llm_model_name = os.getenv("MODEL_NAME", "gpt-4o")
    method = os.getenv("METHOD", "RAG")
    threshold = os.getenv("THRESHOLD", 0.1)
    embedding_model_name =  os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    top_k = os.getenv("TOP_K", 3)
    
    print(method)
    print(user_query)
    if method == "RAG":
        try:
            result = rag_system(
                query=user_query,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                return_metadata=return_metadata,
                temperature=temperature,
                llm_model_name=llm_model_name
            )
            return QueryResponse(response=result["response"], sources=result["sources"])
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    elif method == "Semantic_retrieval":
        try:
            result = retrieval_system(
                query=user_query, 
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                threshold=threshold,
                embedding_model_name=embedding_model_name,
                top_k=top_k
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
    
    else:
        raise HTTPException(status_code=400, detail='Invalid method. Method can take only two values- "RAG" or "Semantic_retrieval"')
    
    return result


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


