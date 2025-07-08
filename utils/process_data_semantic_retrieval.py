import torch
from transformers import AutoTokenizer, AutoModel

from utils import load_data_semantic_retrieval


# Function to embed text using the model
def embed_text(text, embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"):
    
    # Load pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    model = AutoModel.from_pretrained(embedding_model_name)

    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings

# Function to chunk a document into smaller parts
def chunk_document(text, chunk_size=500, chunk_overlap=50):
    # Create chunks of length chunk_size with overlap of chunk_overlap
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks


def split_chunks_and_embed(pdf_folder, chunk_size=500, chunk_overlap=50, embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"):

    documents = load_data_semantic_retrieval.extract_text_from_pdfs_semantic(pdf_folder)

    # Convert documents to chunks and their embeddings
    doc_embeddings = {}
    for name, text in documents.items():
        chunks = chunk_document(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)  # Adjust chunk size and overlap
        chunk_embeddings = [embed_text(chunk, embedding_model_name) for chunk in chunks]  # Get embeddings for each chunk
        doc_embeddings[name] = {
            "chunks": chunks,
            "embeddings": chunk_embeddings
        }

    return doc_embeddings