import langchain_community.document_loaders
from langchain.text_splitter import CharacterTextSplitter
import os

def extract_text_from_pdfs(pdf_folder, chunk_size=500,chunk_overlap=0):
    all_docs = []
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith('.pdf'):
            loader = langchain_community.document_loaders.PyPDFLoader(os.path.join(pdf_folder, pdf_file))
            documents=loader.load()
            text_splitter=CharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
            docs=text_splitter.split_documents(documents)
            all_docs.extend(docs)
    return all_docs


