import PyPDF2
import os 

def read_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_pdfs_semantic(pdf_folder):
    
    # Read and process PDF documents
    documents = {}
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_folder, filename)
            text = read_from_pdf(file_path)
            documents[filename] = text
    return documents