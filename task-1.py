import requests
import PyPDF2
import numpy as np
from io import BytesIO
from sentence_transformers import SentenceTransformer
import faiss
def download_pdf_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        return BytesIO(response.content)  # Return PDF as in-memory file
    else:
        raise Exception(f"Failed to download PDF. HTTP Status Code: {response.status_code}")
def extract_text_from_pdf(pdf_file):
    text = ""
    reader = PyPDF2.PdfReader(pdf_file)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text
def chunk_text(text, chunk_size=512):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
def embed_chunks(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, convert_to_tensor=False)
    return np.array(embeddings)
def store_embeddings(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index
def main():
    pdf_url = "https://www.hunter.cuny.edu/dolciani/pdf_files/workshop-materials/mmc-presentations/tables-charts-and-graphs-with-examples-from.pdf/at_download/file"   
    try:
        pdf_file = download_pdf_from_url(pdf_url)
        print("PDF downloaded successfully.")
        text = extract_text_from_pdf(pdf_file)
        print("Text extracted successfully.")
        chunks = chunk_text(text)
        print(f"Text chunked into {len(chunks)} parts.")
        embeddings = embed_chunks(chunks)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        index = store_embeddings(embeddings)
        print("Embeddings stored successfully in FAISS index.")
        print("Querying the FAISS index with the first embedding...")
        distances, indices = index.search(embeddings[:1], k=3) 
        print("Nearest neighbors:", indices)
        print("Distances:", distances)
    except Exception as e:
        print(f"An error occurred: {e}")
if __name__ == "__main__":
    main()
