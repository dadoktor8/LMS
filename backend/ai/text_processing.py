import re
import fitz  # PyMuPDF for PDF text extraction
from nltk.tokenize import sent_tokenize
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def sanitize_filename(filename: str):
    """Sanitize the filename to avoid special characters."""
    return re.sub(r'[^a-zA-Z0-9.-]', '_', filename)
# 1. Extract Text from PDF
def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extracts text from a PDF file.
    """
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


# 2. Chunk Text into Manageable Pieces
def chunk_text(text: str, max_chunk_size: int = 500) -> list:
    """
    Splits the extracted text into chunks of a manageable size.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# 3. Generate Embeddings for Text Chunks
# Using HuggingFace's "all-MiniLM-L6-v2" model for embeddings.
def embed_chunks(chunks: list) -> torch.Tensor:
    """
    Converts text chunks into embeddings using SentenceTransformer.
    """
    model = SentenceTransformer(r"D:\My Projects And Such\lms\backend\model\all-MiniLM-L6-v2")
    
    # Get embeddings as numpy and convert to torch tensor
    embeddings = model.encode(chunks, convert_to_tensor=True)

    return embeddings


# 4. Save Embeddings to FAISS Vector Database
def save_embeddings_to_faiss(course_id: int, embeddings: torch.Tensor, chunks: list):
    """
    Saves the embeddings and corresponding chunks to a FAISS index.
    """
    # Convert embeddings to numpy array (FAISS needs numpy)
    embeddings_np = embeddings.cpu().numpy()

    # Initialize FAISS index (simple L2 distance index)
    dim = embeddings_np.shape[1]  # The dimensionality of embeddings
    index = faiss.IndexFlatL2(dim)

    # Add embeddings to the FAISS index
    index.add(embeddings_np)

    # Optionally, store FAISS index to a file if persistence is needed
    faiss.write_index(index, f"faiss_index_{course_id}.index")

    # Save the text chunks associated with the embeddings for retrieval
    # You can store them in a database or a file for later use
    with open(f"faiss_chunks_{course_id}.txt", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk + "\n")
