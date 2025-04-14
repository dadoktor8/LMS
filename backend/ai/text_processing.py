import logging
import re
import fitz  # PyMuPDF for PDF text extraction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from nltk.tokenize import sent_tokenize
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from backend.db.models import CourseMaterial, ProcessedMaterial, TextChunk
from backend.db.database import SessionLocal  # if you're using a unified DB setup
from langchain.vectorstores import FAISS
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
#from langchain.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.schema import Document
from transformers import pipeline
from langchain.memory.chat_message_histories import SQLChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from sqlalchemy import create_engine

# ---------------------------------------------
# Utility
def sanitize_filename(filename: str):
    return re.sub(r'[^a-zA-Z0-9.-]', '_', filename)

# ---------------------------------------------
# 1. Extract Text from PDF
def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)
"""
# ---------------------------------------------
# 2. Chunk Text into Manageable Pieces
def chunk_text(text: str, max_chunk_size: int = 500) -> list:
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

"""
def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_text(text)
# ---------------------------------------------
# 3. Embed Chunks
'''
def embed_chunks(chunks: list) -> torch.Tensor:
    model = SentenceTransformer(r"D:\My Projects And Such\lms\backend\model\all-MiniLM-L6-v2")  # or use local path if offline
    return model.encode(chunks, convert_to_tensor=True)
'''
def embed_chunks(chunks: list) -> torch.Tensor:
    model = SentenceTransformer(r"D:\My Projects And Such\lms\backend\model\all-MiniLM-L6-v2")
    normalized_chunks = [chunk.strip().replace("\n", " ") for chunk in chunks]
    return model.encode(normalized_chunks, convert_to_tensor=True)
# ---------------------------------------------
# 4. Save Embeddings
def save_embeddings_to_faiss(course_id: int, embeddings: torch.Tensor, chunks: list, db: Session):
    embeddings_np = embeddings.cpu().numpy()
    dim = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings_np)
    faiss.write_index(index, f"faiss_index_{course_id}.index")

    # Save chunks to .txt for simple retrieval
    with open(f"faiss_chunks_{course_id}.txt", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk + "\n")

    for chunk, embedding in zip(chunks, embeddings):
        db.add(TextChunk(
            course_id=course_id,
            chunk_text=chunk,
            embedding=str(embedding.tolist())
        ))
    db.commit()
    print(f"✅ Saved FAISS index and chunks for course {course_id}")
    logging.info(f"✅ Saved FAISS index and chunks for course {course_id}")

# ---------------------------------------------
# 5. Process PDFs
def process_materials_in_background(course_id: int, db: Session):
    materials = db.query(CourseMaterial).filter_by(course_id=course_id).all()

    for material in materials:
        file_path = f"backend/uploads/{material.filename}"
        if db.query(ProcessedMaterial).filter_by(course_id=course_id, material_id=material.id).first():
            logging.info(f"⏭ Skipping {material.filename} (already processed)")
            print(f"⏭ Skipping {material.filename} (already processed)")
            continue

        try:
            text = extract_text_from_pdf(file_path)
            chunks = chunk_text(text)
            embeddings = embed_chunks(chunks)
            #save_embeddings_to_faiss(course_id, embeddings, chunks, db)
            save_embeddings_to_faiss_langchain(course_id, chunks, db)
            db.add(ProcessedMaterial(course_id=course_id, material_id=material.id))
            db.commit()
        except Exception as e:
            logging.error(f"❌ Failed processing {material.filename}: {e}")
            print(f"❌ Failed processing {material.filename}: {e}")

'''
# ---------------------------------------------
# 6. FAISS Search
def search_faiss(query: str, faiss_index_path: str, chunk_text_path: str, top_k: int = 5) -> list:
    index = faiss.read_index(faiss_index_path)
    model = SentenceTransformer(r"D:\My Projects And Such\lms\backend\model\all-MiniLM-L6-v2")
    query_embedding = model.encode([query])

    distances, indices = index.search(query_embedding, top_k)

    with open(chunk_text_path, "r", encoding="utf-8") as f:
        all_chunks = f.readlines()

    return [all_chunks[i].strip() for i in indices[0] if i < len(all_chunks)]

# ---------------------------------------------
'''
def search_faiss(query: str, faiss_index_path: str, chunk_text_path: str, top_k: int = 5) -> list:
    index = faiss.read_index(faiss_index_path)
    model = SentenceTransformer(r"D:\My Projects And Such\lms\backend\model\all-MiniLM-L6-v2")
    query_embedding = model.encode([query])

    distances, indices = index.search(query_embedding, top_k)

    with open(chunk_text_path, "r", encoding="utf-8") as f:
        all_chunks = [line.strip() for line in f.readlines()]

    results = []
    for idx in indices[0]:
        if idx < len(all_chunks):
            chunk = all_chunks[idx]
            print(f"\n🔹 Retrieved chunk: {chunk[:200]}...")  # debug log
            results.append(chunk)
    return results

# 7. Generate Answer
def get_answer_from_rag(query: str, faiss_index_path: str, top_k: int = 5) -> str:
    chunk_text_path = faiss_index_path.replace("faiss_index_", "faiss_chunks_").replace(".index", ".txt")
    relevant_chunks = search_faiss(query, faiss_index_path, chunk_text_path, top_k)

    context = "\n".join(relevant_chunks)
    rag_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    rag_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

    inputs = rag_tokenizer(query + "\n" + context, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = rag_model.generate(inputs['input_ids'], num_beams=4, max_length=200)

    return rag_tokenizer.decode(outputs[0], skip_special_tokens=True)



def get_answer_from_rag_langchain(query: str, course_id: int, student_id: str) -> str:
    # Load FAISS index
    index_path = f"faiss_index_{course_id}"
    db = FAISS.load_local(
        index_path, 
        HuggingFaceEmbeddings(model_name=r"D:\My Projects And Such\lms\backend\model\all-MiniLM-L6-v2"),
        allow_dangerous_deserialization=True
    )

    # Load LLM pipeline
    rag_pipeline = pipeline("text2text-generation", model="facebook/bart-large-cnn")
    llm = HuggingFacePipeline(pipeline=rag_pipeline)
    engine = create_engine("sqlite:///chat_history.db")
    # Setup persistent memory
    session_id = f"{student_id}_{course_id}"
    chat_history = SQLChatMessageHistory(
        session_id=session_id,
        connection=engine
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        chat_memory=chat_history
    )

    # Setup Conversational RAG chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        return_source_documents=False
    )

    # Run the chain with history
    result = qa_chain.invoke({
            "question": f"""You are an AI tutor. Answer this question using relevant course materials. 
        If the answer is not found in the course content, say 'I don't know' and do not guess.

        Question: {query}"""
        })

    if isinstance(result, dict) and 'answer' in result:
        return result['answer']
    elif 'result' in result:
        return result['result']
    
    return result  # fallback

def save_embeddings_to_faiss_langchain(course_id: int, chunks: list, db: Session):
    # 1. Normalize chunks
    normalized_chunks = [chunk.strip().replace("\n", " ") for chunk in chunks]

    # 2. Prepare LangChain documents
    documents = [Document(page_content=chunk) for chunk in normalized_chunks]

    # 3. Load embedding model
    embeddings_model = HuggingFaceEmbeddings(model_name=r"D:\My Projects And Such\lms\backend\model\all-MiniLM-L6-v2")

    # 4. Build FAISS vectorstore from documents
    vectorstore = FAISS.from_documents(documents, embedding=embeddings_model)

    # 5. Save FAISS index (LangChain format)
    vectorstore.save_local(f"faiss_index_{course_id}")

    # 6. Save chunks + embeddings to DB
    for chunk in normalized_chunks:
        try:
            embedding = embeddings_model.embed_documents([chunk])[0]  # only one chunk

            # Safely convert to list
            if isinstance(embedding, torch.Tensor):
                embedding_value = embedding.cpu().numpy().tolist()
            elif hasattr(embedding, "tolist"):
                embedding_value = embedding.tolist()
            else:
                embedding_value = embedding  # assume already a list

            db.add(TextChunk(
                course_id=course_id,
                chunk_text=chunk,
                embedding=str(embedding_value)
            ))
        except Exception as e:
            db.rollback()
            logging.error(f"❌ Failed to insert chunk: {e}")
            continue

    db.commit()
    print(f"✅ Saved FAISS index and chunks (LangChain) for course {course_id}")
    logging.info(f"✅ Saved FAISS index and chunks (LangChain) for course {course_id}")


def get_past_messages(student_id, course_id):
    session_id = f"{student_id}_{course_id}"
    history = SQLChatMessageHistory(
        session_id=session_id,
        connection_string="sqlite:///chat_history.db"
    )
    return history.messages