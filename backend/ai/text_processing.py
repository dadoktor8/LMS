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
from langchain.llms import LlamaCpp

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

'''

def get_answer_from_rag_langchain(query: str, course_id: int, student_id: str) -> str:
    try:
        # Load FAISS index
        index_path = f"faiss_index_{course_id}"
        embeddings = HuggingFaceEmbeddings(model_name=r"D:\My Projects And Such\lms\backend\model\all-MiniLM-L6-v2")
        
        db = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"📂 FAISS index loaded successfully for course {course_id}")
        
        # Retrieve relevant documents
        docs = db.similarity_search(query, k=5)
        if not docs:
            return "I couldn't find anything relevant in the course materials to answer that question."
        
        print(f"🔍 Retrieved {len(docs)} docs")
        for i, doc in enumerate(docs):
            print(f"Doc {i+1}: {doc.page_content[:100]}...")
        
        # Extract content from documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Set up session ID for chat history
        session_id = f"{student_id}_{course_id}"
        
        # Initialize SQL history only for storing conversation
        from langchain_community.chat_message_histories import SQLChatMessageHistory
        
        sql_history = SQLChatMessageHistory(
            session_id=session_id,
            connection="sqlite:///chat_history.db"
        )
        
        # Set up LLM pipeline - use a simple approach
        rag_pipeline = pipeline("text2text-generation", model="facebook/bart-large-cnn", max_length=512)
        
        # Simple prompt without including previous conversation
        prompt = f"""Answer the following question based only on the provided course materials.
If the answer is not found in the materials, say 'I don't find information about this in the course materials.'

Course materials:
{context}

Question: {query}

Answer:"""
        
        # Generate the answer
        print("Generating answer...")
        generated_text = rag_pipeline(prompt)[0]['generated_text']
        
        # Clean up the response 
        result = generated_text
        if "Answer:" in result:
            result = result.split("Answer:")[-1].strip()
        
        # Store the interaction in chat history for display purposes
        try:
            sql_history.add_user_message(query)
            sql_history.add_ai_message(result)
            print("✅ Successfully saved conversation to history")
        except Exception as e:
            print(f"Warning: Failed to save conversation to history: {str(e)}")
        
        return result
        
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return f"❌ An error occurred: {str(e)}" '''

def get_answer_from_rag_langchain(query: str, course_id: int, student_id: str) -> str:
    try:
        # Load FAISS index
        index_path = f"faiss_index_{course_id}"
        embeddings = HuggingFaceEmbeddings(model_name=r"D:\My Projects And Such\lms\backend\model\all-MiniLM-L6-v2")
        
        db = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"📂 FAISS index loaded successfully for course {course_id}")
        
        # Retrieve relevant documents
        docs = db.similarity_search(query, k=5)
        if not docs:
            return "I couldn't find anything relevant in the course materials to answer that question."
        
        print(f"🔍 Retrieved {len(docs)} docs")
        for i, doc in enumerate(docs):
            print(f"Doc {i+1}: {doc.page_content[:100]}...")
        
        # Extract content from documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Set up session ID for chat history
        session_id = f"{student_id}_{course_id}"
        
        sql_history = SQLChatMessageHistory(
            session_id=session_id,
            connection="sqlite:///chat_history.db"
        )
        llm = LlamaCpp(
            model_path=r"D:\My Projects And Such\lms\backend\model\DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",  # update to your file
            temperature=0.7,
            max_tokens=512,
            n_ctx=2048,
            verbose=True
        )
        
        # Set up LLM pipeline - use a simple approach
        prompt = f"""Answer the following question based only on the provided course materials.
        If the answer is not found in the materials, say 'I don't find information about this in the course materials.'

        Course materials:
        {context}

        Question: {query}

        Answer:"""

        print("🧠 Generating answer using DeepSeek...")
        result = llm(prompt).strip()
        result = clean_deepseek_response(result)
        print("Answer :", result)
        
        # Store the interaction in chat history for display purposes
        try:
            sql_history.add_user_message(query)
            sql_history.add_ai_message(result)
            print("✅ Successfully saved conversation to history")
        except Exception as e:
            print(f"Warning: Failed to save conversation to history: {str(e)}")
        
        return result
        
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return f"❌ An error occurred: {str(e)}"

def save_embeddings_to_faiss_langchain(course_id: int, chunks: list, db: Session):
    # 1. Normalize chunks
    normalized_chunks = [chunk.strip().replace("\n", " ") for chunk in chunks]

    # 2. Prepare LangChain documents
    documents = [Document(page_content=chunk) for chunk in normalized_chunks]

    # 3. Load embedding model
    embeddings_model = HuggingFaceEmbeddings(
        model_name=r"D:\My Projects And Such\lms\backend\model\all-MiniLM-L6-v2"
    )

    # 4. Build FAISS vectorstore from documents
    vectorstore = FAISS.from_documents(documents, embedding=embeddings_model)

    # 5. Check if FAISS index is populated
    if len(vectorstore.docstore._dict) == 0:
        logging.warning("⚠️ FAISS index appears to be empty!")
    else:
        logging.info(f"✅ FAISS index built with {len(vectorstore.docstore._dict)} documents")

    # 6. Save FAISS index (LangChain format)
    vectorstore.save_local(f"faiss_index_{course_id}")
    logging.info(f"📁 FAISS index saved to faiss_index_{course_id}")

    # 7. Save chunks + embeddings to DB
    for chunk in normalized_chunks:
        try:
            # Use embed_query for a single chunk (consistent return format)
            embedding = embeddings_model.embed_query(chunk)

            # Convert to list if it's a tensor or numpy array
            if isinstance(embedding, torch.Tensor):
                embedding_value = embedding.cpu().numpy().tolist()
            elif hasattr(embedding, "tolist"):
                embedding_value = embedding.tolist()
            else:
                embedding_value = embedding

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
    logging.info(f"✅ Saved FAISS index and chunks (LangChain) for course {course_id}")
    print(f"✅ Saved FAISS index and chunks (LangChain) for course {course_id}")


def get_past_messages(student_id, course_id):
    session_id = f"{student_id}_{course_id}"
    history = SQLChatMessageHistory(
        session_id=session_id,
        connection_string="sqlite:///chat_history.db"
    )
    return history.messages

import re

def clean_deepseek_response(text: str) -> str:
    text = text.strip()
    text = text.replace("\u200B", "")
    text = text.replace("\u00A0", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\s\S]*<\/think>\n?", "", text).strip()
    return text
