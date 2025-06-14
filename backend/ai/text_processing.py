from datetime import date
import logging
import os
import re
from typing import Optional
import PyPDF2
from fastapi import HTTPException
from PyPDF2 import PdfReader, PdfWriter
import fitz  # PyMuPDF for PDF text extraction
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from nltk.tokenize import sent_tokenize
#import torch
#from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#import faiss
#import numpy as np
#from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
#import urllib
from backend.ai.aws_ai import S3_BUCKET_NAME, get_s3_client, load_faiss_vectorstore, upload_faiss_index_to_s3
from backend.db.models import CourseMaterial, PDFQuotaUsage, ProcessedMaterial, TextChunk
from backend.db.database import SessionLocal  # if you're using a unified DB setup
from langchain.vectorstores import FAISS
#from langchain.embeddings import HuggingFaceEmbeddings
#from langchain_huggingface import HuggingFaceEmbeddings
#from langchain.llms import HuggingFacePipeline
#from langchain_huggingface import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.schema import Document
#from transformers import pipeline
from langchain.memory.chat_message_histories import SQLChatMessageHistory
#from langchain.memory import ConversationBufferMemory
#from langchain.chains import ConversationalRetrievalChain
#from sqlalchemy import create_engine
#from langchain.llms import LlamaCpp
from langchain_openai import ChatOpenAI, OpenAIEmbeddings 
from langchain.prompts import PromptTemplate


class PDFQuotaConfig:
    # Maximum number of pages allowed to be processed per day per course
    DAILY_PAGE_QUOTA = 2000
    # Maximum size of an individual file in bytes (50MB)
    MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024
    # Error return value for page count function
    ERROR_PAGE_COUNT = 1000

def get_course_retriever(course_id: int):
    try:
        db = load_faiss_vectorstore(course_id, openai_api_key=None)
        return db.as_retriever(search_kwargs={"k": 10})
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No existing knowledge base found for course {course_id}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Problem loading FAISS index: {str(e)}")

def get_context_for_query(retriever, query: str):
    retrieved_docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    if not context.strip():
        raise HTTPException(status_code=404, detail="Insufficient context found for your query.")
    return context

def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key missing in environment variables.")
    try:
        return ChatOpenAI(
            model="gpt-4.1-mini",
            temperature=0.2,
            openai_api_key=api_key
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not setup OpenAI Client: {str(e)}")

# ---------------------------------------------
# Utility
def sanitize_filename(filename: str):
    return re.sub(r'[^a-zA-Z0-9.-]', '_', filename)

# ---------------------------------------------
# 1. Extract Text from PDF
def extract_text_from_pdf(file_path_or_key: str) -> str:
    """Extract text from a PDF file - handles both local and S3 paths"""
    
    # Check if this is an S3 key (starts with "course_materials/")
    if file_path_or_key.startswith("course_materials/"):
        # Create a temporary file
        temp_dir = "temp_pdfs"
        os.makedirs(temp_dir, exist_ok=True)
        local_path = f"{temp_dir}/{os.path.basename(file_path_or_key)}"
        
        # Download from S3
        if not download_file_from_s3(file_path_or_key, local_path):
            raise Exception(f"Failed to download file from S3: {file_path_or_key}")
        
        # Process the local file
        doc = fitz.open(local_path)
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        
        # Clean up the temporary file
        try:
            os.remove(local_path)
        except:
            pass
            
        return text
    else:
        # Original local file handling
        doc = fitz.open(file_path_or_key)
        return "\n".join(page.get_text() for page in doc)

def download_file_from_s3(s3_key: str, local_path: str) -> bool:
    """Download a file from S3 to a local temporary path"""
    s3_client = get_s3_client()
    try:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3_client.download_file(S3_BUCKET_NAME, s3_key, local_path)
        return True
    except Exception as e:
        print(f"Error downloading file from S3: {e}")
        return False


def count_pdf_pages(file_path: str) -> int:
    """
    Counts the number of pages in a PDF file.
    Works with both local files and files that need to be downloaded from S3.
    
    Args:
        file_path: Path to the PDF file (local or S3 key)
        
    Returns:
        int: Number of pages in the PDF, or PDFQuotaConfig.ERROR_PAGE_COUNT on error
    """
    local_path = file_path
    temp_file_created = False
    
    try:
        # Handle S3 paths if needed
        if file_path.startswith("course_materials/") and not os.path.exists(file_path):
            # Create temp directory for downloads if it doesn't exist
            temp_dir = "temp_pdfs"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Set local path for the downloaded file
            local_path = f"{temp_dir}/{os.path.basename(file_path)}"
            temp_file_created = True
            
            # Download from S3
            s3_client = get_s3_client()
            try:
                s3_client.download_file(S3_BUCKET_NAME, file_path, local_path)
                logging.info(f"Downloaded s3://{S3_BUCKET_NAME}/{file_path} to {local_path}")
            except Exception as e:
                logging.error(f"Failed to download file from S3: {file_path}. Error: {e}")
                return PDFQuotaConfig.ERROR_PAGE_COUNT
        
        # Verify the file now exists
        if not os.path.exists(local_path):
            logging.error(f"File not found: {local_path}")
            return PDFQuotaConfig.ERROR_PAGE_COUNT
            
        # Count pages using PyPDF2
        with open(local_path, "rb") as f:
            pdf_reader = PdfReader(f)
            return len(pdf_reader.pages)
            
    except Exception as e:
        logging.error(f"Error counting PDF pages: {e}")
        return PDFQuotaConfig.ERROR_PAGE_COUNT
        
    finally:
        # Clean up temporary file if we created one
        if temp_file_created and os.path.exists(local_path):
            try:
                os.remove(local_path)
                logging.debug(f"Cleaned up temporary file: {local_path}")
            except Exception as e:
                logging.warning(f"Failed to clean up temporary file {local_path}: {e}")

def validate_pdf_for_upload(file_path: str) -> tuple[bool, str, Optional[int]]:
    """
    Validates if a PDF is eligible for upload based on page count limits.
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        tuple: (is_valid, message, page_count)
            - is_valid: Boolean indicating if file is valid for upload
            - message: Validation message
            - page_count: Number of pages in the PDF, or None if error
    """
    # Count pages in the PDF
    page_count = count_pdf_pages(file_path)
    
    # Error counting pages
    if page_count == PDFQuotaConfig.ERROR_PAGE_COUNT:
        return False, "Could not determine page count - file may be corrupted", None
    
    # Check if page count exceeds daily quota
    if page_count > PDFQuotaConfig.DAILY_PAGE_QUOTA:
        return False, f"File contains {page_count} pages, exceeding the maximum of {PDFQuotaConfig.DAILY_PAGE_QUOTA} pages, You can go to PDF-Exteractor.com and select only the neccesary pages.", page_count
    
    return True, f"File contains {page_count} pages", page_count
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

'''

def check_pdf_quota(course_id: int, pdf_path: str, db: Session, daily_page_quota: int = 100) -> tuple[bool, int, int]:
    """
    Checks if processing a PDF would exceed the daily course quota.

    Args:
        course_id: Course identifier.
        pdf_path: Path to the PDF file.
        db: SQLAlchemy Session.
        daily_page_quota: The per-day page quota.

    Returns:
        tuple: (quota_ok, pdf_page_count, remaining_quota_after_processing)
    """
    # Count pages in PDF (with error fallback)
    page_count = count_pdf_pages(pdf_path)

    today = date.today()
    quota_usage = db.query(PDFQuotaUsage).filter(
        PDFQuotaUsage.course_id == course_id,
        PDFQuotaUsage.usage_date == today
    ).first()

    # If no record, create one with 0 pages used
    if not quota_usage:
        quota_usage = PDFQuotaUsage(
            course_id=course_id,
            usage_date=today,
            pages_processed=0
        )
        db.add(quota_usage)
        db.commit()
        db.refresh(quota_usage)

    used_pages = quota_usage.pages_processed
    remaining_quota = daily_page_quota - used_pages

    if page_count > remaining_quota:
        # Not enough quota to process entire PDF
        quota_ok = False
        pages_we_can_process = remaining_quota  # Could process only this many
    else:
        quota_ok = True
        pages_we_can_process = page_count

    # If quota_ok, update the record to account for processed pages
    if quota_ok and pages_we_can_process > 0:
        quota_usage.pages_processed += pages_we_can_process
        db.commit()

    # Return results: (is full file processable, total pages in file, remaining quota AFTER run)
    return quota_ok, page_count, remaining_quota - pages_we_can_process if quota_ok else remaining_quota


# ---------------------------------------------
# 5. Process PDFs
def process_materials_in_background(course_id: int, db: Session):
    materials = db.query(CourseMaterial).filter_by(course_id=course_id).all()
   
    # Track overall processing status
    processed_count = 0
    skipped_count = 0
    quota_exceeded_count = 0
    quota_remaining = None
    
    # Get today's quota usage
    today = date.today()
    quota_usage = db.query(PDFQuotaUsage).filter(
        PDFQuotaUsage.course_id == course_id,
        PDFQuotaUsage.usage_date == today
    ).first()
    
    used_pages = quota_usage.pages_processed if quota_usage else 0
    remaining_quota = PDFQuotaConfig.DAILY_PAGE_QUOTA - used_pages
   
    for material in materials:
        # Check if the material has already been processed
        processed_material = db.query(ProcessedMaterial).filter_by(course_id=course_id, material_id=material.id).first()
        if processed_material:
            logging.info(f"⏭ Skipping {material.filename} (already processed)")
            print(f"⏭ Skipping {material.filename} (already processed)")
            skipped_count += 1
            continue
       
        try:
            file_path = material.filepath
           
            # First check if the file exists locally
            if not os.path.exists(file_path):
                logging.info(f"📥 File not found locally: {file_path}, attempting to download from S3")
                print(f"📥 File not found locally: {file_path}, attempting to download from S3")
               
                # Determine S3 path - assuming files are stored in course_materials/
                s3_key = f"course_materials/{course_id}/{material.filename}"
               
                # Create temp directory for downloads if it doesn't exist
                temp_dir = "temp_pdfs"
                os.makedirs(temp_dir, exist_ok=True)
               
                # Set the local path for the downloaded file
                file_path = f"{temp_dir}/{material.filename}"
               
                # Download from S3
                s3_client = get_s3_client()
                try:
                    s3_client.download_file(S3_BUCKET_NAME, s3_key, file_path)
                    logging.info(f"✅ Downloaded s3://{S3_BUCKET_NAME}/{s3_key} to {file_path}")
                    print(f"✅ Downloaded s3://{S3_BUCKET_NAME}/{s3_key} to {file_path}")
                except Exception as e:
                    raise Exception(f"Failed to download file from S3: {s3_key}. Error: {e}")
           
            # Verify the file now exists
            if not os.path.exists(file_path):
                raise Exception(f"File not found after download attempt: {file_path}")
            
            # Count total pages using our consolidated function
            total_pages = count_pdf_pages(file_path)
            if total_pages == PDFQuotaConfig.ERROR_PAGE_COUNT:
                raise Exception(f"Error counting pages in PDF: {file_path}")
                
            # Check if we have enough quota for the entire document
            if total_pages > remaining_quota:
                logging.warning(f"⚠️ Not enough quota to process {material.filename} ({total_pages} pages needed, {remaining_quota} remaining)")
                print(f"⚠️ Not enough quota to process {material.filename} ({total_pages} pages needed, {remaining_quota} remaining)")
                quota_exceeded_count += 1
                continue

            # Update quota usage BEFORE processing the PDF
            if quota_usage:
                quota_usage.pages_processed += total_pages
            else:
                quota_usage = PDFQuotaUsage(
                    course_id=course_id,
                    usage_date=today,
                    pages_processed=total_pages
                )
                db.add(quota_usage)
            
            db.commit()
            
            # Update remaining quota
            remaining_quota = PDFQuotaConfig.DAILY_PAGE_QUOTA - quota_usage.pages_processed
            quota_remaining = remaining_quota

            # Process entire PDF
            text = extract_text_from_pdf(file_path)
            
            # Process the text and save embeddings
            chunks = chunk_text(text)
            save_embeddings_to_faiss_openai(course_id, chunks, db)
           
            # Mark as processed
            db.add(ProcessedMaterial(course_id=course_id, material_id=material.id))
            db.commit()
            processed_count += 1
            logging.info(f"✅ Processed {material.filename} ({total_pages} pages, {remaining_quota} remaining in quota)")
            print(f"✅ Processed {material.filename} ({total_pages} pages, {remaining_quota} remaining in quota)")
           
        except Exception as e:
            logging.error(f"❌ Failed processing {material.filename}: {e}")
            print(f"❌ Failed processing {material.filename}: {e}")
            
            # If there was an error during processing, rollback any quota usage
            if 'total_pages' in locals() and quota_usage:
                quota_usage.pages_processed -= total_pages
                db.commit()
   
    # Return summary statistics
    return {
        "processed": processed_count,
        "skipped": skipped_count,
        "quota_exceeded": quota_exceeded_count,
        "remaining_quota": quota_remaining
    }
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
        return f"❌ An error occurred: {str(e)}" '''

def detect_study_request(query: str) -> tuple[bool, Optional[str], Optional[str]]:
    query_lower = query.lower()
    
    # Check for study material type
    material_type = None
    if "flashcard" in query_lower or "flash card" in query_lower:
        material_type = "flashcards"
    #elif "quiz" in query_lower or "test" in query_lower or "assess" in query_lower:
        #material_type = "quiz"
    elif "study guide" in query_lower or "notes" in query_lower or "summary" in query_lower:
        material_type = "study_guide"
    
    # Check for study intent keywords
    study_intent_keywords = [
        "help me study", "need to study", "studying for", "learn about",
        "prepare for", "how do i", "can you explain", "what is", "help me understand"
    ]
    
    has_study_intent = any(keyword in query_lower for keyword in study_intent_keywords)
    
    # If we have study intent but no material type specified, default to study guide
    if has_study_intent and not material_type:
        material_type = "study_guide"
    
    # If we have a material type but no clear study intent, assume study intent
    is_study_request = material_type is not None or has_study_intent
    
    # Extract topic if this is a study request
    topic = None
    if is_study_request:
        # Extract topic based on query structure
        if "about" in query_lower:
            topic = query_lower.split("about", 1)[1].strip()
        elif "on" in query_lower:
            topic = query_lower.split("on", 1)[1].strip()
        elif "for" in query_lower:
            topic = query_lower.split("for", 1)[1].strip()
        else:
            # Remove material type and study intent words to extract topic
            topic = query_lower
            for keyword in ["flashcard", "flash card", "quiz", "test", "study guide", 
                           "notes", "summary", "help me study", "studying for"]:
                topic = topic.replace(keyword, "")
            topic = topic.strip()
        
        # Clean up the topic
        for punct in ["?", ".", "!"]:
            topic = topic.replace(punct, "")
        
        # If topic is too short or empty, use a general placeholder
        if not topic or len(topic) < 3:
            topic = "this subject"
    
    return is_study_request, material_type, topic


def get_answer_from_rag_langchain_openai(query: str, course_id: int, student_id: str) -> str:
    
    
    is_study_request, material_type, topic = detect_study_request(query)
    
    if is_study_request and material_type and topic:
        # Redirect to specific study material page based on type
        if material_type == "flashcards":
            study_url = f"/ai/study/flashcards?course_id={course_id}&topic={topic}"
        #elif material_type == "quiz":
            #study_url = f"/ai/study/quiz?course_id={course_id}&topic={topic}"
        elif material_type == "study_guide":
            study_url = f"/ai/study/guide?course_id={course_id}&topic={topic}"
        else:
            # Default to main study page if unsure about material type
            study_url = f"/ai/study?course_id={course_id}&topic={topic}"
        
        response = f"""
        <div>
            <p>I can help you study <strong>{topic}</strong> with <strong>{material_type.capitalize()}</strong>!</p>
            <a href="{study_url}" class="btn btn-primary" style="margin-top:1em;">Open {material_type.capitalize()} Interface</a>
        </div>
        """
        return response
    
    try:
        # Input validation
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if not isinstance(course_id, int) or course_id <= 0:
            raise ValueError("Course ID must be a positive integer")
        if not student_id or not isinstance(student_id, str):
            raise ValueError("Student ID must be a non-empty string")
            
        # Load FAISS index
        try:
            faiss_vectorstore = load_faiss_vectorstore(
                course_id=course_id,
                openai_api_key=os.getenv("OPENAI_API_KEY"),  # Or None to auto-pickup
                temp_dir="tmp"  # Or whatever temp location you want
            )
            retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 5})
        except FileNotFoundError:
            raise ValueError(f"FAISS index not found for course ID: {course_id}")
        except Exception as e:
            raise ConnectionError(f"Error loading or downloading FAISS index: {str(e)}")
            
        # Chat history for memory
        session_id = f"{student_id}_{course_id}"
        try:
            sql_history = SQLChatMessageHistory(session_id=session_id, connection="sqlite:///chat_history.db")
        except Exception as e:
            # Log but continue - history is non-critical
            print(f"Warning: Could not initialize chat history: {e}")
            sql_history = None
            
        # Set up OpenAI GPT-4
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables")
            
        try:
            chat = ChatOpenAI(
                model="gpt-4.1-mini",
                temperature=0.3,
                openai_api_key=api_key
            )
        except Exception as e:
            raise ConnectionError(f"Error initializing OpenAI client: {str(e)}")
            
        # Prompt Template with Personality
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""
            You are Lumi, a warm, caring AI tutor helping students understand complex topics using their course materials.
            Be thoughtful, empathetic, and encouraging. Always thank them for asking and explain clearly.
            
            FORMATTING INSTRUCTIONS:
            1. Start with a brief, friendly greeting and acknowledgment of their question
            2. Structure your response as a set of clear, numbered or bulleted points instead of dense paragraphs
            3. Use <strong> HTML tags to highlight important terms or concepts </strong>
            4. Keep each point focused on a single idea or concept
            5. If explaining a process or sequence, use numbered lists with the <ol> and <li> HTML tags
            6. For general points, use bullet points with the <ul> and <li> HTML tags
            7. For especially important information, wrap it in <div class="key-point">Important information here</div>
            8. Conclude with a brief encouraging note
            9. Let them know that can type "Make Flash-Cards 'Topic'" or "Make Study Guide 'Topic' and you will make one for them"
            
            If the answer is not in the provided materials, let them know gently.
            
            Course Materials:
            {context}
            
            Student's Question:
            {question}
            
            Lumi's Helpful Answer:
            """
        )
        
        # RetrievalQA chain
        try:
            qa_chain = RetrievalQA.from_chain_type(
                llm=chat,
                retriever=retriever,
                return_source_documents=False,
                chain_type_kwargs={"prompt": prompt_template, "verbose":False},
                chain_type="stuff"
            )
            
            # Get response with timeout
            result = qa_chain({"query": query})
            response = result.get("result", "")
            
            if not response:
                return "I apologize, but I couldn't generate a response. Please try again or rephrase your question."
                
        except TimeoutError:
            return "I'm sorry, but the request timed out. Please try again or ask a simpler question."
        except Exception as e:
            raise ConnectionError(f"Error during query processing: {str(e)}")
            
        # Clean formatting
        response = response.replace("\u200B", "")
        response = response.replace("\u00A0", " ")
        response = re.sub(r"\s+", " ", response)
        response = re.sub(r"[\s\S]*<\/think>\n?", "", response).strip()
        response = post_process_response(response)
        # Store in DB history
        if sql_history:
            try:
                sql_history.add_user_message(query)
                sql_history.add_ai_message(response)
            except Exception as e:
                print(f"Warning: Could not save chat history: {e}")
                
        return response
        
    except ValueError as e:
        error_msg = f"Input error: {str(e)}"
        print(error_msg)
        return f"I'm sorry, but there was an issue with your request: {str(e)}. Please contact support if this continues."
        
    except ConnectionError as e:
        error_msg = f"Connection error: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        raise
        
    except Exception as e:
        error_msg = f"Unexpected error in RAG system: {str(e)}"
        print(error_msg)
        # Log the full traceback for debugging
        import traceback
        traceback.print_exc()
        return "I apologize, but I encountered an unexpected error. Our team has been notified, and we're working to fix it."
'''
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

'''
def save_embeddings_to_faiss_openai(course_id: int, chunks: list, db: Session):
    # 1. Normalize chunks
    normalized_chunks = [chunk.strip().replace("\n", " ") for chunk in chunks]

    # 2. Prepare LangChain documents
    documents = [Document(page_content=chunk) for chunk in normalized_chunks]

    # 3. Load OpenAI embedding model
    embeddings_model = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

    # 4. Build FAISS vectorstore
    vectorstore = FAISS.from_documents(documents, embedding=embeddings_model)

    # 5. Check if FAISS index is populated
    if len(vectorstore.docstore._dict) == 0:
        logging.warning("⚠️ FAISS index appears to be empty!")
    else:
        logging.info(f"✅ FAISS index built with {len(vectorstore.docstore._dict)} documents")

    # 6. Save FAISS index (LangChain format)
    index_path = f"faiss_index_{course_id}"
    vectorstore.save_local(index_path)
    logging.info(f"📁 FAISS index saved to {index_path}")
    upload_faiss_index_to_s3(index_path, course_id)
    # 7. Save chunks + embeddings to DB
    for chunk in normalized_chunks:
        try:
            embedding = embeddings_model.embed_query(chunk)

            db.add(TextChunk(
                course_id=course_id,
                chunk_text=chunk,
                embedding=str(embedding)
            ))

        except Exception as e:
            db.rollback()
            logging.error(f"❌ Failed to insert chunk: {e}")
            continue

    db.commit()
    logging.info(f"✅ Saved FAISS index and OpenAI chunks for course {course_id}")
    print(f"✅ Saved FAISS index and OpenAI chunks for course {course_id}")


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


def post_process_response(response):
    """
    Enhance the response with additional HTML formatting if needed.
    Converts markdown-style lists to proper HTML if the AI didn't use HTML tags.
    """
    response = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', response)  # Bold
    response = re.sub(r'\*(.*?)\*', r'<em>\1</em>', response)  # Italic
    
    # Handle Markdown headings
    response = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', response, flags=re.MULTILINE)
    response = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', response, flags=re.MULTILINE)
    response = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', response, flags=re.MULTILINE)
    
    # Handle code blocks - important for showing code examples
    response = re.sub(r'```(?:\w+)?\n(.*?)\n```', r'<pre><code>\1</code></pre>', response, flags=re.DOTALL)
    # Convert markdown-style numbered lists if HTML lists weren't used
    if "<ol>" not in response:
        response = re.sub(r'(\d+\.\s+[^\n]+)(?:\n|$)', r'<li>\1</li>', response)
        if re.search(r'<li>\d+\.', response):
            response = re.sub(r'(<li>\d+\.[^<]+</li>)+', r'<ol>\g<0></ol>', response)
    
    # Convert markdown-style bullet lists if HTML lists weren't used
    if "<ul>" not in response:
        response = re.sub(r'([\*\-]\s+[^\n]+)(?:\n|$)', r'<li>\1</li>', response)
        response = re.sub(r'<li>[\*\-]\s+', r'<li>', response)
        if "<li>" in response and "<ul>" not in response:
            response = re.sub(r'(<li>[^<]+</li>)+', r'<ul>\g<0></ul>', response)
    
    # Ensure paragraphs have p tags
    paragraphs = re.split(r'\n\s*\n', response)
    processed_paragraphs = []
    
    for para in paragraphs:
        if not para.strip():
            continue
        if not (para.strip().startswith('<') and para.strip().endswith('>')):
            # Skip wrapping if it's already in some kind of HTML tag
            if not re.match(r'^<\w+>.*<\/\w+>$', para.strip()):
                para = f'<p>{para}</p>'
        processed_paragraphs.append(para)
    
    response = '\n'.join(processed_paragraphs)
    
    return response