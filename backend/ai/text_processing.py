from datetime import date, datetime
import time
import logging
import os
import re
from typing import List, Optional
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
from sqlalchemy.sql.elements import Tuple as SqlTuple
from typing import Tuple
from sqlalchemy.orm import Session
#import urllib
from backend.ai.aws_ai import S3_BUCKET_NAME, get_s3_client, load_faiss_vectorstore, upload_faiss_index_to_s3
from backend.db.models import CourseMaterial, CourseModule, CourseSubmodule, ModuleTextChunk, PDFQuotaUsage, ProcessedMaterial, TextChunk
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


def extract_pdf_page_ranges(file_path: str) -> List[dict]:
    """
    Analyze PDF structure and suggest natural module divisions
    Returns list of suggested modules with page ranges
    """
    try:
        doc = fitz.open(file_path)
        total_pages = len(doc)
        
        # Basic analysis - you can enhance this with more sophisticated detection
        suggested_modules = []
        
        # Look for potential chapter/section breaks
        chapter_pages = []
        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text()
            
            # Simple heuristics for chapter detection
            lines = text.split('\n')
            for line in lines[:10]:  # Check first 10 lines of each page
                line = line.strip().lower()
                if any(keyword in line for keyword in ['chapter', 'section', 'unit', 'module', 'part']):
                    chapter_pages.append({
                        'page': page_num + 1,
                        'title': line.strip(),
                        'text_preview': ' '.join(lines[:3])
                    })
                    break
        
        # If no clear chapters found, divide by page count
        if not chapter_pages:
            pages_per_module = max(10, total_pages // 5)  # Aim for 5 modules max
            for i in range(0, total_pages, pages_per_module):
                end_page = min(i + pages_per_module, total_pages)
                suggested_modules.append({
                    'title': f'Module {len(suggested_modules) + 1}',
                    'start_page': i + 1,
                    'end_page': end_page,
                    'page_count': end_page - i
                })
        else:
            # Create modules based on detected chapters
            for i, chapter in enumerate(chapter_pages):
                start_page = chapter['page']
                end_page = chapter_pages[i + 1]['page'] - 1 if i + 1 < len(chapter_pages) else total_pages
                
                suggested_modules.append({
                    'title': chapter['title'] or f'Module {i + 1}',
                    'start_page': start_page,
                    'end_page': end_page,
                    'page_count': end_page - start_page + 1
                })
        
        doc.close()
        return suggested_modules
        
    except Exception as e:
        logging.error(f"Error analyzing PDF structure: {e}")
        return []

def extract_text_from_pdf_pages(file_path: str, start_page: int, end_page: int) -> str:
    """Extract text from specific page range of a PDF"""
    try:
        doc = fitz.open(file_path)
        text_content = []
        
        # Ensure page numbers are within bounds
        start_page = max(1, start_page)
        end_page = min(len(doc), end_page)
        
        for page_num in range(start_page - 1, end_page):  # PyMuPDF uses 0-based indexing
            page = doc[page_num]
            page_text = page.get_text()
            if page_text.strip():  # Only add non-empty pages
                text_content.append(f"[Page {page_num + 1}]\n{page_text}")
        
        doc.close()
        return "\n\n".join(text_content)
        
    except Exception as e:
        logging.error(f"Error extracting text from PDF pages {start_page}-{end_page}: {e}")
        return ""

def extract_pdf_pages_to_file(
    source_pdf_path: str, 
    start_page: int, 
    end_page: int, 
    title: str
) -> str:
    """
    Extract specific pages from PDF and save as a new PDF file
    Returns the path to the created PDF file
    """
    try:
        
        source_doc = fitz.open(source_pdf_path)
        
        # Create new PDF with only the specified pages
        new_doc = fitz.open()
        
        # Ensure page numbers are within bounds (convert to 0-based indexing)
        start_idx = max(0, start_page - 1)
        end_idx = min(len(source_doc), end_page)
        
        # Copy pages to new document
        for page_num in range(start_idx, end_idx):
            page = source_doc[page_num]
            new_doc.insert_pdf(source_doc, from_page=page_num, to_page=page_num)
        
        # Create output filename
        temp_dir = "temp_downloads"
        os.makedirs(temp_dir, exist_ok=True)
        
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
        output_filename = f"{safe_title}_pages_{start_page}-{end_page}_{int(time.time())}.pdf"
        output_path = os.path.join(temp_dir, output_filename)
        
        # Save the new PDF
        new_doc.save(output_path)
        
        # Clean up
        new_doc.close()
        source_doc.close()
        
        return output_path
        
    except Exception as e:
        logging.error(f"Error extracting PDF pages: {e}")
        return None


# Optional: Add a cleanup task to remove old temporary files
def cleanup_temp_downloads():
    """Clean up old temporary download files (run this periodically)"""
    try:
        temp_dir = "temp_downloads"
        if not os.path.exists(temp_dir):
            return
        
        current_time = time.time()
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            if os.path.isfile(file_path):
                # Remove files older than 1 hour
                if current_time - os.path.getmtime(file_path) > 3600:
                    os.remove(file_path)
                    
    except Exception as e:
        logging.error(f"Error cleaning up temp downloads: {e}")


def process_submodule_with_quota_check(
    course_id: int, 
    submodule_id: int, 
    db: Session
) -> Tuple[bool, str, int]:
    """
    Process a specific submodule, checking quota constraints
    Returns: (success, message, pages_used)
    """
    submodule = db.query(CourseSubmodule).filter_by(id=submodule_id).first()
    if not submodule:
        return False, "Submodule not found", 0
    
    if submodule.is_processed:
        return False, "Submodule already processed", 0
    
    # Parse page range
    if not submodule.page_range:
        return False, "No page range specified for submodule", 0
    
    try:
        start_page, end_page = map(int, submodule.page_range.split('-'))
        pages_needed = end_page - start_page + 1
    except ValueError:
        return False, "Invalid page range format", 0
    
    # Check quota
    today = date.today()
    quota_usage = db.query(PDFQuotaUsage).filter(
        PDFQuotaUsage.course_id == course_id,
        PDFQuotaUsage.usage_date == today
    ).first()
    
    used_pages = quota_usage.pages_processed if quota_usage else 0
    remaining_quota = PDFQuotaConfig.DAILY_PAGE_QUOTA - used_pages
    
    if pages_needed > remaining_quota:
        return False, f"Not enough quota. Need {pages_needed} pages, have {remaining_quota}", 0
    
    try:
        # Get the source material file path
        material = submodule.material
        if not material:
            return False, "No source material linked to submodule", 0
        
        file_path = material.filepath
        
        # Handle S3 download if needed (using your existing logic)
        if not os.path.exists(file_path):
            s3_key = f"course_materials/{course_id}/{material.filename}"
            temp_dir = "temp_pdfs"
            os.makedirs(temp_dir, exist_ok=True)
            file_path = f"{temp_dir}/{material.filename}"
            
            if not download_file_from_s3(s3_key, file_path):
                return False, "Failed to download source file from S3", 0
        
        # Extract text from specified page range
        text = extract_text_from_pdf_pages(file_path, start_page, end_page)
        
        if not text.strip():
            return False, "No text content extracted from specified pages", 0
        
        # Process the text into chunks
        chunks = chunk_text(text)
        
        # Save embeddings specifically for this submodule
        save_submodule_embeddings(submodule_id, chunks, db)
        
        # Update quota usage
        if quota_usage:
            quota_usage.pages_processed += pages_needed
        else:
            quota_usage = PDFQuotaUsage(
                course_id=course_id,
                usage_date=today,
                pages_processed=pages_needed
            )
            db.add(quota_usage)
        
        # Mark submodule as processed
        submodule.is_processed = True
        submodule.processed_at = datetime.utcnow()
        submodule.metadata = {
            'pages_processed': pages_needed,
            'chunks_created': len(chunks),
            'processing_date': datetime.utcnow().isoformat()
        }
        
        db.commit()
        
        return True, f"Successfully processed {pages_needed} pages", pages_needed
        
    except Exception as e:
        logging.error(f"Error processing submodule {submodule_id}: {e}")
        return False, f"Processing failed: {str(e)}", 0

def save_submodule_embeddings(submodule_id: int, chunks: List[str], db: Session):
    """Save embeddings for a specific submodule"""
    try:
        # Clear existing chunks for this submodule
        db.query(ModuleTextChunk).filter_by(submodule_id=submodule_id).delete()
        
        # Generate embeddings (using your existing embedding logic)
        embeddings = embed_chunks_openai(chunks)  # Assuming you have this function
        
        # Save new chunks with embeddings
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_record = ModuleTextChunk(
                submodule_id=submodule_id,
                chunk_text=chunk,
                chunk_index=i,
                embedding=str(embedding.tolist()) if hasattr(embedding, 'tolist') else str(embedding)
            )
            db.add(chunk_record)
        
        db.commit()
        logging.info(f"Saved {len(chunks)} chunks for submodule {submodule_id}")
        
    except Exception as e:
        logging.error(f"Error saving submodule embeddings: {e}")
        raise

def embed_chunks_openai(chunks: List[str]) -> List[List[float]]:
    """
    Embed a list of text chunks using OpenAIEmbeddings and return
    a list of embeddings (list of floats).
    """
    try:
        embeddings_model = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        results = []
        for chunk in chunks:
            vector = embeddings_model.embed_query(chunk)
            results.append(vector)
        return results
    except Exception as e:
        logging.error(f"Error embedding chunks with OpenAI: {e}")
    # Return an empty list or raise an exception as needed
    return []

def create_modules_from_pdf_analysis(
    course_id: int, 
    material_id: int, 
    suggested_modules: List[dict], 
    db: Session
) -> int:
    """
    Create module structure based on PDF analysis
    Returns the number of modules created
    """
    created_count = 0
    
    for i, module_data in enumerate(suggested_modules):
        # Create the module
        module = CourseModule(
            course_id=course_id,
            title=module_data['title'],
            description=f"Auto-generated module covering pages {module_data['start_page']}-{module_data['end_page']}",
            order_index=i
        )
        db.add(module)
        db.flush()  # Get the ID
        
        # Create a submodule for this page range
        submodule = CourseSubmodule(
            module_id=module.id,
            material_id=material_id,
            title=f"{module_data['title']} - Content",
            page_range=f"{module_data['start_page']}-{module_data['end_page']}",
            order_index=0,
            metadata={
                'page_count': module_data['page_count'],
                'auto_generated': True
            }
        )
        db.add(submodule)
        created_count += 1
    
    db.commit()
    return created_count




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


def retrieve_course_context(course_id, query, module_id=None):
    """Helper function to retrieve context from course knowledge base.
    Optionally filter by specific module."""
    try:
        db_session = next(get_db())
        
        # If module_id is specified, get context only from that module
        if module_id:
            module = db_session.query(CourseModule).filter_by(
                id=module_id, 
                course_id=course_id, 
                is_published=True
            ).first()
            if not module:
                return {"error": f"Module {module_id} not found or not published in course {course_id}"}
            
            # Get text chunks from all published submodules of this module
            chunks = db_session.query(ModuleTextChunk).join(CourseSubmodule).filter(
                CourseSubmodule.module_id == module_id,
                CourseSubmodule.is_published == True
            ).all()
            
            if not chunks:
                return {"error": f"No content found in module '{module.title}'"}
            
            # Combine chunks for context
            context = "\n\n".join([chunk.chunk_text for chunk in chunks])
            
        else:
            # Try to use existing FAISS retrieval for entire course
            try:
                db = load_faiss_vectorstore(course_id, openai_api_key=None)  
                retriever = db.as_retriever(search_kwargs={"k": 10})
                retrieved_docs = retriever.get_relevant_documents(query)
                context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            except (FileNotFoundError, Exception):
                # If FAISS index doesn't exist, try to get content from published modules
                modules = db_session.query(CourseModule).filter_by(
                    course_id=course_id, 
                    is_published=True
                ).all()
                if not modules:
                    return {"error": "No published course content available"}
                
                # Get chunks from all published modules as fallback
                chunks = db_session.query(ModuleTextChunk).join(CourseSubmodule).join(CourseModule).filter(
                    CourseModule.course_id == course_id,
                    CourseModule.is_published == True,
                    CourseSubmodule.is_published == True
                ).limit(50).all()  # Limit to prevent overwhelming context
                
                if chunks:
                    context = "\n\n".join([chunk.chunk_text for chunk in chunks])
                else:
                    return {"error": "No processed content available"}
        
        if not context.strip():
            return {"error": "Insufficient context found for your query."}
        return context
        
    except Exception as e:
        return {"error": f"Problem loading course content: {str(e)}"}

def get_answer_from_rag_langchain_openai(query: str, course_id: int, student_id: str, module_id: Optional[int] = None) -> str:
    
    is_study_request, material_type, topic = detect_study_request(query)
    
    if is_study_request and material_type and topic:
        # Redirect to specific study material page based on type
        if material_type == "flashcards":
            study_url = f"/ai/study/flashcards?course_id={course_id}&topic={topic}"
        elif material_type == "study_guide":
            study_url = f"/ai/study/guide?course_id={course_id}&topic={topic}"
        else:
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
            
        # Try to get course context
        context_result = retrieve_course_context(course_id, query, module_id)
        
        # Check if we have an error (no content available)
        has_course_content = not isinstance(context_result, dict) or "error" not in context_result
        context = context_result if has_course_content else ""
        
        # Chat history for memory
        session_id = f"{student_id}_{course_id}"
        try:
            sql_history = SQLChatMessageHistory(session_id=session_id, connection="sqlite:///chat_history.db")
        except Exception as e:
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
        
        # Enhanced Prompt Template with fallback capability
        if has_course_content:
            if module_id:
                # Module-specific response
                prompt_template = PromptTemplate(
                    input_variables=["context", "question"],
                    template="""
                    You are Lumi, a warm, caring AI tutor helping students understand complex topics using their course materials.
                    You are currently focusing on a specific module within the course.
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
                    9. Let them know that can type "Make Flash-Cards 'Topic'" or "Make Study Guide 'Topic' and you will make one for them
                    
                    If the answer is not in the provided module materials, let them know gently and offer to help with general knowledge.
                    
                    Module Content:
                    {context}
                    
                    Student's Question:
                    {question}
                    
                    Lumi's Helpful Answer:
                    """
                )
            else:
                # Course-wide response
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
                    9. Let them know that can type "Make Flash-Cards 'Topic'" or "Make Study Guide 'Topic' and you will make one for them
                    
                    If the answer is not in the provided materials, let them know gently.
                    
                    Course Materials:
                    {context}
                    
                    Student's Question:
                    {question}
                    
                    Lumi's Helpful Answer:
                    """
                )
        else:
            # Fallback mode - no course content available
            prompt_template = PromptTemplate(
                input_variables=["question"],
                template="""
                You are Lumi, a warm, caring AI tutor. While no specific course materials are available for this course yet, 
                you can still help students learn and understand various topics using your general knowledge.
                Be thoughtful, empathetic, and encouraging. Always thank them for asking and explain clearly.
                
                FORMATTING INSTRUCTIONS:
                1. Start with a brief, friendly greeting and acknowledgment of their question
                2. Mention that you don't have specific course materials available but can still help with general knowledge
                3. Structure your response as a set of clear, numbered or bulleted points instead of dense paragraphs
                4. Use <strong> HTML tags to highlight important terms or concepts </strong>
                5. Keep each point focused on a single idea or concept
                6. If explaining a process or sequence, use numbered lists with the <ol> and <li> HTML tags
                7. For general points, use bullet points with the <ul> and <li> HTML tags
                8. For especially important information, wrap it in <div class="key-point">Important information here</div>
                9. Conclude with a brief encouraging note
                10. Let them know that can type "Make Flash-Cards 'Topic'" or "Make Study Guide 'Topic' and you will make one for them
                
                Student's Question:
                {question}
                
                Lumi's Helpful Answer:
                """
            )
        
        # Create response based on available content
        try:
            if has_course_content:
                # Use retrieval-based response
                retriever = None  # We already have context from retrieve_course_context
                qa_chain = RetrievalQA.from_llm(
                    llm=chat,
                    retriever=None,  # We'll pass context directly
                    return_source_documents=False,
                    chain_type_kwargs={"prompt": prompt_template, "verbose": False},
                    chain_type="stuff"
                )
                # Pass context directly
                result = {"result": chat.predict(prompt_template.format(context=context, question=query))}
            else:
                # Direct LLM response without course materials
                result = {"result": chat.predict(prompt_template.format(question=query))}
            
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