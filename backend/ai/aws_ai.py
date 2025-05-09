import logging
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import os

from fastapi import UploadFile
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS

S3_BUCKET_NAME = "intellaica-filestorage-bucket"  
S3_REGION = "us-east-2" 


def get_s3_client():
    return boto3.client(
        's3',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
        region_name=S3_REGION
    )

async def upload_file_to_s3(file: UploadFile, s3_key: str) -> bool:
    """Upload a file to AWS S3 bucket"""
    s3_client = get_s3_client()
    try:
        contents = await file.read()
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Body=contents,
            ContentType=file.content_type
        )
        return True
    except NoCredentialsError:
        # Handle authentication errors
        print("AWS credentials not available")
        return False
    except ClientError as e:
        # Handle other AWS errors
        print(f"AWS S3 error: {e}")
        return False
    except Exception as e:
        # Handle unexpected errors
        print(f"Unexpected error: {e}")
        return False
    
def generate_presigned_url(s3_key: str, expiration=3600):
    """Generate a presigned URL for an S3 object that expires after a specified time"""
    s3_client = get_s3_client()
    try:
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': S3_BUCKET_NAME,
                'Key': s3_key
            },
            ExpiresIn=expiration
        )
        return url
    except Exception as e:
        print(f"Error generating presigned URL: {e}")
        return None
    
def upload_faiss_index_to_s3(local_folder, course_id):
    """
    Uploads the FAISS index folder (index.faiss, index.pkl) to S3 under faiss_indexes/{course_id}/
    """
    s3_client = get_s3_client()
    files = ["index.faiss", "index.pkl"]
    for fname in files:
        local_file = os.path.join(local_folder, fname)
        if not os.path.exists(local_file):
            logging.error(f"❌ File {local_file} does not exist, skipping upload.")
            continue
        s3_key = f"faiss_indexes/{course_id}/{fname}"
        try:
            with open(local_file, "rb") as f:
                s3_client.upload_fileobj(f, S3_BUCKET_NAME, s3_key)
            logging.info(f"✅ Uploaded {local_file} to s3://{S3_BUCKET_NAME}/{s3_key}")
        except Exception as e:
            logging.error(f"❌ Upload failed for {local_file}: {e}")

def download_faiss_index_from_s3(course_id, dest_root="tmp"):
    """
    Downloads FAISS index files from S3 to a local folder.
    Returns the download path to use with FAISS.load_local
    """
    s3_client = get_s3_client()
    faiss_folder = os.path.join(dest_root, f"faiss_index_{course_id}")
    os.makedirs(faiss_folder, exist_ok=True)
    files = ["index.faiss", "index.pkl"]
    for fname in files:
        s3_key = f"faiss_indexes/{course_id}/{fname}"
        local_file = os.path.join(faiss_folder, fname)
        try:
            s3_client.download_file(S3_BUCKET_NAME, s3_key, local_file)
            logging.info(f"✅ Downloaded s3://{S3_BUCKET_NAME}/{s3_key} to {local_file}")
        except Exception as e:
            logging.error(f"❌ Download failed for {s3_key}: {e}")
    return faiss_folder

def load_faiss_index_from_s3(course_id, embeddings_model, temp_dir="tmp"):
    """
    Downloads the FAISS index from S3, then loads it with LangChain.
    """
    from langchain.vectorstores.faiss import FAISS
    local_path = download_faiss_index_from_s3(course_id, temp_dir)
    logging.debug(f"Attempting to load FAISS index from {local_path}")
    vectorstore = FAISS.load_local(local_path, embeddings_model)
    logging.info(f"✅ Loaded FAISS index for course {course_id} from {local_path}")
    return vectorstore

def ensure_faiss_index_local(course_id: int, embeddings_model=None, temp_dir="tmp") -> str:
    """
    Ensure the FAISS index is present locally for a given course_id; download from S3 if missing.
    Returns the local directory path.
    """
    local_dir = os.path.join(temp_dir, f"faiss_index_{course_id}")
    if os.path.exists(os.path.join(local_dir, "index.faiss")) and os.path.exists(os.path.join(local_dir, "index.pkl")):
        logging.info(f"✅ FAISS index already present locally: {local_dir}")
        return local_dir
    
    logging.info(f"⬇️ FAISS index not found locally, downloading from S3 for course {course_id} ...")
    os.makedirs(local_dir, exist_ok=True)
    s3_client = get_s3_client()
    for fname in ["index.faiss", "index.pkl"]:
        s3_key = f"faiss_indexes/{course_id}/{fname}"
        dest_path = os.path.join(local_dir, fname)
        try:
            s3_client.download_file(S3_BUCKET_NAME, s3_key, dest_path)
            logging.info(f"✅ Downloaded {s3_key} to {dest_path}")
        except Exception as e:
            logging.error(f"❌ Could not download {s3_key} from S3: {e}")
            raise FileNotFoundError(f"Missing FAISS index for course {course_id}: {s3_key}")
    return local_dir

def load_faiss_vectorstore(course_id: int, openai_api_key=None, temp_dir="tmp"):
    """
    Downloads (if needed) and loads the vectorstore for a given course id.
    Returns the FAISS vectorstore.
    """
    if openai_api_key is None:
        openai_api_key = os.getenv("OPENAI_API_KEY")
    embeddings_model = OpenAIEmbeddings(api_key=openai_api_key)
    local_dir = ensure_faiss_index_local(course_id, embeddings_model, temp_dir=temp_dir)
    vectorstore = FAISS.load_local(local_dir, embeddings_model, allow_dangerous_deserialization=True)
    logging.info(f"✅ Loaded FAISS vectorstore for course {course_id} from {local_dir}")
    return vectorstore

def upload_file_to_s3_from_path(file_path, s3_key):
    s3_client = get_s3_client()
    with open(file_path, "rb") as f:
        s3_client.upload_fileobj(f, S3_BUCKET_NAME, s3_key)
    return True

def generate_s3_download_link(s3_key, filename, expiration=600):
    s3_client = get_s3_client()
    try:
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={
                'Bucket': S3_BUCKET_NAME,
                'Key': s3_key,
                # force downloadable filename:
                'ResponseContentDisposition': f'attachment; filename="{filename}"'
            },
            ExpiresIn=expiration
        )
        return url
    except Exception as e:
        print(f"Error generating presigned URL: {e}")
        return None