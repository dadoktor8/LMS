"""
Celery Configuration with SSL Support for Upstash Redis
"""
from celery import Celery
import os
from dotenv import load_dotenv
import ssl

# Load environment variables from .env file
load_dotenv()

# Get Redis URL from environment
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

# Configure SSL options for rediss:// URLs (Upstash Redis)
broker_use_ssl = {
    'ssl_cert_reqs': ssl.CERT_NONE  # Don't verify SSL certificate
}

# Configure Celery with Redis as broker and result backend
celery_app = Celery(
    'lms_tasks',
    broker=redis_url,
    backend=redis_url
)

# Celery configuration
celery_app.conf.update(
    # Serialization
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    

    # Timezone
    timezone='UTC',
    enable_utc=True,
    
    # Task execution
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max per task
    
    # Worker configuration
    worker_prefetch_multiplier=1,  # Process one task at a time per worker
    worker_max_tasks_per_child=50,  # Restart worker after 50 tasks (prevents memory leaks)
    
    # Connection retry
    broker_connection_retry_on_startup=True,
    
    # SSL Configuration for rediss:// (Upstash)
    broker_use_ssl=broker_use_ssl,
    redis_backend_use_ssl=broker_use_ssl,
    
    imports=('backend.ai.ai_routes',),
)