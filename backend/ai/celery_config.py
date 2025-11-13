"""
Celery Configuration with SSL Support for Upstash Redis
"""
from celery import Celery
import os
from dotenv import load_dotenv
import ssl

# Load environment variables from .env file
load_dotenv()

# Get Redis URL and environment
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
is_production = os.getenv('ENVIRONMENT', 'development') == 'production'

# Configure SSL options based on environment
if is_production:
    # Production: Require SSL certificate verification
    broker_use_ssl = {
        'ssl_cert_reqs': ssl.CERT_REQUIRED,
        'ssl_ca_certs': None,  # Use system default CA certs
    }
else:
    # Development: Don't verify SSL (for local testing)
    broker_use_ssl = {
        'ssl_cert_reqs': ssl.CERT_NONE
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
    task_time_limit=1800,  # 30 minutes max per task (reduced from 1 hour)
    task_soft_time_limit=1500,  # 25 minutes soft limit (warning before hard kill)
    
    # Worker configuration
    worker_prefetch_multiplier=1,  # Process one task at a time per worker
    worker_max_tasks_per_child=20,  # Restart worker after 20 tasks (prevents memory leaks)
    worker_disable_rate_limits=True,  # Disable rate limits for better performance
    
    # Connection retry
    broker_connection_retry_on_startup=True,
    broker_connection_retry=True,
    broker_connection_max_retries=10,
    
    # Task execution settings
    task_acks_late=True,  # Acknowledge task after completion (safer)
    task_reject_on_worker_lost=True,  # Re-queue task if worker dies
    
    # Result backend settings
    result_expires=3600,  # Results expire after 1 hour
    result_persistent=False,  # Don't persist results to disk
    
    # SSL Configuration for rediss:// (Upstash)
    broker_use_ssl=broker_use_ssl,
    redis_backend_use_ssl=broker_use_ssl,
    
    # Task imports
    imports=('backend.ai.ai_routes',),
    
    # Performance optimizations
    worker_pool_restarts=True,  # Enable pool restarts
)

# Optional: Add result backend configuration
celery_app.conf.result_backend_transport_options = {
    'retry_policy': {
        'timeout': 5.0
    }
}