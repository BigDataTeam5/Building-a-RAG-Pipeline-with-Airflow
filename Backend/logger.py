import os
import logging
from logging.handlers import RotatingFileHandler
import time
from pathlib import Path

# Create logs directory if it doesn't exist
logs_dir = Path(__file__).parent / "logs"
logs_dir.mkdir(exist_ok=True)

# Configure basic logging format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def setup_logger(name, log_file, level=logging.INFO):
    """
    Setup logger with rotating file handler
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Don't propagate to root logger
    logger.propagate = False
    
    # Create handlers if they don't exist
    if not logger.handlers:
        # Create log file handler
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5           # Keep 5 backup logs
        )
        
        # Create formatter
        formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
        file_handler.setFormatter(formatter)
        
        # Add file handler to logger
        logger.addHandler(file_handler)
        
        # Also log to console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

# Create different loggers for different components
api_logger = setup_logger('api', logs_dir / 'api.log')
pdf_logger = setup_logger('pdf_processing', logs_dir / 'pdf_processing.log')
s3_logger = setup_logger('s3', logs_dir / 's3_operations.log')
error_logger = setup_logger('errors', logs_dir / 'errors.log', level=logging.ERROR)

# Create a request logger to track all API requests
request_logger = setup_logger('requests', logs_dir / 'requests.log')

def log_request(request_details):
    """Log API request details"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    request_info = f"{timestamp} - {request_details}"
    request_logger.info(request_info)

def log_error(error_message, error=None):
    """Log error details"""
    if error:
        error_logger.error(f"{error_message}: {str(error)}", exc_info=True)
    else:
        error_logger.error(error_message)