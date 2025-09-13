"""
Utility Functions and Helper Methods
Common utilities for the Enhanced Puter API Wrapper.
"""

import logging
import functools
import time
import json
import base64
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
import os
import hashlib
import mimetypes

logger = logging.getLogger(__name__)

# DECORATORS

def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff_factor: float = 2.0):
    """
    Decorator to retry function execution on failure with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay after each retry
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        wait_time = delay * (backoff_factor ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")

            raise last_exception
        return wrapper
    return decorator

def log_execution_time(func: Callable) -> Callable:
    """Decorator to log function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed successfully in {execution_time:.2f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.2f}s: {e}")
            raise
    return wrapper

def handle_exceptions(default_return: Any = None, log_error: bool = True):
    """
    Decorator to handle exceptions gracefully.

    Args:
        default_return: Default value to return on exception
        log_error: Whether to log the error
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.error(f"Exception in {func.__name__}: {e}")
                return default_return
        return wrapper
    return decorator

# VALIDATION UTILITIES

def validate_file_size(file_content: bytes, max_size_mb: int = 100) -> bool:
    """
    Validate file size.

    Args:
        file_content: File content as bytes
        max_size_mb: Maximum file size in MB

    Returns:
        True if file size is valid
    """
    max_size_bytes = max_size_mb * 1024 * 1024
    return len(file_content) <= max_size_bytes

def validate_image_url(url: str) -> bool:
    """
    Validate image URL format.

    Args:
        url: URL to validate

    Returns:
        True if URL format is valid
    """
    if not url or not isinstance(url, str):
        return False

    # Check if it's a valid HTTP/HTTPS URL
    if not url.startswith(('http://', 'https://')):
        return False

    # Check for common image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg']
    url_lower = url.lower()

    # If URL has extension, check if it's an image extension
    if '.' in url_lower.split('/')[-1]:
        return any(url_lower.endswith(ext) for ext in image_extensions)

    # If no extension, assume it could be a valid image URL
    return True

def validate_text_length(text: str, min_length: int = 1, max_length: int = 5000) -> bool:
    """
    Validate text length.

    Args:
        text: Text to validate
        min_length: Minimum text length
        max_length: Maximum text length

    Returns:
        True if text length is valid
    """
    if not text or not isinstance(text, str):
        return False
    return min_length <= len(text) <= max_length

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file operations.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove or replace unsafe characters
    unsafe_chars = ['<', '>', ':', '"', '/', '\', '|', '?', '*']
    sanitized = filename

    for char in unsafe_chars:
        sanitized = sanitized.replace(char, '_')

    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(' .')

    # Ensure filename is not empty
    if not sanitized:
        sanitized = f"file_{int(time.time())}"

    return sanitized

# DATA CONVERSION UTILITIES

def encode_file_to_base64(file_content: bytes) -> str:
    """
    Encode file content to base64 string.

    Args:
        file_content: File content as bytes

    Returns:
        Base64 encoded string
    """
    return base64.b64encode(file_content).decode('utf-8')

def decode_base64_to_file(base64_string: str) -> bytes:
    """
    Decode base64 string to file content.

    Args:
        base64_string: Base64 encoded string

    Returns:
        File content as bytes
    """
    return base64.b64decode(base64_string.encode('utf-8'))

def generate_file_hash(file_content: bytes) -> str:
    """
    Generate SHA256 hash of file content.

    Args:
        file_content: File content as bytes

    Returns:
        SHA256 hash as hex string
    """
    return hashlib.sha256(file_content).hexdigest()

def get_file_mime_type(filename: str) -> str:
    """
    Get MIME type for file based on extension.

    Args:
        filename: Filename with extension

    Returns:
        MIME type string
    """
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type or 'application/octet-stream'

# RESPONSE UTILITIES

def create_api_response(data: Any = None, error: str = None, success: bool = None, 
                       status_code: int = 200, additional_data: Dict = None) -> Dict[str, Any]:
    """
    Create standardized API response.

    Args:
        data: Response data
        error: Error message
        success: Success status (auto-determined if None)
        status_code: HTTP status code
        additional_data: Additional data to include

    Returns:
        Standardized API response dictionary
    """
    if success is None:
        success = error is None

    response = {
        'success': success,
        'timestamp': datetime.utcnow().isoformat(),
        'status_code': status_code
    }

    if success:
        if data is not None:
            if isinstance(data, dict):
                response.update(data)
            else:
                response['data'] = data
    else:
        response['error'] = error or 'Unknown error occurred'

    if additional_data and isinstance(additional_data, dict):
        response.update(additional_data)

    return response

def paginate_results(items: List[Any], page: int = 1, per_page: int = 10) -> Dict[str, Any]:
    """
    Paginate list of items.

    Args:
        items: List of items to paginate
        page: Page number (1-indexed)
        per_page: Items per page

    Returns:
        Paginated results with metadata
    """
    total_items = len(items)
    total_pages = (total_items + per_page - 1) // per_page

    start_index = (page - 1) * per_page
    end_index = start_index + per_page

    paginated_items = items[start_index:end_index]

    return {
        'items': paginated_items,
        'pagination': {
            'current_page': page,
            'per_page': per_page,
            'total_items': total_items,
            'total_pages': total_pages,
            'has_next': page < total_pages,
            'has_prev': page > 1
        }
    }

# LOGGING UTILITIES

def setup_structured_logging(log_level: str = 'INFO', log_format: str = None) -> None:
    """
    Setup structured logging for the application.

    Args:
        log_level: Logging level
        log_format: Custom log format (optional)
    """
    if log_format is None:
        log_format = '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def log_request_info(endpoint: str, method: str, data: Dict = None, user_info: Dict = None) -> None:
    """
    Log API request information.

    Args:
        endpoint: API endpoint
        method: HTTP method
        data: Request data (sanitized)
        user_info: User information
    """
    log_data = {
        'endpoint': endpoint,
        'method': method,
        'timestamp': datetime.utcnow().isoformat()
    }

    if data:
        # Sanitize sensitive data
        sanitized_data = sanitize_log_data(data)
        log_data['request_data'] = sanitized_data

    if user_info:
        log_data['user'] = user_info

    logger.info(f"API Request: {json.dumps(log_data)}")

def sanitize_log_data(data: Dict) -> Dict:
    """
    Remove sensitive information from log data.

    Args:
        data: Original data dictionary

    Returns:
        Sanitized data dictionary
    """
    sensitive_keys = ['password', 'token', 'key', 'secret', 'auth', 'credential']
    sanitized = {}

    for key, value in data.items():
        if any(sensitive in key.lower() for sensitive in sensitive_keys):
            sanitized[key] = '*' * len(str(value)) if value else None
        elif isinstance(value, dict):
            sanitized[key] = sanitize_log_data(value)
        else:
            sanitized[key] = value

    return sanitized

# CONFIGURATION UTILITIES

def load_environment_config() -> Dict[str, Any]:
    """
    Load configuration from environment variables.

    Returns:
        Configuration dictionary
    """
    config = {}

    # Required environment variables
    required_vars = ['PUTER_USERNAME', 'PUTER_PASSWORD']

    for var in required_vars:
        value = os.environ.get(var)
        if value:
            config[var] = value
        else:
            logger.warning(f"Required environment variable {var} not set")

    # Optional environment variables with defaults
    optional_vars = {
        'FLASK_ENV': 'development',
        'PORT': '5000',
        'HOST': '0.0.0.0',
        'LOG_LEVEL': 'INFO',
        'PUTER_TEST_MODE': 'false',
        'CORS_ORIGINS': '*'
    }

    for var, default in optional_vars.items():
        config[var] = os.environ.get(var, default)

    return config

# PERFORMANCE UTILITIES

def measure_performance(func: Callable) -> Callable:
    """
    Decorator to measure and log function performance.

    Args:
        func: Function to measure

    Returns:
        Wrapped function with performance logging
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        start_memory = get_memory_usage()

        try:
            result = func(*args, **kwargs)

            end_time = time.perf_counter()
            end_memory = get_memory_usage()

            execution_time = end_time - start_time
            memory_diff = end_memory - start_memory

            logger.debug(f"Performance - {func.__name__}: {execution_time:.4f}s, Memory: {memory_diff:.2f}MB")

            return result

        except Exception as e:
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            logger.error(f"Performance - {func.__name__} failed after {execution_time:.4f}s: {e}")
            raise

    return wrapper

def get_memory_usage() -> float:
    """
    Get current memory usage in MB.

    Returns:
        Memory usage in MB
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return 0.0

# CACHE UTILITIES

class SimpleCache:
    """Simple in-memory cache implementation."""

    def __init__(self, default_ttl: int = 300):
        """
        Initialize cache.

        Args:
            default_ttl: Default time-to-live in seconds
        """
        self.cache = {}
        self.default_ttl = default_ttl

    def get(self, key: str) -> Any:
        """Get value from cache."""
        if key in self.cache:
            value, expiry = self.cache[key]
            if datetime.now() < expiry:
                return value
            else:
                del self.cache[key]
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        ttl = ttl or self.default_ttl
        expiry = datetime.now() + timedelta(seconds=ttl)
        self.cache[key] = (value, expiry)

    def delete(self, key: str) -> None:
        """Delete value from cache."""
        self.cache.pop(key, None)

    def clear(self) -> None:
        """Clear all cached values."""
        self.cache.clear()

    def cleanup_expired(self) -> None:
        """Remove expired entries from cache."""
        now = datetime.now()
        expired_keys = [key for key, (_, expiry) in self.cache.items() if now >= expiry]
        for key in expired_keys:
            del self.cache[key]

# Global cache instance
cache = SimpleCache()
