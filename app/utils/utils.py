import logging

def validate_request(data):
    if not isinstance(data, dict):
        raise ValueError("Request data must be a dictionary")
    if 'messages' not in data:
        raise ValueError("Messages are required")
    if not isinstance(data['messages'], list):
        raise ValueError("Messages must be a list")

def log_request(endpoint, data):
    logging.info(f"Request to {endpoint}: {str(data)[:100]}...")