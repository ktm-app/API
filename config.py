import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-here'
    SESSION_SECRET = os.environ.get('SESSION_SECRET') or 'default-session-secret'
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    RATELIMIT_DEFAULT = os.environ.get('RATELIMIT_DEFAULT', '100 per minute')
    
    # AI Provider Settings
    ENABLE_G4F = True
    ENABLE_GPT4ALL = True
    ENABLE_OLLAMA = True
    ENABLE_POLLINATIONS = True
    
    # Local Model Settings
    GPT4ALL_MODEL_PATH = os.environ.get('GPT4ALL_MODEL_PATH', 'models/')
    OLLAMA_BASE_URL = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
    
    # Provider Priority (higher number = higher priority)
    PROVIDER_PRIORITY = {
        'g4f': 4,
        'ollama': 3,
        'gpt4all': 2,
        'pollinations': 1
    }