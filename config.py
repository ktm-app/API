import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'
    DEBUG = os.environ.get('DEBUG') or True
    # Rate limiting
    RATELIMIT_DEFAULT = "200 per day;50 per hour"
    # Logging
    LOG_LEVEL = 'INFO'