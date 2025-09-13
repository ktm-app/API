"""
Enhanced Puter API Wrapper Configuration
Production-ready configuration with proper environment handling for Render deployment.
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv(os.path.join(basedir, '.env'))

class Config:
    """Base configuration class."""

    # Flask Core Configuration
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'puter-api-secret-key-change-in-production'

    # Puter Configuration - Internal authentication (no user auth required)
    PUTER_USERNAME = os.environ.get('PUTER_USERNAME')
    PUTER_PASSWORD = os.environ.get('PUTER_PASSWORD')

    if not PUTER_USERNAME or not PUTER_PASSWORD:
        print("⚠️  WARNING: PUTER_USERNAME and PUTER_PASSWORD environment variables not set!")
        print("   Set these in your environment or .env file for the API to work properly.")

    # API Configuration
    API_TITLE = "Enhanced Puter API Wrapper"
    API_VERSION = "2.0.0"
    API_DESCRIPTION = "Production-ready API wrapper for Puter services with comprehensive AI and cloud capabilities"

    # Server Configuration  
    PORT = int(os.environ.get('PORT', 5000))
    HOST = os.environ.get('HOST', '0.0.0.0')

    # CORS Configuration
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*')

    # Rate Limiting (optional - works without Redis)
    RATELIMIT_ENABLED = os.environ.get('RATELIMIT_ENABLED', 'false').lower() == 'true'
    RATELIMIT_STORAGE_URL = os.environ.get('REDIS_URL')  # Optional Redis URL
    RATELIMIT_DEFAULT = os.environ.get('RATELIMIT_DEFAULT', '1000 per hour')

    # File Upload Configuration
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size
    UPLOAD_FOLDER = os.path.join(basedir, 'uploads')

    # Logging Configuration
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_TO_STDOUT = os.environ.get('LOG_TO_STDOUT', 'true').lower() == 'true'

    # Health Check Configuration
    HEALTH_CHECK_ENABLED = True

    # Puter API Configuration
    PUTER_TEST_MODE = os.environ.get('PUTER_TEST_MODE', 'false').lower() == 'true'
    PUTER_TIMEOUT = int(os.environ.get('PUTER_TIMEOUT', 30))

    @staticmethod
    def init_app(app):
        """Initialize application with this config."""
        pass

    @classmethod
    def validate_config(cls):
        """Validate configuration and return any errors."""
        errors = []

        if not cls.PUTER_USERNAME:
            errors.append("PUTER_USERNAME environment variable is required")
        if not cls.PUTER_PASSWORD:
            errors.append("PUTER_PASSWORD environment variable is required")

        return errors

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    PUTER_TEST_MODE = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production configuration for Render and other platforms."""
    DEBUG = False
    PUTER_TEST_MODE = False

    @staticmethod
    def init_app(app):
        """Production-specific initialization."""
        Config.init_app(app)

        # Log to stdout for cloud platforms
        import logging
        from logging import StreamHandler
        handler = StreamHandler()
        handler.setLevel(logging.INFO)
        app.logger.addHandler(handler)

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    PUTER_TEST_MODE = True
    PUTER_USERNAME = 'test_user'
    PUTER_PASSWORD = 'test_password'

# Configuration mapping
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config(config_name: Optional[str] = None) -> Config:
    """Get configuration class based on environment."""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')

    return config.get(config_name, config['default'])
