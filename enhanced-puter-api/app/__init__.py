"""
Enhanced Puter API Wrapper - Flask Application Factory
Production-ready Flask app initialization with comprehensive configuration.
"""

import logging
import os
from flask import Flask
from flask_cors import CORS
from config import get_config

def create_app(config_name=None):
    """
    Application factory pattern for creating Flask app instances.

    Args:
        config_name: Configuration name (development, production, testing)

    Returns:
        Flask application instance
    """

    # Create Flask application
    app = Flask(__name__)

    # Load configuration
    config_class = get_config(config_name)
    app.config.from_object(config_class)

    # Initialize configuration
    config_class.init_app(app)

    # Configure CORS
    CORS(app, origins=app.config.get('CORS_ORIGINS', '*'))

    # Configure logging
    configure_logging(app)

    # Initialize optional extensions
    init_extensions(app)

    # Register blueprints
    register_blueprints(app)

    # Register error handlers
    register_error_handlers(app)

    # Validate configuration
    validate_configuration(app)

    app.logger.info(f"Enhanced Puter API Wrapper initialized (env: {config_name})")

    return app

def configure_logging(app):
    """Configure application logging."""
    log_level = getattr(logging, app.config.get('LOG_LEVEL', 'INFO'))

    if app.config.get('LOG_TO_STDOUT', True):
        # Configure stdout logging for cloud platforms
        handler = logging.StreamHandler()
        handler.setLevel(log_level)

        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s in %(module)s: %(message)s'
        )
        handler.setFormatter(formatter)

        app.logger.addHandler(handler)
        app.logger.setLevel(log_level)

    # Configure root logger for the application
    logging.basicConfig(
        level=log_level,
        format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
    )

def init_extensions(app):
    """Initialize Flask extensions."""

    # Initialize rate limiting if Redis is available
    if app.config.get('RATELIMIT_ENABLED') and app.config.get('RATELIMIT_STORAGE_URL'):
        try:
            from flask_limiter import Limiter
            from flask_limiter.util import get_remote_address

            limiter = Limiter(
                app,
                key_func=get_remote_address,
                default_limits=[app.config.get('RATELIMIT_DEFAULT', '1000 per hour')],
                storage_uri=app.config.get('RATELIMIT_STORAGE_URL')
            )
            app.limiter = limiter
            app.logger.info("Rate limiting enabled with Redis storage")

        except ImportError:
            app.logger.warning("Flask-Limiter not available, rate limiting disabled")
        except Exception as e:
            app.logger.warning(f"Rate limiting initialization failed: {e}")

    # Create upload folder if it doesn't exist
    upload_folder = app.config.get('UPLOAD_FOLDER')
    if upload_folder and not os.path.exists(upload_folder):
        os.makedirs(upload_folder, exist_ok=True)

def register_blueprints(app):
    """Register Flask blueprints."""

    # Import and register API blueprints
    from app.api.routes import api_bp
    app.register_blueprint(api_bp, url_prefix='/api')

    # Import and register main routes
    from app.routes import main_bp
    app.register_blueprint(main_bp)

def register_error_handlers(app):
    """Register error handlers for the application."""

    @app.errorhandler(404)
    def not_found_error(error):
        return {
            'error': 'Not Found',
            'message': 'The requested resource was not found.',
            'status_code': 404
        }, 404

    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error(f'Server Error: {error}')
        return {
            'error': 'Internal Server Error',
            'message': 'An internal server error occurred.',
            'status_code': 500
        }, 500

    @app.errorhandler(400)
    def bad_request_error(error):
        return {
            'error': 'Bad Request',
            'message': 'The request was malformed or invalid.',
            'status_code': 400
        }, 400

    @app.errorhandler(429)
    def ratelimit_handler(e):
        return {
            'error': 'Rate Limit Exceeded',
            'message': 'Too many requests. Please try again later.',
            'status_code': 429
        }, 429

def validate_configuration(app):
    """Validate application configuration."""

    from config import Config

    # Validate Puter configuration
    config_errors = Config.validate_config()
    if config_errors:
        for error in config_errors:
            app.logger.warning(f"Configuration warning: {error}")

    # Log important configuration details
    app.logger.info(f"Server will run on {app.config.get('HOST')}:{app.config.get('PORT')}")
    app.logger.info(f"CORS origins: {app.config.get('CORS_ORIGINS')}")
    app.logger.info(f"Test mode: {app.config.get('PUTER_TEST_MODE')}")

    if not app.config.get('PUTER_USERNAME') or not app.config.get('PUTER_PASSWORD'):
        app.logger.error("PUTER_USERNAME and PUTER_PASSWORD must be set in environment variables!")
