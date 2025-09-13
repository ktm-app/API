from flask import Flask, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from config import Config
from .api import api_bp
import logging

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Logging
    logging.basicConfig(level=getattr(logging, app.config['LOG_LEVEL']))

    # CORS
    CORS(app)

    # Rate limiting - Fixed initialization with proper keyword arguments
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=[app.config['RATELIMIT_DEFAULT']]
    )

    # Add root route
    @app.route('/')
    def root():
        return jsonify({
            'message': 'GPT API Server',
            'status': 'running',
            'version': '1.0.0',
            'endpoints': {
                'api_root': '/api/',
                'health': '/api/health',
                'models': '/api/models',
                'chat': '/api/chat',
                'chat_with_model': '/api/chat/<model>',
                'history': '/api/history/<session_id>'
            }
        })

    # Register blueprints
    app.register_blueprint(api_bp, url_prefix='/api')

    return app

app = create_app()