from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from config import Config
from .api import api_bp
import logging
import os

def create_app(config_class=Config):
    app = Flask(__name__, static_folder='../static')
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
                'playground': '/index.html',
                'api_root': '/api/',
                'health': '/api/health',
                'models': '/api/models',
                'chat': '/api/chat (POST)',
                'chat_with_model': '/api/chat/<model> (POST)',
                'history': '/api/history/<session_id>'
            }
        })

    # Serve the HTML playground
    @app.route('/index.html')
    def playground():
        return send_from_directory(app.static_folder, 'index.html')

    # Serve static files
    @app.route('/static/<path:filename>')
    def serve_static(filename):
        return send_from_directory(app.static_folder, filename)

    # Register blueprints
    app.register_blueprint(api_bp, url_prefix='/api')

    return app

app = create_app()