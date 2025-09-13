from flask import Flask
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

    # Rate limiting
    limiter = Limiter(key_func=get_remote_address)\r?\n    limiter.init_app(app, default_limits=[app.config['RATELIMIT_DEFAULT']])

    # Register blueprints
    app.register_blueprint(api_bp, url_prefix='/api')

    return app

app = create_app()


