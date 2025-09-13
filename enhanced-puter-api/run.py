#!/usr/bin/env python3
"""
Enhanced Puter API Wrapper - Production Entry Point
Production-ready Flask application entry point with comprehensive configuration.
"""

import os
import sys
import logging
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app
from config import get_config

def main():
    """Main application entry point."""

    # Get configuration
    flask_env = os.environ.get('FLASK_ENV', 'production')
    config_class = get_config(flask_env)

    # Create Flask application
    app = create_app(flask_env)

    # Display startup information
    print_startup_info(app, config_class, flask_env)

    # Validate configuration
    validate_environment(config_class)

    # Start the application
    if flask_env == 'development':
        # Development server
        app.run(
            host=config_class.HOST,
            port=config_class.PORT,
            debug=config_class.DEBUG,
            threaded=True
        )
    else:
        # Production server (will be handled by gunicorn in deployment)
        app.logger.info(f"Production app ready on {config_class.HOST}:{config_class.PORT}")
        return app

def print_startup_info(app, config_class, flask_env):
    """Print comprehensive startup information."""

    startup_banner = f"""
{'='*80}
🚀 ENHANCED PUTER API WRAPPER
{'='*80}

📋 Configuration:
   • Environment: {flask_env}
   • Host: {config_class.HOST}
   • Port: {config_class.PORT}
   • Debug Mode: {getattr(config_class, 'DEBUG', False)}
   • Test Mode: {getattr(config_class, 'PUTER_TEST_MODE', False)}
   • API Version: {getattr(config_class, 'API_VERSION', '2.0.0')}

🔗 Available Endpoints:
   • GET  /                          - API documentation homepage
   • GET  /api/health               - Health check endpoint
   • GET  /api/user                 - Get user information
   • GET  /api/models               - List available AI models

🤖 AI Services:
   • POST /api/ai/chat              - AI chat completion (GPT-4, GPT-5, Claude, etc.)
   • POST /api/ai/text-to-image     - Generate images from text prompts
   • POST /api/ai/image-to-text     - Extract text/descriptions from images
   • POST /api/ai/text-to-speech    - Convert text to speech audio

📁 File Management:
   • POST /api/files/upload         - Upload files to Puter cloud storage
   • GET  /api/files/download/<path> - Download files from cloud storage
   • GET  /api/files/list           - List files in cloud storage
   • DELETE /api/files/<path>       - Delete files from cloud storage

🗄️ Key-Value Storage:
   • POST /api/kv/set               - Set key-value pairs
   • GET  /api/kv/get/<key>         - Get values by key
   • DELETE /api/kv/delete/<key>    - Delete key-value pairs

✨ Key Features:
   ✅ No user authentication required - your API handles Puter auth internally
   ✅ Multiple AI models support (GPT-4, GPT-5, Claude, o1, o3, etc.)
   ✅ Comprehensive file management with cloud storage
   ✅ Real-time image generation and analysis
   ✅ Text-to-speech conversion capabilities
   ✅ Production-ready with comprehensive error handling
   ✅ CORS enabled for web applications
   ✅ Structured logging and monitoring
   ✅ Rate limiting support (optional Redis)

🔧 Environment Variables Required:
   • PUTER_USERNAME - Your Puter username (REQUIRED)
   • PUTER_PASSWORD - Your Puter password (REQUIRED)

🔧 Optional Environment Variables:
   • FLASK_ENV - Environment mode (development/production)
   • PORT - Server port (default: 5000)
   • HOST - Server host (default: 0.0.0.0)
   • SECRET_KEY - Flask secret key
   • CORS_ORIGINS - CORS allowed origins
   • LOG_LEVEL - Logging level (INFO/DEBUG/WARNING/ERROR)
   • PUTER_TEST_MODE - Enable test mode (true/false)
   • REDIS_URL - Redis URL for rate limiting (optional)

📖 Quick Start Examples:

   Chat with AI:
   curl -X POST http://localhost:{config_class.PORT}/api/ai/chat \
     -H "Content-Type: application/json" \
     -d '{{"messages": [{{"role": "user", "content": "Hello, AI!"}}], "model": "gpt-4"}}'

   Generate Image:
   curl -X POST http://localhost:{config_class.PORT}/api/ai/text-to-image \
     -H "Content-Type: application/json" \
     -d '{{"prompt": "A beautiful sunset over mountains"}}'

   Health Check:
   curl http://localhost:{config_class.PORT}/api/health

🌐 Access Your API:
   • Local: http://localhost:{config_class.PORT}/
   • Documentation: http://localhost:{config_class.PORT}/
   • Health Check: http://localhost:{config_class.PORT}/api/health

📊 Powered by Puter.js - The open-source cloud operating system
   • Website: https://puter.com
   • Documentation: https://docs.puter.com
   • GitHub: https://github.com/HeyPuter/puter

{'='*80}
⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}
"""

    print(startup_banner)

def validate_environment(config_class):
    """Validate environment configuration and show warnings."""

    print("🔍 Environment Validation:")

    # Check required environment variables
    required_vars = ['PUTER_USERNAME', 'PUTER_PASSWORD']
    missing_vars = []

    for var in required_vars:
        if not getattr(config_class, var, None):
            missing_vars.append(var)

    if missing_vars:
        print(f"   ❌ Missing required environment variables: {', '.join(missing_vars)}")
        print(f"   ⚠️  Set these variables for the API to function properly:")
        for var in missing_vars:
            print(f"      export {var}=your_value_here")
        print()
    else:
        print(f"   ✅ All required environment variables are set")

    # Check optional configurations
    if hasattr(config_class, 'PUTER_TEST_MODE') and config_class.PUTER_TEST_MODE:
        print(f"   ℹ️  Test mode is enabled - API calls will use test endpoints")

    if hasattr(config_class, 'DEBUG') and config_class.DEBUG:
        print(f"   ⚠️  Debug mode is enabled - disable in production")

    if hasattr(config_class, 'RATELIMIT_ENABLED') and config_class.RATELIMIT_ENABLED:
        print(f"   ℹ️  Rate limiting is enabled")
    else:
        print(f"   ℹ️  Rate limiting is disabled (set REDIS_URL to enable)")

    print()

def setup_production_logging():
    """Setup production logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

if __name__ == '__main__':
    try:
        # Setup logging for direct execution
        setup_production_logging()

        # Start the application
        app = main()

        # If we get here, we're in production mode and returning the app
        if app:
            print("🚀 Application initialized successfully!")

    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Failed to start application: {e}")
        print(f"\nPlease check your configuration and ensure:")
        print(f"   1. PUTER_USERNAME environment variable is set")
        print(f"   2. PUTER_PASSWORD environment variable is set")
        print(f"   3. All dependencies are installed: pip install -r requirements.txt")
        print(f"\nFor more help, visit: https://docs.puter.com")
        sys.exit(1)

# For WSGI servers (Gunicorn, uWSGI, etc.)
application = main() if __name__ != '__main__' else None
app = application  # Alternative name for WSGI servers
