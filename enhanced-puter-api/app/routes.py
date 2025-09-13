"""
Main Routes - Documentation and Health Endpoints
"""

from flask import Blueprint, jsonify, render_template_string
import os
from datetime import datetime
from config import Config

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    """API documentation homepage."""

    documentation = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Enhanced Puter API Wrapper</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
            .header { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
            .endpoint { background: #f8f9fa; padding: 15px; margin: 10px 0; border-left: 4px solid #3498db; }
            .method { background: #e74c3c; color: white; padding: 3px 8px; border-radius: 3px; font-size: 0.8em; }
            .method.get { background: #27ae60; }
            .method.post { background: #e74c3c; }
            .method.delete { background: #e67e22; }
            code { background: #ecf0f1; padding: 2px 5px; border-radius: 3px; }
            .feature { background: #d5edda; padding: 10px; margin: 5px 0; border-left: 3px solid #28a745; }
        </style>
    </head>
    <body>
        <h1 class="header">🚀 Enhanced Puter API Wrapper</h1>
        <p><strong>Version:</strong> {{ version }} | <strong>Status:</strong> {{ status }} | <strong>Environment:</strong> {{ environment }}</p>

        <div class="feature">
            <h3>✨ Key Features</h3>
            <ul>
                <li>🤖 <strong>AI Services:</strong> Chat with GPT-4, GPT-5, Claude, and more</li>
                <li>🎨 <strong>Image Generation:</strong> Create images from text prompts</li>
                <li>🔍 <strong>Image Analysis:</strong> Extract text and descriptions from images</li>
                <li>🗣️ <strong>Text-to-Speech:</strong> Convert text to natural speech</li>
                <li>📁 <strong>File Management:</strong> Upload, download, and manage cloud files</li>
                <li>🗄️ <strong>Key-Value Storage:</strong> Simple data storage operations</li>
                <li>🔓 <strong>No Authentication Required:</strong> Use all features without API keys</li>
            </ul>
        </div>

        <h2>📋 API Endpoints</h2>

        <h3>🏥 Health & Information</h3>
        <div class="endpoint">
            <span class="method get">GET</span> <code>/api/health</code> - Health check endpoint
        </div>
        <div class="endpoint">
            <span class="method get">GET</span> <code>/api/user</code> - Get user information
        </div>

        <h3>🤖 AI Services</h3>
        <div class="endpoint">
            <span class="method post">POST</span> <code>/api/ai/chat</code> - AI chat completion
            <br><strong>Body:</strong> <code>{"messages": [{"role": "user", "content": "Hello!"}], "model": "gpt-4"}</code>
        </div>
        <div class="endpoint">
            <span class="method post">POST</span> <code>/api/ai/text-to-image</code> - Generate images from text
            <br><strong>Body:</strong> <code>{"prompt": "A beautiful sunset"}</code>
        </div>
        <div class="endpoint">
            <span class="method post">POST</span> <code>/api/ai/image-to-text</code> - Extract text from images
            <br><strong>Body:</strong> <code>{"image_url": "https://example.com/image.jpg"}</code>
        </div>
        <div class="endpoint">
            <span class="method post">POST</span> <code>/api/ai/text-to-speech</code> - Convert text to speech
            <br><strong>Body:</strong> <code>{"text": "Hello world", "voice": "default"}</code>
        </div>

        <h3>📁 File Management</h3>
        <div class="endpoint">
            <span class="method post">POST</span> <code>/api/files/upload</code> - Upload files to cloud storage
        </div>
        <div class="endpoint">
            <span class="method get">GET</span> <code>/api/files/download/&lt;path&gt;</code> - Download files from cloud
        </div>
        <div class="endpoint">
            <span class="method get">GET</span> <code>/api/files/list</code> - List files in cloud storage
        </div>
        <div class="endpoint">
            <span class="method delete">DELETE</span> <code>/api/files/&lt;path&gt;</code> - Delete files from cloud
        </div>

        <h3>🗄️ Key-Value Storage</h3>
        <div class="endpoint">
            <span class="method post">POST</span> <code>/api/kv/set</code> - Set key-value pair
            <br><strong>Body:</strong> <code>{"key": "user_pref", "value": {"theme": "dark"}}</code>
        </div>
        <div class="endpoint">
            <span class="method get">GET</span> <code>/api/kv/get/&lt;key&gt;</code> - Get value by key
        </div>
        <div class="endpoint">
            <span class="method delete">DELETE</span> <code>/api/kv/delete/&lt;key&gt;</code> - Delete key-value pair
        </div>

        <h2>🚀 Getting Started</h2>
        <div class="endpoint">
            <h4>Example: Chat with AI</h4>
            <pre><code>curl -X POST {{ base_url }}/api/ai/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is the meaning of life?"}], "model": "gpt-4"}'</code></pre>
        </div>

        <div class="endpoint">
            <h4>Example: Generate Image</h4>
            <pre><code>curl -X POST {{ base_url }}/api/ai/text-to-image \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A serene mountain landscape at sunset"}'</code></pre>
        </div>

        <p><em>🔗 For more examples and documentation, visit our <a href="https://github.com/your-repo">GitHub repository</a>.</em></p>

        <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666;">
            <p>Powered by <strong>Puter</strong> • Built with ❤️ for developers • Last updated: {{ timestamp }}</p>
        </footer>
    </body>
    </html>
    """

    return render_template_string(
        documentation,
        version=Config.API_VERSION,
        status="Healthy",
        environment=os.environ.get('FLASK_ENV', 'development'),
        base_url=f"http://localhost:{Config.PORT}",
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    )

@main_bp.route('/health')
def health_check():
    """Simple health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'version': Config.API_VERSION,
        'environment': os.environ.get('FLASK_ENV', 'development')
    })
