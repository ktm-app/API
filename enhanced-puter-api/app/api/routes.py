"""
Enhanced Puter API Routes
Comprehensive API endpoints for all Puter services with no user authentication required.
"""

from flask import Blueprint, request, jsonify, current_app, send_file
from werkzeug.utils import secure_filename
import os
import json
import base64
import io
from typing import Dict, Any
from app.puter_client import EnhancedPuterClient
from config import Config
import logging

logger = logging.getLogger(__name__)

api_bp = Blueprint('api', __name__)

def get_puter_client() -> EnhancedPuterClient:
    """Get authenticated Puter client instance."""
    username = current_app.config.get('PUTER_USERNAME')
    password = current_app.config.get('PUTER_PASSWORD')
    test_mode = current_app.config.get('PUTER_TEST_MODE', False)

    if not username or not password:
        raise ValueError("Puter credentials not configured")

    return EnhancedPuterClient(username, password, test_mode)

def handle_api_response(result: Dict[str, Any], success_status_code: int = 200) -> tuple:
    """Handle API response formatting."""
    if result.get('success', False):
        return jsonify(result), success_status_code
    else:
        error_message = result.get('error', 'Unknown error occurred')
        return jsonify({
            'error': error_message,
            'success': False
        }), 500

# HEALTH AND INFO ENDPOINTS

@api_bp.route('/health', methods=['GET'])
def health_check():
    """Comprehensive health check endpoint."""
    try:
        client = get_puter_client()
        health_result = client.health_check()

        return jsonify({
            'status': 'healthy' if health_result.get('success') else 'unhealthy',
            'puter_status': health_result,
            'api_version': Config.API_VERSION,
            'test_mode': Config().PUTER_TEST_MODE,
            'timestamp': health_result.get('timestamp', 'unknown')
        }), 200

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'api_version': Config.API_VERSION
        }), 500

@api_bp.route('/user', methods=['GET'])
def get_user_info():
    """Get user information."""
    try:
        client = get_puter_client()
        result = client.get_user_info()
        return handle_api_response(result)

    except Exception as e:
        logger.error(f"Get user info failed: {str(e)}")
        return jsonify({'error': str(e), 'success': False}), 500

# AI SERVICES ENDPOINTS

@api_bp.route('/ai/chat', methods=['POST'])
def ai_chat():
    """
    AI Chat completion endpoint.

    Body:
    {
        "messages": [{"role": "user", "content": "Hello!"}] or "Simple text prompt",
        "model": "gpt-4" (optional),
        "temperature": 0.7 (optional),
        "max_tokens": 1000 (optional),
        "stream": false (optional),
        "image_url": "https://example.com/image.jpg" (optional)
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided', 'success': False}), 400

        messages = data.get('messages')
        if not messages:
            return jsonify({'error': 'Messages field is required', 'success': False}), 400

        client = get_puter_client()
        result = client.ai_chat(
            messages=messages,
            model=data.get('model', 'gpt-4.1-nano'),
            temperature=data.get('temperature', 0.7),
            max_tokens=data.get('max_tokens', 1000),
            stream=data.get('stream', False),
            image_url=data.get('image_url')
        )

        return handle_api_response(result)

    except Exception as e:
        logger.error(f"AI chat failed: {str(e)}")
        return jsonify({'error': str(e), 'success': False}), 500

@api_bp.route('/ai/text-to-image', methods=['POST'])
def text_to_image():
    """
    Generate image from text prompt.

    Body:
    {
        "prompt": "A beautiful sunset over mountains"
    }
    """
    try:
        data = request.get_json()
        if not data or not data.get('prompt'):
            return jsonify({'error': 'Prompt field is required', 'success': False}), 400

        client = get_puter_client()
        result = client.text_to_image(data['prompt'])

        return handle_api_response(result)

    except Exception as e:
        logger.error(f"Text-to-image failed: {str(e)}")
        return jsonify({'error': str(e), 'success': False}), 500

@api_bp.route('/ai/image-to-text', methods=['POST'])
def image_to_text():
    """
    Extract text/description from image.

    Body:
    {
        "image_url": "https://example.com/image.jpg"
    }
    """
    try:
        data = request.get_json()
        if not data or not data.get('image_url'):
            return jsonify({'error': 'image_url field is required', 'success': False}), 400

        client = get_puter_client()
        result = client.image_to_text(data['image_url'])

        return handle_api_response(result)

    except Exception as e:
        logger.error(f"Image-to-text failed: {str(e)}")
        return jsonify({'error': str(e), 'success': False}), 500

@api_bp.route('/ai/text-to-speech', methods=['POST'])
def text_to_speech():
    """
    Convert text to speech.

    Body:
    {
        "text": "Hello, world!",
        "voice": "default" (optional)
    }
    """
    try:
        data = request.get_json()
        if not data or not data.get('text'):
            return jsonify({'error': 'text field is required', 'success': False}), 400

        client = get_puter_client()
        result = client.text_to_speech(
            text=data['text'],
            voice=data.get('voice', 'default')
        )

        if result.get('success'):
            # Return audio data as base64 encoded string
            audio_data = result.get('audio_data')
            if isinstance(audio_data, bytes):
                result['audio_data_base64'] = base64.b64encode(audio_data).decode('utf-8')
                # Remove the raw bytes for JSON serialization
                del result['audio_data']

        return handle_api_response(result)

    except Exception as e:
        logger.error(f"Text-to-speech failed: {str(e)}")
        return jsonify({'error': str(e), 'success': False}), 500

# FILE MANAGEMENT ENDPOINTS

@api_bp.route('/files/upload', methods=['POST'])
def upload_file():
    """
    Upload file to Puter cloud storage.

    Form data:
    - file: File to upload
    - path: Destination path (optional)
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided', 'success': False}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected', 'success': False}), 400

        # Get file content
        file_content = file.read()

        # Determine file path
        file_path = request.form.get('path', secure_filename(file.filename))

        client = get_puter_client()
        result = client.upload_file(file_content, file_path)

        return handle_api_response(result)

    except Exception as e:
        logger.error(f"File upload failed: {str(e)}")
        return jsonify({'error': str(e), 'success': False}), 500

@api_bp.route('/files/download/<path:file_path>', methods=['GET'])
def download_file(file_path):
    """Download file from Puter cloud storage."""
    try:
        client = get_puter_client()
        result = client.download_file(file_path)

        if result.get('success'):
            file_content = result.get('file_content')
            if isinstance(file_content, bytes):
                return send_file(
                    io.BytesIO(file_content),
                    as_attachment=True,
                    download_name=os.path.basename(file_path)
                )
            else:
                return jsonify({'error': 'Invalid file content', 'success': False}), 500
        else:
            return handle_api_response(result)

    except Exception as e:
        logger.error(f"File download failed: {str(e)}")
        return jsonify({'error': str(e), 'success': False}), 500

@api_bp.route('/files/list', methods=['GET'])
def list_files():
    """List files in Puter cloud storage."""
    try:
        path = request.args.get('path', '/')

        client = get_puter_client()
        result = client.list_files(path)

        return handle_api_response(result)

    except Exception as e:
        logger.error(f"File listing failed: {str(e)}")
        return jsonify({'error': str(e), 'success': False}), 500

@api_bp.route('/files/<path:file_path>', methods=['DELETE'])
def delete_file(file_path):
    """Delete file from Puter cloud storage."""
    try:
        client = get_puter_client()
        result = client.delete_file(file_path)

        return handle_api_response(result)

    except Exception as e:
        logger.error(f"File deletion failed: {str(e)}")
        return jsonify({'error': str(e), 'success': False}), 500

# KEY-VALUE STORAGE ENDPOINTS

@api_bp.route('/kv/set', methods=['POST'])
def kv_set():
    """
    Set key-value pair.

    Body:
    {
        "key": "user_preferences",
        "value": {"theme": "dark", "language": "en"}
    }
    """
    try:
        data = request.get_json()
        if not data or not data.get('key'):
            return jsonify({'error': 'key field is required', 'success': False}), 400

        client = get_puter_client()
        result = client.kv_set(
            key=data['key'],
            value=data.get('value')
        )

        return handle_api_response(result)

    except Exception as e:
        logger.error(f"KV set failed: {str(e)}")
        return jsonify({'error': str(e), 'success': False}), 500

@api_bp.route('/kv/get/<key>', methods=['GET'])
def kv_get(key):
    """Get value by key."""
    try:
        client = get_puter_client()
        result = client.kv_get(key)

        return handle_api_response(result)

    except Exception as e:
        logger.error(f"KV get failed: {str(e)}")
        return jsonify({'error': str(e), 'success': False}), 500

@api_bp.route('/kv/delete/<key>', methods=['DELETE'])
def kv_delete(key):
    """Delete key-value pair."""
    try:
        client = get_puter_client()
        result = client.kv_delete(key)

        return handle_api_response(result)

    except Exception as e:
        logger.error(f"KV delete failed: {str(e)}")
        return jsonify({'error': str(e), 'success': False}), 500

# UTILITY ENDPOINTS

@api_bp.route('/usage', methods=['GET'])
def get_usage():
    """Get API usage statistics."""
    return jsonify({
        'message': 'Usage tracking not implemented yet',
        'note': 'This endpoint will provide usage analytics in future versions',
        'success': True
    })

@api_bp.route('/models', methods=['GET'])
def list_models():
    """List available AI models."""
    return jsonify({
        'models': {
            'chat': [
                'gpt-5', 'gpt-5-mini', 'gpt-5-nano', 'gpt-5-chat-latest',
                'gpt-4.1', 'gpt-4.1-mini', 'gpt-4.1-nano',
                'gpt-4o', 'gpt-4o-mini',
                'o1', 'o1-mini', 'o1-pro',
                'o3', 'o3-mini', 'o4-mini',
                'claude', 'claude-3-5-sonnet'
            ],
            'image_generation': ['dall-e-3', 'stable-diffusion'],
            'text_to_speech': ['default', 'neural']
        },
        'default_models': {
            'chat': 'gpt-4.1-nano',
            'image_generation': 'dall-e-3',
            'text_to_speech': 'default'
        },
        'success': True
    })
