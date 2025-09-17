from flask import request, jsonify, Response
import asyncio
import json
from datetime import datetime
import uuid
import logging
import warnings
from . import api_bp
from app.utils.providers import AIProviderManager

# Initialize logger
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Store chat history in memory (for demo purposes)
chat_history = {}

# Initialize AI Provider Manager
provider_manager = AIProviderManager()
provider_manager.initialize_providers()

def get_session_id():
    """Get or create session ID"""
    session_id = request.args.get('session_id')
    if not session_id and request.method == 'POST':
        try:
            json_data = request.get_json(silent=True)
            if json_data:
                session_id = json_data.get('session_id')
        except:
            pass
    
    if not session_id:
        session_id = request.headers.get('X-Session-ID', 'browser-session')
    
    return session_id

def save_to_history(session_id, role, content, model=None):
    """Save message to chat history"""
    if session_id not in chat_history:
        chat_history[session_id] = []
    
    chat_history[session_id].append({
        'role': role,
        'content': content,
        'model': model,
        'timestamp': datetime.now().isoformat()
    })

def verify_model_functionality(model: str) -> dict:
    """Verify if a model is working by asking 'who are you' question"""
    verification_messages = [
        {
            "role": "user", 
            "content": "Who are you? What is your name and who developed you? Please respond clearly and specifically."
        }
    ]
    
    try:
        response, status, provider = provider_manager.get_response_sync(model, verification_messages)
        
        if status == "success" and response:
            # Check if response contains meaningful information (not generic)
            generic_indicators = [
                "I'm working through g4f API",
                "fallback response",
                "How can I help you today?",
                "You said:"
            ]
            
            is_generic = any(indicator.lower() in response.lower() for indicator in generic_indicators)
            
            return {
                'working': not is_generic,
                'provider': provider,
                'response_preview': response[:100] + "..." if len(response) > 100 else response,
                'status': 'verified' if not is_generic else 'generic'
            }
        else:
            return {
                'working': False,
                'provider': provider,
                'error': f"Failed with status: {status}",
                'status': 'failed'
            }
    except Exception as e:
        return {
            'working': False,
            'provider': None,
            'error': str(e),
            'status': 'error'
        }

@api_bp.route('/', methods=['GET', 'POST'])
def api_info():
    """API information and available endpoints"""
    return jsonify({
        "api": "Enhanced Multi-Provider AI Chat API",
        "version": "2.0",
        "status": "active",
        "providers": list(provider_manager.provider_status.keys()),
        "endpoints": {
            "/api/health": "Health check",
            "/api/models": "List available models",
            "/api/providers": "Get provider status",
            "/api/verify/<model>": "Verify specific model functionality",
            "/api/chat": "Chat with default model (POST: {message, model?, session_id?})",
            "/api/chat/<model>": "Chat with specific model",
            "/api/history/<session_id>": "Get chat history"
        },
        "usage": {
            "POST": "Send JSON with 'message' field",
            "GET": "Add ?test=true&message=your_message for testing"
        }
    })

@api_bp.route('/health', methods=['GET', 'POST'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "providers_available": sum(1 for p in provider_manager.provider_status.values() if p['available']),
        "total_providers": len(provider_manager.provider_status),
        "method": request.method
    })

@api_bp.route('/providers', methods=['GET'])
def get_providers():
    """Get provider status information"""
    return jsonify({
        "providers": provider_manager.provider_status,
        "timestamp": datetime.now().isoformat()
    })

@api_bp.route('/models', methods=['GET', 'POST'])
def get_models():
    """Get available AI models"""
    models = [
        "gpt-3.5-turbo",
        "gpt-4",
        "gpt-4-turbo", 
        "claude-3-haiku",
        "claude-3-sonnet",
        "claude-3.5-sonnet",
        "gemini-pro",
        "gemini-1.5-pro",
        "llama-2-7b",
        "llama-2-13b",
        "llama-3-8b",
        "llama-3-70b",
        "mistral-7b",
        "mistral-8x7b",
        "mixtral-8x7b",
        "phi-3-mini",
        "qwen-2.5-7b",
        "deepseek-coder"
    ]
    
    return jsonify({
        "models": models,
        "total": len(models),
        "method": request.method,
        "note": "All models use multi-provider fallback system (g4f, gpt4all, ollama, pollinations)",
        "providers": list(provider_manager.provider_status.keys())
    })

@api_bp.route('/verify/<model>', methods=['GET', 'POST'])
def verify_model(model):
    """Verify if a specific model is working properly"""
    try:
        # Run verification synchronously
        result = verify_model_functionality(model)
        
        return jsonify({
            "model": model,
            "verification": result,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            "model": model,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@api_bp.route('/chat', methods=['GET', 'POST'])
def chat():
    """Chat with AI using enhanced multi-provider system"""
    if request.method == 'GET':
        # Handle GET request with query parameters
        message = request.args.get('message')
        if not message:
            return jsonify({
                "endpoint": "/api/chat",
                "methods": ["GET", "POST"],
                "description": "Chat with AI models using enhanced multi-provider system",
                "features": [
                    "Multi-provider fallback (g4f, gpt4all, ollama, pollinations)",
                    "Real model responses (no generic fallbacks)",
                    "Model verification system",
                    "Automatic provider selection"
                ],
                "get_usage": {
                    "url": "/api/chat?message=your_message_here&model=gpt-3.5-turbo",
                    "parameters": {
                        "message": "Required - Your message to AI",
                        "model": "Optional - AI model (default: gpt-3.5-turbo)",
                        "session_id": "Optional - Session identifier"
                    }
                },
                "post_usage": {
                    "url": "/api/chat",
                    "body": {
                        "message": "Who are you? What is your name?",
                        "model": "gpt-3.5-turbo",
                        "stream": False,
                        "session_id": "optional"
                    }
                },
                "available_models": [
                    "gpt-3.5-turbo", "gpt-4", "gpt-4-turbo",
                    "claude-3-haiku", "claude-3-sonnet", "claude-3.5-sonnet",
                    "gemini-pro", "gemini-1.5-pro",
                    "llama-2-7b", "llama-2-13b", "llama-3-8b", "llama-3-70b",
                    "mistral-7b", "mistral-8x7b", "mixtral-8x7b",
                    "phi-3-mini", "qwen-2.5-7b", "deepseek-coder"
                ]
            })
        
        model = request.args.get('model', 'gpt-3.5-turbo')
        stream = False  # No streaming for GET requests
        session_id = get_session_id()
    
    else:
        # Handle POST request
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Missing 'message' in request body"}), 400
        
        message = data['message']
        model = data.get('model', 'gpt-3.5-turbo')
        stream = data.get('stream', False)
        session_id = get_session_id()
    
    # Save user message to history
    save_to_history(session_id, 'user', message, model)
    
    # Get AI response using multi-provider system
    try:
        messages = [{"role": "user", "content": message}]
        
        if stream:
            # Handle streaming response (simplified for now)
            def generate_stream():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    try:
                        ai_response, status, provider = loop.run_until_complete(
                            provider_manager.get_response(model, messages)
                        )
                    finally:
                        loop.close()
                    
                    if status == "success" and ai_response:
                        save_to_history(session_id, 'assistant', ai_response, model)
                        
                        # Stream the response in chunks
                        words = ai_response.split()
                        for i, word in enumerate(words):
                            chunk_data = {
                                'chunk': word + ' ',
                                'done': i == len(words) - 1,
                                'provider': provider if i == len(words) - 1 else None
                            }
                            yield f"data: {json.dumps(chunk_data)}\n\n"
                    else:
                        error_data = {
                            'error': f'All providers failed to generate response for {model}',
                            'done': True,
                            'status': 'error'
                        }
                        yield f"data: {json.dumps(error_data)}\n\n"
                        
                except Exception as e:
                    error_data = {'error': str(e), 'done': True}
                    yield f"data: {json.dumps(error_data)}\n\n"
            
            response = Response(generate_stream(), mimetype='text/event-stream')
            response.headers['Cache-Control'] = 'no-cache'
            response.headers['Connection'] = 'keep-alive'
            return response
        
        else:
            # Handle regular response
            ai_response, status, provider = provider_manager.get_response_sync(model, messages)
            
            if status == "success" and ai_response:
                # Save AI response to history
                save_to_history(session_id, 'assistant', ai_response, model)
                
                return jsonify({
                    "response": ai_response,
                    "model": model,
                    "session_id": session_id,
                    "status": "success",
                    "provider": provider,
                    "method": request.method,
                    "note": "Response from actual AI model, not fallback"
                })
            else:
                return jsonify({
                    "error": f"All providers failed to generate response for {model}",
                    "model": model,
                    "session_id": session_id,
                    "status": "failed",
                    "provider": provider,
                    "available_providers": list(provider_manager.provider_status.keys()),
                    "method": request.method
                }), 503
                
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return jsonify({
            "error": f"Failed to get AI response: {str(e)}",
            "model": model,
            "session_id": session_id,
            "status": "error"
        }), 500

@api_bp.route('/chat/<model>', methods=['GET', 'POST'])
def chat_with_model(model):
    """Chat with specific AI model using multi-provider system"""
    if request.method == 'GET':
        message = request.args.get('message')
        if not message:
            return jsonify({
                "endpoint": f"/api/chat/{model}",
                "model": model,
                "method": ["GET", "POST"],
                "description": f"Chat with {model} using enhanced multi-provider fallback",
                "providers": provider_manager.get_provider_for_model(model),
                "get_usage": {
                    "url": f"/api/chat/{model}?message=your_message_here",
                    "parameters": {
                        "message": "Required - Your message to AI",
                        "session_id": "Optional - Session identifier"
                    }
                },
                "post_usage": {
                    "url": f"/api/chat/{model}",
                    "body": {"message": f"Who are you? What is your name?", "session_id": "optional"}
                },
                "examples": [
                    f"/api/chat/{model}?message=Who are you and who developed you?",
                    f"/api/chat/{model}?message=Explain quantum computing",
                    f"/api/chat/{model}?message=Write a Python function"
                ]
            })
        
        session_id = get_session_id()
    else:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Missing 'message' in request body"}), 400
        
        message = data['message']
        session_id = get_session_id()
    
    # Save user message to history
    save_to_history(session_id, 'user', message, model)
    
    # Get AI response using multi-provider system
    try:
        messages = [{"role": "user", "content": message}]
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            ai_response, status, provider = loop.run_until_complete(
                provider_manager.get_response(model, messages)
            )
        finally:
            loop.close()
        
        if status == "success" and ai_response:
            save_to_history(session_id, 'assistant', ai_response, model)
            
            return jsonify({
                "response": ai_response,
                "model": model,
                "session_id": session_id,
                "status": "success",
                "provider": provider,
                "method": request.method,
                "providers_tried": provider_manager.get_provider_for_model(model),
                "note": "Real response from AI model using multi-provider system"
            })
        else:
            return jsonify({
                "error": f"All providers failed for {model}",
                "model": model,
                "session_id": session_id,
                "status": "failed",
                "provider": provider,
                "providers_tried": provider_manager.get_provider_for_model(model),
                "method": request.method
            }), 503
            
    except Exception as e:
        logger.error(f"Model chat error: {e}")
        return jsonify({
            "error": f"Failed to get response from {model}: {str(e)}",
            "model": model,
            "session_id": session_id,
            "status": "error"
        }), 500

@api_bp.route('/history/<session_id>', methods=['GET'])
def get_chat_history(session_id):
    """Get chat history for a session"""
    if session_id in chat_history:
        return jsonify({
            "session_id": session_id,
            "history": chat_history[session_id],
            "total_messages": len(chat_history[session_id])
        })
    else:
        return jsonify({
            "session_id": session_id,
            "history": [],
            "total_messages": 0
        })