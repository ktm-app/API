from flask import Blueprint, request, jsonify
import json
from datetime import datetime
import uuid
import sys
import traceback

api_bp = Blueprint('api', __name__)

# Store chat history in memory (for demo purposes)
chat_history = {}

def get_session_id():
    """Get or create session ID"""
    session_id = request.args.get('session_id') or request.json.get('session_id') if request.json else None
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

def get_ai_response_sync(model, message):
    """Get AI response using g4f with multiple fallback providers"""
    try:
        # Import g4f here to handle import errors gracefully
        import g4f
        print(f"Attempting to get response from {model} for message: {message}")
        
        # Try different approaches to get a response
        methods = [
            # Method 1: Auto provider selection
            lambda: g4f.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": message}]
            ),
            # Method 2: Specific providers
            lambda: g4f.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": message}],
                provider=g4f.Provider.Bing
            ),
            lambda: g4f.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": message}],
                provider=g4f.Provider.ChatgptAi
            ),
            lambda: g4f.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": message}],
                provider=g4f.Provider.GPTalk
            )
        ]
        
        for i, method in enumerate(methods):
            try:
                print(f"Trying method {i+1}...")
                response = method()
                
                if response and isinstance(response, str) and len(response.strip()) > 0:
                    print(f"Success with method {i+1}")
                    return response.strip(), "success", f"method_{i+1}"
                    
            except Exception as e:
                print(f"Method {i+1} failed: {str(e)}")
                continue
        
        print("All methods failed")
        return None, "failed", "all_methods_failed"
        
    except ImportError as e:
        print(f"g4f import error: {str(e)}")
        return None, "import_error", str(e)
    except Exception as e:
        print(f"Error in get_ai_response_sync: {str(e)}")
        traceback.print_exc()
        return None, "error", str(e)

@api_bp.route('/', methods=['GET', 'POST'])
def api_info():
    """API information and available endpoints"""
    return jsonify({
        "api": "G4F Chat API",
        "version": "1.0",
        "status": "active",
        "endpoints": {
            "/api/health": "Health check",
            "/api/models": "List available models",
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
    try:
        import g4f
        g4f_status = "available"
        g4f_version = getattr(g4f, '__version__', 'unknown')
    except ImportError:
        g4f_status = "not_available"
        g4f_version = "not_installed"
    
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "g4f_status": g4f_status,
        "g4f_version": g4f_version,
        "method": request.method,
        "python_version": sys.version
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
        "gemini-pro",
        "llama-2-7b",
        "llama-2-13b",
        "mistral-7b"
    ]
    
    return jsonify({
        "models": models,
        "total": len(models),
        "method": request.method,
        "note": "All models are available through g4f providers"
    })

@api_bp.route('/chat', methods=['GET', 'POST'])
def chat():
    """Chat with AI using default model"""
    if request.method == 'GET':
        # Handle GET request with query parameters
        test_mode = request.args.get('test', '').lower() == 'true'
        if not test_mode:
            return jsonify({
                "endpoint": "/api/chat",
                "method": "POST",
                "required": {"message": "Your message to AI"},
                "optional": {"model": "AI model (default: gpt-3.5-turbo)", "session_id": "Session identifier"},
                "example": {
                    "message": "Hello, what is AI?",
                    "model": "gpt-3.5-turbo"
                },
                "test_url": "/api/chat?test=true&message=Hello&model=gpt-3.5-turbo"
            })
        
        # Test mode - get message from query params
        message = request.args.get('message', 'Hello, how are you?')
        model = request.args.get('model', 'gpt-3.5-turbo')
        session_id = get_session_id()
    else:
        # Handle POST request
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Missing 'message' in request body"}), 400
        
        message = data['message']
        model = data.get('model', 'gpt-3.5-turbo')
        session_id = get_session_id()
    
    # Save user message to history
    save_to_history(session_id, 'user', message, model)
    
    # Get AI response
    try:
        ai_response, status, provider = get_ai_response_sync(model, message)
        
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
                "timestamp": datetime.now().isoformat()
            })
        else:
            # Create intelligent fallback based on the message
            if "ai" in message.lower() or "artificial intelligence" in message.lower():
                fallback_response = f"AI (Artificial Intelligence) refers to computer systems that can perform tasks that typically require human intelligence, such as learning, reasoning, and problem-solving. I'm {model}, an AI assistant powered by the g4f library, though I'm currently experiencing connectivity issues with the AI providers."
            elif "hello" in message.lower() or "hi" in message.lower():
                fallback_response = f"Hello! I'm {model}, an AI assistant. I'm currently having some connectivity issues with the AI providers, but I'm here to help. What would you like to know?"
            else:
                fallback_response = f"I'm {model}, an AI assistant. You asked: '{message}'. I'm currently experiencing connectivity issues with the AI providers, but I'd be happy to help once the connection is restored. Please try again in a moment."
            
            save_to_history(session_id, 'assistant', fallback_response, model)
            
            return jsonify({
                "response": fallback_response,
                "model": model,
                "session_id": session_id,
                "status": "fallback",
                "note": f"Using intelligent fallback - {provider}",
                "method": request.method,
                "timestamp": datetime.now().isoformat(),
                "debug_info": f"Status: {status}, Provider: {provider}"
            })
            
    except Exception as e:
        error_msg = f"Failed to get AI response: {str(e)}"
        return jsonify({
            "error": error_msg,
            "model": model,
            "session_id": session_id,
            "status": "error",
            "timestamp": datetime.now().isoformat()
        }), 500

@api_bp.route('/chat/<model>', methods=['GET', 'POST'])
def chat_with_model(model):
    """Chat with specific AI model"""
    if request.method == 'GET':
        # Handle GET request with query parameters
        test_mode = request.args.get('test', '').lower() == 'true'
        if not test_mode:
            return jsonify({
                "endpoint": f"/api/chat/{model}",
                "model": model,
                "method": "POST",
                "required": {"message": "Your message to AI"},
                "optional": {"session_id": "Session identifier"},
                "example": {"message": f"Hello {model}, what can you do?"},
                "test_url": f"/api/chat/{model}?test=true&message=Hello"
            })
        
        # Test mode - get message from query params
        message = request.args.get('message', f'Hello {model}, how are you?')
        session_id = get_session_id()
    else:
        # Handle POST request
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Missing 'message' in request body"}), 400
        
        message = data['message']
        session_id = get_session_id()
    
    # Save user message to history
    save_to_history(session_id, 'user', message, model)
    
    # Get AI response
    try:
        ai_response, status, provider = get_ai_response_sync(model, message)
        
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
                "timestamp": datetime.now().isoformat()
            })
        else:
            # Model-specific fallback responses
            model_responses = {
                "gpt-4": "I'm GPT-4, OpenAI's advanced language model. I can help with complex reasoning, analysis, and creative tasks.",
                "gpt-3.5-turbo": "I'm GPT-3.5 Turbo, a fast and efficient AI assistant. I can help with various tasks including answering questions and generating text.",
                "claude-3-haiku": "I'm Claude 3 Haiku, Anthropic's AI assistant focused on being helpful, harmless, and honest.",
                "claude-3-sonnet": "I'm Claude 3 Sonnet, designed to be a thoughtful and capable AI assistant.",
                "gemini-pro": "I'm Gemini Pro, Google's advanced AI model capable of understanding and generating human-like text.",
                "llama-2-7b": "I'm Llama 2 7B, Meta's open-source language model designed for dialogue and assistance.",
                "llama-2-13b": "I'm Llama 2 13B, a larger version of Meta's language model with enhanced capabilities.",
                "mistral-7b": "I'm Mistral 7B, an efficient and powerful open-source language model."
            }
            
            fallback_response = model_responses.get(model, f"I'm {model}, an AI assistant.") + f" You said: '{message}'. I'm currently experiencing connectivity issues but I'm here to help once the connection is restored."
            
            save_to_history(session_id, 'assistant', fallback_response, model)
            
            return jsonify({
                "response": fallback_response,
                "model": model,
                "session_id": session_id,
                "status": "fallback",
                "note": f"Using model-specific fallback - {provider}",
                "method": request.method,
                "timestamp": datetime.now().isoformat(),
                "debug_info": f"Status: {status}, Provider: {provider}"
            })
            
    except Exception as e:
        error_msg = f"Failed to get AI response: {str(e)}"
        return jsonify({
            "error": error_msg,
            "model": model,
            "session_id": session_id,
            "status": "error",
            "timestamp": datetime.now().isoformat()
        }), 500

@api_bp.route('/history/<session_id>', methods=['GET'])
def get_history(session_id):
    """Get chat history for a session"""
    history = chat_history.get(session_id, [])
    return jsonify({
        "session_id": session_id,
        "history": history,
        "total_messages": len(history),
        "timestamp": datetime.now().isoformat()
    })

@api_bp.route('/test-g4f', methods=['GET'])
def test_g4f():
    """Test g4f library functionality"""
    try:
        import g4f
        
        # Test basic functionality
        test_message = "Hello, this is a test"
        response, status, provider = get_ai_response_sync("gpt-3.5-turbo", test_message)
        
        return jsonify({
            "g4f_installed": True,
            "test_message": test_message,
            "test_response": response,
            "test_status": status,
            "test_provider": provider,
            "available_providers": [str(p) for p in g4f.Provider.__dict__.values() if hasattr(p, '__name__')],
            "timestamp": datetime.now().isoformat()
        })
        
    except ImportError as e:
        return jsonify({
            "g4f_installed": False,
            "error": f"g4f not installed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500
    except Exception as e:
        return jsonify({
            "g4f_installed": True,
            "error": f"g4f test failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500