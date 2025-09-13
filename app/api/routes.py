from flask import Blueprint, request, jsonify
import g4f
import asyncio
import json
from datetime import datetime
import uuid

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
        print(f"Attempting to get response from {model} for message: {message}")
        
        # List of providers to try in order
        providers = [
            g4f.Provider.Bing,
            g4f.Provider.ChatgptAi,
            g4f.Provider.GPTalk,
            g4f.Provider.Liaobots,
            g4f.Provider.ChatBase,
            None  # Auto provider selection
        ]
        
        # First try without specifying provider (auto-selection)
        try:
            print("Trying auto provider selection...")
            response = g4f.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": message}]
            )
            
            if response and len(response.strip()) > 0:
                print(f"Success with auto provider")
                return response.strip(), "success", "auto"
                
        except Exception as e:
            print(f"Auto provider failed: {str(e)}")
        
        # Try each provider specifically
        for provider in providers[:-1]:  # Exclude None since we tried auto already
            try:
                print(f"Trying provider: {provider}")
                response = g4f.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": message}],
                    provider=provider
                )
                
                if response and len(response.strip()) > 0:
                    print(f"Success with provider: {provider}")
                    return response.strip(), "success", str(provider)
                    
            except Exception as e:
                print(f"Provider {provider} failed: {str(e)}")
                continue
        
        print("All providers failed")
        return None, "failed", "all_providers_failed"
        
    except Exception as e:
        print(f"Error in get_ai_response_sync: {str(e)}")
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
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "g4f_available": True,
        "method": request.method
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
                "method": request.method
            })
        else:
            # Fallback response with more details
            fallback_response = f"Hello! I'm {model} AI assistant. You said: '{message}'. I'm working through g4f API but having connectivity issues. How can I help you today?"
            save_to_history(session_id, 'assistant', fallback_response, model)
            
            return jsonify({
                "response": fallback_response,
                "model": model,
                "session_id": session_id,
                "status": "fallback",
                "note": f"Using fallback response - {provider}",
                "method": request.method,
                "debug_info": f"Status: {status}, Provider: {provider}"
            })
            
    except Exception as e:
        return jsonify({
            "error": f"Failed to get AI response: {str(e)}",
            "model": model,
            "session_id": session_id,
            "status": "error"
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
                "method": request.method
            })
        else:
            # Fallback response
            fallback_response = f"Hello! I'm {model} AI assistant. You said: '{message}'. I'm working through g4f API but having connectivity issues. How can I help you today?"
            save_to_history(session_id, 'assistant', fallback_response, model)
            
            return jsonify({
                "response": fallback_response,
                "model": model,
                "session_id": session_id,
                "status": "fallback",
                "note": f"Using fallback response - {provider}",
                "method": request.method,
                "debug_info": f"Status: {status}, Provider: {provider}"
            })
            
    except Exception as e:
        return jsonify({
            "error": f"Failed to get AI response: {str(e)}",
            "model": model,
            "session_id": session_id,
            "status": "error"
        }), 500

@api_bp.route('/history/<session_id>', methods=['GET'])
def get_history(session_id):
    """Get chat history for a session"""
    history = chat_history.get(session_id, [])
    return jsonify({
        "session_id": session_id,
        "history": history,
        "total_messages": len(history)
    })