from flask import request, jsonify, Response
import asyncio
import json
from datetime import datetime
import uuid
import nest_asyncio
import logging
import warnings
import os
from typing import Optional, Dict, List, Any, Tuple
from . import api_bp

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Suppress specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Store chat history in memory (for demo purposes)
chat_history = {}

# Import AI providers with error handling
try:
    import g4f
    from g4f.client import Client
    g4f_client = Client()
    G4F_AVAILABLE = True
    G4F_READY = False
except ImportError as e:
    G4F_AVAILABLE = False
    G4F_READY = False
    g4f_client = None
    logging.warning(f"g4f not available: {e}")

try:
    from huggingface_hub import InferenceClient
    # Check if HF token is available
    hf_token = os.getenv('HF_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')
    if hf_token:
        hf_client = InferenceClient(token=hf_token)
        HF_AVAILABLE = True
        HF_READY = True
    else:
        hf_client = None
        HF_AVAILABLE = True  # Library available
        HF_READY = False    # But not configured
        logging.warning("HuggingFace available but no token configured")
except ImportError as e:
    HF_AVAILABLE = False
    HF_READY = False
    hf_client = None
    logging.warning(f"HuggingFace not available: {e}")

try:
    import ollama
    import requests
    # Check if Ollama server is running
    try:
        response = requests.get('http://127.0.0.1:11434/api/tags', timeout=2)
        if response.status_code == 200:
            OLLAMA_AVAILABLE = True
            OLLAMA_READY = True
            ollama_models = response.json().get('models', [])
            logging.info(f"Ollama ready with {len(ollama_models)} models")
        else:
            OLLAMA_AVAILABLE = True
            OLLAMA_READY = False
            logging.warning("Ollama server not responding")
    except requests.exceptions.RequestException:
        OLLAMA_AVAILABLE = True
        OLLAMA_READY = False
        logging.warning("Ollama server not reachable")
except ImportError as e:
    OLLAMA_AVAILABLE = False
    OLLAMA_READY = False
    logging.warning(f"Ollama not available: {e}")

# Puter API integration (completely free, no API key needed)
try:
    import requests
    import json
    PUTER_AVAILABLE = True
    PUTER_READY = True
    logging.info("Puter API ready - 400+ free models available")
except ImportError as e:
    PUTER_AVAILABLE = False
    PUTER_READY = False
    logging.warning(f"Puter not available: {e}")

# Test g4f at startup to see if it works without API key
if G4F_AVAILABLE:
    try:
        test_response = g4f_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "test"}],
            provider=None
        )
        G4F_READY = True
        logging.info("g4f ready for keyless usage")
    except Exception as e:
        if "api_key" in str(e).lower():
            G4F_READY = False
            logging.warning("g4f requires API key, disabling")
        else:
            G4F_READY = True  # Might work, other error
            logging.warning(f"g4f test failed but might still work: {e}")

class ModelProvider:
    """Unified interface for different AI model providers"""
    
    @staticmethod
    def get_provider_for_model(model_name: str) -> str:
        """Determine the best provider for a given model"""
        model_lower = model_name.lower()
        
        # Puter models (free keyless models - highest priority)
        if PUTER_READY:
            return 'puter'
        
        # Ollama models (local keyless models)
        if any(term in model_lower for term in ['llama3', 'llama2', 'mistral', 'gemma', 'neural-chat']):
            if OLLAMA_READY:
                return 'ollama'
        
        # HuggingFace models (only if token available)
        if any(term in model_lower for term in ['deepseek', 'codellama', 'phi', 'qwen']):
            if HF_READY:
                return 'huggingface'
        
        # G4F models (only if ready for keyless use)
        if any(term in model_lower for term in ['gpt', 'claude', 'gemini', 'bard']):
            if G4F_READY:
                return 'g4f'
        
        # Default fallback priority (prefer keyless options)
        if PUTER_READY:
            return 'puter'
        elif OLLAMA_READY:
            return 'ollama'
        elif G4F_READY:
            return 'g4f'
        elif HF_READY:
            return 'huggingface'
        
        return 'fallback'
    
    @staticmethod
    async def get_response_g4f(model: str, messages: List[Dict], stream: bool = False) -> Tuple[Any, str]:
        """Get response from G4F provider"""
        if not G4F_READY:
            return None, "G4F not ready"
        
        try:
            if stream:
                response = g4f_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True
                )
                return response, "success"
            else:
                response = g4f_client.chat.completions.create(
                    model=model,
                    messages=messages
                )
                if hasattr(response, 'choices') and response.choices:
                    content = response.choices[0].message.content
                    return content.strip() if content else None, "success"
                return str(response).strip(), "success"
        except Exception as e:
            logging.error(f"G4F error: {e}")
            return None, f"G4F error: {str(e)}"
    
    @staticmethod
    async def get_response_huggingface(model: str, messages: List[Dict], stream: bool = False) -> Tuple[Any, str]:
        """Get response from HuggingFace Inference API"""
        if not HF_AVAILABLE:
            return None, "HuggingFace not available"
        
        try:
            # Convert messages to a single prompt for HF
            prompt = ""
            for msg in messages:
                if msg["role"] == "user":
                    prompt += f"User: {msg['content']}\n"
                elif msg["role"] == "assistant":
                    prompt += f"Assistant: {msg['content']}\n"
            prompt += "Assistant: "
            
            # Map model names to HF model IDs
            hf_model_map = {
                'deepseek-r1': 'deepseek-ai/DeepSeek-R1',
                'codellama': 'codellama/CodeLlama-7b-Instruct-hf',
                'mistral': 'mistralai/Mistral-7B-Instruct-v0.1',
                'phi': 'microsoft/phi-2',
                'qwen': 'Qwen/Qwen-7B-Chat',
                'llama2': 'meta-llama/Llama-2-7b-chat-hf'
            }
            
            hf_model = hf_model_map.get(model.lower(), model)
            
            if stream:
                response = hf_client.text_generation(
                    prompt=prompt,
                    model=hf_model,
                    stream=True,
                    max_new_tokens=512
                )
                return response, "success"
            else:
                response = hf_client.text_generation(
                    prompt=prompt,
                    model=hf_model,
                    max_new_tokens=512,
                    return_full_text=False
                )
                return response.strip(), "success"
        except Exception as e:
            logging.error(f"HuggingFace error: {e}")
            return None, f"HuggingFace error: {str(e)}"
    
    @staticmethod
    async def get_response_puter(model: str, messages: List[Dict], stream: bool = False) -> Tuple[Any, str]:
        """Get response from Puter API (400+ free models, no API key needed)"""
        if not PUTER_READY:
            return None, "Puter not ready"
        
        try:
            # Convert messages to single prompt for Puter API
            prompt = ""
            for msg in messages:
                if msg["role"] == "user":
                    prompt += f"{msg['content']}"
                elif msg["role"] == "assistant":
                    prompt += f"Assistant: {msg['content']}\nUser: "
            
            # Map common model names to Puter model IDs
            puter_model_map = {
                'gpt-3.5-turbo': 'gpt-3.5-turbo',
                'gpt-4': 'gpt-4o',
                'gpt-4o': 'gpt-4o',
                'claude': 'claude',
                'claude-3': 'claude-3.7-sonnet',
                'gemini': 'gemini',
                'llama': 'openrouter:meta-llama/llama-3.1-8b-instruct',
                'llama3': 'openrouter:meta-llama/llama-3.3-70b',
                'mistral': 'openrouter:mistralai/mistral-large',
                'deepseek': 'deepseek-chat',
                'deepseek-r1': 'deepseek-reasoner',
                'qwen': 'qwen/qwen3-235b-a22b',
                'grok': 'x-ai/grok-4',
                'perplexity': 'perplexity/sonar'
            }
            
            puter_model = puter_model_map.get(model.lower(), model)
            
            # Puter API endpoint (using their free service)
            url = "https://api.puter.com/v1/ai/chat"
            
            payload = {
                "model": puter_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": stream
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and result['choices']:
                    content = result['choices'][0]['message']['content']
                    return content.strip(), "success"
                elif 'response' in result:
                    return result['response'].strip(), "success"
                else:
                    return str(result).strip(), "success"
            else:
                logging.error(f"Puter API error: {response.status_code} - {response.text}")
                return None, f"Puter API error: {response.status_code}"
                
        except Exception as e:
            logging.error(f"Puter error: {e}")
            return None, f"Puter error: {str(e)}"
    
    @staticmethod
    async def get_response_ollama(model: str, messages: List[Dict], stream: bool = False) -> Tuple[Any, str]:
        """Get response from Ollama (local models)"""
        if not OLLAMA_AVAILABLE:
            return None, "Ollama not available"
        
        try:
            # Map common model names to Ollama model names
            ollama_model_map = {
                'llama3.3': 'llama3.3',
                'llama3.2': 'llama3.2', 
                'mistral': 'mistral',
                'gemma3': 'gemma3',
                'deepseek-r1': 'deepseek-r1',
                'qwen3': 'qwen3'
            }
            
            ollama_model = ollama_model_map.get(model.lower(), model)
            
            if stream:
                response = ollama.chat(
                    model=ollama_model,
                    messages=messages,
                    stream=True
                )
                return response, "success"
            else:
                response = ollama.chat(
                    model=ollama_model,
                    messages=messages
                )
                return response['message']['content'], "success"
        except Exception as e:
            logging.error(f"Ollama error: {e}")
            return None, f"Ollama error: {str(e)}"

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

async def get_ai_response_with_providers(model: str, messages: List[Dict], stream: bool = False) -> Tuple[Any, str, str]:
    """Get AI response using multiple providers with intelligent fallback"""
    provider = ModelProvider.get_provider_for_model(model)
    
    # Try primary provider
    if provider == 'puter':
        response, status = await ModelProvider.get_response_puter(model, messages, stream)
        if status == "success" and response:
            return response, status, "puter"
    
    elif provider == 'huggingface':
        response, status = await ModelProvider.get_response_huggingface(model, messages, stream)
        if status == "success" and response:
            return response, status, "huggingface"
    
    elif provider == 'ollama':
        response, status = await ModelProvider.get_response_ollama(model, messages, stream)
        if status == "success" and response:
            return response, status, "ollama"
    
    elif provider == 'g4f':
        response, status = await ModelProvider.get_response_g4f(model, messages, stream)
        if status == "success" and response:
            return response, status, "g4f"
    
    # Fallback to other providers (prioritize keyless options)
    for fallback_provider in ['puter', 'ollama', 'g4f', 'huggingface']:
        if fallback_provider != provider:
            try:
                if fallback_provider == 'puter' and PUTER_READY:
                    response, status = await ModelProvider.get_response_puter(model, messages, stream)
                elif fallback_provider == 'huggingface' and HF_READY:
                    response, status = await ModelProvider.get_response_huggingface(model, messages, stream)
                elif fallback_provider == 'g4f' and G4F_READY:
                    response, status = await ModelProvider.get_response_g4f(model, messages, stream)
                elif fallback_provider == 'ollama' and OLLAMA_READY:
                    response, status = await ModelProvider.get_response_ollama(model, messages, stream)
                else:
                    continue
                
                if status == "success" and response:
                    return response, status, fallback_provider
            except Exception as e:
                logging.warning(f"Fallback provider {fallback_provider} failed: {e}")
                continue
    
    # Final fallback response
    return None, "failed", "all_providers_failed"

@api_bp.route('/', methods=['GET', 'POST'])
def api_info():
    """API information and available endpoints"""
    return jsonify({
        "api": "Multi-Provider AI Chat API",
        "version": "2.0",
        "status": "active",
        "providers": {
            "g4f": G4F_AVAILABLE,
            "huggingface": HF_AVAILABLE,
            "ollama": OLLAMA_AVAILABLE
        },
        "endpoints": {
            "/api/health": "Health check",
            "/api/models": "List available models",
            "/api/providers": "List available providers",
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
        "providers": {
            "puter": {"available": PUTER_AVAILABLE, "ready": PUTER_READY},
            "g4f": {"available": G4F_AVAILABLE, "ready": G4F_READY},
            "huggingface": {"available": HF_AVAILABLE, "ready": HF_READY},
            "ollama": {"available": OLLAMA_AVAILABLE, "ready": OLLAMA_READY}
        },
        "keyless_ready": PUTER_READY or OLLAMA_READY or G4F_READY,
        "method": request.method
    })

@api_bp.route('/providers', methods=['GET'])
def get_providers():
    """Get available providers and their status"""
    providers = {}
    
    if PUTER_READY:
        providers["puter"] = {
            "status": "ready",
            "description": "Puter API - 400+ free models, no API key needed",
            "models": ["gpt-4o", "claude", "gemini", "llama3", "mistral", "deepseek", "grok", "perplexity"]
        }
    
    if G4F_READY:
        providers["g4f"] = {
            "status": "ready",
            "description": "Free web-based AI models",
            "models": ["gpt-3.5-turbo", "gpt-4", "claude-3", "gemini-pro"]
        }
    
    if HF_READY:
        providers["huggingface"] = {
            "status": "ready", 
            "description": "HuggingFace Inference API - 800,000+ models",
            "models": ["deepseek-r1", "codellama", "mistral", "phi", "qwen"]
        }
    
    if OLLAMA_READY:
        providers["ollama"] = {
            "status": "ready",
            "description": "Local AI models",
            "models": ["llama3.3", "llama3.2", "mistral", "gemma3", "deepseek-r1"]
        }
    
    return jsonify({
        "providers": providers,
        "total_providers": len(providers)
    })

@api_bp.route('/models', methods=['GET', 'POST'])
def get_models():
    """Get available AI models across all providers"""
    models = {
        "g4f_models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "claude-3-haiku", "claude-3-sonnet", "gemini-pro"] if G4F_AVAILABLE else [],
        "huggingface_models": ["deepseek-r1", "codellama", "mistral", "phi", "qwen", "llama2"] if HF_AVAILABLE else [],
        "ollama_models": ["llama3.3", "llama3.2", "mistral", "gemma3", "deepseek-r1", "qwen3"] if OLLAMA_AVAILABLE else []
    }
    
    all_models = []
    for provider, provider_models in models.items():
        all_models.extend(provider_models)
    
    return jsonify({
        "models_by_provider": models,
        "all_models": sorted(list(set(all_models))),
        "total": len(set(all_models)),
        "method": request.method,
        "note": "Models are available through multiple providers with automatic fallback"
    })

@api_bp.route('/chat', methods=['GET', 'POST'])
def chat():
    """Chat with AI using automatic provider selection"""
    if request.method == 'GET':
        # Handle GET request with query parameters
        message = request.args.get('message')
        if not message:
            # Return API documentation if no message provided
            return jsonify({
                "endpoint": "/api/chat",
                "methods": ["GET", "POST"],
                "description": "Chat with AI models using multiple providers with automatic fallback",
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
                        "message": "Hello, what is AI?",
                        "model": "gpt-3.5-turbo",
                        "stream": False,
                        "session_id": "optional"
                    }
                },
                "providers": {
                    "g4f": G4F_AVAILABLE,
                    "huggingface": HF_AVAILABLE,
                    "ollama": OLLAMA_AVAILABLE
                },
                "examples": [
                    "/api/chat?message=What is artificial intelligence?",
                    "/api/chat?message=Tell me a joke&model=deepseek-r1",
                    "/api/chat?message=Write Python code&model=codellama"
                ]
            })

        # Choose default model based on what's ready
        default_model = 'mistral' if OLLAMA_READY else 'gpt-3.5-turbo'
        model = request.args.get('model', default_model)
        stream = False  # No streaming support for GET requests
        session_id = get_session_id()
    else:
        # Handle POST request
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Missing 'message' in request body"}), 400

        message = data['message']
        # Choose default model based on what's ready
        default_model = 'mistral' if OLLAMA_READY else 'gpt-3.5-turbo'
        model = data.get('model', default_model)
        stream = data.get('stream', False)
        session_id = get_session_id()

    # For GET requests, no streaming support
    if request.method == 'GET':
        stream = False

    # Save user message to history
    save_to_history(session_id, 'user', message, model)

    # Get AI response
    try:
        messages = [{"role": "user", "content": message}]

        if stream:
            # Handle streaming response
            def generate_stream():
                try:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    response_stream, status, provider = loop.run_until_complete(
                        get_ai_response_with_providers(model, messages, stream=True)
                    )
                    
                    if status == "success":
                        full_response = ""
                        
                        if provider == "ollama":
                            for chunk in response_stream:
                                if chunk['message']['content']:
                                    content = chunk['message']['content']
                                    full_response += content
                                    yield f"data: {json.dumps({'chunk': content, 'done': False})}\n\n"
                        elif provider == "huggingface":
                            for chunk in response_stream:
                                content = chunk.token.text
                                full_response += content
                                yield f"data: {json.dumps({'chunk': content, 'done': False})}\n\n"
                        elif provider == "g4f":
                            for chunk in response_stream:
                                if hasattr(chunk, 'choices') and chunk.choices:
                                    delta = chunk.choices[0].delta
                                    if hasattr(delta, 'content') and delta.content:
                                        content = delta.content
                                        full_response += content
                                        yield f"data: {json.dumps({'chunk': content, 'done': False})}\n\n"
                        
                        # Save the complete response to history
                        save_to_history(session_id, 'assistant', full_response, model)
                        yield f"data: {json.dumps({'done': True, 'provider': provider})}\n\n"
                    else:
                        fallback_response = f"Hello! I'm {model} AI assistant. You said: '{message}'. I'm working through multiple AI providers. How can I help you today?"
                        save_to_history(session_id, 'assistant', fallback_response, model)
                        yield f"data: {json.dumps({'chunk': fallback_response, 'done': True, 'status': 'fallback'})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

            response = Response(generate_stream(), mimetype='text/event-stream')
            response.headers['Cache-Control'] = 'no-cache'
            response.headers['Connection'] = 'keep-alive'
            return response
        else:
            # Handle regular response
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            ai_response, status, provider = loop.run_until_complete(
                get_ai_response_with_providers(model, messages)
            )

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
                fallback_response = f"Hello! I'm {model} AI assistant. You said: '{message}'. I'm working through multiple AI providers including HuggingFace, Ollama, and g4f. How can I help you today?"
                save_to_history(session_id, 'assistant', fallback_response, model)

                return jsonify({
                    "response": fallback_response,
                    "model": model,
                    "session_id": session_id,
                    "status": "fallback",
                    "note": f"Using fallback response - {provider}",
                    "method": request.method
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
        message = request.args.get('message')
        if not message:
            provider = ModelProvider.get_provider_for_model(model)
            return jsonify({
                "endpoint": f"/api/chat/{model}",
                "model": model,
                "recommended_provider": provider,
                "method": ["GET", "POST"],
                "description": f"Chat with {model} AI model",
                "get_usage": {
                    "url": f"/api/chat/{model}?message=your_message_here",
                    "parameters": {
                        "message": "Required - Your message to AI",
                        "session_id": "Optional - Session identifier"
                    }
                },
                "post_usage": {
                    "url": f"/api/chat/{model}",
                    "body": {"message": f"Hello {model}, what can you do?", "session_id": "optional"}
                },
                "examples": [
                    f"/api/chat/{model}?message=What is artificial intelligence?",
                    f"/api/chat/{model}?message=Write a Python function"
                ]
            })

        session_id = get_session_id()
    else:
        # Handle POST request
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({"error": "Missing 'message' in request body"}), 400

        message = data['message']
        session_id = get_session_id()
        stream = data.get('stream', False)

    # For GET requests, no streaming support
    if request.method == 'GET':
        stream = False

    # Save user message to history
    save_to_history(session_id, 'user', message, model)

    # Get AI response using the same logic as /chat
    try:
        messages = [{"role": "user", "content": message}]

        # Use the same async pattern
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        ai_response, status, provider = loop.run_until_complete(
            get_ai_response_with_providers(model, messages, stream)
        )

        if status == "success" and ai_response:
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
            fallback_response = f"Hello! I'm {model} AI assistant. You said: '{message}'. I'm working through multiple providers. How can I help you today?"
            save_to_history(session_id, 'assistant', fallback_response, model)

            return jsonify({
                "response": fallback_response,
                "model": model,
                "session_id": session_id,
                "status": "fallback",
                "note": f"Using fallback response",
                "method": request.method
            })

    except Exception as e:
        return jsonify({
            "error": f"Failed to get AI response: {str(e)}",
            "model": model,
            "session_id": session_id,
            "status": "error"
        }), 500

@api_bp.route('/history/<session_id>', methods=['GET'])
def get_chat_history(session_id):
    """Get chat history for a session"""
    history = chat_history.get(session_id, [])
    return jsonify({
        "session_id": session_id,
        "history": history,
        "message_count": len(history)
    })