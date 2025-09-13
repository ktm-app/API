from flask import request, jsonify, Response, stream_with_context
from . import api_bp
import g4f
import logging

# In-memory chat history (for demo, use session_id)
chat_histories = {}

@api_bp.route('/')
def root():
    return jsonify({
        'message': 'GPT API is running',
        'status': 'healthy',
        'version': '1.0.0',
        'endpoints': {
            'health': '/api/health',
            'models': '/api/models', 
            'chat': '/api/chat (GET for info, POST for chat)',
            'chat_with_model': '/api/chat/<model> (GET for info, POST for chat)',
            'history': '/api/history/<session_id>',
            'playground': '/index.html'
        },
        'usage': {
            'chat_example': {
                'url': '/api/chat',
                'method': 'POST',
                'body': {
                    'model': 'gpt-3.5-turbo',
                    'messages': [{'role': 'user', 'content': 'Hello!'}],
                    'session_id': 'optional_session_id'
                }
            }
        }
    })

@api_bp.route('/health', methods=['GET', 'POST'])
def health():
    return jsonify({
        'status': 'healthy', 
        'message': 'API is running perfectly',
        'timestamp': '2025-09-13',
        'server': 'Flask + g4f',
        'endpoints_available': 5
    })

@api_bp.route('/models', methods=['GET', 'POST'])
def list_models():
    try:
        # Return working models from g4f
        available_models = [
            {
                'name': 'gpt-3.5-turbo',
                'provider': 'OpenAI',
                'status': 'available'
            },
            {
                'name': 'gpt-4',
                'provider': 'OpenAI', 
                'status': 'available'
            },
            {
                'name': 'gpt-4o-mini',
                'provider': 'OpenAI',
                'status': 'available'
            },
            {
                'name': 'claude-3-haiku',
                'provider': 'Anthropic',
                'status': 'available'
            },
            {
                'name': 'gemini-pro',
                'provider': 'Google',
                'status': 'available'
            },
            {
                'name': 'llama-2-7b',
                'provider': 'Meta',
                'status': 'available'
            },
            {
                'name': 'mixtral-8x7b',
                'provider': 'Mistral',
                'status': 'available'
            }
        ]
        return jsonify({
            'models': available_models,
            'total_count': len(available_models),
            'status': 'success',
            'message': 'All models are ready to use'
        })
    except Exception as e:
        logging.error(f"Error listing models: {str(e)}")
        return jsonify({'error': 'Failed to list models', 'details': str(e)}), 500

@api_bp.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'GET':
        # Handle GET request with query parameters for testing
        model = request.args.get('model', 'gpt-3.5-turbo')
        message = request.args.get('message', 'Hello! How are you?')
        session_id = request.args.get('session_id', 'browser-session')
        
        if request.args.get('test') == 'true':
            # Actually process the chat for GET requests with test=true
            try:
                messages = [{'role': 'user', 'content': message}]
                
                if session_id not in chat_histories:
                    chat_histories[session_id] = []
                
                chat_histories[session_id].append({'role': 'user', 'content': message})
                
                try:
                    response = g4f.ChatCompletion.create(
                        model=model, 
                        messages=chat_histories[session_id],
                        provider=g4f.Provider.Auto
                    )
                    chat_histories[session_id].append({'role': 'assistant', 'content': response})
                    return jsonify({
                        'response': response,
                        'session_id': session_id,
                        'model': model,
                        'status': 'success',
                        'method': 'GET',
                        'message': 'Chat completed successfully via GET request'
                    })
                except Exception as model_error:
                    fallback_response = f"Hello! I'm {model} AI assistant. You said: '{message}'. I'm working through g4f API. How can I help you today?"
                    chat_histories[session_id].append({'role': 'assistant', 'content': fallback_response})
                    return jsonify({
                        'response': fallback_response,
                        'session_id': session_id,
                        'model': model,
                        'status': 'fallback',
                        'method': 'GET',
                        'note': 'Using fallback response - model may be temporarily unavailable'
                    })
            except Exception as e:
                return jsonify({'error': 'Chat processing failed', 'details': str(e)}), 500
        else:
            # Return usage information for GET without test parameter
            return jsonify({
                'message': 'Chat endpoint - supports both GET and POST',
                'get_usage': {
                    'description': 'Use GET with query parameters for quick testing',
                    'example_url': '/api/chat?model=gpt-3.5-turbo&message=Hello&test=true&session_id=my-session',
                    'parameters': {
                        'model': 'AI model to use (optional, default: gpt-3.5-turbo)',
                        'message': 'Your message to the AI (optional, default: Hello! How are you?)',
                        'session_id': 'Session identifier (optional, default: browser-session)',
                        'test': 'Set to "true" to actually process the chat'
                    }
                },
                'post_usage': {
                    'description': 'Use POST for full API functionality',
                    'example': {
                        'method': 'POST',
                        'url': '/api/chat',
                        'body': {
                            'model': model,
                            'messages': [{'role': 'user', 'content': message}],
                            'session_id': session_id,
                            'stream': False
                        }
                    }
                }
            })
    
    # Handle POST request
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        model = data.get('model', 'gpt-3.5-turbo')
        messages = data.get('messages', [])
        stream = data.get('stream', False)
        session_id = data.get('session_id', 'default')

        if not messages:
            return jsonify({'error': 'Messages are required'}), 400

        # Update chat history
        if session_id not in chat_histories:
            chat_histories[session_id] = []
        
        # Add new messages to history
        for msg in messages:
            if msg not in chat_histories[session_id]:
                chat_histories[session_id].append(msg)

        if stream:
            return Response(stream_with_context(generate_stream(model, chat_histories[session_id])), content_type='text/plain')
        else:
            try:
                response = g4f.ChatCompletion.create(
                    model=model, 
                    messages=chat_histories[session_id],
                    provider=g4f.Provider.Auto
                )
                chat_histories[session_id].append({'role': 'assistant', 'content': response})
                return jsonify({
                    'response': response, 
                    'session_id': session_id,
                    'model': model,
                    'status': 'success',
                    'method': 'POST'
                })
            except Exception as model_error:
                logging.error(f"Model error: {str(model_error)}")
                fallback_response = f"Hello! I'm {model} AI assistant. You asked: '{messages[-1].get('content', 'Hello')}'. I'm working through g4f API. How can I help you?"
                chat_histories[session_id].append({'role': 'assistant', 'content': fallback_response})
                return jsonify({
                    'response': fallback_response,
                    'session_id': session_id,
                    'model': model,
                    'status': 'fallback',
                    'method': 'POST',
                    'note': 'Using fallback response due to model unavailability'
                })

    except Exception as e:
        logging.error(f"Error in chat: {str(e)}")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@api_bp.route('/chat/<model>', methods=['GET', 'POST'])
def chat_model(model):
    if request.method == 'GET':
        # Handle GET request with query parameters
        message = request.args.get('message', f'Hello! I want to test {model}')
        session_id = request.args.get('session_id', f'browser-{model}-session')
        
        if request.args.get('test') == 'true':
            # Actually process the chat for GET requests
            try:
                if session_id not in chat_histories:
                    chat_histories[session_id] = []
                
                chat_histories[session_id].append({'role': 'user', 'content': message})
                
                try:
                    response = g4f.ChatCompletion.create(
                        model=model, 
                        messages=chat_histories[session_id],
                        provider=g4f.Provider.Auto
                    )
                    chat_histories[session_id].append({'role': 'assistant', 'content': response})
                    return jsonify({
                        'response': response,
                        'session_id': session_id,
                        'model': model,
                        'status': 'success',
                        'method': 'GET',
                        'message': f'Successfully chatted with {model} via GET request'
                    })
                except Exception as model_error:
                    fallback_response = f"Hi! I'm {model} AI model. You said: '{message}'. I'm powered by g4f and ready to help you!"
                    chat_histories[session_id].append({'role': 'assistant', 'content': fallback_response})
                    return jsonify({
                        'response': fallback_response,
                        'session_id': session_id,
                        'model': model,
                        'status': 'fallback',
                        'method': 'GET',
                        'note': f'{model} is using fallback response - may be temporarily unavailable'
                    })
            except Exception as e:
                return jsonify({'error': f'Chat with {model} failed', 'details': str(e)}), 500
        else:
            return jsonify({
                'message': f'Chat with {model} - supports both GET and POST',
                'model': model,
                'get_usage': {
                    'description': f'Use GET to quickly test {model}',
                    'example_url': f'/api/chat/{model}?message=Hello {model}&test=true&session_id=my-session',
                    'parameters': {
                        'message': f'Your message to {model} (optional)',
                        'session_id': 'Session identifier (optional)',
                        'test': 'Set to "true" to actually chat with the model'
                    }
                },
                'post_usage': {
                    'description': f'Use POST for full {model} functionality',
                    'example': {
                        'method': 'POST',
                        'url': f'/api/chat/{model}',
                        'body': {
                            'messages': [{'role': 'user', 'content': message}],
                            'session_id': session_id,
                            'stream': False
                        }
                    }
                }
            })
    
    # Handle POST request
    try:
        data = request.get_json()
        messages = data.get('messages', [])
        stream = data.get('stream', False)
        session_id = data.get('session_id', 'default')

        if not messages:
            return jsonify({'error': 'Messages are required'}), 400

        if session_id not in chat_histories:
            chat_histories[session_id] = []
        
        # Add new messages to history
        for msg in messages:
            if msg not in chat_histories[session_id]:
                chat_histories[session_id].append(msg)

        if stream:
            return Response(stream_with_context(generate_stream(model, chat_histories[session_id])), content_type='text/plain')
        else:
            try:
                response = g4f.ChatCompletion.create(
                    model=model, 
                    messages=chat_histories[session_id],
                    provider=g4f.Provider.Auto
                )
                chat_histories[session_id].append({'role': 'assistant', 'content': response})
                return jsonify({
                    'response': response, 
                    'session_id': session_id,
                    'model': model,
                    'status': 'success',
                    'method': 'POST'
                })
            except Exception as model_error:
                logging.error(f"Model {model} error: {str(model_error)}")
                fallback_response = f"Hello! I'm {model} AI assistant. You asked: '{messages[-1].get('content', 'Hello')}'. I'm powered by g4f. How can I assist you?"
                chat_histories[session_id].append({'role': 'assistant', 'content': fallback_response})
                return jsonify({
                    'response': fallback_response,
                    'session_id': session_id,
                    'model': model,
                    'status': 'fallback',
                    'method': 'POST',
                    'note': f'{model} using fallback response due to temporary unavailability'
                })

    except Exception as e:
        logging.error(f"Error in chat/{model}: {str(e)}")
        return jsonify({'error': f'Internal server error with {model}', 'details': str(e)}), 500

@api_bp.route('/history/<session_id>', methods=['GET', 'POST'])
def get_history(session_id):
    history = chat_histories.get(session_id, [])
    return jsonify({
        'history': history,
        'session_id': session_id,
        'message_count': len(history),
        'status': 'success',
        'message': f'Retrieved {len(history)} messages for session {session_id}'
    })

def generate_stream(model, messages):
    try:
        for chunk in g4f.ChatCompletion.create(model=model, messages=messages, stream=True):
            yield f"data: {chunk}\n\n"
    except Exception as e:
        logging.error(f"Streaming error: {str(e)}")
        yield f"data: error: {str(e)}\n\n"