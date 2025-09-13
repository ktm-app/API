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
        'version': '1.0.0',
        'endpoints': {
            'health': '/api/health',
            'models': '/api/models',
            'chat': '/api/chat',
            'chat_with_model': '/api/chat/<model>',
            'history': '/api/history/<session_id>'
        }
    })

@api_bp.route('/health')
def health():
    return jsonify({'status': 'healthy', 'message': 'API is running'})

@api_bp.route('/models')
def list_models():
    try:
        providers = g4f.Provider.__all__
        models = {}
        for provider_name in providers:
            provider = getattr(g4f.Provider, provider_name, None)
            if provider and hasattr(provider, 'models'):
                models[provider_name] = list(provider.models) if provider.models else []
        return jsonify({'models': models})
    except Exception as e:
        logging.error(f"Error listing models: {str(e)}")
        return jsonify({'error': 'Failed to list models'}), 500

@api_bp.route('/chat', methods=['POST'])
def chat():
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
        chat_histories[session_id].extend(messages)

        if stream:
            return Response(stream_with_context(generate_stream(model, chat_histories[session_id])), content_type='text/plain')
        else:
            response = g4f.ChatCompletion.create(model=model, messages=chat_histories[session_id])
            chat_histories[session_id].append({'role': 'assistant', 'content': response})
            return jsonify({'response': response, 'session_id': session_id})

    except Exception as e:
        logging.error(f"Error in chat: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@api_bp.route('/chat/<model>', methods=['POST'])
def chat_model(model):
    try:
        data = request.get_json()
        messages = data.get('messages', [])
        stream = data.get('stream', False)
        session_id = data.get('session_id', 'default')

        if not messages:
            return jsonify({'error': 'Messages are required'}), 400

        if session_id not in chat_histories:
            chat_histories[session_id] = []
        chat_histories[session_id].extend(messages)

        if stream:
            return Response(stream_with_context(generate_stream(model, chat_histories[session_id])), content_type='text/plain')
        else:
            response = g4f.ChatCompletion.create(model=model, messages=chat_histories[session_id])
            chat_histories[session_id].append({'role': 'assistant', 'content': response})
            return jsonify({'response': response, 'session_id': session_id})

    except Exception as e:
        logging.error(f"Error in chat/{model}: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@api_bp.route('/history/<session_id>')
def get_history(session_id):
    history = chat_histories.get(session_id, [])
    return jsonify({'history': history})

def generate_stream(model, messages):
    try:
        for chunk in g4f.ChatCompletion.create(model=model, messages=messages, stream=True):
            yield f"data: {chunk}\n\n"
    except Exception as e:
        logging.error(f"Streaming error: {str(e)}")
        yield f"data: error: {str(e)}\n\n"