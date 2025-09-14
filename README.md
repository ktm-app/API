    # AI API

A comprehensive Flask-based API for accessing multiple AI models without requiring API keys, powered by the g4f library.

## Features

- **Multi-Provider Support**: Access models from OpenAI, Claude, Gemini, and more
- **Chat Completion**: Standard and streaming responses
- **Model Switching**: Dynamic model selection
- **Chat History Management**: Session-based conversation tracking
- **Rate Limiting**: Built-in request throttling
- **CORS Support**: Cross-origin requests enabled
- **Error Handling**: Comprehensive error responses and logging
- **Web Interface**: Simple UI for testing the API
- **Production Ready**: Proper configuration and structure

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python run.py
   ```

The API will be available at `http://localhost:5000`

## Usage

### API Endpoints

#### Health Check
```http
GET /api/health
```
Returns API status.

#### List Available Models
```http
GET /api/models
```
Returns models grouped by provider.

#### Chat Completion
```http
POST /api/chat
Content-Type: application/json

{
  "model": "gpt-3.5-turbo",
  "messages": [
    {"role": "user", "content": "Hello, how are you?"}
  ],
  "session_id": "optional-session-id",
  "stream": false
}
```

#### Model-Specific Chat
```http
POST /api/chat/{model_name}
Content-Type: application/json

{
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "session_id": "optional-session-id",
  "stream": false
}
```

#### Get Chat History
```http
GET /api/history/{session_id}
```

### Web Interface

Visit `http://localhost:5000/static/index.html` for a simple web-based tester.

### Examples

#### Basic Chat
```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [{"role": "user", "content": "What is the capital of France?"}]
  }'
```

#### Streaming Response
```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

#### Using Session for History
```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini",
    "messages": [{"role": "user", "content": "Remember my name is John"}],
    "session_id": "user123"
  }'

curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini",
    "messages": [{"role": "user", "content": "What is my name?"}],
    "session_id": "user123"
  }'
```

## Configuration

Create a `.env` file based on `.env.example`:

```
SECRET_KEY=your-secret-key-here
DEBUG=True
```

## Rate Limiting

- Default: 200 requests per day, 50 per hour per IP
- Configurable in `config.py`

## Error Handling

The API returns appropriate HTTP status codes and JSON error messages:

- `400`: Bad Request (invalid input)
- `429`: Too Many Requests (rate limited)
- `500`: Internal Server Error

## Logging

Requests and errors are logged to the console. Configure log level in `config.py`.

## Production Deployment

For production use:

1. Set `DEBUG=False` in config
2. Use a WSGI server like Gunicorn
3. Set up proper environment variables
4. Consider using a database for chat history persistence

## License

This project is for personal use. Check g4f library license for restrictions.