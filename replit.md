# Enhanced Multi-Provider AI API

## Overview
A comprehensive Flask-based API that provides access to multiple AI models without requiring API keys. The system supports three main providers:

1. **Ollama** (Local models - completely free, no API keys needed)
2. **HuggingFace Inference API** (Free with account, requires token)
3. **G4F** (Free web-based models, may require configuration)

## Current State
- ✅ Flask API server running on port 5000
- ✅ Multi-provider routing system implemented
- ✅ Intelligent fallback mechanisms
- ✅ Provider readiness detection
- ✅ HTML playground for testing at `/index.html`
- ✅ Comprehensive error handling

## Provider Status (As of Sep 14, 2025)
- **Ollama**: Available but not ready (server not running)
- **HuggingFace**: Available but not ready (no token configured) 
- **G4F**: Available but not ready (requires API key in current version)

## Setup for Keyless Operation

### Option 1: Ollama (Recommended for keyless usage)
1. Install Ollama on your system: https://ollama.com/
2. Start the Ollama server: `ollama serve`
3. Pull a model: `ollama pull mistral`
4. Test: `ollama run mistral "Hello"`

Once Ollama is running, the API will automatically detect it and prioritize local models.

### Option 2: HuggingFace (Free but requires token)
1. Create account at https://huggingface.co/
2. Get token from https://huggingface.co/settings/tokens
3. Set environment variable: `export HF_TOKEN=your_token_here`

### Option 3: G4F (May work without keys)
Some G4F providers work without API keys, but current version may require configuration.

## API Endpoints

### Health Check
```bash
curl http://localhost:5000/api/health
```

### List Providers  
```bash
curl http://localhost:5000/api/providers
```

### Chat with Default Model
```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, what can you do?"}'
```

### Chat with Specific Model
```bash
curl -X POST http://localhost:5000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Write Python code", "model": "mistral"}'
```

## Available Models by Provider

### Ollama Models (Keyless)
- mistral
- llama3.3
- llama3.2
- gemma3
- deepseek-r1

### HuggingFace Models (Token required)
- deepseek-r1
- codellama
- phi
- qwen

### G4F Models (Configuration dependent)
- gpt-3.5-turbo
- gpt-4
- claude-3
- gemini-pro

## Architecture Features
- **Intelligent Routing**: Automatically selects best provider for each model
- **Provider Fallbacks**: Falls back to alternative providers if primary fails
- **Readiness Detection**: Distinguishes between installed vs. configured providers
- **Model Compatibility**: Maps model names to appropriate provider APIs
- **Session Management**: Maintains chat history across requests
- **Streaming Support**: Supports real-time streaming responses
- **Error Handling**: Comprehensive error handling with fallback responses

## Recent Changes (Sep 14, 2025)
- Enhanced provider readiness detection
- Fixed model routing to prefer keyless options
- Added proper HuggingFace token detection
- Implemented Ollama server connectivity checks
- Updated default models based on provider availability
- Created HTML playground for testing
- Improved error handling and logging

## User Preferences
- **Keyless Operation**: Priority on solutions that work without API keys
- **Multiple Models**: Support for various AI models from different providers
- **Reliability**: Robust fallback mechanisms when providers are unavailable
- **Backward Compatibility**: Maintain existing API structure for deployment