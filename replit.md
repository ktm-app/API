# Overview

This is a Flask-based AI API server that provides a unified interface to multiple AI providers. The application acts as an aggregator and router for different AI services including G4F, GPT4All, Ollama, and Pollinations. It features a web-based chat playground, automatic provider fallback, circuit breaker patterns for reliability, and in-memory session management. The system is designed to maximize AI availability by trying multiple providers in priority order when one fails.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Web Framework
- **Flask**: Chosen as the core web framework for its simplicity and flexibility in building REST APIs
- **Blueprint Architecture**: API routes are organized using Flask blueprints for better code organization and modularity
- **Factory Pattern**: Application creation uses the factory pattern (create_app) for better configuration management and testing

## AI Provider Management
- **Multi-Provider Strategy**: Implements a provider manager that supports multiple AI backends (G4F, GPT4All, Ollama, Pollinations) with configurable priority levels
- **Circuit Breaker Pattern**: Each provider has failure tracking with automatic circuit breaking to prevent cascading failures
- **Automatic Fallback**: When a provider fails, the system automatically tries the next available provider in priority order
- **Provider Status Tracking**: Real-time monitoring of each provider's availability and error states

## Session and State Management
- **In-Memory Storage**: Chat history is stored in memory using Python dictionaries, suitable for development but not production-scale
- **Session-Based Conversations**: Each chat session is identified by a session ID, allowing for conversation continuity
- **Stateless API Design**: The API itself is stateless, with session data managed separately

## Security and Rate Limiting
- **Flask-Limiter**: Implements configurable rate limiting to prevent abuse (default: 100 requests per minute)
- **CORS Support**: Cross-origin resource sharing enabled for web client integration
- **Environment-Based Configuration**: Sensitive configuration managed through environment variables

## Frontend Architecture
- **Static File Serving**: Flask serves a single-page HTML application for the chat playground
- **REST API Integration**: Frontend communicates with backend through JSON APIs
- **Responsive Design**: CSS styling includes responsive design patterns for various screen sizes

# External Dependencies

## AI Provider Services
- **G4F (GPT4Free)**: Primary AI provider offering access to various language models
- **Ollama**: Local AI model serving with configurable base URL (default: localhost:11434)
- **GPT4All**: Local language model runtime with configurable model path
- **Pollinations**: Additional AI service provider for model diversity

## Infrastructure Dependencies
- **Gunicorn**: WSGI HTTP Server for production deployment
- **Python-dotenv**: Environment variable management for configuration
- **Flask Extensions**: 
  - Flask-CORS for cross-origin requests
  - Flask-Limiter for rate limiting

## Network and HTTP Libraries
- **Requests**: HTTP client library for API communications
- **aiohttp**: Async HTTP client for concurrent provider requests
- **curl_cffi**: HTTP client with CloudFlare bypass capabilities
- **nest_asyncio**: Enables nested event loops for async operations

## Development and Runtime
- **Asyncio-throttle**: Async request throttling for provider management
- **Logging**: Built-in Python logging with configurable levels
- **Concurrent.futures**: Thread-based execution for provider timeout handling