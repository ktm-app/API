# 🚀 Enhanced Puter API Wrapper

A comprehensive, production-ready Flask API wrapper that provides seamless access to **all Puter capabilities** without requiring end-user authentication. This enhanced version integrates the complete Puter Python SDK, offering developers a powerful, unified API for AI services, cloud storage, and more.

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/flask-3.0+-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Deploy](https://img.shields.io/badge/deploy-render-purple.svg)](https://render.com)

## ✨ Key Features

### 🎯 **No User Authentication Required**
- **Zero-config for end users** - Your API handles all Puter authentication internally
- **Single endpoint access** - Users interact only with your API, not Puter directly
- **Seamless integration** - Drop-in replacement for direct Puter API calls

### 🤖 **Comprehensive AI Services**
- **Chat Completion**: GPT-4, GPT-5, Claude, o1, o3, and more models
- **Image Generation**: DALL-E 3, Stable Diffusion support
- **Image Analysis**: Extract text and descriptions from images  
- **Text-to-Speech**: Natural voice synthesis
- **Vision AI**: Image understanding with GPT-4o

### 📁 **Complete File Management**
- **Cloud Storage**: Upload, download, list, and delete files
- **Large File Support**: Up to 100MB file uploads
- **Secure Operations**: File validation and sanitization
- **Multi-format Support**: Images, documents, audio, video

### 🗄️ **Key-Value Storage**
- **Simple Data Storage**: Store and retrieve JSON data
- **Session Management**: User preferences and state
- **Configuration Storage**: App settings and metadata

### 🚀 **Production-Ready Features**
- **Deployment Optimized**: Ready for Render, Heroku, Docker
- **Comprehensive Logging**: Structured logging with multiple levels
- **Error Recovery**: Robust retry logic and fallback handling
- **Rate Limiting**: Optional Redis-based rate limiting
- **CORS Support**: Cross-origin resource sharing enabled
- **Health Monitoring**: Built-in health check endpoints

## 📋 Available Models

### Chat Models
- `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-5-chat-latest`
- `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`
- `gpt-4o`, `gpt-4o-mini` (vision capable)
- `o1`, `o1-mini`, `o1-pro`
- `o3`, `o3-mini`, `o4-mini`
- `claude`, `claude-3-5-sonnet`

### Image Generation
- `dall-e-3` (default)
- `stable-diffusion`

### Text-to-Speech
- `default`, `neural`

## 🚀 Quick Start

### 1. **Clone & Setup**
```bash
git clone https://github.com/your-username/enhanced-puter-api.git
cd enhanced-puter-api

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies  
pip install -r requirements.txt
```

### 2. **Configure Environment**
```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

**Required Configuration:**
```bash
PUTER_USERNAME=your_puter_username
PUTER_PASSWORD=your_puter_password
SECRET_KEY=your_random_secret_key
```

### 3. **Run the API**
```bash
# Development
python run.py

# Production
gunicorn --bind 0.0.0.0:5000 --workers 2 run:app
```

### 4. **Test the API**
```bash
# Health check
curl http://localhost:5000/api/health

# Chat with AI
curl -X POST http://localhost:5000/api/ai/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

## 📚 API Documentation

### 🏥 **Health & Information**

#### Health Check
```http
GET /api/health
```
**Response:**
```json
{
  "status": "healthy",
  "puter_status": {...},
  "api_version": "2.0.0",
  "timestamp": "2025-01-01T12:00:00Z"
}
```

#### Available Models
```http
GET /api/models
```

### 🤖 **AI Services**

#### Chat Completion
```http
POST /api/ai/chat
Content-Type: application/json

{
  "messages": [
    {"role": "user", "content": "What is artificial intelligence?"}
  ],
  "model": "gpt-4",
  "temperature": 0.7,
  "max_tokens": 1000,
  "stream": false
}
```

#### Text to Image
```http
POST /api/ai/text-to-image
Content-Type: application/json

{
  "prompt": "A futuristic city skyline at sunset",
  "size": "512x512"
}
```

#### Image to Text  
```http
POST /api/ai/image-to-text
Content-Type: application/json

{
  "image_url": "https://example.com/image.jpg"
}
```

#### Text to Speech
```http
POST /api/ai/text-to-speech
Content-Type: application/json

{
  "text": "Hello, this is a test of text-to-speech.",
  "voice": "default"
}
```

### 📁 **File Management**

#### Upload File
```http
POST /api/files/upload
Content-Type: multipart/form-data

file=[binary file data]
path=/my-folder/filename.txt
```

#### Download File
```http
GET /api/files/download/my-folder/filename.txt
```

#### List Files
```http
GET /api/files/list?path=/my-folder
```

#### Delete File
```http
DELETE /api/files/my-folder/filename.txt
```

### 🗄️ **Key-Value Storage**

#### Set Data
```http
POST /api/kv/set
Content-Type: application/json

{
  "key": "user_settings",
  "value": {"theme": "dark", "notifications": true}
}
```

#### Get Data
```http
GET /api/kv/get/user_settings
```

#### Delete Data
```http
DELETE /api/kv/delete/user_settings
```

## 🚀 Deployment

### **Render.com (Recommended)**

1. **Fork this repository** to your GitHub account

2. **Create a new Web Service** on [Render](https://render.com)

3. **Connect your repository** - Render will auto-detect `render.yaml`

4. **Set environment variables** in Render dashboard:
   ```
   PUTER_USERNAME=your_puter_username
   PUTER_PASSWORD=your_puter_password  
   SECRET_KEY=your_generated_secret_key
   ```

5. **Deploy!** - Your API will be available at `https://your-service.onrender.com`

### **Heroku**

```bash
# Install Heroku CLI and login
heroku create your-app-name

# Set environment variables
heroku config:set PUTER_USERNAME=your_username
heroku config:set PUTER_PASSWORD=your_password
heroku config:set SECRET_KEY=your_secret_key

# Deploy
git push heroku main
```

### **Docker**

```bash
# Build image
docker build -t enhanced-puter-api .

# Run container
docker run -d -p 5000:5000 \
  -e PUTER_USERNAME=your_username \
  -e PUTER_PASSWORD=your_password \
  -e SECRET_KEY=your_secret_key \
  enhanced-puter-api
```

## 🔧 Configuration

### **Environment Variables**

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PUTER_USERNAME` | ✅ | - | Your Puter username |
| `PUTER_PASSWORD` | ✅ | - | Your Puter password |
| `SECRET_KEY` | ✅ | - | Flask secret key |
| `FLASK_ENV` | ❌ | `production` | Environment mode |
| `PORT` | ❌ | `5000` | Server port |
| `HOST` | ❌ | `0.0.0.0` | Server host |
| `LOG_LEVEL` | ❌ | `INFO` | Logging level |
| `CORS_ORIGINS` | ❌ | `*` | CORS origins |
| `PUTER_TEST_MODE` | ❌ | `false` | Test mode |
| `REDIS_URL` | ❌ | - | Redis for rate limiting |

### **Optional Features**

#### Rate Limiting
```bash
# Enable with Redis
RATELIMIT_ENABLED=true
REDIS_URL=redis://localhost:6379/0
RATELIMIT_DEFAULT=1000 per hour
```

#### Enhanced Logging
```bash
LOG_LEVEL=DEBUG
LOG_TO_STDOUT=true
```

## 📊 Usage Examples

### **Python Client Example**
```python
import requests

# API base URL
API_BASE = "https://your-api.onrender.com"

# Chat with AI
response = requests.post(f"{API_BASE}/api/ai/chat", json={
    "messages": [{"role": "user", "content": "Explain quantum computing"}],
    "model": "gpt-4"
})
result = response.json()
print(result["response"])

# Generate image
response = requests.post(f"{API_BASE}/api/ai/text-to-image", json={
    "prompt": "A serene mountain landscape"
})
image_data = response.json()
print(f"Image URL: {image_data['image_url']}")

# Upload file
with open("document.pdf", "rb") as f:
    files = {"file": f}
    data = {"path": "/documents/document.pdf"}
    response = requests.post(f"{API_BASE}/api/files/upload", 
                           files=files, data=data)
print(response.json())
```

### **JavaScript Client Example**
```javascript
const API_BASE = "https://your-api.onrender.com";

// Chat with AI
async function chatWithAI(message) {
  const response = await fetch(`${API_BASE}/api/ai/chat`, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      messages: [{role: "user", content: message}],
      model: "gpt-4"
    })
  });
  const data = await response.json();
  return data.response;
}

// Generate image
async function generateImage(prompt) {
  const response = await fetch(`${API_BASE}/api/ai/text-to-image`, {
    method: "POST", 
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({prompt})
  });
  const data = await response.json();
  return data.image_url;
}

// Usage
chatWithAI("What is machine learning?").then(console.log);
generateImage("A robot in a garden").then(console.log);
```

### **cURL Examples**
```bash
# Health check
curl https://your-api.onrender.com/api/health

# Chat completion
curl -X POST https://your-api.onrender.com/api/ai/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello AI!"}]}'

# Image generation  
curl -X POST https://your-api.onrender.com/api/ai/text-to-image \
  -H "Content-Type: application/json" \
  -d '{"prompt":"A beautiful sunset"}'

# File upload
curl -X POST https://your-api.onrender.com/api/files/upload \
  -F "file=@document.pdf" \
  -F "path=/uploads/document.pdf"
```

## 🧪 Testing

### **Run Test Suite**
```bash
# Install test dependencies
pip install pytest pytest-flask

# Run tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=app
```

### **Manual API Testing**
```bash
# Use the included test script
python test_api.py

# Or test individual endpoints
curl http://localhost:5000/api/health
```

## 🔒 Security

### **Best Practices**
- ✅ **Environment Variables**: Never commit credentials to code
- ✅ **HTTPS**: Always use HTTPS in production
- ✅ **Secret Keys**: Generate random secret keys
- ✅ **Input Validation**: All inputs are validated and sanitized
- ✅ **Error Handling**: Secure error messages without data leaks
- ✅ **Rate Limiting**: Implement rate limiting for production

### **Security Headers**
The API includes standard security headers:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY` 
- `X-XSS-Protection: 1; mode=block`

## 🐛 Troubleshooting

### **Common Issues**

#### Authentication Errors
```
Error: Authentication required but failed
```
**Solution**: Verify `PUTER_USERNAME` and `PUTER_PASSWORD` in environment variables.

#### Port Conflicts
```
Error: Address already in use
```
**Solution**: Change the port in `.env` file or kill existing processes.

#### Missing Dependencies
```
Error: No module named 'putergenai'
```
**Solution**: Install requirements: `pip install -r requirements.txt`

#### Deployment Errors
```
Error: Application failed to start
```
**Solution**: Check logs for specific errors, verify environment variables are set.

### **Debug Mode**
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export FLASK_ENV=development
python run.py
```

### **Health Diagnostics**
```bash
# Check API health
curl http://localhost:5000/api/health

# Check Puter connection
curl http://localhost:5000/api/user
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### **Development Setup**
```bash
# Clone and setup
git clone https://github.com/your-username/enhanced-puter-api.git
cd enhanced-puter-api
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install development dependencies
pip install -r requirements-dev.txt

# Run in development mode
export FLASK_ENV=development
python run.py
```

### **Code Standards**
- Follow PEP 8 style guidelines
- Add type hints where possible
- Include docstrings for functions
- Write tests for new features
- Update documentation

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[Puter](https://puter.com)** - The open-source cloud operating system powering this API
- **[putergenai](https://github.com/Nerve11/putergenai)** - Python SDK for Puter integration
- **Flask Community** - For the excellent web framework
- **All Contributors** - Thank you for making this project better!

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/enhanced-puter-api/issues)
- **Documentation**: [API Docs](https://your-api.onrender.com/)
- **Puter Help**: [Puter Documentation](https://docs.puter.com)
- **Community**: [Puter Discord](https://discord.gg/puter)

## 🚀 What's Next?

- [ ] WebSocket support for real-time AI streaming
- [ ] Database integration for persistent data
- [ ] Advanced rate limiting and analytics
- [ ] Docker container optimization
- [ ] Additional AI model integrations
- [ ] GraphQL API interface
- [ ] OpenAPI/Swagger documentation
- [ ] Monitoring and alerting integration

---

**Built with ❤️ for developers who want to harness the full power of Puter without the complexity.**

*Ready to deploy? Click the button below to deploy to Render in seconds:*

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)
