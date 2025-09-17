import asyncio
import logging
import json
import time
import concurrent.futures
from typing import Optional, Tuple, Dict, Any
from functools import wraps

logger = logging.getLogger(__name__)

class AIProviderManager:
    """Manages multiple AI providers with automatic fallback"""
    
    def __init__(self):
        self.providers = {}
        self.provider_status = {}
        self.model_verification_prompt = "Who are you? What is your name and who developed you? Please respond clearly and specifically."
        self.provider_timeouts = {
            'g4f': 15,
            'gpt4all': 30,
            'ollama': 10,
            'pollinations': 10
        }
        self.circuit_breaker = {
            'g4f': {'failures': 0, 'last_failure': 0, 'threshold': 3, 'cooldown': 300},
            'gpt4all': {'failures': 0, 'last_failure': 0, 'threshold': 2, 'cooldown': 600},
            'ollama': {'failures': 0, 'last_failure': 0, 'threshold': 3, 'cooldown': 180},
            'pollinations': {'failures': 0, 'last_failure': 0, 'threshold': 3, 'cooldown': 300}
        }
        
    def initialize_providers(self):
        """Initialize all available AI providers"""
        # Initialize G4F
        try:
            import g4f
            from g4f.client import Client
            self.providers['g4f'] = G4FProvider()
            self.provider_status['g4f'] = {'available': True, 'last_error': None}
            logger.info("G4F provider initialized successfully")
        except Exception as e:
            self.provider_status['g4f'] = {'available': False, 'last_error': str(e)}
            logger.error(f"Failed to initialize G4F: {e}")
            
        # Initialize GPT4All
        try:
            from gpt4all import GPT4All
            self.providers['gpt4all'] = GPT4AllProvider()
            self.provider_status['gpt4all'] = {'available': True, 'last_error': None}
            logger.info("GPT4All provider initialized successfully")
        except Exception as e:
            self.provider_status['gpt4all'] = {'available': False, 'last_error': str(e)}
            logger.error(f"Failed to initialize GPT4All: {e}")
            
        # Initialize Ollama
        try:
            import ollama
            self.providers['ollama'] = OllamaProvider()
            self.provider_status['ollama'] = {'available': True, 'last_error': None}
            logger.info("Ollama provider initialized successfully")
        except Exception as e:
            self.provider_status['ollama'] = {'available': False, 'last_error': str(e)}
            logger.error(f"Failed to initialize Ollama: {e}")
            
        # Initialize Pollinations
        try:
            import requests
            self.providers['pollinations'] = PollinationsProvider()
            self.provider_status['pollinations'] = {'available': True, 'last_error': None}
            logger.info("Pollinations provider initialized successfully")
        except Exception as e:
            self.provider_status['pollinations'] = {'available': False, 'last_error': str(e)}
            logger.error(f"Failed to initialize Pollinations: {e}")
    
    def _is_provider_available(self, provider_name: str) -> bool:
        """Check if provider is available (not in circuit breaker state)"""
        if not self.provider_status.get(provider_name, {}).get('available', False):
            return False
            
        breaker = self.circuit_breaker.get(provider_name, {})
        current_time = time.time()
        
        # Check if provider is in cooldown period
        if breaker['failures'] >= breaker['threshold']:
            if current_time - breaker['last_failure'] < breaker['cooldown']:
                return False
            else:
                # Reset circuit breaker after cooldown
                breaker['failures'] = 0
                
        return True
    
    def _record_provider_failure(self, provider_name: str):
        """Record provider failure and update circuit breaker"""
        if provider_name in self.circuit_breaker:
            breaker = self.circuit_breaker[provider_name]
            breaker['failures'] += 1
            breaker['last_failure'] = time.time()
            
    def _record_provider_success(self, provider_name: str):
        """Record provider success and reset failures"""
        if provider_name in self.circuit_breaker:
            self.circuit_breaker[provider_name]['failures'] = 0
    
    def get_provider_for_model(self, model: str) -> list:
        """Get prioritized list of providers for a specific model"""
        model_lower = model.lower()
        
        if 'gpt' in model_lower or 'openai' in model_lower:
            return ['g4f', 'pollinations', 'ollama', 'gpt4all']
        elif 'claude' in model_lower:
            return ['g4f', 'pollinations', 'ollama']
        elif 'gemini' in model_lower or 'bard' in model_lower:
            return ['g4f', 'pollinations', 'ollama']
        elif 'llama' in model_lower:
            return ['gpt4all', 'ollama', 'g4f', 'pollinations']
        elif 'mistral' in model_lower:
            return ['gpt4all', 'ollama', 'g4f', 'pollinations']
        else:
            return ['g4f', 'ollama', 'gpt4all', 'pollinations']
    
    def get_response_sync(self, model: str, messages: list, stream: bool = False) -> Tuple[Optional[str], str, str]:
        """Get AI response using provider fallback chain (synchronous)"""
        provider_order = self.get_provider_for_model(model)
        
        for provider_name in provider_order:
            if provider_name not in self.providers:
                continue
                
            if not self._is_provider_available(provider_name):
                continue
                
            try:
                provider = self.providers[provider_name]
                timeout = self.provider_timeouts.get(provider_name, 10)
                
                # Use ThreadPoolExecutor for timeout handling
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(provider.get_response_sync, model, messages, stream)
                    try:
                        response = future.result(timeout=timeout)
                        
                        if response and len(response.strip()) > 10:
                            # Verify this is not a generic fallback response
                            if not self._is_fallback_response(response):
                                self._record_provider_success(provider_name)
                                logger.info(f"Success with provider: {provider_name}")
                                return response, "success", provider_name
                        
                    except concurrent.futures.TimeoutError:
                        logger.error(f"Provider {provider_name} timed out after {timeout}s")
                        self._record_provider_failure(provider_name)
                        continue
                        
            except Exception as e:
                logger.error(f"Provider {provider_name} failed: {e}")
                self.provider_status[provider_name]['last_error'] = str(e)
                self._record_provider_failure(provider_name)
                continue
        
        return None, "failed", "all_providers_failed"
    
    def _is_fallback_response(self, response: str) -> bool:
        """Check if response appears to be a generic fallback"""
        fallback_indicators = [
            "I'm working through g4f API",
            "You said:",
            "How can I help you today?",
            "I'm an AI assistant",
            "fallback response"
        ]
        
        for indicator in fallback_indicators:
            if indicator.lower() in response.lower():
                return True
        return False

class G4FProvider:
    """G4F provider implementation"""
    
    def __init__(self):
        import g4f
        from g4f.client import Client
        self.client = Client()
        self.g4f = g4f
        
    def get_response_sync(self, model: str, messages: list, stream: bool = False) -> Optional[str]:
        try:
            if stream:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True
                )
                full_response = ""
                for chunk in response:
                    if hasattr(chunk, 'choices') and chunk.choices:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content:
                            full_response += delta.content
                return full_response.strip() if full_response.strip() else None
            else:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages
                )
                if hasattr(response, 'choices') and response.choices:
                    content = response.choices[0].message.content
                    return content.strip() if content else None
                return None
        except Exception as e:
            logger.error(f"G4F error: {e}")
            return None

class GPT4AllProvider:
    """GPT4All provider implementation"""
    
    def __init__(self):
        from gpt4all import GPT4All
        self.GPT4All = GPT4All
        self.models = {}
        
    def _get_gpt4all_model_name(self, model: str) -> str:
        """Map API model names to GPT4All model names"""
        model_lower = model.lower()
        if 'llama' in model_lower:
            if '7b' in model_lower:
                return "Meta-Llama-3-8B-Instruct.Q4_0.gguf"
            elif '13b' in model_lower:
                return "nous-hermes-llama2-13b.Q4_0.gguf"
            else:
                return "Meta-Llama-3-8B-Instruct.Q4_0.gguf"
        elif 'mistral' in model_lower:
            return "mistral-7b-openorca.Q4_0.gguf"
        else:
            return "Meta-Llama-3-8B-Instruct.Q4_0.gguf"
    
    def get_response_sync(self, model: str, messages: list, stream: bool = False) -> Optional[str]:
        try:
            gpt4all_model = self._get_gpt4all_model_name(model)
            
            if gpt4all_model not in self.models:
                self.models[gpt4all_model] = self.GPT4All(gpt4all_model)
            
            gpt4all_instance = self.models[gpt4all_model]
            
            # Convert messages to single prompt
            prompt = ""
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if role == 'user':
                    prompt += f"User: {content}\n"
                elif role == 'assistant':
                    prompt += f"Assistant: {content}\n"
            
            prompt += "Assistant: "
            
            with gpt4all_instance.chat_session():
                response = gpt4all_instance.generate(
                    prompt,
                    max_tokens=1024,
                    temp=0.7
                )
                return response.strip() if response else None
                
        except Exception as e:
            logger.error(f"GPT4All error: {e}")
            return None

class OllamaProvider:
    """Ollama provider implementation"""
    
    def __init__(self):
        import ollama
        self.client = ollama
        
    def _get_ollama_model_name(self, model: str) -> str:
        """Map API model names to Ollama model names"""
        model_lower = model.lower()
        if 'gpt' in model_lower:
            return 'llama3'
        elif 'claude' in model_lower:
            return 'llama3'
        elif 'gemini' in model_lower:
            return 'llama3'
        elif 'llama' in model_lower:
            return 'llama3'
        elif 'mistral' in model_lower:
            return 'mistral'
        else:
            return 'llama3'
    
    def get_response_sync(self, model: str, messages: list, stream: bool = False) -> Optional[str]:
        try:
            ollama_model = self._get_ollama_model_name(model)
            
            response = self.client.chat(
                model=ollama_model,
                messages=messages
            )
            
            if 'message' in response and 'content' in response['message']:
                return response['message']['content'].strip()
            return None
            
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return None

class PollinationsProvider:
    """Pollinations.ai provider implementation"""
    
    def __init__(self):
        import requests
        self.session = requests.Session()
        self.base_url = "https://text.pollinations.ai"
        
    def get_response_sync(self, model: str, messages: list, stream: bool = False) -> Optional[str]:
        try:
            # Convert messages to single prompt
            prompt = ""
            for msg in messages:
                role = msg.get('role', 'user')
                content = msg.get('content', '')
                if role == 'user':
                    prompt += content
            
            # Make request to Pollinations API
            response = self.session.get(
                self.base_url,
                params={
                    'prompt': prompt,
                    'model': 'openai',
                    'jsonMode': 'false'
                },
                timeout=30
            )
            
            if response.status_code == 200:
                return response.text.strip()
            return None
            
        except Exception as e:
            logger.error(f"Pollinations error: {e}")
            return None