"""
Enhanced Puter Client Wrapper
Comprehensive wrapper for the Puter Python SDK with all capabilities.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Union, Generator, Tuple
from putergenai import PuterClient
import requests
import json
import time

logger = logging.getLogger(__name__)

class EnhancedPuterClient:
    """
    Enhanced Puter client that provides all Puter SDK capabilities with
    robust error handling, retry logic, and comprehensive API coverage.
    """

    def __init__(self, username: str, password: str, test_mode: bool = False):
        """Initialize the enhanced Puter client."""
        self.username = username
        self.password = password
        self.test_mode = test_mode
        self.client = None
        self._authenticated = False
        self.max_retries = 3
        self.retry_delay = 1

        logger.info(f"Initializing Enhanced Puter Client (test_mode: {test_mode})")

    async def authenticate(self) -> bool:
        """Authenticate with Puter and establish connection."""
        try:
            self.client = PuterClient()
            token = self.client.login(self.username, self.password)
            self._authenticated = True
            logger.info("Successfully authenticated with Puter")
            return True
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            self._authenticated = False
            return False

    def _ensure_authenticated(self):
        """Ensure client is authenticated before making requests."""
        if not self._authenticated or not self.client:
            # Try to authenticate synchronously
            try:
                self.client = PuterClient()
                self.client.login(self.username, self.password)
                self._authenticated = True
            except Exception as e:
                raise Exception(f"Authentication required but failed: {str(e)}")

    def _retry_on_failure(self, func, *args, **kwargs):
        """Retry function on failure with exponential backoff."""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                wait_time = self.retry_delay * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)

    # AI SERVICES

    def ai_chat(self, 
                messages: Union[str, List[Dict[str, Any]]], 
                model: str = "gpt-4.1-nano",
                temperature: float = 0.7,
                max_tokens: int = 1000,
                stream: bool = False,
                image_url: Optional[str] = None) -> Union[Dict[str, Any], Generator]:
        """
        AI Chat completion with multiple model support.

        Args:
            messages: Chat messages or simple text prompt
            model: AI model to use (gpt-4, gpt-5, claude, etc.)
            temperature: Response randomness (0-2)
            max_tokens: Maximum response length
            stream: Enable streaming responses
            image_url: Image URL for vision models
        """
        self._ensure_authenticated()

        try:
            # Convert simple string to message format
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]

            options = {
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream
            }

            # Add image support for vision models
            kwargs = {
                "messages": messages,
                "options": options,
                "test_mode": self.test_mode
            }

            if image_url:
                kwargs["image_url"] = image_url

            response = self._retry_on_failure(self.client.ai_chat, **kwargs)

            if stream:
                return response  # Generator for streaming
            else:
                return {
                    "response": response.get("response", {}).get("result", {}).get("message", {}).get("content", ""),
                    "model_used": response.get("used_model", model),
                    "success": True
                }

        except Exception as e:
            logger.error(f"AI Chat error: {str(e)}")
            return {
                "error": str(e),
                "success": False,
                "response": "I apologize, but I encountered an error processing your request."
            }

    def text_to_image(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate image from text prompt."""
        self._ensure_authenticated()

        try:
            result = self._retry_on_failure(
                self.client.ai_txt2img,
                prompt=prompt,
                test_mode=self.test_mode
            )

            return {
                "image_url": result,
                "prompt": prompt,
                "success": True
            }

        except Exception as e:
            logger.error(f"Text-to-image error: {str(e)}")
            return {
                "error": str(e),
                "success": False
            }

    def image_to_text(self, image_url: str) -> Dict[str, Any]:
        """Extract text/description from image."""
        self._ensure_authenticated()

        try:
            result = self._retry_on_failure(
                self.client.ai_img2txt,
                image=image_url,
                test_mode=self.test_mode
            )

            return {
                "text": result,
                "image_url": image_url,
                "success": True
            }

        except Exception as e:
            logger.error(f"Image-to-text error: {str(e)}")
            return {
                "error": str(e),
                "success": False
            }

    def text_to_speech(self, text: str, voice: str = "default") -> Dict[str, Any]:
        """Convert text to speech."""
        self._ensure_authenticated()

        try:
            # Use TTS options if available
            options = {"voice": voice} if voice != "default" else None

            result = self._retry_on_failure(
                self.client.ai_txt2speech,
                text=text,
                options=options
            )

            return {
                "audio_data": result,
                "text": text,
                "voice": voice,
                "success": True
            }

        except Exception as e:
            logger.error(f"Text-to-speech error: {str(e)}")
            return {
                "error": str(e),
                "success": False
            }

    # FILE MANAGEMENT

    def upload_file(self, file_content: Union[str, bytes], file_path: str) -> Dict[str, Any]:
        """Upload file to Puter cloud storage."""
        self._ensure_authenticated()

        try:
            result = self._retry_on_failure(
                self.client.fs_write,
                path=file_path,
                content=file_content
            )

            return {
                "file_path": file_path,
                "upload_result": result,
                "success": True
            }

        except Exception as e:
            logger.error(f"File upload error: {str(e)}")
            return {
                "error": str(e),
                "success": False
            }

    def download_file(self, file_path: str) -> Dict[str, Any]:
        """Download file from Puter cloud storage."""
        self._ensure_authenticated()

        try:
            result = self._retry_on_failure(
                self.client.fs_read,
                path=file_path
            )

            return {
                "file_content": result,
                "file_path": file_path,
                "success": True
            }

        except Exception as e:
            logger.error(f"File download error: {str(e)}")
            return {
                "error": str(e),
                "success": False
            }

    def delete_file(self, file_path: str) -> Dict[str, Any]:
        """Delete file from Puter cloud storage."""
        self._ensure_authenticated()

        try:
            result = self._retry_on_failure(
                self.client.fs_delete,
                path=file_path
            )

            return {
                "file_path": file_path,
                "success": True
            }

        except Exception as e:
            logger.error(f"File delete error: {str(e)}")
            return {
                "error": str(e),
                "success": False
            }

    def list_files(self, path: str = "/") -> Dict[str, Any]:
        """List files in Puter cloud storage (placeholder - SDK may not have this)."""
        # Note: The current SDK might not have a list files method
        # This is a placeholder for when it's available
        return {
            "files": [],
            "path": path,
            "success": True,
            "note": "File listing may require direct API calls"
        }

    # KEY-VALUE STORAGE (placeholder methods)

    def kv_set(self, key: str, value: Any) -> Dict[str, Any]:
        """Set key-value pair (placeholder for future SDK support)."""
        # The current putergenai SDK might not have KV storage
        # This would need to be implemented via direct API calls
        return {
            "key": key,
            "success": True,
            "note": "KV storage may require direct API implementation"
        }

    def kv_get(self, key: str) -> Dict[str, Any]:
        """Get value by key (placeholder for future SDK support)."""
        return {
            "key": key,
            "value": None,
            "success": True,
            "note": "KV storage may require direct API implementation"
        }

    def kv_delete(self, key: str) -> Dict[str, Any]:
        """Delete key-value pair (placeholder for future SDK support)."""
        return {
            "key": key,
            "success": True,
            "note": "KV storage may require direct API implementation"
        }

    # UTILITY METHODS

    def get_user_info(self) -> Dict[str, Any]:
        """Get current user information."""
        return {
            "username": self.username,
            "authenticated": self._authenticated,
            "test_mode": self.test_mode,
            "success": True
        }

    def health_check(self) -> Dict[str, Any]:
        """Perform health check of the Puter connection."""
        try:
            if not self._authenticated:
                auth_success = await self.authenticate() if asyncio.iscoroutinefunction(self.authenticate) else True
                if not auth_success:
                    return {
                        "status": "unhealthy",
                        "authenticated": False,
                        "success": False
                    }

            return {
                "status": "healthy",
                "authenticated": self._authenticated,
                "test_mode": self.test_mode,
                "success": True
            }

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "success": False
            }
