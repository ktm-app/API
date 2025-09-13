"""
Data Models and Validation
Pydantic models for request/response validation and data structures.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

# AI MODELS

class ChatMessage(BaseModel):
    """Chat message model."""
    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")

    @validator('role')
    def validate_role(cls, v):
        if v not in ['user', 'assistant', 'system']:
            raise ValueError('Role must be user, assistant, or system')
        return v

class ChatRequest(BaseModel):
    """AI Chat request model."""
    messages: Union[List[ChatMessage], str] = Field(..., description="Chat messages or simple text prompt")
    model: str = Field("gpt-4.1-nano", description="AI model to use")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Response randomness")
    max_tokens: int = Field(1000, ge=1, le=4000, description="Maximum response length")
    stream: bool = Field(False, description="Enable streaming responses")
    image_url: Optional[str] = Field(None, description="Image URL for vision models")

class ChatResponse(BaseModel):
    """AI Chat response model."""
    response: str = Field(..., description="AI response text")
    model_used: str = Field(..., description="Model that was actually used")
    success: bool = Field(True, description="Whether the request was successful")
    error: Optional[str] = Field(None, description="Error message if any")

class TextToImageRequest(BaseModel):
    """Text to image generation request."""
    prompt: str = Field(..., min_length=1, max_length=1000, description="Text prompt for image generation")
    size: str = Field("512x512", description="Image size")
    quality: str = Field("standard", description="Image quality")
    style: str = Field("natural", description="Image style")

class TextToImageResponse(BaseModel):
    """Text to image generation response."""
    image_url: str = Field(..., description="Generated image URL")
    prompt: str = Field(..., description="Original text prompt")
    success: bool = Field(True, description="Whether the request was successful")
    error: Optional[str] = Field(None, description="Error message if any")

class ImageToTextRequest(BaseModel):
    """Image to text analysis request."""
    image_url: str = Field(..., description="URL of the image to analyze")

    @validator('image_url')
    def validate_image_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('Image URL must be a valid HTTP/HTTPS URL')
        return v

class ImageToTextResponse(BaseModel):
    """Image to text analysis response."""
    text: str = Field(..., description="Extracted text or description")
    image_url: str = Field(..., description="Original image URL")
    success: bool = Field(True, description="Whether the request was successful")
    error: Optional[str] = Field(None, description="Error message if any")

class TextToSpeechRequest(BaseModel):
    """Text to speech conversion request."""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to convert to speech")
    voice: str = Field("default", description="Voice to use for speech synthesis")

class TextToSpeechResponse(BaseModel):
    """Text to speech conversion response."""
    audio_data_base64: str = Field(..., description="Base64 encoded audio data")
    text: str = Field(..., description="Original text")
    voice: str = Field(..., description="Voice used")
    success: bool = Field(True, description="Whether the request was successful")
    error: Optional[str] = Field(None, description="Error message if any")

# FILE MODELS

class FileUploadResponse(BaseModel):
    """File upload response model."""
    file_path: str = Field(..., description="Path where file was uploaded")
    upload_result: Dict[str, Any] = Field(..., description="Upload operation result")
    success: bool = Field(True, description="Whether the upload was successful")
    error: Optional[str] = Field(None, description="Error message if any")

class FileListResponse(BaseModel):
    """File list response model."""
    files: List[Dict[str, Any]] = Field(..., description="List of files")
    path: str = Field(..., description="Directory path")
    success: bool = Field(True, description="Whether the request was successful")
    note: Optional[str] = Field(None, description="Additional notes")

class FileDeleteResponse(BaseModel):
    """File deletion response model."""
    file_path: str = Field(..., description="Path of deleted file")
    success: bool = Field(True, description="Whether the deletion was successful")
    error: Optional[str] = Field(None, description="Error message if any")

# KEY-VALUE STORAGE MODELS

class KVSetRequest(BaseModel):
    """Key-value set request model."""
    key: str = Field(..., min_length=1, max_length=255, description="Key name")
    value: Any = Field(..., description="Value to store")

class KVSetResponse(BaseModel):
    """Key-value set response model."""
    key: str = Field(..., description="Key that was set")
    success: bool = Field(True, description="Whether the operation was successful")
    note: Optional[str] = Field(None, description="Additional notes")

class KVGetResponse(BaseModel):
    """Key-value get response model."""
    key: str = Field(..., description="Key that was requested")
    value: Any = Field(None, description="Retrieved value")
    success: bool = Field(True, description="Whether the operation was successful")
    note: Optional[str] = Field(None, description="Additional notes")

class KVDeleteResponse(BaseModel):
    """Key-value delete response model."""
    key: str = Field(..., description="Key that was deleted")
    success: bool = Field(True, description="Whether the deletion was successful")
    note: Optional[str] = Field(None, description="Additional notes")

# SYSTEM MODELS

class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="System status (healthy/unhealthy)")
    puter_status: Dict[str, Any] = Field(..., description="Puter client status")
    api_version: str = Field(..., description="API version")
    test_mode: bool = Field(..., description="Whether in test mode")
    timestamp: str = Field(..., description="Timestamp of health check")

class UserInfoResponse(BaseModel):
    """User information response model."""
    username: str = Field(..., description="Puter username")
    authenticated: bool = Field(..., description="Whether user is authenticated")
    test_mode: bool = Field(..., description="Whether in test mode")
    success: bool = Field(True, description="Whether the request was successful")

class ModelsResponse(BaseModel):
    """Available models response."""
    models: Dict[str, List[str]] = Field(..., description="Available models by category")
    default_models: Dict[str, str] = Field(..., description="Default models for each category")
    success: bool = Field(True, description="Whether the request was successful")

class UsageResponse(BaseModel):
    """API usage response model."""
    message: str = Field(..., description="Usage message")
    note: str = Field(..., description="Additional notes")
    success: bool = Field(True, description="Whether the request was successful")

# ERROR MODELS

class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error message")
    success: bool = Field(False, description="Always false for errors")
    status_code: Optional[int] = Field(None, description="HTTP status code")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

# VALIDATION UTILITIES

def validate_json_request(request_data: Dict[str, Any], model_class: BaseModel) -> Union[BaseModel, ErrorResponse]:
    """
    Validate JSON request data against a Pydantic model.

    Args:
        request_data: Request data to validate
        model_class: Pydantic model class to validate against

    Returns:
        Validated model instance or ErrorResponse
    """
    try:
        return model_class(**request_data)
    except Exception as e:
        return ErrorResponse(
            error=f"Validation error: {str(e)}",
            details={"validation_errors": str(e)}
        )

def create_success_response(data: Dict[str, Any], model_class: BaseModel) -> BaseModel:
    """
    Create a successful response using a Pydantic model.

    Args:
        data: Response data
        model_class: Response model class

    Returns:
        Model instance with success=True
    """
    data['success'] = True
    return model_class(**data)

def create_error_response(error_message: str, status_code: int = 500, details: Optional[Dict[str, Any]] = None) -> ErrorResponse:
    """
    Create an error response.

    Args:
        error_message: Error message
        status_code: HTTP status code
        details: Additional error details

    Returns:
        ErrorResponse instance
    """
    return ErrorResponse(
        error=error_message,
        status_code=status_code,
        details=details
    )
