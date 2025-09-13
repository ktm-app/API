#!/usr/bin/env python3
"""
Comprehensive API Testing Script for Puter API Wrapper
=====================================================

This script validates all API endpoints and functionality.
Run this after deploying to ensure everything works correctly.

Usage:
    python test_api.py [BASE_URL]

Examples:
    python test_api.py                           # Test localhost:5000
    python test_api.py http://your-app.onrender.com  # Test deployed app
"""

import sys
import json
import time
import requests
from typing import Dict, Any, Optional
import base64
from io import BytesIO


class PuterAPITester:
    """Comprehensive tester for Puter API wrapper endpoints."""

    def __init__(self, base_url: str = "http://localhost:5000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.timeout = 30
        self.results = {
            'passed': 0,
            'failed': 0,
            'errors': []
        }

    def log(self, message: str, level: str = "INFO"):
        """Log test messages with timestamp."""
        timestamp = time.strftime("%H:%M:%S")
        prefix = {
            'INFO': '📋',
            'SUCCESS': '✅',
            'WARNING': '⚠️',
            'ERROR': '❌',
            'TEST': '🧪'
        }.get(level, 'ℹ️')
        print(f"[{timestamp}] {prefix} {message}")

    def assert_response(self, response: requests.Response, 
                       expected_status: int = 200, 
                       test_name: str = "Test") -> Optional[Dict[Any, Any]]:
        """Assert response status and return JSON if valid."""
        try:
            if response.status_code == expected_status:
                self.log(f"{test_name}: PASSED (Status: {response.status_code})", "SUCCESS")
                self.results['passed'] += 1
                try:
                    return response.json()
                except:
                    return {'status': 'success', 'raw_response': response.text}
            else:
                error_msg = f"{test_name}: Status {response.status_code}, Expected {expected_status}"
                self.log(error_msg, "ERROR")
                self.results['failed'] += 1
                self.results['errors'].append(error_msg)
                return None
        except Exception as e:
            error_msg = f"{test_name}: Exception - {str(e)}"
            self.log(error_msg, "ERROR")
            self.results['failed'] += 1
            self.results['errors'].append(error_msg)
            return None

    def test_health_check(self):
        """Test basic health check endpoint."""
        self.log("Testing health check endpoint...", "TEST")
        try:
            response = self.session.get(f"{self.base_url}/api/health")
            data = self.assert_response(response, 200, "Health Check")
            if data and data.get('status') == 'healthy':
                self.log("Health check returned healthy status", "INFO")
            return data is not None
        except Exception as e:
            self.log(f"Health check failed: {e}", "ERROR")
            return False

    def test_user_info(self):
        """Test user info endpoint."""
        self.log("Testing user info endpoint...", "TEST")
        try:
            response = self.session.get(f"{self.base_url}/api/user/info")
            data = self.assert_response(response, 200, "User Info")
            if data and 'user' in data:
                self.log(f"Retrieved user info: {data['user'].get('username', 'Unknown')}", "INFO")
            return data is not None
        except Exception as e:
            self.log(f"User info test failed: {e}", "ERROR")
            return False

    def test_ai_chat(self):
        """Test AI chat completion endpoint."""
        self.log("Testing AI chat completion...", "TEST")
        try:
            payload = {
                "message": "Hello! Can you tell me a short joke?",
                "model": "gpt-4",
                "max_tokens": 100
            }
            response = self.session.post(
                f"{self.base_url}/api/ai/chat", 
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            data = self.assert_response(response, 200, "AI Chat")
            if data and 'response' in data:
                self.log(f"AI Response: {data['response'][:100]}...", "INFO")
            return data is not None
        except Exception as e:
            self.log(f"AI chat test failed: {e}", "ERROR")
            return False

    def test_text_to_image(self):
        """Test text-to-image generation."""
        self.log("Testing text-to-image generation...", "TEST")
        try:
            payload = {
                "prompt": "A simple blue circle on white background",
                "model": "dall-e-3",
                "size": "512x512"
            }
            response = self.session.post(
                f"{self.base_url}/api/ai/text-to-image",
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            data = self.assert_response(response, 200, "Text-to-Image")
            if data and ('image_url' in data or 'image_data' in data):
                self.log("Image generation successful", "INFO")
            return data is not None
        except Exception as e:
            self.log(f"Text-to-image test failed: {e}", "ERROR")
            return False

    def test_text_to_speech(self):
        """Test text-to-speech conversion."""
        self.log("Testing text-to-speech conversion...", "TEST")
        try:
            payload = {
                "text": "Hello, this is a test of the text to speech system.",
                "voice": "alloy"
            }
            response = self.session.post(
                f"{self.base_url}/api/ai/text-to-speech",
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            data = self.assert_response(response, 200, "Text-to-Speech")
            if data and ('audio_url' in data or 'audio_data' in data):
                self.log("TTS generation successful", "INFO")
            return data is not None
        except Exception as e:
            self.log(f"Text-to-speech test failed: {e}", "ERROR")
            return False

    def test_kv_storage(self):
        """Test key-value storage operations."""
        self.log("Testing KV storage operations...", "TEST")
        test_key = f"test_key_{int(time.time())}"
        test_value = {"message": "Hello from test!", "timestamp": time.time()}

        # Test SET operation
        try:
            payload = {"key": test_key, "value": test_value}
            response = self.session.post(
                f"{self.base_url}/api/kv/set",
                json=payload,
                headers={'Content-Type': 'application/json'}
            )
            set_result = self.assert_response(response, 200, "KV Set")
            if not set_result:
                return False

            # Test GET operation
            response = self.session.get(f"{self.base_url}/api/kv/get/{test_key}")
            get_result = self.assert_response(response, 200, "KV Get")
            if get_result and get_result.get('value') == test_value:
                self.log("KV storage round-trip successful", "INFO")

                # Test DELETE operation
                response = self.session.delete(f"{self.base_url}/api/kv/delete/{test_key}")
                del_result = self.assert_response(response, 200, "KV Delete")
                return del_result is not None
            return False
        except Exception as e:
            self.log(f"KV storage test failed: {e}", "ERROR")
            return False

    def test_file_operations(self):
        """Test file upload and management."""
        self.log("Testing file operations...", "TEST")
        try:
            # Create a simple test file
            test_content = "This is a test file for API validation."
            test_filename = f"test_file_{int(time.time())}.txt"

            # Test file upload
            files = {
                'file': (test_filename, test_content, 'text/plain')
            }
            response = self.session.post(
                f"{self.base_url}/api/files/upload",
                files=files
            )
            upload_result = self.assert_response(response, 200, "File Upload")
            if not upload_result:
                return False

            file_path = upload_result.get('file_path')
            if not file_path:
                self.log("File upload did not return file_path", "ERROR")
                return False

            # Test file list
            response = self.session.get(f"{self.base_url}/api/files/list")
            list_result = self.assert_response(response, 200, "File List")
            if list_result and 'files' in list_result:
                self.log(f"Found {len(list_result['files'])} files", "INFO")

            # Test file download
            response = self.session.get(f"{self.base_url}/api/files/download/{file_path}")
            download_result = self.assert_response(response, 200, "File Download")
            if download_result:
                self.log("File operations completed successfully", "INFO")
                return True
            return False
        except Exception as e:
            self.log(f"File operations test failed: {e}", "ERROR")
            return False

    def test_error_handling(self):
        """Test API error handling."""
        self.log("Testing error handling...", "TEST")

        # Test invalid endpoint
        response = self.session.get(f"{self.base_url}/api/nonexistent")
        self.assert_response(response, 404, "Invalid Endpoint")

        # Test invalid JSON
        response = self.session.post(
            f"{self.base_url}/api/ai/chat",
            data="invalid json",
            headers={'Content-Type': 'application/json'}
        )
        # Should return 400 for bad request
        if response.status_code in [400, 422]:
            self.log("Error handling: PASSED (Bad JSON handled)", "SUCCESS")
            self.results['passed'] += 1
        else:
            self.log(f"Error handling: FAILED (Expected 400/422, got {response.status_code})", "ERROR")
            self.results['failed'] += 1

        # Test missing required fields
        response = self.session.post(
            f"{self.base_url}/api/ai/chat",
            json={},  # Empty payload
            headers={'Content-Type': 'application/json'}
        )
        if response.status_code in [400, 422]:
            self.log("Error handling: PASSED (Missing fields handled)", "SUCCESS")
            self.results['passed'] += 1
        else:
            self.log(f"Error handling: FAILED (Expected 400/422, got {response.status_code})", "ERROR")
            self.results['failed'] += 1

    def run_all_tests(self):
        """Run comprehensive test suite."""
        self.log(f"Starting comprehensive API tests for: {self.base_url}", "INFO")
        self.log("=" * 60, "INFO")

        # Check if API is accessible
        if not self.test_health_check():
            self.log("❌ API is not accessible. Stopping tests.", "ERROR")
            return False

        # Core functionality tests
        tests = [
            ("User Info", self.test_user_info),
            ("AI Chat", self.test_ai_chat),
            ("Text-to-Image", self.test_text_to_image),
            ("Text-to-Speech", self.test_text_to_speech),
            ("KV Storage", self.test_kv_storage),
            ("File Operations", self.test_file_operations),
            ("Error Handling", self.test_error_handling),
        ]

        self.log("Running core functionality tests...", "INFO")
        for test_name, test_func in tests:
            self.log(f"\n--- {test_name} Test ---", "TEST")
            try:
                test_func()
            except Exception as e:
                self.log(f"{test_name} test crashed: {e}", "ERROR")
                self.results['failed'] += 1
                self.results['errors'].append(f"{test_name}: {str(e)}")

        # Print final results
        self.log("=" * 60, "INFO")
        self.log("🎯 TEST RESULTS SUMMARY", "INFO")
        self.log(f"✅ Passed: {self.results['passed']}", "SUCCESS")
        self.log(f"❌ Failed: {self.results['failed']}", "ERROR")

        if self.results['errors']:
            self.log("\n📋 Error Details:", "WARNING")
            for error in self.results['errors']:
                self.log(f"  • {error}", "ERROR")

        success_rate = (self.results['passed'] / 
                       (self.results['passed'] + self.results['failed']) * 100 
                       if (self.results['passed'] + self.results['failed']) > 0 else 0)

        self.log(f"\n🎯 Success Rate: {success_rate:.1f}%", "INFO")

        if success_rate >= 80:
            self.log("🎉 API is functioning well!", "SUCCESS")
            return True
        elif success_rate >= 60:
            self.log("⚠️ API has some issues but is mostly functional", "WARNING")
            return True
        else:
            self.log("❌ API has significant issues", "ERROR")
            return False


def main():
    """Main test runner."""
    # Determine base URL
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = "http://localhost:5000"

    print("🚀 Puter API Wrapper - Comprehensive Test Suite")
    print("=" * 50)
    print(f"Target URL: {base_url}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    # Run tests
    tester = PuterAPITester(base_url)
    success = tester.run_all_tests()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
