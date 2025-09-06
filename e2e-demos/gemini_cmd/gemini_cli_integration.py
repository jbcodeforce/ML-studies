"""
Proof of Concept: Gemini CLI Integration Tests

This module demonstrates how to call the Gemini CLI from Python using subprocess.
It includes wrapper classes, error handling, and comprehensive test cases.
"""

import subprocess
import json
import os
import tempfile

from unittest.mock import patch, Mock, MagicMock
from typing import Optional, Dict, Any, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeminiCLIError(Exception):
    """Custom exception for Gemini CLI related errors."""
    pass


class GeminiCLI:
    """
    Wrapper class for interacting with Google's Gemini CLI.
    
    This class provides a Python interface to the Gemini command-line tool,
    handling process execution, error management, and response parsing.
    """
    
    def __init__(self, cli_path: str = "gemini", timeout: int = 30):
        """
        Initialize the Gemini CLI wrapper.
        
        Args:
            cli_path: Path to the Gemini CLI executable
            timeout: Default timeout for CLI commands in seconds
        """
        self.cli_path = cli_path
        self.timeout = timeout
        self.api_key = os.getenv('GEMINI_API_KEY')
        
    def check_availability(self) -> bool:
        """
        Check if Gemini CLI is available and properly configured.
        
        Returns:
            True if CLI is available, False otherwise
        """
        try:
            result = subprocess.run(
                [self.cli_path, '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            print(result)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return False
    
    def generate_text(self, prompt: str, model: str = "gemini-pro", **kwargs) -> Dict[str, Any]:
        """
        Generate text using Gemini CLI.
        
        Args:
            prompt: Text prompt for generation
            model: Gemini model to use
            **kwargs: Additional CLI parameters
            
        Returns:
            Dictionary containing the response and metadata
            
        Raises:
            GeminiCLIError: If the CLI command fails
        """
        cmd = [
            self.cli_path,
            '--model', model,
            '--prompt', prompt
        ]       
        return self._execute_command(cmd)
    
    def generate_from_file(self, file_path: str, model: str = "gemini-2.5-pro") -> Dict[str, Any]:
        """
        Generate text from a file prompt.
        
        Args:
            file_path: Path to file containing the prompt
            model: Gemini model to use
            
        Returns:
            Dictionary containing the response and metadata
        """
        if not os.path.exists(file_path):
            raise GeminiCLIError(f"Prompt file not found: {file_path}")
            
        cmd = [
            self.cli_path,
            '--model', model,
            '--file', file_path
        ]
        
        if self.api_key:
            cmd.extend(['--api-key', self.api_key])
            
        return self._execute_command(cmd)
    
    def chat_session(self, messages: List[Dict[str, str]], model: str = "gemini-pro") -> Dict[str, Any]:
        """
        Start a chat session with multiple messages.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Gemini model to use
            
        Returns:
            Dictionary containing the response and metadata
        """
        # Create temporary file with chat history
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({'messages': messages}, f)
            temp_file = f.name
        
        try:
            cmd = [
                self.cli_path,
                'chat',
                '--model', model,
                '--history', temp_file
            ]
            
            if self.api_key:
                cmd.extend(['--api-key', self.api_key])
                
            return self._execute_command(cmd)
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file):
                os.unlink(temp_file)
    
    def analyze_image(self, image_path: str, prompt: str = "Describe this image", 
                     model: str = "gemini-pro-vision") -> Dict[str, Any]:
        """
        Analyze an image using Gemini's vision capabilities.
        
        Args:
            image_path: Path to the image file
            prompt: Text prompt for image analysis
            model: Vision model to use
            
        Returns:
            Dictionary containing the response and metadata
        """
        if not os.path.exists(image_path):
            raise GeminiCLIError(f"Image file not found: {image_path}")
            
        cmd = [
            self.cli_path,
            'analyze',
            '--model', model,
            '--image', image_path,
            '--prompt', prompt
        ]
        
        if self.api_key:
            cmd.extend(['--api-key', self.api_key])
            
        return self._execute_command(cmd)
    
    def _execute_command(self, cmd: List[str]) -> Dict[str, Any]:
        """
        Execute a Gemini CLI command and parse the response.
        
        Args:
            cmd: Command list for subprocess
            
        Returns:
            Dictionary containing parsed response
            
        Raises:
            GeminiCLIError: If command execution fails
        """
        try:
            logger.info(f"Executing Gemini CLI command: {' '.join(cmd[:3])}...")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode != 0:
                error_msg = result.stderr or result.stdout or "Unknown error"
                raise GeminiCLIError(f"CLI command failed: {error_msg}")
            
            # Try to parse JSON response, fallback to plain text
            try:
                response = json.loads(result.stdout)
            except json.JSONDecodeError:
                response = {
                    'text': result.stdout.strip(),
                    'raw_output': True
                }
            
            return {
                'success': True,
                'response': response,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'return_code': result.returncode
            }
            
        except subprocess.TimeoutExpired:
            raise GeminiCLIError(f"Command timed out after {self.timeout} seconds")
        except FileNotFoundError:
            raise GeminiCLIError(f"Gemini CLI not found at: {self.cli_path}")
        except Exception as e:
            raise GeminiCLIError(f"Unexpected error: {str(e)}")







def test_gemini_cli_error_custom_exception():
    """Test custom exception class."""
    error = GeminiCLIError("Test error message")
    assert str(error) == "Test error message"
    assert isinstance(error, Exception)


# Example usage and demonstration
def example_usage():
    """
    Example of how to use the GeminiCLI wrapper in practice.
    This function demonstrates various use cases.
    """
    print("=== Gemini CLI Integration Examples ===\n")
    
    # Initialize the CLI wrapper
    gemini = GeminiCLI(timeout=60)
    
    # Check if CLI is available
    if not gemini.check_availability():
        print("❌ Gemini CLI is not available")
        print("   Please install the Gemini CLI and set GEMINI_API_KEY environment variable")
        return
    
    print("✅ Gemini CLI is available\n")
    
    try:
        # Example 1: Simple text generation
        print("1. Simple Text Generation:")
        result = gemini.generate_text(
            "Write a haiku about programming"
        )
        if result['success']:
            print(f"   Response: {result['response'].get('text', 'No text in response')}")
        else:
            return
        # Example 2: Chat session
        print("2. Chat Session:")
        messages = [
            {"role": "user", "content": "What is Apache Flink?"},
            {"role": "assistant", "content": "Apache Flink is a stream processing framework."},
            {"role": "user", "content": "What are its main benefits?"}
        ]
        result = gemini.chat_session(messages)
        if result['success']:
            print(f"   Chat Response: {result['response'].get('text', 'No text in response')}")
        print()
        
        # Example 3: Generation with specific parameters
        print("3. Parametrized Generation:")
        result = gemini.generate_text(
            "Explain microservices in exactly 50 words",
            output_format="json"
        )
        if result['success']:
            print(f"   Structured Response: {result['response']}")
        
    except GeminiCLIError as e:
        print(f"❌ Error during example execution: {e}")
    
    print("\n=== Examples completed ===")



if __name__ == "__main__":
    example_usage()