import pytest
from unittest.mock import patch, Mock, MagicMock
import json
import os
from gemini_cli_integration import GeminiCLI, GeminiCLIError

def example_usage():
    """Example usage of GeminiCLI."""
    gemini = GeminiCLI()
    print(gemini.generate_text("Hello, how are you?"))

class TestGeminiCLIIntegration:
    """Test cases for Gemini CLI integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.gemini_cli = GeminiCLI()
    
    def test_gemini_cli_initialization(self):
        """Test GeminiCLI class initialization."""
        cli = GeminiCLI(cli_path="/usr/local/bin/gemini", timeout=60)
        assert cli.cli_path == "/usr/local/bin/gemini"
        assert cli.timeout == 60
        assert cli.api_key == os.getenv('GEMINI_API_KEY')
    
    @patch('subprocess.run')
    def test_check_availability_success(self, mock_run):
        """Test successful CLI availability check."""
        mock_run.return_value = Mock(returncode=0)
        
        result = self.gemini_cli.check_availability()
        
        assert result is True
        mock_run.assert_called_once_with(
            ['gemini', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
    
    @patch('subprocess.run')
    def test_check_availability_failure(self, mock_run):
        """Test CLI availability check when CLI is not available."""
        mock_run.side_effect = FileNotFoundError("Command not found")
        
        result = self.gemini_cli.check_availability()
        
        assert result is False
    
    @patch('subprocess.run')
    def test_generate_text_success(self, mock_run):
        """Test successful text generation."""
        mock_response = {
            'text': 'This is a generated response from Gemini.',
            'model': 'gemini-pro',
            'usage': {'input_tokens': 10, 'output_tokens': 8}
        }
        
        mock_run.return_value = Mock(
            returncode=0,
            stdout=json.dumps(mock_response),
            stderr=""
        )
        
        result = self.gemini_cli.generate_text("Hello, how are you?")
        
        assert result['success'] is True
        assert result['response'] == mock_response
        assert 'This is a generated response' in result['response']['text']
        
        # Verify the command was constructed correctly
        expected_cmd = [
            'gemini', 'generate', '--model', 'gemini-pro',
            '--prompt', 'Hello, how are you?'
        ]
        mock_run.assert_called_once()
        actual_cmd = mock_run.call_args[0][0]
        assert actual_cmd[:6] == expected_cmd
    
    @patch('subprocess.run')
    def test_generate_text_with_parameters(self, mock_run):
        """Test text generation with additional parameters."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout='{"text": "Generated with parameters"}',
            stderr=""
        )
        
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'test_key'}):
            cli = GeminiCLI()
            result = cli.generate_text(
                "Test prompt",
                temperature=0.7,
                max_tokens=100,
                output_format="json"
            )
        
        assert result['success'] is True
        
        # Check that parameters were included in the command
        mock_run.assert_called_once()
        cmd_args = mock_run.call_args[0][0]
        assert '--temperature' in cmd_args
        assert '0.7' in cmd_args
        assert '--max-tokens' in cmd_args
        assert '100' in cmd_args
        assert '--format' in cmd_args
        assert 'json' in cmd_args
        assert '--api-key' in cmd_args
        assert 'test_key' in cmd_args
    
    @patch('subprocess.run')
    def test_generate_text_cli_error(self, mock_run):
        """Test handling of CLI errors."""
        mock_run.return_value = Mock(
            returncode=1,
            stdout="",
            stderr="API key not found"
        )
        
        with pytest.raises(GeminiCLIError, match="CLI command failed: API key not found"):
            self.gemini_cli.generate_text("Test prompt")
    
    @patch('subprocess.run')
    def test_generate_text_timeout(self, mock_run):
        """Test handling of command timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired(['gemini'], 30)
        
        with pytest.raises(GeminiCLIError, match="Command timed out after 30 seconds"):
            self.gemini_cli.generate_text("Test prompt")
    
    @patch('subprocess.run')
    def test_generate_text_file_not_found(self, mock_run):
        """Test handling when CLI executable is not found."""
        mock_run.side_effect = FileNotFoundError("gemini: command not found")
        
        with pytest.raises(GeminiCLIError, match="Gemini CLI not found at: gemini"):
            self.gemini_cli.generate_text("Test prompt")
    
    @patch('os.path.exists')
    @patch('subprocess.run')
    def test_generate_from_file_success(self, mock_run, mock_exists):
        """Test generating text from a file."""
        mock_exists.return_value = True
        mock_run.return_value = Mock(
            returncode=0,
            stdout='{"text": "Response from file prompt"}',
            stderr=""
        )
        
        result = self.gemini_cli.generate_from_file("/path/to/prompt.txt")
        
        assert result['success'] is True
        assert result['response']['text'] == "Response from file prompt"
        
        # Verify file parameter was used
        mock_run.assert_called_once()
        cmd_args = mock_run.call_args[0][0]
        assert '--file' in cmd_args
        assert '/path/to/prompt.txt' in cmd_args
    
    @patch('os.path.exists')
    def test_generate_from_file_not_found(self, mock_exists):
        """Test error when prompt file doesn't exist."""
        mock_exists.return_value = False
        
        with pytest.raises(GeminiCLIError, match="Prompt file not found: /nonexistent/file.txt"):
            self.gemini_cli.generate_from_file("/nonexistent/file.txt")
    
    @patch('os.path.exists')
    @patch('tempfile.NamedTemporaryFile')
    @patch('subprocess.run')
    @patch('os.unlink')
    def test_chat_session_success(self, mock_unlink, mock_run, mock_temp_file, mock_exists):
        """Test chat session functionality."""
        # Mock path exists for cleanup
        mock_exists.return_value = True
        
        # Mock temporary file
        mock_file = MagicMock()
        mock_file.name = '/tmp/chat_history.json'
        mock_temp_file.return_value.__enter__.return_value = mock_file
        
        # Mock subprocess result
        mock_run.return_value = Mock(
            returncode=0,
            stdout='{"text": "Chat response", "conversation_id": "abc123"}',
            stderr=""
        )
        
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        result = self.gemini_cli.chat_session(messages)
        
        assert result['success'] is True
        assert result['response']['text'] == "Chat response"
        
        # Verify chat command was used
        mock_run.assert_called_once()
        cmd_args = mock_run.call_args[0][0]
        assert 'chat' in cmd_args
        assert '--history' in cmd_args
        
        # Verify temporary file was cleaned up
        mock_unlink.assert_called_once_with('/tmp/chat_history.json')
    
    @patch('os.path.exists')
    @patch('subprocess.run')
    def test_analyze_image_success(self, mock_run, mock_exists):
        """Test image analysis functionality."""
        mock_exists.return_value = True
        mock_run.return_value = Mock(
            returncode=0,
            stdout='{"text": "This image contains a cat sitting on a table."}',
            stderr=""
        )
        
        result = self.gemini_cli.analyze_image("/path/to/image.jpg", "What do you see?")
        
        assert result['success'] is True
        assert "cat sitting on a table" in result['response']['text']
        
        # Verify image analysis command
        mock_run.assert_called_once()
        cmd_args = mock_run.call_args[0][0]
        assert 'analyze' in cmd_args
        assert '--image' in cmd_args
        assert '/path/to/image.jpg' in cmd_args
        assert '--prompt' in cmd_args
        assert 'What do you see?' in cmd_args
    
    @patch('os.path.exists')
    def test_analyze_image_file_not_found(self, mock_exists):
        """Test error when image file doesn't exist."""
        mock_exists.return_value = False
        
        with pytest.raises(GeminiCLIError, match="Image file not found: /nonexistent/image.jpg"):
            self.gemini_cli.analyze_image("/nonexistent/image.jpg")
    
    @patch('subprocess.run')
    def test_non_json_response_handling(self, mock_run):
        """Test handling of non-JSON responses from CLI."""
        mock_run.return_value = Mock(
            returncode=0,
            stdout="Plain text response without JSON formatting",
            stderr=""
        )
        
        result = self.gemini_cli.generate_text("Simple prompt")
        
        assert result['success'] is True
        assert result['response']['text'] == "Plain text response without JSON formatting"
        assert result['response']['raw_output'] is True


class TestGeminiCLIIntegrationE2E:
    """End-to-end integration tests (require actual Gemini CLI)."""
    
    def setup_method(self):
        """Set up for integration tests."""
        self.gemini_cli = GeminiCLI()
        
    @pytest.mark.integration
    @pytest.mark.skipif(not os.getenv('GEMINI_API_KEY'), reason="No API key provided")
    def test_real_gemini_cli_availability(self):
        """Test against real Gemini CLI (requires API key)."""
        # This test will only run if GEMINI_API_KEY environment variable is set
        # and if the actual Gemini CLI is installed
        
        if not self.gemini_cli.check_availability():
            pytest.skip("Gemini CLI not available")
            
        # Simple test to verify CLI works
        try:
            result = self.gemini_cli.generate_text(
                "Say hello in one word",
                temperature=0.1,
                max_tokens=5
            )
            assert result['success'] is True
            assert 'response' in result
            logger.info(f"Real CLI response: {result['response']}")
        except GeminiCLIError as e:
            pytest.fail(f"Real CLI test failed: {e}")
    
    @pytest.mark.integration
    def test_cli_not_installed_graceful_failure(self):
        """Test graceful failure when CLI is not installed."""
        # Test with non-existent CLI path
        cli = GeminiCLI(cli_path="/nonexistent/gemini/path")
        
        assert cli.check_availability() is False
        
        with pytest.raises(GeminiCLIError, match="Gemini CLI not found"):
            cli.generate_text("Test prompt")

if __name__ == "__main__":
    # Run examples if this file is executed directly
    example_usage()
    
    # Run tests
    print("\n" + "="*50)
    print("Running basic tests...")
    pytest.main([__file__, "-v", "--tb=short"])