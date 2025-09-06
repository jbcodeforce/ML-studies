# Gemini CLI Integration - Proof of Concept

This directory contains a comprehensive proof of concept for integrating Google's Gemini CLI with Python applications. The implementation demonstrates how to call the Gemini CLI from within Python processes using subprocess management, proper error handling, and comprehensive testing.

## Files Overview

### `test_gemini_cli_integration.py`
The main proof of concept file containing:
- **`GeminiCLI` class**: A wrapper for interacting with the Gemini CLI
- **Test suite**: Unit tests with mocking and integration tests
- **Error handling**: Custom exceptions and graceful error management
- **Multiple interaction patterns**: Text generation, chat sessions, file-based prompts, image analysis

### `gemini_cli_example.py`
Practical examples demonstrating real-world usage scenarios:
- Code review automation
- Documentation generation
- Unit test creation
- Troubleshooting assistance
- Interactive chat sessions
- Performance testing

## 🚀 Quick Start

### Prerequisites

1. **Install Google's Gemini CLI** (if available):
   ```bash
   # Installation method depends on Google's official distribution
   # Check Google's documentation for latest installation instructions
   ```

2. **Set up API credentials**:
   ```bash
   export GEMINI_API_KEY="your-api-key-here"
   ```

3. **Install Python dependencies**:
   ```bash
   pip install pytest
   ```

### Running the Examples

```bash
# Run the interactive examples
python gemini_cli_example.py

# Or make it executable and run directly
chmod +x tests/gemini_cli_example.py
./tests/gemini_cli_example.py
```

### Running Tests

```bash
# Run all tests (uses mocking, doesn't require actual CLI)
python -m pytest tests/test_gemini_cli_integration.py -v

# Run only unit tests (with mocking)
python -m pytest tests/test_gemini_cli_integration.py::TestGeminiCLIIntegration -v

# Run integration tests (requires actual CLI and API key)
GEMINI_API_KEY=your-key python -m pytest tests/test_gemini_cli_integration.py::TestGeminiCLIIntegrationE2E -v -m integration
```

## 🔧 Core Features

### 1. CLI Wrapper Class

The `GeminiCLI` class provides a Python interface to the Gemini command-line tool:

```python
from tests.test_gemini_cli_integration import GeminiCLI, GeminiCLIError

# Initialize with custom settings
gemini = GeminiCLI(cli_path="gemini", timeout=30)

# Check if CLI is available
if gemini.check_availability():
    # Generate text
    result = gemini.generate_text(
        "Explain quantum computing in simple terms",
        temperature=0.7,
        max_tokens=200
    )
    print(result['response']['text'])
```

### 2. Multiple Interaction Modes

**Text Generation:**
```python
result = gemini.generate_text(
    prompt="Write a Python function to calculate factorial",
    model="gemini-pro",
    temperature=0.3,
    max_tokens=150
)
```

**File-based Prompts:**
```python
result = gemini.generate_from_file("/path/to/prompt.txt")
```

**Chat Sessions:**
```python
messages = [
    {"role": "user", "content": "What is Apache Flink?"},
    {"role": "assistant", "content": "Apache Flink is a stream processing framework."},
    {"role": "user", "content": "How does it handle backpressure?"}
]
result = gemini.chat_session(messages)
```

**Image Analysis:**
```python
result = gemini.analyze_image(
    "/path/to/image.jpg", 
    "Describe what you see in this image",
    model="gemini-pro-vision"
)
```

### 3. Robust Error Handling

```python
try:
    result = gemini.generate_text("Hello, world!")
    if result['success']:
        print(result['response']['text'])
    else:
        print("Generation failed")
except GeminiCLIError as e:
    print(f"CLI Error: {e}")
```

### 4. Comprehensive Testing

The test suite includes:
- **Unit tests** with subprocess mocking
- **Integration tests** for real CLI interaction
- **Error condition testing**
- **Timeout handling**
- **Parameter validation**
- **Response parsing**

## 🎯 Use Cases Demonstrated

### 1. Code Review Automation
```python
def review_code_with_gemini(code_snippet):
    prompt = f"Review this code and suggest improvements:\n```python\n{code_snippet}\n```"
    result = gemini.generate_text(prompt, temperature=0.3)
    return result['response']['text']
```

### 2. Documentation Generation
```python
def generate_docstring(function_code):
    prompt = f"Generate a comprehensive docstring for:\n```python\n{function_code}\n```"
    result = gemini.generate_text(prompt, temperature=0.2)
    return result['response']['text']
```

### 3. Test Generation
```python
def generate_unit_tests(function_code):
    prompt = f"Generate pytest unit tests for:\n```python\n{function_code}\n```"
    result = gemini.generate_text(prompt, temperature=0.3, max_tokens=600)
    return result['response']['text']
```

### 4. Troubleshooting Assistant
```python
def analyze_error_logs(error_log):
    prompt = f"Analyze this error and provide troubleshooting steps:\n{error_log}"
    result = gemini.generate_text(prompt, temperature=0.4, max_tokens=500)
    return result['response']['text']
```

## ⚙️ Configuration Options

### Environment Variables
- `GEMINI_API_KEY`: Your Gemini API key (required for real CLI usage)

### CLI Parameters
- `cli_path`: Path to the Gemini CLI executable (default: "gemini")
- `timeout`: Command timeout in seconds (default: 30)
- `temperature`: Response randomness (0.0-1.0)
- `max_tokens`: Maximum response length
- `model`: Gemini model to use ("gemini-pro", "gemini-pro-vision", etc.)

## 🧪 Testing Strategy

### 1. Unit Tests (No CLI Required)
- Uses `unittest.mock` to simulate CLI responses
- Tests all code paths and error conditions
- Fast execution for development workflow

### 2. Integration Tests (Requires CLI)
- Tests against real Gemini CLI when available
- Marked with `@pytest.mark.integration`
- Requires API key and CLI installation

### 3. Performance Tests
- Measures response times and success rates
- Tests concurrent requests
- Provides performance metrics

## 🔒 Security Considerations

1. **API Key Management**: Store API keys securely, never commit to version control
2. **Input Validation**: The wrapper validates inputs before sending to CLI
3. **Error Information**: Sensitive information is not logged in error messages
4. **Timeout Protection**: All CLI calls have configurable timeouts
5. **Temporary Files**: Chat history files are securely cleaned up

## 🚦 Error Handling

The implementation provides multiple layers of error handling:

```python
# Custom exception hierarchy
GeminiCLIError
├── Command execution failures
├── Timeout errors  
├── CLI not found errors
└── Response parsing errors

# Error response structure
{
    'success': False,
    'error': 'Error description',
    'stdout': 'CLI stdout',
    'stderr': 'CLI stderr',
    'return_code': 1
}
```

## 📊 Performance Considerations

- **Subprocess Overhead**: Each CLI call creates a new subprocess
- **Response Parsing**: JSON responses are parsed automatically
- **Timeout Management**: Configurable timeouts prevent hanging
- **Memory Management**: Large responses are handled efficiently
- **Concurrent Requests**: Multiple requests can be made in parallel

## 🔄 Extension Points

The proof of concept is designed for easy extension:

1. **Additional Models**: Easy to add support for new Gemini models
2. **Custom Parameters**: Simple to add new CLI parameters
3. **Response Formats**: Flexible response parsing and formatting
4. **Async Support**: Can be extended with asyncio for concurrent operations
5. **Caching**: Response caching can be added for repeated queries

## 📝 Example Output

When running the examples, you'll see output like:

```
🚀 Basic Gemini CLI Usage Demo
========================================
Checking Gemini CLI availability...
✅ Gemini CLI is available!

📝 Example: Code Review with Gemini
----------------------------------------
🤖 Gemini's Code Review:
   The recursive Fibonacci implementation has exponential time complexity O(2^n).
   Consider using dynamic programming or memoization for better performance...

📚 Example: Documentation Generation
----------------------------------------
📋 Generated Documentation:
   def process_flink_job_metrics(metrics_data: List[Dict], threshold: float = 0.8) -> List[Dict]:
       """
       Process Flink job metrics and categorize them based on utilization threshold.
       
       Args:
           metrics_data: List of metric dictionaries containing utilization data
           threshold: Utilization threshold for high load classification (default: 0.8)
           
       Returns:
           List of processed metrics with added status field
       """
```

## 🤝 Contributing

This proof of concept demonstrates patterns that can be applied to other CLI integrations:

1. **Subprocess Management**: Proper process handling with timeouts
2. **Error Handling**: Comprehensive error management
3. **Testing Strategy**: Unit tests with mocking + integration tests  
4. **Response Parsing**: Flexible JSON and text response handling
5. **Configuration Management**: Environment-based configuration

## 📚 References

- [Google Gemini API Documentation](https://ai.google.dev/docs)
- [Python subprocess Documentation](https://docs.python.org/3/library/subprocess.html)
- [pytest Documentation](https://docs.pytest.org/)
- [unittest.mock Documentation](https://docs.python.org/3/library/unittest.mock.html)

---

*This proof of concept demonstrates production-ready patterns for CLI integration with comprehensive error handling, testing, and real-world usage examples.*
