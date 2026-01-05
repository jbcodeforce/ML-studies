#!/usr/bin/env python3
"""
Simple example script demonstrating Gemini CLI integration.

This script shows practical examples of how to use the GeminiCLI wrapper
for various AI tasks in a real-world scenario.
"""

import os
import sys
import logging
from typing import List, Dict

# Add the parent directory to Python path to import our module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gemini_cli_integration import GeminiCLI, GeminiCLIError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demonstrate_basic_usage():
    """Demonstrate basic Gemini CLI usage."""
    print("🚀 Basic Gemini CLI Usage Demo")
    print("=" * 40)
    
    # Initialize the CLI wrapper
    gemini = GeminiCLI(timeout=30)
    
    # Check availability first
    print("Checking Gemini CLI availability...")
    if not gemini.check_availability():
        print("❌ Gemini CLI is not available on this system")
        print("   To use this demo, please:")
        print("   1. Install Google's Gemini CLI tool")
        print("   2. Set the GEMINI_API_KEY environment variable")
        print("   3. Ensure the CLI is in your system PATH")
        return False
    
    print("✅ Gemini CLI is available!")
    return True


def example_code_review():
    """Example: Use Gemini to review code."""
    print("\n📝 Example: Code Review with Gemini")
    print("-" * 40)
    
    gemini = GeminiCLI()
    
    # Sample code to review
    sample_code = """
def calculate_fibonacci(n):
    if n <= 1:
        return n
    else:
        return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

# Usage
result = calculate_fibonacci(10)
print(f"Fibonacci(10) = {result}")
"""
    
    prompt = f"""
Please review this Python code and provide suggestions for improvement:

```python
{sample_code}
```

Focus on:
1. Performance optimizations
2. Code readability
3. Best practices
4. Potential issues

Provide specific, actionable feedback.
"""
    
    try:
        result = gemini.generate_text(
            prompt,
            temperature=0.3,  # Lower temperature for more focused analysis
            max_tokens=500
        )
        
        if result['success']:
            print("🤖 Gemini's Code Review:")
            response_text = result['response'].get('text', 'No response text')
            print(f"   {response_text}")
        else:
            print("❌ Failed to get code review")
            
    except GeminiCLIError as e:
        print(f"❌ Error during code review: {e}")


def example_documentation_generation():
    """Example: Generate documentation for a function."""
    print("\n📚 Example: Documentation Generation")
    print("-" * 40)
    
    gemini = GeminiCLI()
    
    function_code = """
def process_flink_job_metrics(metrics_data, threshold=0.8):
    filtered_metrics = []
    for metric in metrics_data:
        if metric['utilization'] > threshold:
            metric['status'] = 'high_load'
        else:
            metric['status'] = 'normal'
        filtered_metrics.append(metric)
    return filtered_metrics
"""
    
    prompt = f"""
Generate comprehensive documentation for this Python function:

```python
{function_code}
```

Include:
- Function description
- Parameter documentation
- Return value documentation
- Usage example
- Type hints suggestion

Format as proper Python docstring.
"""
    
    try:
        result = gemini.generate_text(
            prompt,
            temperature=0.2,
            max_tokens=400
        )
        
        if result['success']:
            print("📋 Generated Documentation:")
            response_text = result['response'].get('text', 'No response text')
            print(f"   {response_text}")
            
    except GeminiCLIError as e:
        print(f"❌ Error generating documentation: {e}")


def example_test_generation():
    """Example: Generate unit tests for a function."""
    print("\n🧪 Example: Unit Test Generation")
    print("-" * 40)
    
    gemini = GeminiCLI()
    
    function_to_test = """
def validate_flink_config(config):
    required_fields = ['job_name', 'parallelism', 'memory_mb']
    
    if not isinstance(config, dict):
        raise ValueError("Config must be a dictionary")
    
    for field in required_fields:
        if field not in config:
            raise KeyError(f"Missing required field: {field}")
    
    if config['parallelism'] <= 0:
        raise ValueError("Parallelism must be positive")
    
    if config['memory_mb'] < 512:
        raise ValueError("Memory must be at least 512 MB")
    
    return True
"""
    
    prompt = f"""
Generate comprehensive pytest unit tests for this function:

```python
{function_to_test}
```

Include tests for:
- Valid input scenarios
- Invalid input types
- Missing required fields
- Edge cases and boundary conditions
- Error conditions with appropriate exception testing

Use pytest fixtures if appropriate.
"""
    
    try:
        result = gemini.generate_text(
            prompt,
            temperature=0.3,
            max_tokens=600
        )
        
        if result['success']:
            print("🔬 Generated Unit Tests:")
            response_text = result['response'].get('text', 'No response text')
            print(f"   {response_text}")
            
    except GeminiCLIError as e:
        print(f"❌ Error generating tests: {e}")


def example_troubleshooting_assistant():
    """Example: Use Gemini as a troubleshooting assistant."""
    print("\n🔍 Example: Troubleshooting Assistant")
    print("-" * 40)
    
    gemini = GeminiCLI()
    
    error_scenario = """
Error Log:
2024-01-15 10:30:45 ERROR JobManager - Job failed with exception
java.lang.OutOfMemoryError: GC overhead limit exceeded
    at org.apache.flink.streaming.runtime.tasks.StreamTask.invoke(StreamTask.java:123)
    at org.apache.flink.runtime.taskmanager.Task.run(Task.java:456)

Job Configuration:
- Parallelism: 8
- Memory per TaskManager: 2GB
- Processing 50,000 records/second
- State size: ~10GB
- Checkpointing interval: 60 seconds
"""
    
    prompt = f"""
Analyze this Apache Flink error and provide troubleshooting guidance:

{error_scenario}

Please provide:
1. Root cause analysis
2. Immediate steps to resolve the issue
3. Long-term optimization suggestions
4. Configuration recommendations
5. Monitoring recommendations

Be specific and actionable.
"""
    
    try:
        result = gemini.generate_text(
            prompt,
            temperature=0.4,
            max_tokens=700
        )
        
        if result['success']:
            print("🩺 Troubleshooting Analysis:")
            response_text = result['response'].get('text', 'No response text')
            print(f"   {response_text}")
            
    except GeminiCLIError as e:
        print(f"❌ Error in troubleshooting analysis: {e}")


def example_interactive_chat():
    """Example: Interactive chat session with context."""
    print("\n💬 Example: Interactive Chat Session")
    print("-" * 40)
    
    gemini = GeminiCLI()
    
    # Build conversation context
    chat_messages = [
        {
            "role": "user", 
            "content": "I'm working on a Flink streaming job that processes financial transactions. What are the key considerations for ensuring exactly-once processing?"
        },
        {
            "role": "assistant", 
            "content": "For exactly-once processing in Flink financial applications, key considerations include: 1) Enable checkpointing with appropriate intervals, 2) Use transactional sinks, 3) Configure proper restart strategies, 4) Ensure idempotent operations, 5) Handle late-arriving data appropriately."
        },
        {
            "role": "user", 
            "content": "What checkpoint interval would you recommend for a job processing 100,000 transactions per second with low-latency requirements?"
        }
    ]
    
    try:
        result = gemini.chat_session(chat_messages, model="gemini-pro")
        
        if result['success']:
            print("💡 Chat Response:")
            response_text = result['response'].get('text', 'No response text')
            print(f"   {response_text}")
            
            # Show conversation context
            print("\n📝 Conversation Context:")
            for i, msg in enumerate(chat_messages):
                role_emoji = "👤" if msg['role'] == 'user' else "🤖"
                print(f"   {role_emoji} {msg['role'].title()}: {msg['content'][:100]}...")
            
    except GeminiCLIError as e:
        print(f"❌ Error in chat session: {e}")


def run_performance_test():
    """Run a simple performance test of the CLI wrapper."""
    print("\n⚡ Performance Test")
    print("-" * 40)
    
    import time
    
    gemini = GeminiCLI(timeout=10)
    
    if not gemini.check_availability():
        print("❌ Skipping performance test - CLI not available")
        return
    
    # Test multiple quick requests
    test_prompts = [
        "Say hello",
        "Count to 3",
        "Name a color",
        "What is 2+2?",
        "Complete: The sky is..."
    ]
    
    print(f"Testing {len(test_prompts)} quick requests...")
    
    start_time = time.time()
    successful_requests = 0
    
    for i, prompt in enumerate(test_prompts, 1):
        try:
            print(f"   Request {i}/{len(test_prompts)}: '{prompt}'")
            result = gemini.generate_text(
                prompt, 
                temperature=0.1, 
                max_tokens=10
            )
            if result['success']:
                successful_requests += 1
                response_text = result['response'].get('text', 'No response')
                print(f"   ✅ Response: {response_text[:50]}...")
            else:
                print(f"   ❌ Failed")
                
        except GeminiCLIError as e:
            print(f"   ❌ Error: {e}")
        except KeyboardInterrupt:
            print(f"   ⏹️  Test interrupted by user")
            break
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"\n📊 Performance Results:")
    print(f"   Total time: {elapsed:.2f} seconds")
    print(f"   Successful requests: {successful_requests}/{len(test_prompts)}")
    print(f"   Average time per request: {elapsed/len(test_prompts):.2f} seconds")
    print(f"   Success rate: {(successful_requests/len(test_prompts))*100:.1f}%")


def main():
    """Main function to run all examples."""
    print("🎯 Gemini CLI Integration Examples")
    print("=" * 50)
    
    # Check basic availability
    if not demonstrate_basic_usage():
        print("\n⚠️  Examples will be skipped due to CLI unavailability")
        print("   However, you can still run the unit tests with mocking!")
        return
    
    try:
        # Run different examples
        example_code_review()
        example_documentation_generation()
        example_test_generation()
        example_troubleshooting_assistant()
        example_interactive_chat()
        
        # Performance test (optional)
        print("\n" + "="*50)
        user_input = input("Would you like to run a performance test? (y/N): ")
        if user_input.lower() in ['y', 'yes']:
            run_performance_test()
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error during demo: {e}")
        logger.exception("Demo error")
    
    print("\n✨ Demo completed!")
    print("\nTo run the unit tests (with mocking):")
    print("   cd /path/to/flink-estimator/src")
    print("   python -m pytest tests/test_gemini_cli_integration.py -v")


if __name__ == "__main__":
    main()
