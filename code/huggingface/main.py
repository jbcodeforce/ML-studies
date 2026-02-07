from huggingface_hub import InferenceClient
import json

# 1. Initialize the client pointing to your LM Studio local server
# Note: Use 'base_url' to point to the local instance
client = InferenceClient(
    base_url="http://localhost:1234/v1", 
    api_key="lm-studio"  # Dummy key required by the SDK for OpenAI compatibility
)

# 2. Define your tool (function)
def get_weather(city: str):
    # This is a mock function; in a real app, you'd call a weather API
    return f"The weather in {city} is 22°C and sunny."

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a specific city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The name of the city"}
                },
                "required": ["city"],
            },
        },
    }
]

# 3. Initial Chat Request
messages = [{"role": "user", "content": "What's the weather like in Paris?"}]

response = client.chat.completions.create(
    model="local-model", # LM Studio handles model selection; this can often be anything
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

# 4. Handle Tool Calls
message = response.choices[0].message

if message.tool_calls:
    for tool_call in message.tool_calls:
        # Extract function details
        function_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        
        if function_name == "get_weather":
            # Execute the actual Python function
            result = get_weather(arguments.get("city"))
            
            # Append the model's request and the tool's result to history
            messages.append(message) 
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": result
            })
            
    # 5. Get the final response from the model
    final_response = client.chat.completions.create(
        model="local-model",
        messages=messages
    )
    print(final_response.choices[0].message.content)


