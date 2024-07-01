
from taipy.gui import Gui, State, notify
import sys,os
module_path = "."
sys.path.append(os.path.abspath(module_path))
from prompt_builder_graph import send_user_msg

context = "Hi"
conversation = {
    "Conversation": []
}
current_user_message = ""



def invoke_model(state: State, user: str) -> str:
    """
    Send a message to graph for processing

    Args:
        - state: The current conversation state.
        - prompt: The prompt to send to the API.

    Returns:
        The response from the API.
    """
    for output in send_user_msg(user):
        for key, value in output.items():
                    print(f"Output from node '{key}':")
                    print("---")
                    print(value.content)
                    return value.content

def send_message(state: State) -> None:
    """
    Send the user's message to the API and update the conversation.

    Args:
        - state: The current state.
    """
    global current_user_message
    # Add the user's message to the context
    state.context += state.current_user_message
    # Send the user's message to the API and get the response
    answer = invoke_model(state, state.context)
    # Add the response to the context for future messages
    state.context += answer
    # Update the conversation
    conv = state.conversation._dict.copy()
    conv["Conversation"] += [state.current_user_message, answer]
    state.conversation = conv
    # Clear the input field
    state.current_user_message = ""
    current_user_message=""


def style_conv(state: State, idx: int, row: int) -> str:
    """
    Apply a style to the conversation table depending on the message's author.

    Args:
        - state: The current state of the app.
        - idx: The index of the message in the table.
        - row: The row of the message in the table.

    Returns:
        The style to apply to the message.
    """
    if idx is None:
        return None
    elif idx % 2 == 0:
        return "user_message"
    else:
        return "bot_message"

page = """
<|{conversation}|table|show_all|style=style_conv|>
<|{current_user_message}|input|label=Write your message here...|on_action=send_message|class_name=fullwidth|>
"""



if __name__ == "__main__":
    Gui(page, css_file="./main.css").run(title="Prompt Builder Chat")