from nicegui import ui
import requests
from dto_models import ConversationControl, ResponseControl
from app_config import load_config
config=load_config()

@ui.page('/')
def main():
    async def send() -> None:
        query = text.value
        text.value = ''
        convControl = ConversationControl.model_validate({
                    "callWithVectorStore": True, 
                    "callWithDecisionService": True,  
                    "type": "chat", 
                    "query": query,
                    "chat_history": [] })
        with message_container:
            ui.chat_message(text=query, name='You', sent=True)
            response_message = ui.chat_message(name='Bot', sent=False)
            spinner = ui.spinner(type='dots')

        response = ''
        for chunk in requests.post(url=config["athena"]["backend"]["chat_url"],data=convControl.model_dump_json()):
            response += chunk
            response_message.clear()
            with response_message:
                ui.html(response)
            ui.run_javascript('window.scrollTo(0, document.body.scrollHeight)')
        message_container.remove(spinner)

    ui.add_css(r'a:link, a:visited {color: inherit !important; text-decoration: none; font-weight: 500}')

    # the queries below are used to expand the contend down to the footer (content can then use flex-grow to expand)
    ui.query('.q-page').classes('flex')
    ui.query('.nicegui-content').classes('w-full')
    message_container = ui.row().classes('items-stretch')
    with ui.footer().classes('bg-white'), ui.column().classes('w-full max-w-3xl mx-auto my-6'):
        with ui.row().classes('w-full no-wrap items-center'):
            placeholder = "message"
            text = ui.input(placeholder=placeholder).props('rounded outlined input-class=mx-3') \
                .classes('w-full self-center').on('keydown.enter', send)
        ui.markdown('simple chat app built with [NiceGUI](https://nicegui.io)') \
            .classes('text-xs self-end mr-8 m-[-1em] text-primary')

ui.run(title='Chat with a LLM', port=8081)