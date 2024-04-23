
from nicegui import ui
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from log_callback_handler import NiceGuiLogElementCallbackHandler
"""
a simple chatbot to interact with Llama3
"""

def buildLamaChatChain():
    llm = ChatOllama(model="llama3")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a earth environment expert."),
        ("user", "{input}")
    ])
    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser
    return chain


@ui.page('/')
def main():
    llm = buildLamaChatChain()
    async def send() -> None:
        question = text.value
        text.value = ''

        with message_container:
            ui.chat_message(text=question, name='You', sent=True)
            response_message = ui.chat_message(name='Bot', sent=False)
            spinner = ui.spinner(type='dots')

        response = ''
        async for chunk in llm.astream(question, config={'callbacks': [NiceGuiLogElementCallbackHandler(log)]}):
            response += chunk.content
            response_message.clear()
            with response_message:
                ui.html(response)
            ui.run_javascript('window.scrollTo(0, document.body.scrollHeight)')
        message_container.remove(spinner)

    ui.add_css(r'a:link, a:visited {color: inherit !important; text-decoration: none; font-weight: 500}')

    # the queries below are used to expand the contend down to the footer (content can then use flex-grow to expand)
    ui.query('.q-page').classes('flex')
    ui.query('.nicegui-content').classes('w-full')

    with ui.tabs().classes('w-full') as tabs:
        chat_tab = ui.tab('Chat')
        logs_tab = ui.tab('Logs')
    with ui.tab_panels(tabs, value=chat_tab).classes('w-full max-w-2xl mx-auto flex-grow items-stretch'):
        message_container = ui.tab_panel(chat_tab).classes('items-stretch')
        with ui.tab_panel(logs_tab):
            log = ui.log().classes('w-full h-full')
    with ui.footer().classes('bg-white'), ui.column().classes('w-full max-w-3xl mx-auto my-6'):
            with ui.row().classes('w-full no-wrap items-center'):
                placeholder = 'message' 
                text = ui.input(placeholder=placeholder).props('rounded outlined input-class=mx-3') \
                    .classes('w-full self-center').on('keydown.enter', send)
            ui.markdown('simple chat app built with [NiceGUI](https://nicegui.io)') \
                .classes('text-xs self-end mr-8 m-[-1em] text-primary')


ui.run(title="Chat with Lama 3")