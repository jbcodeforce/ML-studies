# Develop UI with Gradio

Gradio is the fastest way to demo stuff with a user interface. One of its advantages against StreamLit is the capacity to share data between pages and elements.

[Quickstart](https://www.gradio.app/guides/quickstart)

```sh
# For continuous development
gradio main.py
# For running a server
python main.app
```

Open in a browser on http://localhost:7860

[My first app](https://github.com/jbcodeforce/ML-studies/tree/master/techno/gradio/1_basic.py)

## Main concepts

* A python function can be wrapped with a user interface
* Use `gr.Blocks` to develop custom interface. Components are automatically added to the Blocks as they are created within the `with` clause. It is a classic python function. 
* Any Component that acts as an input to an event listener is made interactive
* Global state can be set via variable at the top of the program, and are shared between connected session

* [Chatbot](https://www.gradio.app/docs/gradio/chatbot) and [ChatInterface](https://www.gradio.app/guides/creating-a-chatbot-fast) to build simple user interface for a chat bot demo.

## How to

* Add [accordion to an interface](https://www.gradio.app/guides/the-interface-class#additional-inputs-within-an-accordion)
* To make an output R/W:  `output = gr.Textbox(label="Output", interactive=True)`
* [Updating Component Configurations](https://www.gradio.app/guides/blocks-and-event-listeners#updating-component-configurations)
