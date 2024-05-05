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

[My first app]()

## Main concepts

* A python function can be wrapped with a user interface
* use gr.Blocks to develop custom interface. Components are automatically added to the Blocks as they are created within the with clause
* any Component that acts as an input to an event listener is made interactive

## How tos