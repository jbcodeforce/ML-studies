from taipy.gui import Gui 
from taipy.gui import Markdown
from taipy.gui import Html

name=""
show_dialog=False

def button_action(state):
    state.show_dialog = True

def dialog_action(state, id, payload):
    with state as st:
        if payload["args"][0] == 0:
            name=name
        st.show_dialog = False

dialog_props = {
  "title":       "Select an item in the list",
  "labels":      "Cancel;Validate",
  "page_id":     "page",
  "close_label": "Cancel"
}

root_md = """
   # Page title

    {name}

   <|Open Me|button|on_action=button_action|class_name=plain|>

   <|{show_dialog}|dialog|labels=Save;Cancel|on_action=dialog_action |
   Enter a name:

   <|{name}|input|>
   |>

   """

if __name__ == "__main__":
    Gui(page=root_md).run(title="Dialog chart", debug=True, use_reloader=True)