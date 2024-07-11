from taipy.gui import Gui 
from taipy.gui import Markdown
from taipy.gui import Html

name = "Boule and Bill"
show_dialog=False

def button_action(state):
    state.show_dialog = True

def dialog_action(state):
    with state as st:
        name=st.name
        st.show_dialog = False

dialog_props = {
  "title":       "Select an item in the list",
  "labels":      "Cancel;Validate",
  "page_id":     "page",
  "close_label": "Cancel"
}

root_md = """
   # Page title

    <|{name}|>

    <|{test}|input|>
   <|Open Me|button|on_action=button_action|class_name=plain|>

   <|{show_dialog}|dialog|labels=Save;Cancel|on_action=dialog_action |
   Enter a name:

   <|{name}|input|>
   |>

   """

if __name__ == "__main__":
    Gui(page=root_md).run(title="Demo", use_reloader=True)