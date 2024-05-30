from taipy.gui import Gui 
from taipy.gui import Markdown
from taipy.gui import Html

root_md = """
<|navbar|>
# Multi-page application
<|content|>

   This application was created with [Taipy](https://www.taipy.io/).
 
"""

page2 = Html("""
<h1>Html Page 2</h1>

Go to <a href="/page1"><i> First Page </i></a> for more information

""")

page1 = Markdown("""
# MD Page 1

Any [*Markdown*](https://en.wikipedia.org/wiki/Markdown) content can be used here.

Go to [Second Page](/page2) for more information
""")

pages = {
    "/": root_md,
    "page1": page1,
    "page2": page2
}

if __name__ == "__main__":
    Gui(pages=pages).run(title="Dynamic chart", debug=True, use_reloader=True)