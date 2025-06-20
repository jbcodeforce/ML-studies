from taipy.gui import Gui, Icon, navigate
import taipy.gui.builder as tgb
import pandas as pd
from chart import generate_map

data = pd.read_csv("data.csv")
map_fig = generate_map(data)
chart_data = (
        data.groupby("State")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
start_date = "2015-01-01"
start_date = pd.to_datetime(start_date)
end_date = "2018-12-31"
end_date = pd.to_datetime(end_date)

selected_category = "Furniture"
categories = list(data["Category"].unique())
selected_subcategory = "Bookcases"
subcategories = list(
    data[data["Category"] == selected_category]["Sub-Category"].unique()
)

layout = {
        "yaxis": {"title": "Revenue (USD)"},
        "title": f"Sales by State",
    }

def change_category(state):
    """
    update the chart data based on the selected category
    """
    state.subcategories = list(
        data[data["Category"] == state.selected_category]["Sub-Category"].unique()
    )
    state.selected_subcategory = state.subcategories[0]
    

def apply_changes(state):
    new_data = data[
        (
            pd.to_datetime(data["Order Date"], format="%d/%m/%Y")
            >= pd.to_datetime(state.start_date)
        )
        & (
            pd.to_datetime(data["Order Date"], format="%d/%m/%Y")
            <= pd.to_datetime(state.end_date)
        )
    ]
    new_data = new_data[new_data["Category"] == state.selected_category]
    new_data = new_data[new_data["Sub-Category"] == state.selected_subcategory]
    state.data = new_data
    state.chart_data = (
        state.data.groupby("State")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    state.layout = {
        "yaxis": {"title": "Revenue (USD)"},
        "title": f"Sales by State for {state.selected_category} - {state.selected_subcategory}",
    }
    state.map_fig = generate_map(state.data)

def menu_option_selected(state, action, info):
    page = info["args"][0]
    navigate(state, to=page)

with tgb.Page() as root_page:
    tgb.menu(
        label="Menu",
        lov=[
            ("page1", Icon("images/map.png", "Sales")),
            ("page2", Icon("images/person.png", "Account")),
        ],
        on_action=menu_option_selected,
    )

with tgb.Page() as page_2:
    tgb.text("# Account **Management**", mode="md")
    tgb.button("Logout", class_name="plain login-button", width="50px")

with tgb.Page() as page_1:
    with tgb.part(class_name="container"):
        tgb.text("# Sales by **state**", mode = "md")
        with tgb.part(class_name="card"):
            with tgb.layout(columns="1 2 1"):
                with tgb.part():
                    tgb.text("Filter **From**", mode="md")
                    tgb.date("{start_date}")
                    tgb.text("To")
                    tgb.date("{end_date}")
                with tgb.part():
                    tgb.text("Filter Product **Category**", mode="md")
                    tgb.selector(
                        value="{selected_category}",
                        lov=categories,
                        on_change=change_category,
                        dropdown=True,
                    )
                    tgb.text("Filter Product **Subcategory**", mode="md")
                    tgb.selector(
                        value="{selected_subcategory}",
                        lov="{subcategories}",
                        dropdown=True,
                    )
                with tgb.part(class_name="text-center"):
                    tgb.button(
                        "Apply",
                        class_name="plain apply_button",
                        on_action=apply_changes,
                    )

    tgb.html("br")
    with tgb.layout(columns="2 3"):
        tgb.chart(
            data="{chart_data}",
            x="State",
            y="Sales",
            type="bar",
            layout="{layout}",
        )
        tgb.chart(figure="{map_fig}")
    tgb.html("br")
    tgb.table(data="{data}")

pages = {"/": root_page, "page1": page_1, "page2": page_2}

if __name__ == "__main__":
    Gui(pages=pages).run(title="Sales", dark_mode=False, debug=True, use_reloader=True, port=5001)



