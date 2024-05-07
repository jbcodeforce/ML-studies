import gradio as gr

"""
Keep global state in an array. Sort the elements and place highest score first
"""
scores = []

def track_score(score):
    scores.append(score)
    top_scores = sorted(scores, reverse=True)[:3]
    return top_scores

"""
Interface to present any python function result
"""
demo = gr.Interface(
    track_score,       # function
    gr.Number(label="Score"),   # input
    gr.JSON(label="Top Scores"),  # output
    title="1st Gradio app",
    description = "Present one input form + sorted score result as json using global state.",
    article =" Based on [tutorial](https://www.gradio.app/guides/the-interface-class)",
    live=True
)
demo.launch()