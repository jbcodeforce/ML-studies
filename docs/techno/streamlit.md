# [Streamlit](https://streamlit.io/)

Streamlit is an open-source Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science. It uses an integrated cloud based IDE with integration to github codespaces and deploy on streamlit SaaS servers.

See [Getting started](https://docs.streamlit.io/library/get-started/create-an-app)

## Main concepts

* Use a CLI to start an Streamlit server.

```sh
streamlit run your_script.py
# or 
python -m streamlit your_script.py
# or using a script in a git url
streamlit run https://raw.githubusercontent.com/streamlit/demo-uber-nyc-pickups/master/streamlit_app.py
```

* Support continuous interactive loop development experience
* Can present Pandas dataframe directly in the page withing a table widget.
* Streamlit makes it easy to organize the widgets in a left panel sidebar with `st.sidebar`. 
* It supports Light and Dark themes out of the box, and custom theme.
* Support multiple pages application.


## Samples

* [Getting Started](https://docs.streamlit.io/library/get-started/create-an-app) with [matching code]() to test it.
* Run the app in Docker, see the Dockerfile [in this project / folder (llm-ref-arch-demo/sa-tools/user-interface)](https://github.com/jbcodeforce/llm-ref-arch-demo/blob/main/sa-tools/user-interface/Dockerfile)
* Some [best practices here](https://pmbaumgartner.github.io/streamlitopedia/front/introduction.html).
* [Streamlit Cheat Sheet app in Streamlit](https://cheat-sheet.streamlit.app/).

## Some How To

???- question "How to share data between pages?"
    Use st.session_state. For example a page get some settings in a form and a save button. The supporting function needs to use the session_state.

    ```python
        if save_button:
            data={"callWithVectorStore":callWithVectorStore, "callWithDecisionService": callWithDecisionService, "llm_provider": llm_provider }
            st.write(data)
            st.session_state["app_config"]=data
    ```
    In other page use something as:

    ```python
    if 'app_config' not in st.session_state:
    st.session_state['app_config']= {
        "callWithVectorStore":False, 
        "callWithDecisionService": False, 
        "llm_provider": "openAI" 
        }

    app_config=st.session_state['app_config']
    ```
