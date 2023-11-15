# [Streamlit](https://streamlit.io/)

Streamlit is an open-source Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science. It uses an integrated cloud based IDE with integration to github codespaces and deploy on

## Main concepts

* Use a CLI to start an streamlit server.

```sh
streamlit run your_script.py
# or 
python -m streamlit your_script.py
# or an url
streamlit run https://raw.githubusercontent.com/streamlit/demo-uber-nyc-pickups/master/streamlit_app.py
```

* Support continuous interactive loop development experience
* Can present Pandas dataframe directly in the page withing a table widget.
* Streamlit makes it easy to organize your widgets in a left panel sidebar with st.sidebar. 
* It supports Light and Dark themes out of the box, and custom theme.
* Support multiple pages application.
* Streamlit apps are Python scripts that run from top to bottom

## Samples

* [Getting Started](https://docs.streamlit.io/library/get-started/create-an-app) with [matching code]() to test it.
* Run the app in Docker, see the Dockerfile [in this project / folder (llm-ref-arch-demo/sa-tools/user-interface)](https://github.com/jbcodeforce/llm-ref-arch-demo/blob/main/sa-tools/user-interface/Dockerfile)
