FROM jupyter/scipy-notebook

USER $USER

# Update pip
RUN pip install --upgrade pip
# install TensorFlow
RUN conda install --quiet --yes 'tensorflow=1.0*'

# install tflearn and keras:
RUN pip install tflearn==0.3.2 && pip install keras==2.0.8


# install NLP packages: 
RUN pip install -U PySide2 nltk==3.2.4 gensim==2.3.0 wget pixiedust

