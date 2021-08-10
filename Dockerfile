
FROM python:3.7.4-stretch
ENV PATH=/root/.local/bin:$PATH

ENV PYTHONPATH=/app

ENV TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.1.0-cp36-cp36m-linux_x86_64.whl
RUN pip install --upgrade pip \
  && pip install wget requests  numpy pandas tflearn keras matplotlib
USER $USER

# install NLP packages: 
RUN pip install -U PySide2 nltk  gensim  pixiedust
