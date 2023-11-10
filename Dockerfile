
FROM python:3.11.6
ENV PATH=/root/.local/bin:$PATH

ENV PYTHONPATH=/app

RUN pip install --upgrade pip \
  && pip install wget requests  numpy pandas xlrd scikit-learn keras matplotlib seaborn bs4
USER $USER

# install NLP packages: 
RUN pip install -U  nltk  gensim  pixiedust

WORKDIR /app