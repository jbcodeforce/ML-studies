
FROM python:3.9.16
ENV PATH=/root/.local/bin:$PATH

ENV PYTHONPATH=/app

RUN pip install --upgrade pip \
  && pip install wget requests  numpy pandas xlrd sklearn keras matplotlib seaborn
USER $USER

# install NLP packages: 
RUN pip install -U  nltk  gensim  pixiedust
