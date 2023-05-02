
FROM python:latest
ENV PATH=/root/.local/bin:$PATH

ENV PYTHONPATH=/app

RUN pip install --upgrade pip \
  && pip install wget requests  numpy pandas xlrd scikit-learn keras matplotlib seaborn
USER $USER

# install NLP packages: 
RUN pip install -U  nltk  gensim  pixiedust
WORKDIR /app