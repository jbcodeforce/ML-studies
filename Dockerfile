
FROM python:latest
ENV PATH=/root/.local/bin:$PATH

ENV PYTHONPATH=/app

RUN pip install --upgrade pip \
  && pip install wget requests  numpy pandas xlrd scikit-learn keras matplotlib seaborn 
USER $USER

# install NLP packages: 
RUN pip install -U  nltk  gensim  pixiedust
# install llm package
RUN pip install langchain openai boto3
WORKDIR /app