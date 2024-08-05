# Natural Language Processing (NLP)


## Embedding

An embedding is a mathematical representation of a set of data points in a lower-dimensional space that captures their underlying relationships and patterns. There are different embedding types: image, word, graph, video embeddings.

The vector numbers intent to capture the attributes of the object and the semantic and syntactic relationships between words. Dense embeddings were introduced by Google’s [**Word2vec**](https://arxiv.org/abs/1301.3781) (Mikolov et al) in 2014 and used in GPT model. The transformation of word to vector, gives the capability to compute arithmetics with words, like **similarity** computation. Vectors which are closer together, mean they represent semantically similar concepts. 
The technique works by training a neural network on a large corpus of text data, to predict the context in which a given word appears. Principal Component Analysis (PCA) and Singular Value Decomposition (SVD), auto-encoder, are dimensionality reduction techniques. 

See [this basic code](https://github.com/jbcodeforce/ML-studies/blob/master/llm-langchain/RAG/embeddings_hf.py) which uses `SentenceTransformer all-MiniLM-L6-v2` model to encode sentences of 100 tokens, construct from a markdown file.

Embeddings are created using a pre-trained LLM, and a set of documents used to fine-tune the model. The fine-tuning process is done using a small subset of the documents, and the LLM is trained to predict the next word in the document. 

The fine-tuned LLM is then used to generate the embeddings. The embedding size is usually between 200 to 1000 dimensions. 

The embedding process is time consuming, and may take several days to complete. The embedding model is usually saved and re-used, and most of the time in open access. 

Embeddings are used to compare the query with the document chunks. The cosine similarity is used to compute the similarity between the query and the document chunks. 

The cosine similarity is a measure of the similarity between two non-zero vectors of an inner product space. It is defined to equal the cosine of the angle between them, which is also the same as the inner product of the same vectors normalized to both have length 1. 

Embedding can improve data quality, reduce the need for manual data labeling, and enable more efficient computation.

See the [Encord's guide to embeddings in machine learning](https://encord.com/blog/embeddings-machine-learning/)

### Use cases

* LLM for token embedding
* Image embedding to represent images in the vector space.
* Audio and video embedding 
* RAG with sentence embedding and similarity search.
* Product recommendations, via similarity search
* Anomaly detection 


## BERT

Bidirectional Encoder Representations from Transformers (BERT) is a family of masked-language models published in 2018 by researchers at Google. It is much smaller than current LLMs, so if the task can be accomplished by BERT it can be very helpful for developers - however it usually does not perform as well as other foundation models because it is not large enough. 

## Named Entity Recognition

Named Entity Recognition (NER) is a Natural Language Processing (NLP) technique used to identify and extract important entities from unstructured text data. It is achieved by using NN trained on labeled data to recognize patterns and extract entity from text.

Some techniques uses Gen AI model to do NER with a good prompt.


## Deeper Dive

* [PyTorch based NLP tutorial](https://github.com/graykode/nlp-tutorial)