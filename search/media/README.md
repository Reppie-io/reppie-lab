## Help Center Search

This app provides a demonstration illustrating the application of hybrid search to enhance search relevance in media and content production industries.

<img width="760" alt="image" src="https://github.com/Reppie-io/reppie-labs/assets/20309154/baed9dfd-d36c-429f-860e-b19f370abe83">

### Embeddings

#### all-MiniLM-L6-v2
In this demonstration, we use the `all-MiniLM-L6-v2` model, a variant of the MiniLM model, which is is a general-purpose model that can be used for a variety of tasks, including:
* Semantic Search
* Question Answering
* Text summarization

The all-MiniLM-L6-v2 model is available for download from the Hugging (Face Model Hub.)[https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2]

#### BM25

For the keyword embeddings, we use BM25 embedding model. BM25 is a popular information retrieval (IR) model that is often used for ranking documents in search results. It is a statistical model that takes into account the frequency of terms in a document, as well as the length of the document, to determine its relevance to a query.

BM25 can be used for sparse vector embeddings, which are high-dimensional vectors that represent documents. In this context, each dimension of the vector corresponds to a term in the vocabulary, and the value of each dimension represents the frequency of that term in the document.

Check Pinecone enconde sparse vectors (documentation)[https://docs.pinecone.io/docs/encode-sparse-vectors] for more details.

### Dataset
The demo utilizes a (sample dataset)[https://huggingface.co/datasets/BEE-spoke-data/medium-articles-en] suitable for media and content production industries.

### How to Run
To run the demo, follow these three steps:

1. Add your Pinecone API Key to the docker-compose.yml file.
2. Ingest the dataset into the vector store by executing the following command in this directory: docker-compose run app python search/media/sample_data/ingest.py. Note that for performance reasons, only a subset of the dataset may be ingested.
3. With the data ready in the vector store, run the demo using docker-compose up and access localhost:8501 in your browser.

### Limitations
Please note that this is a demonstration intended for testing purposes only. If you require this use case with additional requirements or in a production environment, please contact us at contato@reppie.io.
