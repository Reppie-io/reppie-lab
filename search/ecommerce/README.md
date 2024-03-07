## Ecommerce Search

This app provides a demonstration illustrating the application of hybrid search to enhance search relevance in e-commerce platforms. The demo showcases two primary functionalities: searching for media content by image and searching for media content by text.

<img width="736" alt="image" src="https://github.com/Reppie-io/reppie-labs/assets/20309154/e4d2ce44-c780-4e29-8ef2-e68da6276fd0">

### Embeddings

#### CLIP
 In this demonstration, we leverage the [CLIP model](https://huggingface.co/sentence-transformers/clip-ViT-B-32) developed by OpenAI, allowing the creation of vectors from both text and images. It is a multimodal embedding modal which refers to representations that capture both textual and visual information.

### BM25

For the keyword embeddings, we use BM25 embedding model. BM25 is a popular information retrieval (IR) model that is often used for ranking documents in search results. It is a statistical model that takes into account the frequency of terms in a document, as well as the length of the document, to determine its relevance to a query.

BM25 can be used for sparse vector embeddings, which are high-dimensional vectors that represent documents. In this context, each dimension of the vector corresponds to a term in the vocabulary, and the value of each dimension represents the frequency of that term in the document.

Check Pinecone encode sparse vectors [documentation](https://docs.pinecone.io/docs/encode-sparse-vectors) for more details.

## Dataset
The demo utilizes a [sample dataset](https://huggingface.co/datasets/ashraq/fashion-product-images-small) suitable for ecommerce applications.

### How to Run
To run the demo, follow these three steps:

1. Add your Pinecone API Key to the docker-compose.yml file.
2. Ingest the dataset into the vector store by executing the following command in this directory: `docker-compose run app python search/media/sample_data/ingest.py`. Note that for performance reasons, only a subset of the dataset may be ingested (10000 products).
3. With the data ready in the vector store, run the demo using `docker-compose up` and access `localhost:8501` in your browser.

### Limitations
Please note that this is a demonstration intended for testing purposes only. If you require this use case with additional requirements or in a production environment, please contact us: contato@reppie.io.
