## Ecommerce Search

This app provides a demonstration illustrating the application of hybrid search to enhance search relevance in media and content production industries. The demo showcases two primary functionalities: searching for media content by image and searching for media content by text.

<img width="760" alt="image" src="https://github.com/Reppie-io/reppie-labs/assets/20309154/a098476b-765b-48b0-a026-f740b59b45dc">

### Multimodal Embedding
Multimodal embeddings refer to representations that capture both textual and visual information. In this demonstration, we leverage the (CLIP model)[https://huggingface.co/sentence-transformers/clip-ViT-B-32] developed by OpenAI, allowing the creation of vectors from both text and images. For this demo, searches are conducted based on images using CLIP embeddings.

## Dataset
The demo utilizes a (sample dataset)[https://huggingface.co/datasets/ashraq/fashion-product-images-small] suitable for ecommerce applications.

### How to Run
To run the demo, follow these three steps:

1. Add your Pinecone API Key to the docker-compose.yml file.
2. Ingest the dataset into the vector store by executing the following command in this directory: `docker-compose run app python search/media/sample_data/ingest.py`. Note that for performance reasons, only a subset of the dataset may be ingested (10000 products).
3. With the data ready in the vector store, run the demo using `docker-compose up` and access `localhost:8501` in your browser.

### Limitations
Please note that this is a demonstration intended for testing purposes only. If you require this use case with additional requirements or in a production environment, please contact us: contato@reppie.io.
