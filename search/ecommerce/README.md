## Ecommerce Search

This app provides a demonstration illustrating the application of hybrid search to enhance search relevance in e-commerce platforms. The demo showcases two primary functionalities: searching for products by image and searching for products by text.

<img width="736" alt="image" src="https://github.com/Reppie-io/reppie-labs/assets/20309154/e4d2ce44-c780-4e29-8ef2-e68da6276fd0">

### Embeddings

#### CLIP
 In this demonstration, we leverage the [CLIP model](https://huggingface.co/sentence-transformers/clip-ViT-B-32) developed by OpenAI, allowing the creation of vectors from both text and images. It is a multimodal embedding modal which refers to representations that capture both textual and visual information.

### BM25

For the keyword embeddings, we use BM25 embedding model. BM25 is a popular information retrieval (IR) model that is often used for ranking documents in search results. It is a statistical model that takes into account the frequency of terms in a document, as well as the length of the document, to determine its relevance to a query.

BM25 can be used for sparse vector embeddings, which are high-dimensional vectors that represent documents. In this context, each dimension of the vector corresponds to a term in the vocabulary, and the value of each dimension represents the frequency of that term in the document.

Check Pinecone encode sparse vectors [documentation](https://docs.pinecone.io/docs/encode-sparse-vectors) for more details.

## Dataset
The demo utilizes a [sample dataset](https://huggingface.co/datasets/ashraq/fashion-product-images-small) suitable for e-commerce applications.

### How to Run
To run the demo, follow these three steps:

1. Add your Pinecone API Key to the docker-compose.yml file.
2. Ingest the dataset into the vector store by executing the following command in this directory: `docker-compose run app python search/media/sample_data/ingest.py`. Note that for performance reasons, only a subset of the dataset may be ingested (10000 products).
3. With the data ready in the vector store, run the demo using `docker-compose up` and access `localhost:8501` in your browser.

----
Here's a breakdown of each step:

**Step 1: Add Your Pinecone API Key**

1. Locate the file named `docker-compose.yml` in the current directory. This file contains configurations for running the demo using Docker.
2. Open the `docker-compose.yml` file using a text editor.
3. Replace the placeholder <PINECONE_API_KEY> with your actual Pinecone API Key. You can obtain your API key from your Pinecone account dashboard.
4. Save the changes you made to the `docker-compose.yml` file.

**Step 2: Ingest Dataset**

1. Open your terminal application (Command Prompt on Windows, Terminal on macOS/Linux).
2. Make sure you are in the same directory where the `docker-compose.yml` file is located. You can use the `pwd` command to verify your current directory.
3. Run the following command to ingest the dataset into the vector store:

   #shouldn't be docker-compose run app python reppie-labs/search/ecommerce/sample_data/ingest.py ??

```bash
docker-compose run app python search/media/sample_data/ingest.py
```

**Note:**
This script likely handles loading the sample data and uploading it to the vector store using your Pinecone API Key.
For performance reasons, only a subset of the dataset (10,000 products) might be ingested.

**Step 3: Run the Demo and Access It**

1. After the data ingestion is complete (indicated by the terminal showing success messages), run the following command in your terminal:

```bash
docker-compose up
```

2. Once the services are up and running, open your web browser and navigate to `http://localhost:8501`. This should launch the product search demo application.

**By following these steps, you should be able to run the ecommerce search demo application.**

### Limitations
Please note that this is a demonstration intended for testing purposes only. If you require this use case with additional requirements or in a production environment, please contact us: contato@reppie.io.
