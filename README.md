# Semantic Image Search with Pinecone and CLIP

This repository demonstrates semantic image search using Pinecone for vector storage and CLIP for multimodal embeddings.

## Overview

Semantic image search allows users to find images based on their content rather than just metadata. In this project, we leverage Pinecone for vector storage and retrieval, and CLIP for generating multimodal embeddings that enable semantic understanding of images.

## Requirements

To use this application, you need to have the following:

- Docker
- Pinecone API key (obtainable from Pinecone's [documentation](https://docs.pinecone.io/v1/docs/quickstart#2-get-your-api-key))

## Setting up secrets.toml

To securely store your Pinecone API key, follow these steps:

1. Create a file named `.streamlit/secrets.toml` in the root directory of the project.
2. Add the following lines to `.streamlit/secrets.toml`:

```toml
[PINECONE]
API_KEY = "YOUR_PINECONE_API_KEY"
```

Replace "YOUR_PINECONE_API_KEY" with your actual Pinecone API key.

## Usage

To use the semantic image search application:

1. Clone this repository.
2. Set up your secrets.toml file as described above.
3. Build docker image and run: `docker-compose up --build`

## Generating Data in Pinecone

To generate data in Pinecone, you can use the provided ingest.py script located inside the vectorstore.data.open_fashion folder. This script downloads the [Open Fashion](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) dataset and inserts it into the vector store.

To generate the data, follow these steps:

1. Navigate to the vectorstore.data.open_fashion directory.
2. Run the ingest.py script by executing `python vectorstore/data/open_fashion/ingest.py` in your terminal.

The script will download the dataset and insert it into the Pinecone vector store.

## References

[Pinecone Documentation](https://docs.pinecone.io/docs/overview) <br>
[OpenAI CLIP Documentation](https://openai.com/research/clip)

## Contributing
Contributions are welcome! Please fork this repository and submit pull requests with your enhancements.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
