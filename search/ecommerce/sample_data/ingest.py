# docker-compose run app python search/ecommerce/sample_data/ingest.py

import os
from tqdm.auto import tqdm
from search.ecommerce.sample_data.dataset import (
    EcommerceDataset,
)
from libs.vectorstore.pinecone import Pinecone


def load_ecommerce_dataset(
    num_rows_to_ingest: int = None,
):  # if num_rows_to_ingest is None, ingest full dataset
    if num_rows_to_ingest:
        return EcommerceDataset().load_dataset().select(list(range(num_rows_to_ingest)))

    return EcommerceDataset().load_dataset()


def ingest_dataset_into_vectorstore(
    ecommerce_dataset: EcommerceDataset,
    vectorstore: Pinecone,
    batch_size: int = 200,
):
    imgs = ecommerce_dataset["image"]
    metadata = ecommerce_dataset.remove_columns("image").to_pandas()

    for i in tqdm(range(0, len(ecommerce_dataset), batch_size)):

        # find end of batch
        i_end = min(i + batch_size, len(ecommerce_dataset))

        # extract metadata batch
        meta_batch = metadata.iloc[i:i_end]
        metadata = meta_batch.to_dict(orient="records")

        # concatinate all metadata field except for id and year to form a single string
        keyword_texts = [
            " ".join(x)
            for x in meta_batch.loc[
                :, ~meta_batch.columns.isin(["id", "year"])
            ].values.tolist()
        ]

        # extract image batch
        images = imgs[i:i_end]

        # create unique IDs
        ids = [str(x) for x in range(i, i_end)]

        vectorstore.upsert_images(
            ids=ids, images=images, keyword_texts=keyword_texts, metadata=metadata
        )

    print("Vectors Upserted!")


ecommerce_dataset = load_ecommerce_dataset(num_rows_to_ingest=10)

index_name = os.getenv("PINECONE_INDEX_NAME")

vectorstore = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    index_name=index_name,
    bm25_fit_corpus=ecommerce_dataset["productDisplayName"],
)

if index_name not in vectorstore.pinecone.list_indexes().names():
    vectorstore.create_index(index_name=index_name, dimension=512)

ingest_dataset_into_vectorstore(
    ecommerce_dataset=ecommerce_dataset,
    vectorstore=vectorstore,
    batch_size=200,
)
