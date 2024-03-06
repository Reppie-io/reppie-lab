import os
from tqdm.auto import tqdm
from search.ecommerce.sample_data.dataset import (
    EcommerceDataset,
)
from libs.vectorstore.pinecone import PineconeIndex


def load_ecommerce_dataset(
    num_rows_to_ingest: int = None,
):  # if num_rows_to_ingest is None, ingest full dataset
    if num_rows_to_ingest:
        return EcommerceDataset().load_dataset().select(list(range(num_rows_to_ingest)))

    return EcommerceDataset().load_dataset()


def ingest_dataset_into_vectorstore(
    ecommerce_dataset: EcommerceDataset,
    vectorstore_index: PineconeIndex,
    batch_size: int = 200,
):
    images = ecommerce_dataset["image"]
    metadata = ecommerce_dataset.remove_columns("image").to_pandas()

    for i in tqdm(range(0, len(ecommerce_dataset), batch_size)):

        # find end of batch
        i_end = min(i + batch_size, len(ecommerce_dataset))

        # extract metadata batch
        meta_batch = metadata.iloc[i:i_end]
        meta_dict = meta_batch.to_dict(orient="records")

        # concatinate all metadata field except for id and year to form a single string
        meta_batch = [
            " ".join(x)
            for x in meta_batch.loc[
                :, ~meta_batch.columns.isin(["id", "year"])
            ].values.tolist()
        ]

        # extract image batch
        img_batch = images[i:i_end]

        # create unique IDs
        ids = [str(x) for x in range(i, i_end)]

        vectorstore_index.upsert_images(ids=ids, images=img_batch, meta_list=meta_batch, meta_dict=meta_dict)

    print("Vectors Upserted!")


ecommerce_dataset = load_ecommerce_dataset(num_rows_to_ingest=10)
pinecone_index = PineconeIndex(
    api_key=os.getenv("PINECONE_API_KEY"),
    index_name=os.getenv("PINECONE_INDEX_NAME"),
    bm25_fit_corpus=ecommerce_dataset["productDisplayName"],
)

ingest_dataset_into_vectorstore(
    ecommerce_dataset=ecommerce_dataset,
    vectorstore_index=pinecone_index,
    batch_size=200,
)
