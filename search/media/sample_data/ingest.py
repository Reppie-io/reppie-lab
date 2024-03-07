# docker-compose run app python search/media/sample_data/ingest.py

import os
from tqdm.auto import tqdm
from search.media.sample_data.dataset import (
    MediaDataset,
)
from libs.vectorstore.pinecone import Pinecone


def load_media_dataset(
    num_rows_to_ingest: int = None,
):  # if num_rows_to_ingest is None, ingest full dataset
    if num_rows_to_ingest:
        return MediaDataset().load_dataset().select(list(range(num_rows_to_ingest)))

    return MediaDataset().load_dataset()


def ingest_dataset_into_vectorstore(
    media_dataset: MediaDataset,
    vectorstore: Pinecone,
    batch_size: int,
):

    pandas_dataset = media_dataset.remove_columns(
        ["token_count", "timestamp"]
    ).to_pandas()

    # for i in range(30):
    #     print(type(pandas_dataset["text"].tolist()[i]))

    for i in tqdm(range(0, len(media_dataset), batch_size)):

        # find end of batch
        i_end = min(i + batch_size, len(media_dataset))

        # extract metadata batch
        metadata_batch = pandas_dataset.iloc[i:i_end]

        # concatinate all metadata field except for url and text to form a single string
        keyword_texts = [
            " ".join(x)
            for x in metadata_batch.loc[
                :, ~metadata_batch.columns.isin(["url", "text"])
            ].values.tolist()
        ]

        # extract texts batch
        texts = metadata_batch["text"].tolist()

        # meta_batch to dict
        metadata = metadata_batch.to_dict(orient="records")

        # create unique IDs
        ids = [str(x) for x in range(i, i_end)]

        vectorstore.upsert_texts(
            ids=ids,
            texts=texts,
            keyword_texts=keyword_texts,
            metadata=metadata,
        )

    print("Vectors Upserted!")


media_dataset = load_media_dataset(num_rows_to_ingest=1000)

index_name = os.getenv("PINECONE_INDEX_NAME")

vectorstore = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    index_name=index_name,
    bm25_fit_corpus=media_dataset["text"],
)

if index_name not in vectorstore.pinecone.list_indexes().names():
    vectorstore.create_index(index_name, dimension=384)

ingest_dataset_into_vectorstore(
    media_dataset=media_dataset,
    vectorstore=vectorstore,
    batch_size=10,
)
