import sys
import os
from tqdm.auto import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))

levels_up = 3
app_dir = script_dir
for _ in range(levels_up):
    app_dir = os.path.dirname(app_dir)

sys.path.append(app_dir)

from embedding.CLIP.sentence_transformer import CLIPSentenceTransformer  # noqa: E402
from vectorstore.data.open_fashion.dataset import OpenFashionDataset  # noqa: E402
from vectorstore.pinecone import PineconeIndex  # noqa: E402


PINECONE_INDEX_NAME = "cosine-image-search-2"

# ingest full dataset
open_fashion_dataset = OpenFashionDataset().load_dataset()

# NUM_PRODUCTS_TO_INGEST = 10
# ingest only a subset of the dataset
# open_fashion_dataset = (
#     OpenFashionDataset().load_dataset().select(list(range(NUM_PRODUCTS_TO_INGEST)))
# )

clip_model = CLIPSentenceTransformer()
pinecone_index = PineconeIndex(PINECONE_INDEX_NAME)

images = open_fashion_dataset["image"]
metadata = open_fashion_dataset.remove_columns("image").to_pandas()

batch_size = 200
for i in tqdm(range(0, len(open_fashion_dataset), batch_size)):
    # find end of batch
    i_end = min(i + batch_size, len(open_fashion_dataset))

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

    # create vectors values
    values = clip_model.encode(img_batch)

    # create unique IDs
    ids = [str(x) for x in range(i, i_end)]

    vectors = []

    # loop through the data and create dictionaries for uploading to pinecone index
    for _id, value, meta in zip(ids, values, meta_dict):
        vectors.append({"id": _id, "values": value, "metadata": meta})

    # upload the documents to the new hybrid index
    pinecone_index.upsert_vectors(vectors)

print("Vectors Upserted!")
