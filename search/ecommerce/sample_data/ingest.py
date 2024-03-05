import base64
import sys
import os
import io
from tqdm.auto import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))

levels_up = 3
app_dir = script_dir
for _ in range(levels_up):
    app_dir = os.path.dirname(app_dir)

sys.path.append(app_dir)

from libs.embedding.clip.encoder import CLIPEncoder  # noqa: E402
from libs.embedding.bm25.encoder import Bm25Encoder  # noqa: E402
from search.ecommerce.sample_data.open_fashion_dataset import OpenFashionDataset  # noqa: E402
from libs.vectorstore.pinecone import PineconeIndex  # noqa: E402


PINECONE_INDEX_NAME = "ecommerce-hybrid-image-search"

# ingest full dataset
open_fashion_dataset = OpenFashionDataset().load_dataset()

# ingest only a subset of the dataset
# NUM_PRODUCTS_TO_INGEST = 10
# open_fashion_dataset = (
#     OpenFashionDataset().load_dataset().select(list(range(NUM_PRODUCTS_TO_INGEST)))
# )

clip_model = CLIPEncoder()
bm25_model = Bm25Encoder(fit_corpus=open_fashion_dataset["productDisplayName"])
pinecone_index = PineconeIndex(
    PINECONE_INDEX_NAME, bm25_fit_corpus=open_fashion_dataset["productDisplayName"]
)

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

    # create sparse BM25 vectors
    sparse_embeds = bm25_model.encode_documents(texts=[text for text in meta_batch])

    # create vectors values
    values = clip_model.encode(img_batch)

    # create unique IDs
    ids = [str(x) for x in range(i, i_end)]

    vectors = []

    # loop through the data and create dictionaries for uploading to pinecone index
    for _id, value, sparse, meta in zip(ids, values, sparse_embeds, meta_dict):
        img_bytes = io.BytesIO()
        image = images[int(_id)]

        # Save the image to the in-memory stream in JPEG format
        image.save(img_bytes, format="JPEG")

        meta["image_b64"] = base64.b64encode(img_bytes.getvalue()).decode("utf-8")

        vectors.append(
            {"id": _id, "values": value, "sparse_values": sparse, "metadata": meta}
        )

    # upload the documents to the new hybrid index
    pinecone_index.upsert_vectors(vectors)

print("Vectors Upserted!")
