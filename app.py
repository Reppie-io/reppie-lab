import streamlit as st

from PIL import Image
from typing import List
from utils import base64_to_pil
from vectorstore.pinecone import PineconeIndex
from vectorstore.data.open_fashion.dataset import OpenFashionDataset

DEFAULT_QUERY = "roupas elegantes para homens"
PINECONE_INDEX_NAME = "hybrid-image-search"


@st.cache_data
def load_open_fashion_dataset():
    return OpenFashionDataset().load_dataset()


def update_results(results: List) -> None:
    results_images = []

    for result in results:
        image_b64 = result["metadata"]["image_b64"]
        image_score = result["score"]

        results_images.append({"image_b64": image_b64, "score": image_score})

    st.session_state.images = results_images


def display_images_grid(images: List[dict]) -> None:
    col_count = 4
    image_count = len(images)
    row_count = image_count // col_count + int(image_count % col_count > 0)
    for i in range(row_count):
        cols = st.columns(col_count)
        for j in range(col_count):
            idx = i * col_count + j
            if idx < image_count:
                cols[j].image(
                    base64_to_pil(images[idx]["image_b64"]),
                    width=100,
                )
                cols[j].write(f"Similary Score: {float(images[idx]['score']):.4f}")


if "images" not in st.session_state:
    st.session_state["images"] = []

bm25_fit_corpus = load_open_fashion_dataset()["productDisplayName"]
vectorstore = PineconeIndex(PINECONE_INDEX_NAME)

with st.container():
    st.title("Busca semântica de imagens com CLIP e Pinecone.")

    uploaded_file = st.file_uploader(
        "Pesquise por imagens similares:",
        type=["jpg", "png"],
        accept_multiple_files=False,
    )

    with st.spinner("Pesquisando imagens semelhantes..."):
        if uploaded_file:
            image = Image.open(uploaded_file)
            results = vectorstore.image_search(image)
            update_results(results)

    query = st.text_input(
        "Ou pesquise por imagens usando busca semântica:",
        disabled=uploaded_file is not None,
        value="",
    )

    with st.spinner("Pesquisando imagens semelhantes..."):
        if query and not uploaded_file:
            results = vectorstore.hybrid_search(
                query=query,
                alpha=1.0,  # float between 0 and 1 where 0 == sparse only and 1 == dense only
            )
            update_results(results)

    display_images_grid(st.session_state.images)
