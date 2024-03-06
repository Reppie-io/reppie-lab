import io
import os
import base64
import streamlit as st

from PIL import Image
from typing import List
from libs.vectorstore.pinecone import PineconeIndex
from search.ecommerce.sample_data.dataset import EcommerceDataset


@st.cache_data
def load_ecommerce_dataset():
    return EcommerceDataset().load_dataset()


@st.cache_resource
def load_vectorstore(bm25_fit_corpus: List[str]):
    return PineconeIndex(
        api_key=os.getenv("PINECONE_API_KEY"),
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        bm25_fit_corpus=bm25_fit_corpus,
    )


def base64_to_pil(base64_str):
    img_data = base64.b64decode(base64_str)
    pil_image = Image.open(io.BytesIO(img_data))

    return pil_image


def update_results(results: List) -> None:
    results_images = []

    for result in results:
        image_b64 = result["metadata"]["image_b64"]
        image_score = result["score"]
        image_description = result["metadata"]["productDisplayName"]

        results_images.append(
            {
                "image_b64": image_b64,
                "description": image_description,
                "score": image_score,
            }
        )

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
                cols[j].write(images[idx]["description"])
                cols[j].write(f"Similary Score: {float(images[idx]['score']):.4f}")


if "images" not in st.session_state:
    st.session_state.images = []

bm25_fit_corpus = load_ecommerce_dataset()["productDisplayName"]
vectorstore = load_vectorstore(bm25_fit_corpus=bm25_fit_corpus)

with st.container():
    st.title("Busca híbrida de imagens com CLIP, BM25 e Pinecone.")

    on = st.toggle("Pesquisa por imagem")

    if on:
        st.session_state.images = []

        uploaded_file = st.file_uploader(
            "Pesquisa por imagem:",
            type=["jpg", "png"],
            accept_multiple_files=False,
        )

        with st.spinner("Pesquisando imagens semelhantes..."):
            if uploaded_file:
                image = Image.open(uploaded_file)
                results = vectorstore.image_search(image)
                update_results(results)
    else:
        st.session_state.images = []

        query = st.text_input(
            "Pesquisa por texto:",
            value="",
        )

        hybrid_alpha = st.slider(
            "Selecione o valor da busca híbrida: 0 (apenas keyword)"
            " e 1 (apenas semântica).",
            0.00,
            1.0,
        )

        with st.spinner("Pesquisando..."):
            if query:
                results = vectorstore.hybrid_search(
                    query=query,
                    alpha=hybrid_alpha,
                )
                update_results(results)

    display_images_grid(st.session_state.images)
