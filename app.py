import streamlit as st

from PIL import Image
from PIL.Image import Image as ImageFile
from typing import List
from utils import base64_to_pil
from vectorstore.pinecone import PineconeIndex

DEFAULT_QUERY = "roupas elegantes para homens"
PINECONE_INDEX_NAME = "cosine-image-search"

if "images" not in st.session_state:
    st.session_state["images"] = []


@st.spinner(text="Pesquisando imagens...")
def search(query: ImageFile | str, index_name: str = PINECONE_INDEX_NAME) -> List:
    vectorstore = PineconeIndex(index_name)
    return vectorstore.semantic_search(query)


def update_results(results: List) -> None:
    results_images = []

    for result in results:
        image_b64 = result["metadata"]["image_b64"]
        image_score = result["score"]

        results_images.append({"image_b64": image_b64, "score": image_score})

    st.session_state.images = results_images


with st.container():
    st.title("Busca semântica de imagens com CLIP e Pinecone.")

    uploaded_file = st.file_uploader(
        "Pesquise por imagens similares:",
        type=["jpg", "png"],
        accept_multiple_files=False,
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        results = search(image)
        update_results(results)

    query = st.text_input(
        "Ou pesquise por imagens usando busca semântica:",
        disabled=uploaded_file is not None,
        value="",
    )

    if query and not uploaded_file:
        results = search(query)
        update_results(results)

    col_count = 4
    image_count = len(st.session_state.images)
    row_count = image_count // col_count + int(image_count % col_count > 0)
    for i in range(row_count):
        cols = st.columns(col_count)
        for j in range(col_count):
            idx = i * col_count + j
            if idx < image_count:

                cols[j].image(
                    base64_to_pil(st.session_state.images[idx]["image_b64"]), width=100
                )
                cols[j].write(
                    f"Similary Score: {float(st.session_state.images[idx]['score']):.4f}"
                )
