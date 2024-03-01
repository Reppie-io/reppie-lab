import os
from PIL import Image
import streamlit as st

from utils import base64_to_pil
from vectorstores.pinecone import PineconeVectorStore

DEFAULT_QUERY = "roupas elegantes para homens"

PINECONE_INDEX_NAME = "cosine-image-search"
pinecone = PineconeVectorStore(
    api_key=os.getenv("PINECONE_API_KEY"), index_name=PINECONE_INDEX_NAME
)

if "images" not in st.session_state:
    st.session_state["images"] = []

if "init_query" not in st.session_state:
    st.session_state["init_query"] = False

if "search_bar_value" not in st.session_state:
    st.session_state["search_bar_value"] = DEFAULT_QUERY

if "searching_by_image" not in st.session_state:
    st.session_state["searching_by_image"] = False


@st.spinner(text="Pesquisando imagens...")
def search(query):
    results = pinecone.query(query)

    parsed_results = []

    for result in results:
        image_b64 = result["metadata"]["image_b64"]
        image_score = result["score"]

        parsed_results.append({"image_b64": image_b64, "score": image_score})

    st.session_state.images = parsed_results


with st.container():
    st.title("Busca semântica de imagens com CLIP e Pinecone.")

    uploaded_file = st.file_uploader(
        "Pesquise por imagens similares:",
        type=["jpg", "png"],
        accept_multiple_files=False,
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        search(image)

    query = st.text_input(
        "Ou pesquise por imagens usando busca semântica:",
        placeholder=st.session_state.search_bar_value,
        value="" if uploaded_file is not None else st.session_state.search_bar_value,
        disabled=uploaded_file is not None,
    )

    if query:
        st.session_state.search_bar_value = query
        search(query)

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
