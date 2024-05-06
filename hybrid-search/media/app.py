import os
import streamlit as st

from typing import List
from libs.vectorstore.pinecone import Pinecone


@st.cache_resource
def load_vectorstore():
    return Pinecone(
        api_key=os.getenv("PINECONE_API_KEY"),
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        bm25_params_path="libs/embedding/bm25/bm25_media.json",
    )


def update_results(results: List) -> None:
    results_blog_posts = []

    for result in results:
        blog_post_title = result["metadata"]["title"]
        blog_post_score = result["score"]
        blog_post_text = result["metadata"]["text"]
        blog_post_link = result["metadata"]["url"]
        blog_post_authors = result["metadata"]["authors"]
        blog_post_tags = result["metadata"]["tags"]

        results_blog_posts.append(
            {
                "title": blog_post_title,
                "url": blog_post_link,
                "text": blog_post_text,
                "score": blog_post_score,
                "authors": blog_post_authors,
                "tags": blog_post_tags,
            }
        )

    st.session_state.blog_posts = results_blog_posts


def display_blog_posts(blog_posts: List[dict]) -> None:
    for blog_post in blog_posts:
        st.write(f"<b>Title:</b> {blog_post['title']}", unsafe_allow_html=True)
        st.write(f"<b>Link:</b> {blog_post['url']}", unsafe_allow_html=True)
        st.write(f"<b>Authors:</b> {blog_post['authors']}", unsafe_allow_html=True)
        st.write(f"<b>Tags</b>: {blog_post['tags']}", unsafe_allow_html=True)
        st.write(
            f"<b>Similary Score:</b> {float(blog_post['score']):.4f}",
            unsafe_allow_html=True,
        )

        with st.expander("Text", expanded=False):
            st.write(blog_post["text"])

        st.divider()


if "blog_posts" not in st.session_state:
    st.session_state.blog_posts = []

vectorstore = load_vectorstore()

with st.container():
    st.title("Busca híbrida de blog posts com All_MiniLM, BM25 e Pinecone.")

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

    display_blog_posts(st.session_state.blog_posts)
