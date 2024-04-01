import os
from typing import List
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.chains.combine_documents.stuff import StuffDocumentsChain


def save_file(file):
    folder = "tmp"
    if not os.path.exists(folder):
        os.makedirs(folder)

    file_path = f"./{folder}/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file.getvalue())
    return file_path


def stuff_summarize(docs: List[dict], llm: ChatOpenAI):
    # Define prompt
    prompt_template = """
        Write a concise summary in portuguese-BR of the following document:
        "{document}"
        CONCISE SUMMARY:
    """

    prompt = PromptTemplate.from_template(prompt_template)

    # Define LLM chain
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Define StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain, document_variable_name="document"
    )

    return stuff_chain.invoke(docs)


def summarize_file(file, chain_type="stuff"):
    file_path = save_file(file)

    file_loader = UnstructuredFileLoader(file_path)
    file_docs = file_loader.load()

    llm = ChatOpenAI(
        model="gpt-4-turbo-preview",
        api_key="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",
    )

    st.write(stuff_summarize(docs=file_docs, llm=llm))


with st.container():
    st.title("Resumo de documento.")

    uploaded_file = st.file_uploader(
        "Faça o upload do documento que deseja resumir:",
        type=["txt", "pdf"],
        accept_multiple_files=False,
    )

    with st.spinner("Gerando o resumo do documento, por favor aguarde..."):
        if uploaded_file:
            summarize_file(uploaded_file)
