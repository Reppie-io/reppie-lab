import os
import streamlit as st

from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import load_summarize_chain
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.callbacks import get_openai_callback
from langchain_community.document_loaders import UnstructuredFileLoader

from prompts_templates.stuff import stuff_template
from prompts_templates.refine import seed_template, refine_template
from prompts_templates.map_reduce import map_template, reduce_template


def save_file(file) -> str:
    folder = "summarization/sales/tmp"
    if not os.path.exists(folder):
        os.makedirs(folder)

    file_path = f"./{folder}/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file.getvalue())
    return file_path


def split_transcriptions(transcriptions: str) -> List[str]:
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )

    return text_splitter.split_documents(transcriptions)


def stuff_summarization(transcriptions: List[dict], llm: ChatOpenAI):
    stuff_chain = load_summarize_chain(
        llm=llm,
        chain_type="stuff",
        prompt=PromptTemplate.from_template(stuff_template),
        document_variable_name="call_transcriptions",
    )

    return stuff_chain.invoke(transcriptions)


def map_reduce_summarization(transcriptions: List[dict], llm: ChatOpenAI) -> str:
    map_reduce_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=PromptTemplate.from_template(map_template),
        combine_prompt=PromptTemplate.from_template(reduce_template),
        combine_document_variable_name="call_transcriptions_summaries",
        map_reduce_document_variable_name="call_transcriptions",
    )

    splitted_transcriptions = split_transcriptions(transcriptions)

    return map_reduce_chain.invoke(splitted_transcriptions)


def refine_summarization(transcriptions: List[dict], llm: ChatOpenAI) -> str:
    refine_chain = load_summarize_chain(
        llm=llm,
        chain_type="refine",
        question_prompt=PromptTemplate.from_template(seed_template),
        refine_prompt=PromptTemplate.from_template(refine_template),
        document_variable_name="call_transcriptions",
        initial_response_name="existing_summary",
    )

    splitted_transcriptions = split_transcriptions(transcriptions)

    return refine_chain.invoke(splitted_transcriptions)


def generate_summary(
    transcriptions: List[dict], llm: ChatOpenAI, summarization_mode: str
) -> str:
    summary = None
    match summarization_mode:
        case "Map-Reduce":
            summary = map_reduce_summarization(transcriptions=transcriptions, llm=llm)
        case "Refine":
            summary = refine_summarization(transcriptions=transcriptions, llm=llm)
        case _:
            summary = stuff_summarization(transcriptions=transcriptions, llm=llm)

    return summary["output_text"]


def summarize_file(file, summarization_mode):
    file_path = save_file(file)

    file_loader = UnstructuredFileLoader(file_path)
    transcriptions = file_loader.load()

    llm = ChatOpenAI(
        model="gpt-4-turbo-preview",
    )

    with get_openai_callback() as cb:
        summary = generate_summary(transcriptions, llm, summarization_mode)

        st.write(summary)
        st.write(cb)


with st.container():
    st.title("Resumo de chamada de vendas.")

    uploaded_file = st.file_uploader(
        "Faça o upload da transcrição da chamada que deseja resumir:",
        type=["txt"],
        accept_multiple_files=False,
    )

    with st.expander("Selecione o modo de resumo."):
        summarization_mode = st.radio(
            "Modos disponíveis:",
            ["Stuff", "Map-Reduce", "Refine"],
            captions=[
                "Insere a transcrição inteira no contexto do modelo de linguagem (LLM). Ideal para transcrições com até 500k palavras. (GPT-4-Turbo)",  # noqa: E501
                'Primeiro resume os trechos da transcrição separadamente e depois "combina" todos os resumos em um resumo final. Ideal para transcrições com mais de 500k palavras. (GPT-4-Turbo)',  # noqa: E501
                "Gera o resumo iterativamente refinando o resumo a cada trecho da ligação, é um pouco mais barato que o modo Map-Reduce. Ideal para transcrições com mais de 500k palavras. (GPT-4-Turbo)",  # noqa: E501
            ],
            label_visibility="collapsed",
        )

    with st.spinner("Gerando o resumo do documento, pode levar alguns minutos..."):
        if uploaded_file:
            summarize_file(uploaded_file, summarization_mode)
