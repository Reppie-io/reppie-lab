import os
from typing import List
import streamlit as st

from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_community.callbacks import get_openai_callback

from langchain.chains import (
    MapReduceDocumentsChain,
    ReduceDocumentsChain,
    load_summarize_chain,
)
from langchain_text_splitters import CharacterTextSplitter


def save_file(file):
    folder = "summary/tmp"
    if not os.path.exists(folder):
        os.makedirs(folder)

    file_path = f"./{folder}/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file.getvalue())
    return file_path


def refine_summarize(docs: List[dict], llm: ChatOpenAI):
    prompt_template = """Write a concise summary of the following:
    {text}
    CONCISE SUMMARY:"""

    prompt = PromptTemplate.from_template(prompt_template)

    refine_template = """
        Your job is to produce a final summary
        We have provided an existing summary up to a certain point: {existing_answer}
        We have the opportunity to refine the existing summary
        (only if needed) with some more context below.
        ------------
        {text}
        ------------
        If the context isn't useful, return the original summary.
        Given the new context, refine the original summary following the instructions:

        You are a helpful assistant that helps a sales rep to summarize information from the sales call transcriptions provided above.
        You have two specific goals:
            
            1. Write a summary from the perspective of the sales rep that will highlight key points that will be relevant to making this sale.
            
            2. Identify the next steps agreed between the sales meeting participants from the perspective of the sales rep that will highlight 
            key points that will be relevant to making this sale. In the nexts steps, you should identify the dates, time and deadlines agreed between the meeting participants always as possible.

        The output format must follow the basic structure shown below:

            ```
            Resumo da chamada:
            <the generated summary>

            Próximos passos acordados:
            <the generated next steps. Use structured data such as lists, tables and bullet points always as possible>
            ```

        Do not respond with anything outside of the call transcript. If you don't know, answer with "I don't know"
        The output must be in brazilian portuguese.
    """

    refine_prompt = PromptTemplate.from_template(refine_template)
    chain = load_summarize_chain(
        llm=llm,
        chain_type="refine",
        question_prompt=prompt,
        refine_prompt=refine_prompt,
        return_intermediate_steps=False,
        input_key="input_documents",
        output_key="output_text",
    )

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    split_docs = text_splitter.split_documents(docs)

    return chain.invoke(split_docs)


def map_reduce_summarize(docs: List[dict], llm: ChatOpenAI):
    # Map
    map_template = """
    You are a helpful assistant that helps a sales rep to summarize information from the following sales call transcriptions: {docs}.
        You have two specific goals:
            
            1. Write a summary from the perspective of the sales rep that will highlight key points that will be relevant to making this sale.
            
            2. Identify the next steps agreed between the sales meeting participants from the perspective of the sales rep that will highlight 
            key points that will be relevant to making this sale. In the nexts steps, you should identify the dates, time and deadlines agreed between the meeting participants always as possible.

        The output format must follow the basic structure shown below:

            ```
            Resumo da chamada:
            <the generated summary>

            Próximos passos acordados:
            <the generated next steps. Use structured data such as lists, tables and bullet points always as possible>
            ```

        Do not respond with anything outside of the call transcript. If you don't know, answer with "I don't know"
        The output must be in brazilian portuguese.
    """

    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=llm, prompt=map_prompt)

    # Reduce
    reduce_template = """The following is set of summaries about a sales call transcriptions:
    "{call_transcriptions_summaries}"
    Take these and distill it into a final, consolidated summary following the instructions below:
    
    You are a helpful assistant that helps a sales rep to summarize information from the sales call transcriptions provided above.
    You have two specific goals:
            
        1. Write a summary from the perspective of the sales rep that will highlight key points that will be relevant to making this sale.
            
        2. Identify the next steps agreed between the sales meeting participants from the perspective of the sales rep that will highlight 
        key points that will be relevant to making this sale. In the nexts steps, you should identify the dates, time and deadlines agreed between the meeting participants always as possible.

    The output format must follow the basic structure shown below:

        ```
        Resumo da chamada:
        <the generated summary>

        Próximos passos acordados:
        <the generated next steps. Use structured data such as lists, tables and bullet points always as possible>
        ```

    Do not respond with anything outside of the call transcript. If you don't know, answer with "I don't know"
    The output must be in brazilian portuguese.
    """

    reduce_prompt = PromptTemplate.from_template(reduce_template)
    reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

    # Takes a list of documents, combines them into a single string, and passes this to a LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="call_transcriptions_summaries"
    )

    # Combines and iteratively reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=4000,
    )

    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="docs",
        # Return the results of the map steps in the output
        return_intermediate_steps=False,
    )

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    split_docs = text_splitter.split_documents(docs)

    return map_reduce_chain.invoke(split_docs)


def stuff_summarize(docs: List[dict], llm: ChatOpenAI):
    # Define prompt
    prompt_template = """
        You are a helpful assistant that helps a sales rep to summarize information from the following sales call transcriptions: {call_transcriptions}.
        You have two specific goals:
            
            1. Write a summary from the perspective of the sales rep that will highlight key points that will be relevant to making this sale.
            
            2. Identify the next steps agreed between the sales meeting participants from the perspective of the sales rep that will highlight 
            key points that will be relevant to making this sale. In the nexts steps, you should identify the dates, time and deadlines agreed between the meeting participants always as possible.

        The output format must follow the basic structure shown below:

            ```
            Resumo da chamada:
            <the generated summary>

            Próximos passos acordados:
            <the generated next steps. Use structured data such as lists, tables and bullet points always as possible>
            ```

        Do not respond with anything outside of the call transcript. If you don't know, answer with "I don't know"
        The output must be in brazilian portuguese.
    """

    prompt = PromptTemplate.from_template(prompt_template)

    # Define LLM chain
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Define StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain, document_variable_name="call_transcriptions"
    )

    return stuff_chain.invoke(docs)


def summarize_file(file, summarization_mode):
    file_path = save_file(file)

    file_loader = UnstructuredFileLoader(file_path)
    file_docs = file_loader.load()

    llm = ChatOpenAI(
        model="gpt-4-turbo-preview",
    )

    with get_openai_callback() as cb:
        if summarization_mode == "Map-Reduce":
            st.write(
                map_reduce_summarize(docs=file_docs, llm=llm)["output_text"],
            )
            st.write(cb)
        elif summarization_mode == "Refine":
            st.write(
                refine_summarize(docs=file_docs, llm=llm)["output_text"],
            )
            st.write(cb)
        else:
            st.write(
                stuff_summarize(docs=file_docs, llm=llm)["output_text"],
            )
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
                "Insere a transcrição inteira no contexto do modelo de linguagem (LLM). Ideal para transcrições com até 500k palavras. (GPT-4-Turbo)",
                'Primeiro resume os trechos da transcrição separadamente e depois "combina" todos os resumos em um resumo final. Ideal para transcrições com mais de 500k palavras. (GPT-4-Turbo)',
                "Gera o resumo iterativamente refinando o resumo a cada trecho da ligação, é um pouco mais barato que o modo Map-Reduce. Ideal para transcrições com mais de 500k palavras. (GPT-4-Turbo)",
            ],
            label_visibility="collapsed",
        )

    with st.spinner("Gerando o resumo do documento, pode levar alguns minutos..."):
        if uploaded_file:
            summarize_file(uploaded_file, summarization_mode)
