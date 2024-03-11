from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

from dotenv import load_dotenv, find_dotenv

load_dotenv((find_dotenv()))


loader = TextLoader("sample_product_catalog.txt")
documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

index_name = "reppie-labs-sales-agent"
PineconeVectorStore.from_documents(docs, embeddings, index_name=index_name)
