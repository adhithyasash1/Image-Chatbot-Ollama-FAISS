# vector_store.py

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List
import config

def create_vector_store(text_chunks: List[str]):
    """Creates a FAISS vector store from text chunks."""
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
    vectorstore = FAISS.from_documents(documents=text_chunks, embedding=embeddings)
    return vectorstore