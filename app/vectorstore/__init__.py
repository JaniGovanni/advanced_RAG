from dotenv import load_dotenv
from langchain_chroma import Chroma
import os
from uuid import uuid4
from langchain_community.vectorstores.utils import filter_complex_metadata
from .embeddings import get_ollama_embeddings, JinaEmbeddings
from app.source_handling import save_uuids

load_dotenv()


def get_chroma_store_as_retriever(embeddings=None):
    """
    To easily create a vectorstore.
    :return: chroma vectorstore setup with appropriate embeddings
    """
    if embeddings is None:
        embeddings = get_ollama_embeddings()
    
    vectorstore = Chroma(persist_directory=os.getenv("CHROMA_PATH"),
                         embedding_function=embeddings)

    return vectorstore.as_retriever()



def add_docs_to_store(retriever, docs):
    """
    Adds a list of documents, which are part of a single source to a given vectorstore
    :param retriever: The retriever object containing the vectorstore
    :param docs: List of documents to add
    """
    uuids = [str(uuid4()) for _ in range(len(docs))]
    source = docs[0].metadata['source'] 
    save_uuids(source, uuids, retriever.vectorstore)
    
    filtered_docs = filter_complex_metadata(docs)
    retriever.vectorstore.add_documents(filtered_docs, ids=uuids)




