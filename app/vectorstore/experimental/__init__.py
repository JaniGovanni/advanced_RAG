from langchain.vectorstores import FAISS
import os
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
from app.vectorstore import save_uuids
from app.vectorstore.embeddings import JinaEmbeddings

def get_faiss_store_as_retriever():
    """
    To easily create or load a FAISS vectorstore.
    :return: FAISS vectorstore setup with appropriate embeddings as a retriever
    """
    faiss_index_path = os.getenv("FAISS_INDEX_PATH", "faiss_index")
    
    # Try to load existing index if it exists
    if os.path.exists(faiss_index_path):
        print(f"Loading existing FAISS index from {faiss_index_path}")
        vectorstore = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        print(f"Creating new FAISS index")
        embeddings = JinaEmbeddings()
        index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
        vectorstore = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

    return vectorstore.as_retriever()


def add_docs_to_faiss_store(retriever, docs):
    """
    Adds a list of documents to the FAISS vectorstore and saves it to disk
    :param retriever: The retriever object containing the vectorstore
    :param docs: List of documents to add
    """
    vectorstore = retriever.vectorstore
    
    text_embeddings = [(doc.page_content, doc.metadata['embedding']) for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    ids = [str(uuid4()) for _ in range(len(docs))]
    save_uuids(docs[0].metadata['source'], ids, retriever.vectorstore)
    vectorstore.add_embeddings(text_embeddings, metadatas, ids)
    
    # Save the updated index
    faiss_index_path = os.getenv("FAISS_INDEX_PATH", "faiss_index")
    vectorstore.save_local(faiss_index_path)
    print(f"FAISS index saved to {faiss_index_path}")
