from langchain.vectorstores import FAISS
import os
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
from app.vectorstore import save_uuids
from app.vectorstore.embeddings import JinaEmbeddings
from dotenv import load_dotenv
from typing import List, Tuple
import numpy as np
from langchain.schema import Document

load_dotenv()

def cos_sim(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))

class CustomFAISS(FAISS):
    def similarity_search_by_cosine(self, query: str, k: int = 4, filter=None, oversampling_factor: int = 10) -> List[Tuple[Document, float]]:
        # Retrieve more candidates than needed
        candidates_k = k * oversampling_factor

        # Call the parent class method for initial search
        docs_and_scores = super().similarity_search_with_score(query, k=candidates_k, filter=filter)
        
        query_embedding = self.embedding_function.embed_query(query)
        
        # Recalculate scores using cosine similarity
        rescored_docs = []
        for doc, _ in docs_and_scores:
            similarity = cos_sim(query_embedding, doc.metadata['embedding'])
            rescored_docs.append((doc, similarity))
        
        # Sort by cosine similarity and return top k
        return sorted(rescored_docs, key=lambda x: x[1], reverse=True)[:k]

    def similarity_search(self, query: str, k: int = 4, filter=None) -> List[Document]:
        docs_and_scores = self.similarity_search_by_cosine(query, k, filter)
        return [doc for doc, _ in docs_and_scores]
    

def get_faiss_store_as_retriever(custom: bool = False):
    """
    To easily create or load a FAISS vectorstore.
    :param custom: If True, use CustomFAISS instead of regular FAISS
    :return: FAISS vectorstore setup with appropriate embeddings as a retriever
    """
    faiss_index_path = os.getenv("FAISS_STORE_PATH", "FAISS_STORE/faiss_index")
    embeddings = JinaEmbeddings()
    
    # Try to load existing index if it exists
    if os.path.exists(faiss_index_path):
        print(f"Loading existing FAISS index from {faiss_index_path}")
        vectorstore_class = CustomFAISS if custom else FAISS
        vectorstore = vectorstore_class.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
    else:
        print(f"Creating new FAISS index")
        # to get embed dim
        index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
        vectorstore_class = CustomFAISS if custom else FAISS
        vectorstore = vectorstore_class(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        vectorstore.save_local(faiss_index_path)
        print(f"New FAISS index saved to {faiss_index_path}")

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
    save_uuids(docs[0].metadata['source'],
               ids,
               retriever.vectorstore,
               file_path=os.getenv("FAISS_STORE_PATH") + "/source_to_id.json")
    vectorstore.add_embeddings(text_embeddings, metadatas, ids)
    
    # Save the updated index
    faiss_index_path = os.getenv("FAISS_STORE_PATH", "FAISS_STORE/faiss_index")
    vectorstore.save_local(faiss_index_path)
    print(f"FAISS index saved to {faiss_index_path}")
