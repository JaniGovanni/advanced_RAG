from sentence_transformers import CrossEncoder
import numpy as np
from .prompts import MULTI_QUERY_PROMPT, HYDE_PROMPT_V2
from typing import List, Any, Dict
from pydantic import BaseModel, Field

class MultiQueryOutput(BaseModel):
    queries: List[str] = Field(..., min_length=5, max_length=5)


def generate_multi_query(query: str, llm: Any) -> List[str]:
    """
    Generate multiple queries similar to the provided one using an LLM.

    :param query: User's query
    :param llm: Language model to perform query expansion technique
    :return: List of generated queries
    """
    messages = [
        ("system", MULTI_QUERY_PROMPT),
        ("human", f"Original query: {query}\n\nGenerate 5 alternative queries:"),
    ]
    response = llm.invoke(messages)
    
    try:
        parsed_output = MultiQueryOutput.model_validate_json(response.content)
        return parsed_output.queries
    except Exception as e:
        print(f"Error parsing LLM output: {e}")
        # Fallback to simple splitting if parsing fails
        return response.content.split('\n')[:5]


def get_joint_query_results(retriever: Any, joint_query: List[str], filter: Dict[str, Any], k: int = 5) -> List[Any]:
    """
    Perform a similarity search in the vectorstore for a list of queries.

    :param retriever: Retriever object with a vectorstore
    :param joint_query: List of queries
    :param filter: Filter dictionary for the search
    :param k: Number of results to retrieve per query
    :return: List of unique documents
    """
    joint_query_results = []
    for query in joint_query:
        results = retriever.vectorstore.similarity_search(query, k=k, filter=filter)
        joint_query_results.extend([doc for doc in results if doc not in joint_query_results])
    return joint_query_results

def augment_query_generated(query: str, llm: Any) -> str:
    """
    Generate a hallucinated answer for the given query for HyDE search.

    :param query: User query
    :param llm: Language model to generate the fictional answer
    :return: Fictional answer to the query
    """
    messages = [
        ("system", HYDE_PROMPT_V2),
        ("human", query),
    ]
    response = llm.invoke(messages)
    return response.content

def project_embeddings(embeddings, umap_transform):
    """
    ONLY FOR TESTING PURPOSES
    Projects the given embeddings using the provided UMAP transformer.

    Args:
    embeddings (numpy.ndarray): The embeddings to project.
    umap_transform (umap.UMAP): The trained UMAP transformer.

    Returns:
    numpy.ndarray: The projected embeddings.
    """
    projected_embeddings = umap_transform.transform(embeddings)
    return projected_embeddings

def rerank_by_crossencoder(retrieved_docs: List[Any], original_query: str, top_k: int = 3) -> List[Any]:
    """
    Rerank retrieved documents using a cross-encoder model.

    :param retrieved_docs: List of retrieved documents
    :param original_query: Original user query
    :param top_k: Number of top documents to return
    :return: List of top reranked documents
    """
    try:
        cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [[original_query, doc] for doc in retrieved_docs]
        scores = cross_encoder.predict(pairs)

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [retrieved_docs[i] for i in top_indices]
    except Exception as e:
        print(f"An error occurred during reranking: {str(e)}")
        return retrieved_docs[:top_k]  # Return top k documents without reranking