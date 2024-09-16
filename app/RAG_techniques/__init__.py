import re
from sentence_transformers import CrossEncoder
import numpy as np

#=====intersting Prompts======
# for multi query
template = """You are an AI language model assistant. Your task is to generate five 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""

# for HyDE
prompt = example_prompt = prompt = """
    Assume you have access to a hypothetical document. 
    This could be a website, a scientific paper, a report, a Wikipedia article, or anything else. 
    Your task is to provide an text snippet from such a hypothetical document 
    that could contain the answer 
    to a user's question. Please dont output anything else. Here is an example:

    user_question_1: Is Discretization necessary?
    
    answer_1: However, the parameterization of Mamba still used the same discretization 
    step as in prior structured SSMs, where there is another parameter Δ being modeled. 
    We do this because the discretization step has other side effects such as properly normalizing the activations 
    which is important for performance.
    The initializations and parameterizations from the 
    previous  SSMs still work out-of-the-box, so why fix what’s not broken?
    Despite this, we’re pretty sure that the discretization step isn’t necessary for Mamba. 
    In the Mamba-2 paper, we chose to work directly with the “discrete parameters” 𝐴 and 𝐵, 
    which in all previous structured SSM papers (including Mamba-1) were denoted
    
    user_question_2: What is the Hooded Man?
    
    answer_2: The Hooded Man (or The Man on the Box)[1] is an image showing 
    a prisoner at Abu Ghraib prison with wires attached to his fingers, 
    standing on a box with a covered head. The photo has been portrayed as an iconic 
    photograph of the Iraq War, "the defining image of the scandal" 
    and "symbol of the torture at Abu Ghraib". The image was published on the 
    cover of The Economist's 8 May 2004 issue, the opening photo of The New 
    Yorker on 10 May 2004, and on 11 March 2006 in The New York Times's 
    first section at the top left-hand corner.
    """

def generate_multi_query(query, llm):
    """
    Uses a llm, to generate multiple queries,
    which should be similar to the provided one.
    :param query: users query
    :param llm: llm, to perform query expansion technique
    :return: list of generated queries
    """
    # general formulated, works ok.
    example_prompt = """
    You are a knowledgeable research assistant. 
    Your users are inquiring about a specific topic. 
    For the given question, propose up to five related questions to assist them
    in finding the information they need. 
    Provide concise, single-topic questions (without compounding sentences)
    that cover various aspects of the topic. 
    Ensure each question is complete and directly related 
    to the original inquiry. List each question on a separate line. Do Not number or index
    them.
    """

    messages = [
        (
            "system",
            example_prompt
        ),
        (
            "human",
            query
        ),
    ]
    response = llm.invoke(messages)
    content = response.content
    # split generated queries
    # phi 3.5 create indices in every case......
    content = re.findall(r'\d+\.\s+(.*)', content, re.MULTILINE)
    return content


def get_joint_query_results(retriever, joint_query, filter: dict, k=5):
    """
    Performs a similarity search in the vectorstore, for a list of queries.
    Duplicate founded documents gets filtered. Is part of the query-expansion /
    multi query technique
    """
    joint_query_results = []
    for query in joint_query:
        results = retriever.vectorstore.similarity_search(query,
                                                          k=k,
                                                          filter=filter)
        for doc in results:
            if doc not in joint_query_results:
                joint_query_results.append(doc)

    return joint_query_results

def augment_query_generated(query, llm):
    """
    Generates an hallucinated answer for the given query, which can be
    used to perform HyDe search. HyDe search is a RAG-technique, where a
    query is enhanced by an fictional answer, to increase the cosine similarity
    between search-query and context
    :param query: user query
    :param llm: llm, to generate the fictional answer
    :return: the fictional answer to the query
    """
    # general formulated, works ok.
    example_prompt = """
    You are a knowledgeable and helpful research assistant.
    Provide an example answer to the given question,
    that could be found in a scholarly or professional document or an 
    scientific paper, and 
    might actually be true.
    """

    messages = [
        (
            "system",
            example_prompt
        ),
        (
            "human",
            query
        ),
    ]
    response = llm.invoke(messages)
    content = response.content
    return content

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

def rerank_by_crossencoder(retrieved_docs, original_query, top_k=3):
    """
    Uses an crossencoder (from sentence-transformers), to determine if the query
    is connected to the retrieved documents. After this, the documents, which are
    most connected to the original query gets returned
    :param retrieved_docs: by an similarity search founded docs
    :param original_query: The original user query
    :param top_k: How many docs should be returned
    :return: top_k to the query most connected documents
    """

    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    pairs = [[original_query, doc] for doc in retrieved_docs]
    scores = cross_encoder.predict(pairs)

    top_indices = np.argsort(scores)[::-1][:top_k]
    top_documents = [retrieved_docs[i] for i in top_indices]
    return top_documents

