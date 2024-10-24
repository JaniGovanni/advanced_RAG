import app.RAG_techniques as RAG_techniques
from langchain_core.messages import HumanMessage
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from app.vectorstore import get_documents_by_tag
from app.chat import ChatConfig
from app.chains import HistoryAwareQueryChain

def history_aware_query(chat_config: ChatConfig, query: str) -> str:
    """
    Process the query based on the chat configuration.

    :param chat_config: Configuration for the chat session
    :param query: Original user query
    :return: Processed query
    """
    if chat_config.history_awareness and chat_config.conversation_history:
        history_chain = HistoryAwareQueryChain(chat_config.llm)
        processed_query = history_chain.reformulate(chat_config.conversation_history, query)
    else:
        processed_query = query
        # Append the query to the conversation history
        #chat_config.conversation_history.append({"type": "human", "content": query})
    return processed_query


def retrieve_documents(chat_config: ChatConfig, query: str, retriever, search_filter: dict) -> Tuple[List[Document], str]:
    """
    Retrieve documents based on the chat configuration and query.

    :param chat_config: Configuration for the chat session
    :param query: Processed user query
    :param retriever: Retriever object to use for document retrieval
    :param search_filter: Filter to apply to the search
    :return: Tuple of (list of retrieved documents, joint query used)
    """
    
    if chat_config.expand_by_mult_queries:
        generated_queries = RAG_techniques.generate_multi_query(query=query, llm=chat_config.llm)
        joint_query = [query] + generated_queries
        result = RAG_techniques.get_joint_query_results(
            retriever=retriever,
            joint_query=joint_query,
            filter=search_filter,
            k=chat_config.k
        )
    elif chat_config.expand_by_answer:
        hypothetical_answer = RAG_techniques.augment_query_generated(query=query,
                                                                    llm=chat_config.llm)
        joint_query = f"{query} {hypothetical_answer}"
        result = retriever.vectorstore.similarity_search(
            joint_query,
            k=chat_config.k,
            filter=search_filter
        )
    
    else:
        result = retriever.vectorstore.similarity_search(
            query,
            k=chat_config.k,
            filter=search_filter
        )
        joint_query = query
    return result, joint_query

def additional_bm25_retrieval(chat_config: ChatConfig,
                              query: str, 
                              retriever,
                              result: List[Document]) -> Tuple[List[Document], str]:
    # TODO: suboptimal to do this for every query.
    all_docs = get_documents_by_tag(retriever, chat_config.tag)
    bm25_retriever = BM25Retriever.from_documents([Document(page_content=doc.page_content, metadata=doc.metadata) for doc in all_docs])
        
    # Combine results from both searches
    bm25_results = bm25_retriever.get_relevant_documents(query)
        
    # Merge and deduplicate results
    combined_results = list({doc.page_content: doc for doc in result + bm25_results}.values())
    result = combined_results[:chat_config.k]
    return result
