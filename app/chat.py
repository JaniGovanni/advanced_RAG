from typing import List, Union, Optional, Dict
from langchain_core.language_models import BaseLanguageModel
from app.vectorstore import get_chroma_store_as_retriever
import app.llm
import app.RAG_techniques as RAG_techniques




# every tag needs a seperate chat history: tag needs to be stored in FileChatMessageHistory
# there should be an option to delete the history for a tag (or create an extra one)
# build chat should return an chain
# TODO: should be seperated in extra file
class ChatConfig:
    """Configuration class for chat settings."""

    def __init__(
        self,
        tag: str,
        expand_by_answer: bool = False,
        expand_by_mult_queries: bool = False,
        history_awareness: bool = False,
        reranking: bool = True,
        k: int = 10,
        llm_choice: str = "groq",
        use_bm25: bool = False,
        conversation_history: Optional[List[Union[Dict]]] = None
    ):
        self.tag = tag
        self.expand_by_answer = expand_by_answer
        self.expand_by_mult_queries = expand_by_mult_queries
        self.history_awareness = history_awareness
        self.reranking = reranking
        self.k = k
        self.llm_choice = llm_choice
        self.use_bm25 = use_bm25
        self.llm = self.get_llm()
        
        self.conversation_history = conversation_history or []

    def get_llm(self) -> BaseLanguageModel:
        """Get the language model based on the chosen option."""
        return app.llm.get_ollama_llm() if self.llm_choice == "ollama" else app.llm.get_groq_llm()
      
      

from app.utils_chat import history_aware_query, retrieve_documents, additional_bm25_retrieval

def get_result_docs(chat_config: ChatConfig, query: str, retriever=None) -> tuple[List[str], str]:
    """
    Retrieve and process documents based on the given query and chat configuration.
    
    :param chat_config: Configuration for the chat session
    :param query: User's query
    :param retriever: Optional custom retriever
    :return: Tuple of (list of result texts, joint query)
    """
    retriever = retriever or get_chroma_store_as_retriever()
    search_filter = {'tag': chat_config.tag}
    
    orig_query = query
    query = history_aware_query(chat_config, query)
    result, joint_query = retrieve_documents(chat_config, query, retriever, search_filter)
    if chat_config.use_bm25:
        result = additional_bm25_retrieval(chat_config, query, retriever, result)

    result_texts = [re.page_content for re in result]

    if chat_config.reranking:
        result_texts = RAG_techniques.rerank_by_crossencoder(retrieved_docs=result_texts,
                                                             original_query=orig_query,
                                                             top_k=4)
    return result_texts, joint_query


def create_RAG_output(context, query, llm):
    """
    creates an answer of the query, based on the provided context
    :param context: list of context documents
    :param query: users query
    :param llm: llm, to answer the question
    :return:
    """
    prompt = f"""
        You are an assistant for question-answering tasks. Use the retrieved context to answer the question. The context could include table and text data. 
        If you don't know the answer, just say that you don't know and dont use informations from your own knowledge base to
        answer the questions. 
        Use three sentences maximum and keep the answer concise.
        """

    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {
            "role": "user",
            "content": f"based on the following context:\n\n{context}\n\nAnswer the query: '{query}'",
        },
    ]

    response = llm.invoke(messages)
    content = response.content
    return content









