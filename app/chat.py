from app.vectorstore import get_chroma_store_as_retriever, get_documents_by_tag
from langchain_community.llms import Ollama
import app.llm
from app.chains import HistoryAwareQueryChain
from app.memory import build_window_buffer_memory, build_conversation_buffer_memory
import app.RAG_techniques as RAG_techniques
from langchain.memory import (ConversationBufferWindowMemory,
                              ConversationBufferMemory)
from langchain_core.messages import HumanMessage
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
import time



# every tag needs a seperate chat history: tag needs to be stored in FileChatMessageHistory
# there should be an option to delete the history for a tag (or create an extra one)
# build chat should return an chain
class ChatConfig:
    tag: str
    expand_by_answer: bool
    expand_by_mult_queries: bool
    reranking: bool
    history_aware: bool
    k: int
    memory: ConversationBufferWindowMemory | ConversationBufferWindowMemory
    history_chain: HistoryAwareQueryChain
    llm_choice: str  # Add this line
    bm25_retriever: BM25Retriever

    def __init__(self,
                 tag,
                 expand_by_answer=False,
                 expand_by_mult_queries=False,
                 reranking=True,
                 llm=None,
                 k=10,
                 llm_choice="groq",
                 use_bm25=False):
        self.tag = tag
        self.expand_by_answer = expand_by_answer
        self.expand_by_mult_queries = expand_by_mult_queries
        self.reranking = reranking
        self.k = k
        self.llm_choice = llm_choice
        self.use_bm25 = use_bm25
        if llm:
            self.llm = llm
        else:
            self.llm = self.get_llm()  
        #self.memory = build_window_buffer_memory(tag=self.tag)
        self.memory = ConversationBufferWindowMemory(memory_key="history",
                                                     output_key="response",
                                                     return_messages=True,
                                                     k=4)

    def history_awareness(self, yes):
        if yes:
            self.history_aware = True
            self.history_chain = HistoryAwareQueryChain(memory=self.memory, llm=self.llm)
        else:
            self.history_aware = False

    def set_use_bm25(self, flag):
        self.use_bm25 = flag
    def set_mult_queries(self, flag):
        self.expand_by_mult_queries = flag

    def set_exp_by_answer(self, flag):
        self.expand_by_answer = flag

    def set_reranking(self, flag):
        self.reranking = flag
    def get_llm(self):
        if self.llm_choice == "ollama":
            return app.llm.get_ollama_llm()
        else:
            return app.llm.get_groq_llm()


def get_result_docs(ChatConfig, query):
    retriever = get_chroma_store_as_retriever()
    # not necessary
    #ChatConfig.memory.chat_memory.add_user_message(query)
    search_filter = {'tag': ChatConfig.tag}
    orig_query = query
    print(ChatConfig.tag)

    if ChatConfig.history_aware:
        query = ChatConfig.history_chain.reformulate(input=orig_query)
    else:
        # in case the chain doesnt append the query to the messageHistory
        ChatConfig.memory.chat_memory.messages.append(HumanMessage(content=orig_query))
    

    if ChatConfig.expand_by_mult_queries:
        generated_queries = RAG_techniques.generate_multi_query(query=query,
                                                                llm=ChatConfig.llm)
        joint_query = [query] + generated_queries
        result = RAG_techniques.get_joint_query_results(retriever=retriever,
                                                        joint_query=joint_query,
                                                        filter=search_filter,
                                                        k=ChatConfig.k)
    elif ChatConfig.expand_by_answer:
        hypothetical_answer = RAG_techniques.augment_query_generated(query=query,
                                                                     llm=ChatConfig.llm)
        joint_query = f"{query} {hypothetical_answer}"
        result = retriever.vectorstore.similarity_search(joint_query,
                                                         k=ChatConfig.k,
                                                         filter=search_filter)
    else:
        result = retriever.vectorstore.similarity_search(query,
                                                         k=ChatConfig.k,
                                                         filter=search_filter)
        joint_query = query
    if ChatConfig.use_bm25:
        # it isnt optimal to do this for every query. Better is in the init method
        # of chatConfig. Adjust this.
        #start_time = time.time()
        all_docs = get_documents_by_tag(retriever, ChatConfig.tag)
        bm25_retriever = BM25Retriever.from_documents([Document(page_content=doc.page_content, metadata=doc.metadata) for doc in all_docs])
        end_time = time.time()
        #print(f"Time taken to create BM25Retriever: {end_time - start_time} seconds")
        
        # Combine results from both retrievers
        bm25_results = bm25_retriever.get_relevant_documents(query)
        
        # Merge and deduplicate results
        combined_results = list({doc.page_content: doc for doc in result + bm25_results}.values())
        result = combined_results[:ChatConfig.k]


    result_texts = [re.page_content for re in result]

    if ChatConfig.reranking:
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









