from langchain.chains import ConversationChain
from langchain_ollama import ChatOllama
from app.memory import build_window_buffer_memory, build_conversation_buffer_memory


# rewrite this with runnables
# https://python.langchain.com/v0.2/docs/tutorials/chatbot/
# def history_aware_query_chain(memory, verbose=False):
#     """
#     creates a chain, which reformulates a users query, based on the previous
#     conversation stored in memory, as a standalone question
#     :param memory: memory object
#     :return: history_aware_query_chain
#     """
#     # currently only phi3.5
#     llm = ChatOllama(
#         model="phi3.5",
#         temperature=0,
#     )
#
#     conversation = ConversationChain(
#         llm=llm,
#         verbose=verbose,
#         memory=memory
#     )
#
#     conversation.prompt.template = """Given a chat history and the latest user question
#     which might reference context in the chat history,
#     formulate a standalone question which can be understood
#     without the chat history. Be as short as possible.
#     Do NOT answer the question, just
#     reformulate it if needed and otherwise return it as is.
#
#     Current conversation:
#     {history}
#     Human: {input}
#     AI:"""
#     return conversation

# work in progress
# class Reformulate_Query_Object:
#     def __init__(self, ChatConfig):
#         """
#         Object is used to reformulate a users query (with different advanced RAG techniques) to
#         be find better results during a similarity search.
#         :param ChatConfig: Configuration of the object
#         """
#         self.tag = ChatConfig.tag
#         self.expand_by_answer = ChatConfig.expand_by_answer
#         self.expand_by_mult_queries = ChatConfig.expand_by_mult_queries
#         self.reranking = ChatConfig.reranking
#         self.history_aware = ChatConfig.history_aware
#         self.k = ChatConfig.k
#         self.tag = ChatConfig.tag
#
#         self.llm = ChatOllama(
#             model="phi3.5",
#             temperature=0,
#         )
#         self.memory = build_window_buffer_memory(self.tag)
#         self.history_aware_chain = HistoryAwareQueryChain(self.llm, self.memory)
#

class HistoryAwareQueryChain:
    def __init__(self, memory, verbose=False):
        """
        Creates a chain that reformulates a user's query, based on the previous
        conversation stored in memory, as a standalone question.

        :param memory: memory object
        :param verbose: whether to print verbose output
        """
        self.memory = memory
        self.llm = ChatOllama(
            model="phi3.5",
            temperature=0,
        )
        # rewrite with runnables
        # https://python.langchain.com/v0.2/docs/tutorials/chatbot/
        self.conversation = ConversationChain(
            llm=self.llm,
            verbose=verbose,
            memory=self.memory
        )

        self.conversation.prompt.template = """Given a chat history and the latest user question 
        which might reference context in the chat history, 
        formulate a standalone question which can be understood 
        without the chat history. Be as short as possible.
        Do NOT answer the question, just 
        reformulate it if needed and otherwise return it as is.

        Current conversation:
        {history}
        Human: {input}
        AI:"""

    def reformulate(self, input):
        """
        Reformulates the given input query using the conversation chain and
        dont adds the output to the memory

        :param input: the input query
        :return: the reformulated query
        """
        response = self.conversation.invoke(input=input, return_only_output=True)
        # to clear the generated answer from the memory, probably there
        # is a better way to do this....
        del self.memory.chat_memory.messages[-1]
        #self.memory.chat_memory.messages.pop()
        return response['response']
