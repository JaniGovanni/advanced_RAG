from langchain.chains import ConversationChain
from langchain_ollama import ChatOllama
from app.memory import build_window_buffer_memory, build_conversation_buffer_memory
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMemory

class HistoryAwareQueryChain:
    def __init__(self, memory: BaseMemory, llm, verbose: bool = False):
        """
        Creates a chain that reformulates a user's query based on conversation history.

        :param memory: Memory object to store conversation history
        :param llm: Language model to use for reformulation
        :param verbose: Whether to print verbose output
        """
        self.memory = memory
        self.llm = llm

        prompt_template = """
        Given a chat history and the latest user question, formulate a standalone question
        that can be understood without the chat history. Be concise.
        Do NOT answer the question, just reformulate if needed or return as is.

        Current conversation:
        {history}
        Human: {input}
        AI:
        """

        self.conversation = ConversationChain(
            llm=self.llm,
            verbose=verbose,
            memory=self.memory,
            prompt=PromptTemplate.from_template(prompt_template)
        )

    def reformulate(self, input_query: str) -> str:
        """
        Reformulates the given input query using the conversation chain.

        :param input_query: The input query to reformulate
        :return: The reformulated query
        """
        response = self.conversation.invoke(input=input_query, return_only_output=True)
        self._clear_last_memory_entry()
        return response['response']

    def _clear_last_memory_entry(self):
        """Removes the last entry from the memory to avoid storing the reformulated query."""
        if hasattr(self.memory, 'chat_memory') and self.memory.chat_memory.messages:
            self.memory.chat_memory.messages.pop()
