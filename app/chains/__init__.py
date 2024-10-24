from langchain.chains import ConversationChain
from langchain_ollama import ChatOllama
from app.memory import build_window_buffer_memory, build_conversation_buffer_memory
from langchain.prompts import PromptTemplate
from langchain.schema import BaseMemory
from langchain_core.messages import HumanMessage, AIMessage
import traceback

# class HistoryAwareQueryChain:
#     def __init__(self, memory: BaseMemory, llm, verbose: bool = False):
#         """
#         Creates a chain that reformulates a user's query based on conversation history.

#         :param memory: Memory object to store conversation history
#         :param llm: Language model to use for reformulation
#         :param verbose: Whether to print verbose output
#         """
#         self.memory = memory
#         self.llm = llm

#         prompt_template = """
#         Given a chat history and the latest user question, formulate a standalone question
#         that can be understood without the chat history. Be concise.
#         Do NOT answer the question, just reformulate if needed or RETURN the question as is.

#         Current conversation:
#         {history}
#         Human: {input}
#         AI:
#         """
#         self.conversation = ConversationChain(
#             llm=self.llm,
#             verbose=verbose,
#             memory=self.memory,
#             prompt=PromptTemplate.from_template(prompt_template)
#         )

#     def reformulate(self, input_query: str) -> str:
#         """
#         Reformulates the given input query using the conversation chain.

#         :param input_query: The input query to reformulate
#         :return: The reformulated query
#         """
#         response = self.conversation.invoke(input=input_query, return_only_output=True)
#         self._clear_last_memory_entry()
#         return response['response']

#     def _clear_last_memory_entry(self):
#         """Removes the last entry from the memory to avoid storing the reformulated query."""
#         if hasattr(self.memory, 'chat_memory') and self.memory.chat_memory.messages:
#             self.memory.chat_memory.messages.pop()

#     def _print_memory_messages(self):
#         """Prints all messages in the memory for debugging purposes."""
#         print("--- Debug: Memory Messages ---")
#         if hasattr(self.memory, 'chat_memory'):
#             for message in self.memory.chat_memory.messages:
#                 print(f"{message.type}: {message.content}")
#         else:
#             print("Memory does not have chat_memory attribute")
#         print("-------------------------------")

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from typing import List, Dict
from langchain_core.messages import HumanMessage, AIMessage, messages_from_dict

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableWithMessageHistory
from typing import List, Dict, Callable
from langchain_core.messages import messages_from_dict

class HistoryAwareQueryChain:
    def __init__(self, llm):
        """
        Creates a runnable that reformulates a user's query based on conversation history.

        :param llm: Language model to use for reformulation
        """
        self.llm = llm

        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an AI assistant specialized in reformulating user queries based on conversation context. Your task is to:

            1. Analyze the given chat history and the latest user question.
            2. Determine if the question needs reformulation based on context from the chat history.
            3. If reformulation is needed, create a standalone question that incorporates necessary context.
            4. If the original question is clear and self-contained, return it unchanged.

            Guidelines:
            - Be concise and precise in your reformulation.
            - Maintain the original intent of the user's question.
            - Do NOT answer the question or provide any additional information.
            - Focus solely on creating a clear, context-aware question.

            Example:
            History: "User asked about Python libraries for data analysis."
            User: "What about visualization?"
            Reformulation: "What are some Python libraries for data visualization?"

            Your output should be ONLY the reformulated or original question, nothing else.
            """),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])

        self.runnable = prompt | llm

        self.runnable_with_history = RunnableWithMessageHistory(
            self.runnable,
            lambda session_id: [],  # Placeholder for session history
            input_messages_key="input",
            history_messages_key="history",
        )

    def reformulate(self, conversation_history: List[Dict], input_query: str) -> str:
        """
        Reformulates the given input query using the conversation history.

        :param conversation_history: List of previous conversation messages
        :param input_query: The input query to reformulate
        :return: The reformulated query
        """
        
        # Convert the list of dictionaries to a list of Message objects
        try:
            formatted_history = []
            # last message is the current user query, so we don't need it
            for message in conversation_history[:-1]:
                if message['type'] == 'human':
                    formatted_history.append(HumanMessage(content=message['content']))
                elif message['type'] == 'ai':
                    formatted_history.append(AIMessage(content=message['content']))
            response = self.runnable.invoke(
                {"input": input_query, "history": formatted_history}
            )
            return response.content.strip()
        except Exception as e:
            error_message = f"Error in reformulate method:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            print(error_message)
            # You might want to log this error or handle it in a way that fits your application
            raise  # Re-raise the ex