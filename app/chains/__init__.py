from langchain_core.messages import HumanMessage, AIMessage
import traceback
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict
from langchain_core.messages import HumanMessage, AIMessage

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from typing import List, Dict

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
    
            raise  