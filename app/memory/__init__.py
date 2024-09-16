from langchain.memory import (ConversationBufferWindowMemory,
                              ConversationBufferMemory)
from langchain_community.chat_message_histories import FileChatMessageHistory
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path='/app/.env')


# not needed right now. It is better, to store memory in session state
def build_window_buffer_memory(tag):
    storage_path = os.getenv("MEMORY_STORAGE_PATH")
    return ConversationBufferWindowMemory(
        memory_key="history",
        output_key="response",
        return_messages=True,
        chat_memory=FileChatMessageHistory(storage_path + tag + '.json'),
        k=4
    )


def build_conversation_buffer_memory(tag):
    storage_path = os.getenv("MEMORY_STORAGE_PATH")
    return ConversationBufferMemory(
        # This is a workaround for the message storing and retrieving,
        # since the custom class (storing and retrieving from database)
        # doesnt work the opportunity for the custom implementation of the messages Property seems to be buggy
        chat_memory=FileChatMessageHistory(storage_path + tag + '.json'),
        return_messages=True,
        memory_key="chat_history",
        output_key="answer"
    )