from langchain_ollama import ChatOllama
import os
import dotenv
from langchain_groq import ChatGroq

dotenv.load_dotenv()
def get_ollama_llm():
    return ChatOllama(model="phi3.5", temperature=0)

def get_groq_llm():
    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        temperature=0,
        api_key=os.environ.get('GROQ_API_KEY')
    )
    return llm

