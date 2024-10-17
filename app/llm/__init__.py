from langchain_ollama import ChatOllama
import os
import dotenv
from langchain_groq import ChatGroq
import app.llm
from transformers import AutoTokenizer

dotenv.load_dotenv()
def get_ollama_llm():
    return ChatOllama(#model="phi3.5",
                      model="llama3.2",
                      temperature=0,
                      base_url=os.getenv('OLLAMA_BASE_URL'))

def get_groq_llm():
    llm = ChatGroq(
        model="llama-3.1-70b-versatile",
        #model="llama-3.2-90b-text-preview",
        temperature=0,
        api_key=os.environ.get('GROQ_API_KEY')
    )
    return llm

def get_model_type(llm):
    return getattr(llm, '_llm_type', None)

# currently unused
def get_model_tokenizer(llm):
    """
    loads the tokenizer for the given llm through huggingface
    """
    # better store this as json or something like that
    llm_to_tokenizer = {
        'chat-ollama': "microsoft/Phi-3.5-mini-instruct",
        'chat-groq': "mattshumer/Reflection-Llama-3.1-70B"  # needed to login to get original llama3.1 tokenizer
    }
    
    
    return AutoTokenizer.from_pretrained(llm_to_tokenizer[get_model_type(llm)])

