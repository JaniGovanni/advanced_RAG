from langchain_community.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from app.chains import HistoryAwareQueryChain
from app.RAG_techniques import generate_multi_query, augment_query_generated
from app.custom_conversations import chat_history_5 as chat_history

input = chat_history[-1]
context = chat_history[:-1]

memory = ConversationBufferMemory()
for i in context:
  if i["type"] == "human":
    memory.chat_memory.add_user_message(i["data"])
  elif i["type"] == "ai":
    memory.chat_memory.add_ai_message(i["data"])

llm = ChatOllama(
        model="phi3.5",
        temperature=0,
    )
summary_query = HistoryAwareQueryChain(memory, verbose=True)
reformulated_query = summary_query.reformulate(input=input['data'])

multi_queries = generate_multi_query(llm=llm, query=reformulated_query)
#augmented_query = augment_query_generated(query=reformulated_query['response'], llm=llm)


