import app.llm
from langchain.memory import ConversationBufferMemory
from app.chains import HistoryAwareQueryChain
from app.RAG_techniques import generate_multi_query, augment_query_generated
from app.custom_conversations import chat_history_4 as chat_history


input = chat_history[-1]
context = chat_history[:-1]

memory = ConversationBufferMemory()
for i in context:
  if i["type"] == "human":
    memory.chat_memory.add_user_message(i["data"])
  elif i["type"] == "ai":
    memory.chat_memory.add_ai_message(i["data"])

#llm = app.llm.get_ollama_llm()
llm = app.llm.get_groq_llm()
summary_query = HistoryAwareQueryChain(memory, verbose=True, llm=llm)
reformulated_query = summary_query.reformulate(input=input['data'])

multi_queries = generate_multi_query(llm=llm, query=reformulated_query)
#augmented_query = augment_query_generated(query=reformulated_query['response'], llm=llm)


