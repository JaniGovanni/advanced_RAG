from app.chat import get_result_docs, ChatConfig, create_RAG_output
from langchain_community.llms import Ollama
from langchain_ollama import ChatOllama

query = "What is attention?"
# it is very difficult, to get the prompt for expansion_answer right, for
# general usecases, so multi query might better work
config_multi_query = ChatConfig(tag='attention',
                                expand_by_answer=False,
                                expand_by_mult_queries=True,
                                reranking=True,
                                k=10)
config_multi_query.history_awareness(False)
result_texts, joint_query = get_result_docs(config_multi_query, query=query)

context = ''.join(result_texts)
llm = ChatOllama(model="phi3.5",
                 temperature=0)
final_answer = create_RAG_output(context, query, llm)
