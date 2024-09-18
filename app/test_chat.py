from app.chat import get_result_docs, ChatConfig, create_RAG_output
import app.llm
import os
import dotenv


dotenv.load_dotenv('app/.env')

query = "What is attention?"
# it is very difficult, to get the prompt for expansion_answer right, for
# general usecases, so multi query might better work
config_multi_query = ChatConfig(tag='attention',
                                k=10,
                                llm=app.llm.get_groq_llm()
                                )
config_multi_query.history_awareness(False)
config_multi_query.set_mult_queries(True)
config_multi_query.set_reranking(True)
config_multi_query.set_exp_by_answer(False)

result_texts, joint_query = get_result_docs(config_multi_query, query=query)

context = ''.join(result_texts)

final_answer = create_RAG_output(context, query, config_multi_query.llm)
