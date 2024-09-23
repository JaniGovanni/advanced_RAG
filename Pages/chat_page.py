import streamlit as st
from app.chat import get_result_docs, create_RAG_output
from langchain_core.messages import AIMessage, HumanMessage

st.header("chat with the docs from a tag")

# LLM selection
llm_choice = st.radio("Choose LLM", ("groq", "ollama"))
st.session_state['chat_config'].llm_choice = llm_choice
st.session_state['chat_config'].llm = st.session_state['chat_config'].get_llm()

# define rag options

st.session_state['chat_config'].history_awareness(st.checkbox("Use history awareness", value=1))
st.session_state['chat_config'].set_exp_by_answer(st.checkbox("Use HyDe to expand the similarity search by an fictional answer",
                                                              value=0))
st.session_state['chat_config'].set_mult_queries(st.checkbox("formulate multiple search querys for the similarity search",
                                                             value=0))
st.session_state['chat_config'].set_reranking(st.checkbox("Perform an reranking of the founded documents based on relevance, by an crossencoder",
                                                          value=1))
st.session_state['chat_config'].set_use_bm25(st.checkbox("Use BM25 for keyword search", value=0))

user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":

    # create a function to do this
    context, joint_query = get_result_docs(query=user_query,
                                           ChatConfig=st.session_state['chat_config'])

    context = ''.join(context)
    ai_answer = create_RAG_output(context=context,
                                  query=user_query,
                                  llm=st.session_state['chat_config'].llm)
    st.session_state['chat_config'].memory.chat_memory.messages.append(AIMessage(content=ai_answer))

    # debugging
    #st.write(joint_query)

# conversation
for message in st.session_state['chat_config'].memory.chat_memory.messages:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)
st.sidebar.page_link('main.py', label='Home')
#st.write(st.session_state['chat_config'].expand_by_mult_query)