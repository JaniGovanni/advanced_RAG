import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from pages.utils import api_get_result

def clear_conversation():
    if 'conversation' in st.session_state:
        del st.session_state['conversation']
if st.sidebar.button('Clear Conversation'):
    clear_conversation()
st.sidebar.page_link('main.py', label='Home')

st.header("chat with the docs from a tag")

# LLM selection
llm_choice = st.radio("Choose LLM", ("groq", "ollama"))
st.session_state['chat_config'].llm_choice = llm_choice

# define rag options
st.session_state['chat_config'].history_awareness = st.checkbox("Use history awareness", value=1)
st.session_state['chat_config'].expand_by_answer = st.checkbox("Use HyDe to expand the similarity search by an fictional answer", value=0)
st.session_state['chat_config'].expand_by_mult_queries = st.checkbox("formulate multiple search querys for the similarity search", value=0)
st.session_state['chat_config'].reranking = st.checkbox("Perform an reranking of the founded documents based on relevance, by an crossencoder", value=1)
st.session_state['chat_config'].use_bm25 = st.checkbox("Use BM25 for keyword search", value=0)

user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    if 'conversation' not in st.session_state:
        st.session_state['conversation'] = []

    st.session_state['conversation'].append(HumanMessage(content=user_query))
    st.session_state['chat_config'].conversation_history = st.session_state['conversation']
    
    api_response = api_get_result(user_query, st.session_state['chat_config'].model_dump())
    
    if api_response:
        context = ''.join(api_response['result_texts'])
        joint_query = api_response['joint_query']
        # debugging
        st.write(joint_query)
        ai_answer = api_response['rag_output']
        st.session_state['conversation'].append(AIMessage(content=ai_answer))


for message in st.session_state.get('conversation', []):
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)
#st.write(st.session_state['chat_config'].expand_by_mult_query)