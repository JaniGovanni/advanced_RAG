

import os
import streamlit as st
from app.vectorstore import get_chroma_store_as_retriever, get_stored_files_and_tags, get_stored_tags_and_files
from app.doc_processing import ProcessDocConfig
from app.chat import ChatConfig

def save_file(file):
    """
    saves the uploaded file in a specified directory
    """
    if not os.path.exists('uploaded_files'):
        os.makedirs('uploaded_files')
    filepath = os.path.join('uploaded_files', file.name)
    with open(filepath, 'wb') as f:
        f.write(file.getbuffer())
    st.success('file saved.')
    return filepath


st.header("File Upload")
retriever = get_chroma_store_as_retriever()
tag_to_files = get_stored_tags_and_files(retriever)
if "tag" not in st.session_state:
    st.session_state['tag'] = ''


st.session_state['tag'] = st.text_input("define a source tag",
                                        placeholder="tag")

if st.session_state['tag']:
    st.subheader("upload a file or define an URL")
    col1, col2 = st.columns(2)
    with col1:
        file = st.file_uploader("Upload File (.pdf, .docx, .pptx, .xlsx,...)")
        if file is not None:
            filepath = save_file(file)
            st.session_state['process_config'] = ProcessDocConfig(tag=st.session_state['tag'],
                                                                  filepath=filepath)
            st.write(":green[Go to the next page shown in the sidebar]")
            st.sidebar.page_link('pages/second_page.py', label='document processing config')
            # debugging
            print(st.session_state['process_config'].source)
            print(st.session_state['process_config'].filepath)
            print(st.session_state['process_config'].tag)


    with col2:
        url = st.text_input("or define an url. PLEASE DONT DO BOTH", placeholder='url')
        if url:
            st.session_state['process_config'] = ProcessDocConfig(tag=st.session_state['tag'],
                                                                  url=url)
            st.write(":green[Go to the next page shown in the sidebar]")
            st.sidebar.page_link('pages/second_page.py', label='document processing config')

            # debugging
            print(st.session_state['process_config'].source)

#st.write("\n\n")
#st.subheader("Files and their tags in the vectorstore:")

tag_to_button = {}
for i, tag in enumerate(tag_to_files.keys()):
    col_tag, col_file, col_chat = st.columns(3)
    with col_tag:
        if i == 0:
            st.subheader("tags:")
        st.write(tag)
    with col_file:
        if i == 0:
            st.subheader("Files:")
        for file in tag_to_files[tag]:
            st.write(file)
    with col_chat:
        if i == 0:
            st.subheader("Chat with this tag")
        tag_to_button[tag] = st.button("chat", key=tag)

    st.write("\n")

for tag, clicked in tag_to_button.items():
    if clicked:
        st.write(f":green[To chat about {tag} go to the page shown in the sidebar]")
        st.session_state['chat_config'] = ChatConfig(tag=tag, llm_choice="groq")  # Set initial choice to "groq"
        st.sidebar.page_link('pages/chat_page.py', label='chatting page')

# URI = 'bolt://localhost'   # neo4j:// also works
# AUTH = ('neo4j', '12345678')
# from neo4j import GraphDatabase
#
# with GraphDatabase.driver(URI, auth=AUTH) as driver:
#     driver.verify_connectivity()


