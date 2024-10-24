

import os
import streamlit as st
from pages.utils import upload_file
from dotenv import load_dotenv
from app.api_setup.dataclasses_api import ProcessDocConfigAPI, ChatConfigAPI
from pages.utils import get_stored_tags_and_files

load_dotenv()


st.header("File Upload")
tag_to_files = get_stored_tags_and_files()
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
            filepath = upload_file(file)
            if filepath:
                st.session_state['process_config'] = ProcessDocConfigAPI(tag=st.session_state['tag'],
                                                                         filepath=filepath)
                st.write(":green[Go to the next page shown in the sidebar]")
                st.sidebar.page_link('pages/second_page.py', 
                                     label='document processing config')
                # debugging
                #print(st.session_state['process_config'].filepath)
                #print(st.session_state['process_config'].tag)


    with col2:
        url = st.text_input("or define an url. PLEASE DONT DO BOTH", placeholder='url')
        if url:
            st.session_state['process_config'] = ProcessDocConfigAPI(tag=st.session_state['tag'],
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
        st.session_state['chat_config'] = ChatConfigAPI(tag=tag, llm_choice="groq")  
        st.sidebar.page_link('pages/chat_page.py', label='chatting page')


