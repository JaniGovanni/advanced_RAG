import streamlit as st
from app.doc_processing import process_doc
from app.doc_processing.filters import (unwanted_titles_list_default,
                                        unwanted_categories_default)
from app.vectorstore import get_chroma_store_as_retriever, add_docs_to_store


if 'unwanted_titles' and 'unwanted_categories' not in st.session_state:
    st.session_state['unwanted_titles'] = unwanted_titles_list_default
    st.session_state['unwanted_categories'] = unwanted_categories_default

st.subheader("At first you have to choose some configurations for the document processing. Default should also work fine.")

st.write("Define the titles of your document that should not be transferred to the database.")


st.write("Add new unwanted title. Case does not matter.")

def submit():
    st.session_state['unwanted_titles'].append(st.session_state["new_title"])
    st.session_state.new_title = ''

new_title = st.text_input("Add a new title:",
                          # defines the name of the input string in session state
                          key='new_title',
                          placeholder="unwanted title",
                          on_change=submit)

st.session_state['process_config'].unwanted_title_list = st.multiselect("These titles are selected",
                                                                options=st.session_state['unwanted_titles'],
                                                                default=st.session_state['process_config'].unwanted_titles_list)

st.write("Define the section types of your document that should not be transferred to the database.")
st.session_state['process_config'].unwanted_categories_list = st.multiselect("These categories are selected",
                                                                      # these should be a list of all categories existing (Table, Image,...)
                                                                             options=st.session_state['unwanted_categories'],
                                                                             default=st.session_state['unwanted_categories'])
st.write(st.session_state['process_config'].unwanted_categories_list)
go_further = st.button("Click when finished")
if go_further:
    retriever = get_chroma_store_as_retriever()
    add_docs_to_store(retriever, process_doc(st.session_state['process_config']))
    st.subheader("Finished processing. Go to the page shown in the sidebar")
    st.sidebar.page_link('main.py', label='Home')

