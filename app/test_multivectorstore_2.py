from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from app.doc_processing import process_doc, ProcessDocConfig
from app.vectorstore import (get_chroma_store_as_retriever,
                             add_docs_to_store,
                             delete_file_from_store)

html = 'https://datatables.net/examples/basic_init/multiple_tables.html'

table_processing = ['Header', 'Footer', 'Image', 'FigureCaption', 'Formula']

pdf_config = ProcessDocConfig(url=html, tag='test',
                              unwanted_categories_list=table_processing,
                              search_by_summaries=True)

docs = process_doc(pdf_config)

retriever = get_chroma_store_as_retriever()
add_docs_to_store(retriever, docs)
# answer is contained in founded docs
results = retriever.vectorstore.similarity_search(k=4,
                                                  query='What is the salary of Shad Decker ?')

# ==========Cross validation without summary=========

from app.doc_processing import process_doc, ProcessDocConfig
from app.vectorstore import (get_chroma_store_as_retriever,
                             add_docs_to_store,
                             delete_file_from_store)
html = 'https://datatables.net/examples/basic_init/multiple_tables.html'

table_processing = ['Header', 'Footer', 'Image', 'FigureCaption', 'Formula']

pdf_config = ProcessDocConfig(url=html, tag='test',
                              unwanted_categories_list=table_processing,
                              search_by_summaries=False)

docs = process_doc(pdf_config)

retriever = get_chroma_store_as_retriever()
add_docs_to_store(retriever, docs)
# answer is still founded docs
results = retriever.vectorstore.similarity_search(k=4,
                                                  query='What is the salary of Shad Decker ?')
