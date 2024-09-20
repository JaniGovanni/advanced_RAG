from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from app.doc_processing import process_doc, ProcessDocConfig
from app.vectorstore import (get_chroma_store_as_retriever,
                             add_docs_to_store,
                             delete_file_from_store)
from app.chat import get_result_docs, ChatConfig, create_RAG_output
import app.llm

# inspired by
# https://github.com/langchain-ai/langchain/blob/master/cookbook/Semi_Structured_RAG.ipynb?ref=blog.langchain.dev

#===========Test 1 on attention is all you need===========
# this shows that the summaries of tables isnt necessary
pdf = '/Users/jan/Desktop/advanced_rag/dev_tests/test_data/attention_is_all.pdf'
table_processing = unwanted_categories_default = ['Header', 'Footer', 'Image',
                                                  'FigureCaption', 'Formula']
pdf_config = ProcessDocConfig(filepath=pdf, tag='attention',
                              unwanted_categories_list=table_processing,
                              search_by_summaries=False)

docs = process_doc(pdf_config)
retriever = get_chroma_store_as_retriever()
add_docs_to_store(retriever, docs)

query = 'What BLEU score did the Transformer (base-model) achieve?'
# answer not in results
results = retriever.vectorstore.similarity_search(k=4,
                                                  query=query)

context = ''.join(results)
llm = ChatOllama(model="phi3.5",
                 temperature=0)
# it creates the right answer
final_answer = create_RAG_output(context, query, llm)