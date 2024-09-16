from app.doc_processing import process_doc, ProcessDocConfig
from app.vectorstore import (get_chroma_store_as_retriever,
                             add_docs_to_store,
                             delete_file_from_store)

pdf_path = '/Users/jan/Desktop/advanced_rag/dev_tests/test_data/2406.07887v1.pdf'
url_1 = 'https://pli.princeton.edu/blog/2024/mamba-2-algorithms-and-systems'
pptx_path = '/Users/jan/Desktop/advanced_rag/dev_tests/test_data/Prasentation_aktuell.pptx'


html_config = ProcessDocConfig(url=url_1, tag='mamba2')
# for testing purposes
pptx_config = ProcessDocConfig(filepath=pptx_path, tag='unused')

pdf_config = ProcessDocConfig(filepath=pdf_path, tag='mamba2')

html_docs = process_doc(html_config)
pptx_doc = process_doc(pptx_config)
pdf_docs = process_doc(pdf_config)

retriever = get_chroma_store_as_retriever()
add_docs_to_store(retriever, html_docs)
add_docs_to_store(retriever, pptx_doc)
add_docs_to_store(retriever, pdf_docs)

delete_file_from_store(retriever, pptx_config.source)





