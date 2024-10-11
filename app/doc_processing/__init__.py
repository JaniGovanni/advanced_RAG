import os
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
import re
from app.doc_processing.API import process_api
from app.doc_processing.filters import (filter_elements_by_title,
                                        filter_elements_by_unwanted_categories,
                                        unwanted_titles_list_default,
                                        unwanted_categories_default)
from app.doc_processing.metadata import convert_to_document
from langchain_ollama import ChatOllama
import app.llm
import logging
import os
import dotenv
from app.contextual_embedding import create_contextual_embeddings_with_progress
import streamlit as st



class ProcessDocConfig:
    local: bool
    unwanted_title_list: list
    unwanted_categories_list: list
    tag: str
    filepath: str | None
    url: str | None
    source: str
    situate_context: bool  

    def __init__(self,
                 tag,
                 unwanted_titles_list=unwanted_titles_list_default,
                 unwanted_categories_list=unwanted_categories_default,
                 local=True,
                 filepath=None,
                 url=None,
                 situate_context=False  # Initialize the new attribute
                 ):
        self.local = local
        self.unwanted_titles_list = unwanted_titles_list
        self.unwanted_categories_list = unwanted_categories_list
        self.tag = tag
        self.filepath = filepath
        self.url = url
        self.situate_context = situate_context  # Set the new attribute
        if filepath is not None:
            self.source = os.path.basename(filepath)
        else:
            self.source = url


def process_doc(config):
    
    if config.local:

        pdf_elements = partition(filename=config.filepath,
                                 strategy="hi_res",
                                 hi_res_model_name="yolox",
                                 # the document (pdf, html,..) types, for which table elements should not be extracted
                                 skip_infer_table_types=[],
                                 # if document should be loaded from url (like html)
                                 url=config.url,
                                 # language could be added for improved ocr
                                 # all of these categories will saved under 'extract_image_block_output_dir' or encoded within metadata
                                 #extract_image_block_types=['Table', 'Image'],
                                 # direction for extracted image block types, crucial for multimodal RAG
                                 #extract_image_block_output_dir = '/Users/jan/Desktop/advanced_rag/dev_tests/test_data/extracted_categories'
                                 )
    else:
        pdf_elements = process_api(config.filepath)

    # filter unwanted sections of the document grouped under specific titles
    pdf_elements = filter_elements_by_title(pdf_elements, config.unwanted_titles_list)
    # filter unwanted elements contained in a specific section
    pdf_elements = filter_elements_by_unwanted_categories(pdf_elements, config.unwanted_categories_list)
    # default values seems to be ok for this, max char value is 500
    pdf_chunks = chunk_by_title(pdf_elements)
    
    if config.situate_context:
        created_contents = create_contextual_embeddings_with_progress(pdf_chunks)
    else:
        created_contents = []
    
    pdf_chunks = convert_to_document(elements=pdf_chunks,
                                      tag=config.tag,
                                      created_contents=created_contents)
    return pdf_chunks

