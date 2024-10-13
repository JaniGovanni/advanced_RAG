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
from app.doc_processing.late_chunking import apply_late_chunking  



class ProcessDocConfig:
    local: bool
    unwanted_title_list: list
    unwanted_categories_list: list
    tag: str
    filepath: str | None
    url: str | None
    source: str
    situate_context: bool
    late_chunking: bool

    def __init__(self,
                 tag,
                 unwanted_titles_list=unwanted_titles_list_default,
                 unwanted_categories_list=unwanted_categories_default,
                 local=True,
                 filepath=None,
                 url=None,
                 situate_context=False,
                 late_chunking=False  
                 ):
        self.local = local
        self.unwanted_titles_list = unwanted_titles_list
        self.unwanted_categories_list = unwanted_categories_list
        self.tag = tag
        self.filepath = filepath
        self.url = url
        self.situate_context = situate_context
        self.late_chunking = late_chunking  
        if filepath is not None:
            self.source = os.path.basename(filepath)
        else:
            self.source = url


def process_doc(config):
    
    if config.local:
        pdf_elements = partition(filename=config.filepath,
                                 strategy="hi_res",
                                 hi_res_model_name="yolox",
                                 skip_infer_table_types=[],
                                 url=config.url)
    else:
        pdf_elements = process_api(config.filepath)

    pdf_elements = filter_elements_by_title(pdf_elements, config.unwanted_titles_list)
    pdf_elements = filter_elements_by_unwanted_categories(pdf_elements, config.unwanted_categories_list)
    pdf_chunks = chunk_by_title(pdf_elements)
    
    if config.situate_context:
        created_contents = create_contextual_embeddings_with_progress(pdf_chunks)
    else:
        created_contents = []
    
    pdf_chunks = convert_to_document(elements=pdf_chunks,
                                     tag=config.tag,
                                     created_contents=created_contents)
    
    if config.late_chunking:
        # Apply late chunking after convert_to_document
        pdf_chunks = apply_late_chunking(pdf_chunks)
    
    return pdf_chunks

