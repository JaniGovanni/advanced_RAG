import os
from unstructured.partition.auto import partition
from unstructured.chunking.title import chunk_by_title
from app.doc_processing.API import process_api
from app.doc_processing.filters import (filter_elements_by_title,
                                        filter_elements_by_unwanted_categories,
                                        unwanted_titles_list_default,
                                        unwanted_categories_default)
from app.doc_processing.metadata import convert_to_document
import logging
import os
from app.contextual_embedding import create_contextual_embeddings_with_progress
from app.doc_processing.late_chunking import apply_late_chunking  
from typing import List



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


def partition_document(config: ProcessDocConfig):
    if config.local:
        return partition(
            filename=config.filepath,
            strategy="hi_res",
            hi_res_model_name="yolox",
            skip_infer_table_types=[],
            url=config.url
        )
    return process_api(config.filepath)

def filter_and_chunk(elements: List, config: ProcessDocConfig):
    filtered_elements = filter_elements_by_title(elements, config.unwanted_titles_list)
    filtered_elements = filter_elements_by_unwanted_categories(filtered_elements, config.unwanted_categories_list)
    return chunk_by_title(filtered_elements)


def process_doc(config: ProcessDocConfig):
    try:
        pdf_elements = partition_document(config)
        pdf_chunks = filter_and_chunk(pdf_elements, config)
        
        created_contents = []
        if config.situate_context:
            created_contents = create_contextual_embeddings_with_progress(pdf_chunks)
        
        pdf_chunks = convert_to_document(
            elements=pdf_chunks,
            tag=config.tag,
            created_contents=created_contents
        )
        
        if config.late_chunking:
            pdf_chunks = apply_late_chunking(pdf_chunks)
        
        return pdf_chunks
    except Exception as e:
        logging.error(f"Error processing document: {str(e)}")
        raise


