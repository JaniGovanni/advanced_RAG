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
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import app.llm
from langchain.graphs import Neo4jGraph
from langchain.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_experimental.graph_transformers import LLMGraphTransformer
import logging
import os
import dotenv

dotenv.load_dotenv()


def extract_knowledge_graph(pdf_chunks, llm):
    """
    Extracts a knowledge graph from a list of PDF chunks (Langchain documents) and returns a Neo4j graph object.
    """
    # only possible via api
    graph = None
    
    try:
        # Initialize Neo4j graph
        graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            database=os.getenv("NEO4J_DATABASE"),
        )
        
        # Initialize LLMGraphTransformer
        graph_transformer = LLMGraphTransformer.from_llm(llm=llm)
        
        # Process each chunk and extract entities and relationships
        for chunk in pdf_chunks:
            try:
                graph_documents = graph_transformer.convert_to_graph_documents([chunk])
                for graph_document in graph_documents:
                    graph.add_graph_documents([graph_document])
                
            except Exception as e:
                logging.error(f"Error processing chunk: {e}")
                continue  # Skip to the next chunk if there's an error
        
        return graph
    except Exception as e:
        logging.error(f"Error in knowledge graph extraction: {e}")
        return None
    finally:
        if graph:
            try:
                graph.close()
            except Exception as e:
                logging.error(f"Error closing Neo4j connection: {e}")



class ProcessDocConfig:
    local: bool
    unwanted_title_list: list
    unwanted_categories_list: list
    tag: str
    filepath: str | None
    url: str | None
    source: str
    search_by_summaries: bool
    extract_knowledge_graph: bool
    # language could be added for improved ocr

    def __init__(self,
                 tag,
                 unwanted_titles_list=unwanted_titles_list_default,
                 unwanted_categories_list=unwanted_categories_default,
                 local=True,
                 filepath=None,
                 url=None,
                 search_by_summaries=False,
                 extract_knowledge_graph=False
                 ):
        self.local = local
        self.unwanted_titles_list = unwanted_titles_list
        self.unwanted_categories_list = unwanted_categories_list
        self.tag = tag
        self.filepath = filepath
        self.url = url
        self.search_by_summaries = search_by_summaries   # is not recommended
        self.extract_knowledge_graph = extract_knowledge_graph
        if filepath is not None:
            self.source = os.path.basename(filepath)
        else:
            self.source = url


def process_doc(config):
    # if local -> get_ollama_llm
    # if not -> get_groq_llm
    llm = app.llm.get_groq_llm()
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
    if config.search_by_summaries:
        # if table data is included it might be better to build
        # table and text summaries for the similarity search
        summaries = create_table_text_summaries(pdf_chunks, llm)
    else:
        summaries = []
    pdf_chunks = convert_to_document(elements=pdf_chunks, tag=config.tag, summaries=summaries)
    if config.extract_knowledge_graph:
        knowledge_graph = extract_knowledge_graph(pdf_chunks, llm)
    else:
        knowledge_graph = None

    return pdf_chunks


def create_table_text_summaries(chunks, model):
    """
    Creates a summary for table and text chunks and adds this to the metadata
    """
    # maybe like this but doesnt seem necessary
    # tables = [i.metadata.text_as_html for i in table_chunks]
    # tables = [chunk for chunk in chunks if chunk.category == 'Table']
    # texts = [chunk for chunk in chunks if chunk.category == 'CompositeElement']

    prompt_text = """You are an assistant tasked with summarizing tables and text for retrieval. \
        These summaries will be embedded and used to retrieve the raw text or table elements. \
        Give a concise summary of the table or text that is well optimized for retrieval. Table or text: {element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)

    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    # max_concurrency is batch size
    summaries = summarize_chain.batch(chunks, {"max_concurrency": 5})

    return summaries

