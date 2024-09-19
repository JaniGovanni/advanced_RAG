from dotenv import load_dotenv
from langchain_community.embeddings import OllamaEmbeddings
from langchain_chroma import Chroma
import os
from uuid import uuid4
import json
from langchain_community.vectorstores.utils import filter_complex_metadata

load_dotenv()


def get_chroma_store_as_retriever():
    """
    To easily create a vecotstore.
    :return: chroma vectorstore setup with Ollama embeddings
    """
    embeddings = OllamaEmbeddings(model="nomic-embed-text",
                                  show_progress=True)
    vectorstore = Chroma(persist_directory=os.getenv("CHROMA_PATH"),
                         embedding_function=embeddings)

    return vectorstore.as_retriever()



def add_docs_to_store(retriever, docs):
    """
    Adds a list of documents, which are part of a single source to a given vectorstore
    """
    # creating ids for handling the documents in the store
    uuids = [str(uuid4()) for _ in range(len(docs))]
    source = docs[0].metadata['source']
    save_uuids(source, uuids, retriever.vectorstore)
    retriever.vectorstore.add_documents(filter_complex_metadata(docs), ids=uuids)


def save_uuids(source, new_uuids, vectorstore):
    """
    saves a source to doc_id mapping to a seperate file. Overwrite any data,
    with the same source name.
    """
    file_path = os.getenv("SOURCE_TO_ID_PATH")
    try:
        with open(file_path, 'r') as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        existing_data = {}
    # check if source already exist
    if source in existing_data:
        existing_uuids = existing_data[source]
        # delete existing source for newer version
        vectorstore.delete(ids=existing_uuids)
        existing_data.update({source: new_uuids})
    else:
        existing_data.update({source: new_uuids})
    with open(file_path, 'w') as f:
        json.dump(existing_data, f, indent=4)


def delete_file_from_store(retriever, source:str):
    """
    Deletes a source from the vectorstore and actualize the source_to_ids file.
    """
    try:
        with open(os.getenv("SOURCE_TO_ID_PATH"), 'r') as f:
            existing_data = json.load(f)
        ids_to_delete = existing_data[source]
        retriever.vectorstore.delete(ids=ids_to_delete)
        del existing_data[source]
        with open(os.getenv("SOURCE_TO_ID_PATH"), 'w') as f:
            json.dump(existing_data, f, indent=4)
    except:
        print("Source not found.")
        return


def get_stored_files_and_tags(retriever):
    """
    gets all files and the tags from them in a vectorstore.
    """
    try:
        with open(os.getenv("SOURCE_TO_ID_PATH"), 'r') as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        return {}
    filenames = existing_data.keys()
    file_to_tag = {}
    for filename in filenames:
        # there must be a better way to do this
        ex_doc = retriever.vectorstore.similarity_search('', filter={'source':filename})[0]
        tag = ex_doc.metadata['tag']
        file_to_tag[filename] = tag
    return file_to_tag

def get_stored_tags_and_files(retriever):
    """
    gets all tags and their files in a vectorstore.
    """
    try:
        with open(os.getenv("SOURCE_TO_ID_PATH"), 'r') as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        return {}
    filenames = existing_data.keys()
    tag_to_file = {}
    for filename in filenames:
        # there must be a better way to do this
        ex_doc = retriever.vectorstore.similarity_search('', filter={'source':filename})[0]
        tag = ex_doc.metadata['tag']
        if ex_doc.metadata['tag'] not in tag_to_file:
            tag_to_file[ex_doc.metadata['tag']] = [filename]
        else:
            tag_to_file[ex_doc.metadata['tag']].append(filename)
    return tag_to_file


