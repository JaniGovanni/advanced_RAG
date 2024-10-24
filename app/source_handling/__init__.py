from langchain_core.documents import Document
import json
import os
from dotenv import load_dotenv

load_dotenv()

def get_filepath_from_id(file_id):
    mapping_file = os.getenv('FILEPATH_TO_ID_PATH')
    
    try:
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
    except FileNotFoundError:
        return None
    
    return mapping.get(file_id)

def filepath_to_id(filepath, file_id):
    mapping_file = os.getenv('FILEPATH_TO_ID_PATH')
    
    try:
        with open(mapping_file, 'r') as f:
            mapping = json.load(f)
    except FileNotFoundError:
        mapping = {}
    
    # Check if the filepath already exists as a value
    existing_id = next((key for key, value in mapping.items() if value == filepath), None)
    if existing_id:
        del mapping[existing_id]  # Remove the existing entry
    
    mapping[file_id] = filepath  # Add the new file_id: filepath pair
    
    with open(mapping_file, 'w') as f:
        json.dump(mapping, f, indent=4)
    

def get_documents_by_tag(retriever, tag):
    """
    Retrieves all documents from the vectorstore that have a specific tag.
    
    :param retriever: The retriever object containing the vectorstore
    :param tag: The tag to filter documents by
    :return: A list of Document objects with the specified tag
    """
    collection = retriever.vectorstore._collection

    # Get all documents for the given tag using Chroma's where clause
    results = collection.get(
        where={"tag": tag},
        include=['metadatas', 'documents']
    )

    # Create Document objects from the results
    documents = [
        Document(page_content=doc, metadata=metadata) 
        for doc, metadata in zip(results['documents'], results['metadatas'])
    ]

    return documents

def save_uuids(source, new_uuids, vectorstore, file_path=None):
    """
    saves a source to doc_id mapping to a seperate file. Overwrite any data,
    with the same source name.
    """
    if file_path is None:
        file_path = os.getenv("SOURCE_TO_ID_PATH")
    # Try to load existing data, or create an empty dict if file doesn't exist
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
    

def get_stored_tags_and_files(retriever, filename=None):
    """
    gets all tags and their files in a vectorstore.
    """
    if filename is None:
        filename = os.getenv("SOURCE_TO_ID_PATH")
    try:
        with open(filename, 'r') as f:
            existing_data = json.load(f)
    except FileNotFoundError:
        return {}
    filenames = existing_data.keys()
    tag_to_file = {}
    for filename in filenames:
        # there must be a better way to do this
        ex_doc = retriever.vectorstore.similarity_search('', filter={'source':filename})[0]
        if ex_doc.metadata['tag'] not in tag_to_file:
            tag_to_file[ex_doc.metadata['tag']] = [filename]
        else:
            tag_to_file[ex_doc.metadata['tag']].append(filename)
    return tag_to_file