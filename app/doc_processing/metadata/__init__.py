from langchain_core.documents import Document


def convert_to_document(elements, tag, created_contents=None):
    """
    Adds some metadata to the given chunks and convert them into
    langchain document objects for further storing.

    :param elements: list of elements
    :param tag: The topic to which the document belongs
    :param summaries: list of generated summaries, for the elements, to increase efficiency
    during similarity search
    :return: list of documents
    """
    documents = []
    if not created_contents:
        for element in elements:
            metadata = modify_metadata(element, tag)
            documents.append(Document(page_content=element.text,
                                      metadata=metadata))
    else:
        for element, content in zip(elements, created_contents):
            metadata = modify_metadata(element, tag)
            # original text becomes a metadata
            metadata['orig_text'] = element.text
            # this is a little workaround, instead of using a multi vectorstore.
            # found this approach better because the necessary structural changes are smaller
            documents.append(Document(page_content=content,
                                      metadata=metadata))
    return documents


def modify_metadata(element, tag):
    """
    creates a dictionary from the metadata property and stores some
    extra information.
    """
    metadata = element.metadata.to_dict()
    if 'filename' in metadata:
        metadata["source"] = metadata["filename"]
    else:
        # html elements have no attribute filename
        metadata["source"] = metadata["url"]
    metadata['category'] = element.category
    metadata["tag"] = tag
    return metadata
