import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)


import pytest
from app.doc_processing.filters import (
    filter_elements_by_title,
    filter_elements_by_unwanted_categories,
    unwanted_titles_list_default,
    unwanted_categories_default
)
from unstructured.documents.elements import Text, Title
from app.doc_processing import process_doc, ProcessDocConfig
import os
import dotenv

# pytest app/doc_processing/test_doc_processing.py

# Load environment variables
dotenv.load_dotenv()


def test_process_doc():
   
    url = 'https://pli.princeton.edu/blog/2024/mamba-2-algorithms-and-systems'
    
    config = ProcessDocConfig(
        tag="test",
        filepath=None,
        url=url,
        local=True,
        situate_context=True,
        late_chunking=False
    )

    result = process_doc(config)

    # Test the structure of the result
    assert isinstance(result, list), "Result should be a list"
    assert len(result) > 0, "Result should not be empty"
    
    # Test the properties of each document in the result
    for doc in result:
        assert hasattr(doc, 'page_content'), "Each document should have page_content"
        assert hasattr(doc, 'metadata'), "Each document should have metadata"
        
        # Test metadata
        assert 'tag' in doc.metadata, "Metadata should include 'tag'"
        assert doc.metadata['tag'] == 'test', "Tag should match the input config"
        assert 'source' in doc.metadata, "Metadata should include 'source'"
        assert url in doc.metadata['source'], "Source should include the input URL"
        assert len(doc.page_content) > 0, "Page content should not be empty"
    config_no_context = ProcessDocConfig(
        tag="no_context",
        filepath=None,
        url=url,
        local=True,
        situate_context=False,
        late_chunking=False
    )
    result_no_context = process_doc(config_no_context)
    
    assert len(result_no_context[0].page_content.split('\n')) < len(result[0].page_content.split('\n')), \
        "Result without context should have less content"

    # Test error handling
    with pytest.raises(Exception):
        invalid_config = ProcessDocConfig(
            tag="invalid",
            filepath=None,
            url="https://invalid-url.com",
            local=True,
            situate_context=False,
            late_chunking=False
        )
        process_doc(invalid_config)

def test_filter_elements_by_title():
    elements = [
        Title(text="Table of Contents"),
        Text(text="Chapter 1"),
        Title(text="Introduction"),
        Text(text="This is the introduction."),
        Title(text="References"),
        Text(text="1. Author, A. (2023)"),
    ]

    for i, elem in enumerate(elements):
        elem.metadata.parent_id = str(elements[i-1].id) if i % 2 == 1 else None


    

    filtered_elements = filter_elements_by_title(elements, unwanted_titles_list_default)

    print("Filtered elements:")
    for elem in filtered_elements:
        print(f"- {elem.text} (type: {type(elem).__name__})")

    print(f"Number of filtered elements: {len(filtered_elements)}")
    print(f"Unwanted titles: {unwanted_titles_list_default}")

    assert len(filtered_elements) == 2
    assert filtered_elements[0].text == "Introduction"
    assert filtered_elements[1].text == "This is the introduction."



def test_filter_elements_by_unwanted_categories():
    elements = [
        Text(text="This is regular text."),
        Title(text="This is a title"),
        Text(text="This is a header"),
        Text(text="This is a footer"),
    ]

    elements[0].category = "NarrativeText"
    elements[1].category = "Title"
    elements[2].category = "Header"
    elements[3].category = "Footer"

    filtered_elements = filter_elements_by_unwanted_categories(elements, unwanted_categories_default)

    assert len(filtered_elements) == 2
    assert filtered_elements[0].text == "This is regular text."
    assert filtered_elements[1].text == "This is a title"

def test_filter_and_chunk():
    from app.doc_processing import filter_and_chunk, ProcessDocConfig

    elements = [
        Title(text="Table of Contents"),
        Text(text="Chapter 1"),
        Title(text="Introduction"),
        Text(text="This is the introduction."),
        Title(text="References"),
        Text(text="1. Author, A. (2023)"),
        Text(text="This is a header"),
        Text(text="This is a footer"),
    ]

    # Set metadata after creation
    for i, elem in enumerate(elements):
        elem.metadata.parent_id = str(elements[i-1].id) if i % 2 == 1 else None
        if i == 5:
            break
    elements[6].category = "Header"
    elements[7].category = "Footer"

    config = ProcessDocConfig(
        tag="test",
        filepath=None,
        url="https://example.com",
        local=True,
        situate_context=False,
        late_chunking=False
    )

    filtered_chunks = filter_and_chunk(elements, config)

    assert len(filtered_chunks) == 1
    assert filtered_chunks[0].text == "Introduction\n\nThis is the introduction."

if __name__ == "__main__":
    #test_process_doc()
    test_filter_elements_by_title()
    test_filter_elements_by_unwanted_categories()
    test_filter_and_chunk()