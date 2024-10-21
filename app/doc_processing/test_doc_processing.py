import sys
import os
import pytest

# pytest app/doc_processing/test_doc_processing.py

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)


from app.doc_processing import process_doc, ProcessDocConfig
import os
import dotenv

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

if __name__ == "__main__":
    test_process_doc()