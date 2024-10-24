import sys
import os
import pytest


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from app.doc_processing import process_doc, ProcessDocConfig
import os
import dotenv


dotenv.load_dotenv()


def test_process_doc_with_contextual_embedding():
    app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    pdf_path = os.path.join(app_dir, 'test', 'documents_test', 'el_nino.pdf')

    # Ensure the file exists
    assert os.path.exists(pdf_path), f"Test file not found: {pdf_path}"

    # Configure the ProcessDocConfig
    config = ProcessDocConfig(
            tag="test",
            filepath=pdf_path,
            local=True,
            situate_context=True,
            late_chunking=False
        )

    # Process the document
    result = process_doc(config)

    # Assertions
    assert result is not None
    assert len(result) > 0
    assert hasattr(result[0], 'page_content')
    assert hasattr(result[0], 'metadata')
    assert 'tag' in result[0].metadata
    assert result[0].metadata['tag'] == 'test'
    assert 'source' in result[0].metadata
    assert os.path.basename(pdf_path) in result[0].metadata['source']

    # Check if contextual embedding was applied
    assert len(result[0].page_content.split('\n')) > 1  # The original content plus the added context
    # toDo: find a better way to assert the contextual summary
    assert "El Niño" in result[0].page_content

    # pytest app/contextual_embedding/test_contextual_embedding.py

if __name__ == "__main__":
    test_process_doc_with_contextual_embedding()