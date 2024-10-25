import pytest
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from app.RAG_techniques import (
    generate_multi_query,
    augment_query_generated,
    rerank_by_crossencoder,
)
from app.llm import get_groq_llm, get_ollama_llm
import numpy as np



llm = get_ollama_llm()

def test_generate_multi_query():
    original_query = "What are the benefits of exercise?"
    result = generate_multi_query(original_query, llm)
    
    assert len(result) == 5
    assert all(isinstance(q, str) for q in result)
    assert all(q != original_query for q in result)

def test_augment_query_generated():
    query = "What are the main causes of climate change?"
    result = augment_query_generated(query, llm)
    
    assert isinstance(result, str)
    assert len(result) > 0
    # check that the result is not the same as the query
    assert result != query

def test_rerank_by_crossencoder():
    docs = [
        "Exercise improves cardiovascular health.",
        "Regular exercise can boost mood and mental health.",
        "A balanced diet is important for overall health.",
        "Exercise helps in maintaining a healthy weight.",
    ]
    query = "What are the benefits of exercise?"
    
    result = rerank_by_crossencoder(docs, query, top_k=2)
    
    assert len(result) == 2
    assert all(doc in docs for doc in result)
    assert "diet" not in result[0].lower()  # The diet-related document should not be in the top 2

if __name__ == "__main__":
    test_generate_multi_query()
    test_augment_query_generated()
    test_rerank_by_crossencoder()

# pytest app/RAG_techniques/test_RAG_techniques.py