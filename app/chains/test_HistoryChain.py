import pytest
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from app.chains import HistoryAwareQueryChain
from app.llm import get_groq_llm, get_ollama_llm

def test_reformulate_with_history():
    # function is concepted, that the last query is already part of the history
    conversation_history = [
        {"type": "human", "content": "What's the capital of France?"},
        {"type": "ai", "content": "The capital of France is Paris."},
        {"type": "human", "content": "What's its population?"},
    ]

    history_aware_query_chain = HistoryAwareQueryChain(get_ollama_llm())
    
    input_query = "What's its population?"
    
    reformulated_query = history_aware_query_chain.reformulate(conversation_history, input_query)
    
    assert isinstance(reformulated_query, str)
    assert len(reformulated_query) > 0
    print(reformulated_query)
    assert "Paris" in reformulated_query or "France" in reformulated_query


def test_reformulate_with_unrelated_history():
    conversation_history = [
        {"type": "human", "content": "What's the largest planet in our solar system?"},
        {"type": "ai", "content": "The largest planet in our solar system is Jupiter."},
        {"type": "human", "content": "What's the capital of France?"},
    ]
    history_aware_query_chain = HistoryAwareQueryChain(get_ollama_llm())
    input_query = "What's the capital of France?"
    
    reformulated_query = history_aware_query_chain.reformulate(conversation_history, input_query)
    print(reformulated_query)
    assert "France" in reformulated_query and "capital" in reformulated_query and "planet" not in reformulated_query

if __name__ == "__main__":
    test_reformulate_with_history()
    test_reformulate_with_unrelated_history()