import sys
import os
import unittest
from unittest.mock import patch
import dotenv
from groq import Groq
from app.chat import get_result_docs, ChatConfig, create_RAG_output
import app.llm
from app.doc_processing import process_doc, ProcessDocConfig
from app.vectorstore import get_chroma_store_as_retriever, add_docs_to_store
import shutil

import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# to run: python -m unittest app.test.test_rag_pipeline

# This is an example, who an RAG pipeline test could look like.
# The test is not perfect, but it is a good starting point.
# Note that there is an library called deepeval that could be used to test RAG pipelines.
# However, i dont have an OPENAI key, so i cannot use it without much effort and create a custom llm wrapper.
dotenv.load_dotenv('app/.env')

def evaluate_answer(query, answer):
    llm = app.llm.get_groq_llm()
    prompt = f"""
    Query: {query}
    Answer: {answer}

    Evaluate the relevance and correctness of the answer to the given query. 
    Provide a score from 0 to 100, where:
    0-20: Completely irrelevant or incorrect
    21-40: Mostly irrelevant or incorrect
    41-60: Partially relevant and correct
    61-80: Mostly relevant and correct
    81-100: Highly relevant and correct

    Output only the numeric score.
    """
    
    response = llm.predict(prompt)
    return int(response.strip())

class TestChatRelevance(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up the document processing and store creation
        pdf = '/Users/jan/Desktop/advanced_rag/dev_tests/test_data/attention_is_all.pdf'
        table_processing = ['Header', 'Footer', 'Image', 'FigureCaption', 'Formula']
        pdf_config = ProcessDocConfig(filepath=pdf, tag='attention',
                                      unwanted_categories_list=table_processing)

        docs = process_doc(pdf_config)
        cls.retriever = get_chroma_store_as_retriever()
        add_docs_to_store(cls.retriever, docs)
    
    @classmethod
    def tearDownClass(cls):
        # Clean up the data directory
        data_dir = os.getenv("CHROMA_PATH")
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
        print(f"Cleaned up {data_dir} directory")

    
    def test_bm25_retrieval_effectiveness(self):
        # Create a ChatConfig instance
        chat_config = ChatConfig(
        tag="attention",
        use_bm25=True,
        k=5,
        llm_choice="groq",
        reranking=False,
        expand_by_mult_queries=False)

        # Disable history awareness for this test
        chat_config.history_awareness(False)

        # Define a query that is likely to be challenging for pure vector similarity search
        query = "What are the computational requirements for the Transformer model?"

        # Get the results with BM25 enabled
        result_docs_with_bm25, _ = get_result_docs(chat_config, query)

        # Disable BM25 and get results with only similarity search
        chat_config.use_bm25 = False
        result_docs_without_bm25, _ = get_result_docs(chat_config, query)

        # Function to check if a document is relevant
        def is_relevant(doc):
            relevant_terms = ['computational', 'requirements', 'complexity', 'efficiency', 'hardware', 'gpu', 'memory']
            return any(term in doc.lower() for term in relevant_terms)

        # Count relevant documents in each result set
        relevant_with_bm25 = sum(1 for doc in result_docs_with_bm25 if is_relevant(doc))
        relevant_without_bm25 = sum(1 for doc in result_docs_without_bm25 if is_relevant(doc))

        # Assert that BM25 retrieval finds more relevant documents
        self.assertGreater(relevant_with_bm25, relevant_without_bm25, 
                        "BM25 retrieval did not improve the relevance of search results")

        print(f"Relevant docs with BM25: {relevant_with_bm25}")
        print(f"Relevant docs without BM25: {relevant_without_bm25}")

    def test_document_processing_and_storage(self):
        # Perform a similarity search to check if documents were added correctly
        results = self.retriever.vectorstore.similarity_search("attention", filter={"tag": "attention"}, k=1)
        
        self.assertGreater(len(results), 0, "No documents found with the 'attention' tag")
        self.assertIn("attention", results[0].page_content.lower(), "The retrieved document doesn't contain 'attention'")

    def test_attention_query(self):
        
        query = "What is attention?"
        config_multi_query = ChatConfig(tag='attention',
                                        k=10,
                                        llm=app.llm.get_ollama_llm())
        config_multi_query.history_awareness(False)
        config_multi_query.set_mult_queries(True)
        config_multi_query.set_reranking(True)
        config_multi_query.set_exp_by_answer(False)

        result_texts, joint_query = get_result_docs(config_multi_query, query=query)
        context = ''.join(result_texts)
        final_answer = create_RAG_output(context, query, config_multi_query.llm)

        score = evaluate_answer(query, final_answer)
        # Assert that the score is above a certain threshold (e.g., 70)
        self.assertGreaterEqual(score, 70, f"The answer relevance score ({score}) is below the acceptable threshold.")


    def test_transformer_bleu_score(self):
        query = "What BLEU score did the Transformer (base-model) achieve?"
        config_multi_query = ChatConfig(tag='attention',
                                        k=10,
                                        llm=app.llm.get_groq_llm())
        config_multi_query.history_awareness(False)
        config_multi_query.set_mult_queries(True)
        config_multi_query.set_reranking(True)
        config_multi_query.set_exp_by_answer(False)

        result_texts, joint_query = get_result_docs(config_multi_query, query=query)
        context = ''.join(result_texts)
        final_answer = create_RAG_output(context, query, config_multi_query.llm)

        # Assert that the correct BLEU score is mentioned in the answer
        self.assertIn("27.3", final_answer, "The correct BLEU score (27.3) is not mentioned in the answer.")

        # Evaluate the overall relevance and correctness
        score = evaluate_answer(query, final_answer)
        self.assertGreaterEqual(score, 70, f"The answer relevance score ({score}) is below the acceptable threshold.")

if __name__ == '__main__':
    unittest.main()