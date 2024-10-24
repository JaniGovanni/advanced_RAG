import sys
import os

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import unittest
import json
import requests
from app.api_setup.dataclasses_api import ProcessDocConfigAPI, ChatConfigAPI
from langchain_core.messages import HumanMessage, AIMessage
import subprocess
import time
from dotenv import load_dotenv

load_dotenv('app/.env')

class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = os.getenv('BASE_URL') 
        cls.filepath = '/Users/jan/Desktop/advanced_rag/dev_tests/test_data/el_nino.pdf'
        cls.tag = 'el_nino'

        # Start the Flask server
        cls.server_process = subprocess.Popen(['python', 'app/api_setup/api.py'])
        time.sleep(5)
        
        # Process the document before running any tests
        cls.process_doc()
    
    @classmethod
    def tearDownClass(cls):
        # Shut down the Flask server
        cls.server_process.terminate()
        cls.server_process.wait()

    @classmethod
    def process_doc(cls):
        test_data = ProcessDocConfigAPI(
            tag=cls.tag,
            local=True,
            filepath=cls.filepath,
            situate_context=False,
            late_chunking=False
        )
        response = requests.post(f'{cls.base_url}/process_doc', json=test_data.model_dump())
        assert response.status_code == 200, f"Failed to process document in setup. Status code: {response.status_code}"

        
    def test_history_awareness(self):
        
        chat_config = ChatConfigAPI(
        tag='el_nino',
        expand_by_answer=False,
        expand_by_mult_queries=False,
        reranking=True,
        k=5,
        llm_choice="groq",
        use_bm25=False,
        history_awareness=True
        )

        # First query
        first_query = "What is the main topic of the document?"
        chat_config.conversation_history = [
            HumanMessage(content=first_query)
        ]
        first_payload = {
            "query": first_query,
            "config": chat_config.model_dump()
        }
        first_response = requests.post(f'{self.base_url}/get_result', json=first_payload)
        self.assertEqual(first_response.status_code, 200)
        first_data = first_response.json()
        self.assertEqual(first_data['status'], 'success')
        print(f"First query response: {first_data['rag_output']}")
        chat_config.conversation_history.append(AIMessage(content=first_data['rag_output']))
        # Second query that relies on context from the first
        second_query = "Can you provide more details about that?"
        chat_config.conversation_history.append(HumanMessage(content=second_query))
        second_payload = {
            "query": second_query,
            "config": chat_config.model_dump()
        }
        second_response = requests.post(f'{self.base_url}/get_result', json=second_payload)
        self.assertEqual(second_response.status_code, 200)
        second_data = second_response.json()
        self.assertEqual(second_data['status'], 'success')
        print(f"Second query response: {second_data['rag_output']}")
        
    def test_get_result(self):
        test_data = ChatConfigAPI(
            tag=self.tag,
            expand_by_answer=False,
            expand_by_mult_queries=False,
            reranking=True,
            k=5,
            llm_choice="groq",
            use_bm25=False,
        )
        query = "What is El Niño?"
        payload = {
            "query": query,
            "config": test_data.model_dump()
        }
        response = requests.post(f'{self.base_url}/get_result', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('result_texts', data)
        self.assertIn('joint_query', data)
        self.assertIn('rag_output', data)
        print(f"RAG output for PDF document: {data['rag_output']}")
    def test_get_stored_tags_and_files(self):
        response = requests.get(f'{self.base_url}/get_stored_tags_and_files')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('tag_to_files', data)
        print(f"Stored tags and files: {data['tag_to_files']}")
        
    def test_get_result_missing_query(self):
        test_data = ChatConfigAPI(
            tag=self.tag,
            expand_by_answer=False,
            expand_by_mult_queries=False,
            reranking=True,
            k=5,
            llm_choice="groq",
            use_bm25=False,
        )
        query = None
        payload = {
            "query": query,
            "config": test_data.model_dump()
        }
        response = requests.post(f'{self.base_url}/get_result', json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data['status'], 'error')
        self.assertEqual(data['message'], 'Query is required')

    def test_process_doc_with_url_and_get_result(self):
        url_tag = 'mamba2'
        url = 'https://pli.princeton.edu/blog/2024/mamba-2-algorithms-and-systems'
        process_doc_data = ProcessDocConfigAPI(
            tag=url_tag,
            local=True,
            url=url,
            situate_context=False,
            late_chunking=False
        )
        process_response = requests.post(f'{self.base_url}/process_doc', json=process_doc_data.model_dump())
        self.assertEqual(process_response.status_code, 200, "Failed to process document with URL")
        
        test_data = ChatConfigAPI(
            tag=url_tag,
            expand_by_answer=False,
            expand_by_mult_queries=False,
            reranking=True,
            k=5,
            llm_choice="groq",
            use_bm25=False,
        )
        query = "What is Mamba2?"
        payload = {
            "query": query,
            "config": test_data.model_dump()
        }
        result_response = requests.post(f'{self.base_url}/get_result', json=payload)
        self.assertEqual(result_response.status_code, 200, "Failed to get result docs")
        
        data = result_response.json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('result_texts', data)
        self.assertIn('joint_query', data)
        self.assertIn('rag_output', data)
        print(f"RAG output for URL document: {data['rag_output']}")

if __name__ == '__main__':
    unittest.main()
    #test_history_awareness()

    # python app/api_setup/test_api.py