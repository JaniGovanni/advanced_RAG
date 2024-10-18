import unittest
import json
import requests

class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = 'http://localhost:5000'  # Adjust if your server runs on a different port
        cls.filepath = '/Users/jan/Desktop/advanced_rag/dev_tests/test_data/el_nino.pdf'
        cls.tag = 'el_nino'
        
        # Process the document before running any tests
        cls.process_doc()

    @classmethod
    def process_doc(cls):
        test_data = {
            "tag": cls.tag,
            "local": True,
            "filepath": cls.filepath,
            "situate_context": False,
            "late_chunking": False
        }
        response = requests.post(f'{cls.base_url}/process_doc', json=test_data)
        assert response.status_code == 200, f"Failed to process document in setup. Status code: {response.status_code}"

    def test_get_result(self):
        test_data = {
            "config": {
                "tag": self.tag,
                "expand_by_answer": False,
                "expand_by_mult_queries": False,
                "reranking": True,
                "k": 5,
                "llm_choice": "groq",
                "use_bm25": False
            },
            "query": "What is El Niño?"
        }
        response = requests.post(f'{self.base_url}/get_result', json=test_data)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('result_texts', data)
        self.assertIn('joint_query', data)
        self.assertIn('rag_output', data)
        print(f"RAG output for PDF document: {data['rag_output']}")

    def test_process_doc_validation_error(self):
        test_data = {
            "tag": self.tag,
            "local": "not_a_boolean"  # This should cause a validation error
        }
        response = requests.post(f'{self.base_url}/process_doc', json=test_data)
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data['status'], 'error')

    def test_get_result_missing_query(self):
        test_data = {
            "config": {
                "tag": self.tag
            }
            # Missing 'query' field
        }
        response = requests.post(f'{self.base_url}/get_result', json=test_data)
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data['status'], 'error')
        self.assertEqual(data['message'], 'Query is required')

    def test_process_doc_with_url_and_get_result(self):
        url_tag = 'mamba2'
        url = 'https://pli.princeton.edu/blog/2024/mamba-2-algorithms-and-systems'
        process_doc_data = {
            "tag": url_tag,
            "local": True,
            "url": url,  
            "situate_context": False,
            "late_chunking": False
        }
        process_response = requests.post(f'{self.base_url}/process_doc', json=process_doc_data)
        self.assertEqual(process_response.status_code, 200, "Failed to process document with URL")
        
        get_result_data = {
            "config": {
                "tag": url_tag,
                "expand_by_answer": False,
                "expand_by_mult_queries": False,
                "reranking": True,
                "k": 5,
                "llm_choice": "groq",
                "use_bm25": False
            },
            "query": "What is Mamba2?"
        }
        result_response = requests.post(f'{self.base_url}/get_result', json=get_result_data)
        self.assertEqual(result_response.status_code, 200, "Failed to get result docs")
        
        data = result_response.json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('result_texts', data)
        self.assertIn('joint_query', data)
        self.assertIn('rag_output', data)
        print(f"RAG output for URL document: {data['rag_output']}")

if __name__ == '__main__':
    unittest.main()

# python -m unittest app/test/test_api.py