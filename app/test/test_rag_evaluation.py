import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import unittest
import json
from app.doc_processing import process_doc, ProcessDocConfig
from app.vectorstore import get_chroma_store_as_retriever, add_docs_to_store
from app.chat import ChatConfig, get_result_docs, create_RAG_output
import app.llm
from app.test.grader_agent import evaluate_answer_v1
import tempfile
import os
import shutil


class TestRAGEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up the document processing and store creation
        cls.retriever = get_chroma_store_as_retriever()
        cls.evaluation_data = cls.load_jsonl('app/test/evaluation_set.jsonl', num_lines=20)
        cls.add_golden_docs_to_store()
    @classmethod
    def create_test_chat_config(cls):
        config = ChatConfig(
            tag="test",
            k=10,
            llm_choice="groq",
            expand_by_answer=True,
            expand_by_mult_queries=True,
            reranking=True,
            use_bm25=False,
            history_awareness=False
    )
        return config


    @staticmethod
    def load_jsonl(file_path, num_lines=None):
        """
        Reads the specified number of lines from the jsonl file and returns a list of dictionaries.
        """
        data = []
        with open(file_path, 'r') as file:
            for i, line in enumerate(file):
                if num_lines is not None and i >= num_lines:
                    break
                data.append(json.loads(line))
        return data
    
    @classmethod
    def add_golden_docs_to_store(cls):
        for entry in cls.evaluation_data:
            golden_docs = entry['golden_documents']
            for doc in golden_docs:
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
                    temp_file.write(doc['content'])
                    temp_file_path = temp_file.name

                try:
                    config = ProcessDocConfig(
                        tag="test",
                        local=True,
                        filepath=temp_file_path,
                        url=None,
                        situate_context=False)
                    # Process the golden document content
                    processed_chunks = process_doc(config)
                    # Add processed chunks to the vector store
                    add_docs_to_store(cls.retriever, processed_chunks)
                finally:
                    # Ensure the temporary file is removed
                    os.unlink(temp_file_path)
    @classmethod
    def tearDownClass(cls):
        # Clean up the data directory
        data_dir = os.getenv("CHROMA_PATH")
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
        print(f"Cleaned up {data_dir} directory")


    def test_answer_relevance(self):
        relevance_scores = []
        results = []
        
        # Create the test chat config
        chat_config = self.create_test_chat_config()
        
        # Prepare the config properties to be written to JSON
        config_properties = {
            "tag": chat_config.tag,
            "k": chat_config.k,
            "llm_choice": chat_config.llm_choice,
            "expand_by_answer": chat_config.expand_by_answer,
            "expand_by_mult_queries": chat_config.expand_by_mult_queries,
            "reranking": chat_config.reranking,
            "use_bm25": chat_config.use_bm25,
            "history_awareness": chat_config.history_awareness
        }

        for entry in self.evaluation_data:
            query = entry['query']
            expected_answer = entry['answer']
        
            result_docs, _ = get_result_docs(chat_config, query)
            context = ''.join(result_docs)
            final_answer = create_RAG_output(context, query, chat_config.llm)
        
            relevance_score = evaluate_answer_v1(query, final_answer, expected_answer)
            relevance_scores.append(relevance_score)
            
            results.append({
                "query": query,
                "expected_answer": expected_answer,
                "given_answer": final_answer,
                "relevance_score": relevance_score
            })

        average_score = sum(relevance_scores) / len(relevance_scores)
        
        # Write results to a JSON file, including the chat config properties
        output = {
            "chat_config": config_properties,
            "results": results,
            "average_score": average_score
        }
        
        with open('/Users/jan/Desktop/advanced_rag/app/test/relevance_results.json', 'w') as f:
            json.dump(output, f, indent=2)

        print(f"Average relevance score: {average_score:.2f}")
        print("Detailed results have been saved to 'relevance_results.json'")
        # Assert the average score
        self.assertGreaterEqual(average_score, 40,
                                f"Low average relevance score ({average_score:.2f})")
        
if __name__ == '__main__':
    unittest.main()

# python -m unittest app/test/test_rag_evaluation.py