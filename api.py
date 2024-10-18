from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError
from typing import List, Optional
from app.doc_processing import ProcessDocConfig, process_doc
from app.doc_processing.filters import (filter_elements_by_title,
                                        filter_elements_by_unwanted_categories,
                                        unwanted_titles_list_default,
                                        unwanted_categories_default)
from app.chat import ChatConfig, get_result_docs, create_RAG_output
from app.llm import get_ollama_llm, get_groq_llm
from app.vectorstore import get_chroma_store_as_retriever, add_docs_to_store

# python api.py

app = Flask(__name__)

class ProcessDocConfigAPI(BaseModel):
    tag: str
    unwanted_titles_list: Optional[List[str]] = unwanted_titles_list_default
    unwanted_categories_list: Optional[List[str]] = unwanted_categories_default
    local: bool = True
    filepath: Optional[str] = None
    url: Optional[str] = None
    situate_context: bool = False
    late_chunking: bool = False

class ChatConfigAPI(BaseModel):
    tag: str
    expand_by_answer: bool = False
    expand_by_mult_queries: bool = False
    reranking: bool = True
    k: int = 10
    llm_choice: str = "groq"
    use_bm25: bool = False

@app.route('/process_doc', methods=['POST'])
def api_process_doc():
    try:
        config_data = request.json
        app.logger.info(f"Received config data: {config_data}")
        config = ProcessDocConfigAPI(**config_data)
        doc_config = ProcessDocConfig(**config.model_dump())
        result = process_doc(doc_config)
        # Create Chroma vectorstore and store the processed documents
        retriever = get_chroma_store_as_retriever()
        add_docs_to_store(retriever, result)
        
        return jsonify({"status": "success", "message": "Documents processed and stored successfully"}), 200
    except ValidationError as e:
        app.logger.error(f"Validation error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/get_result', methods=['POST'])
def api_get_result_docs():
    try:
        data = request.json
        config_data = data.get('config', {})
        query = data.get('query')
        
        if not query:
            return jsonify({"status": "error", "message": "Query is required"}), 400
        
        config = ChatConfigAPI(**config_data)
        chat_config = ChatConfig(**config.model_dump())
        
        if chat_config.llm_choice == "ollama":
            chat_config.llm = get_ollama_llm()
        else:
            chat_config.llm = get_groq_llm()
        # deactivate history_awareness
        chat_config.history_awareness(False)
        result_texts, joint_query = get_result_docs(chat_config, query)
        
        # Call create_RAG_output with the retrieved context
        context = ' '.join(result_texts)
        rag_output = create_RAG_output(context, query, chat_config.llm)
        
        return jsonify({
            "status": "success", 
            "result_texts": result_texts, 
            "joint_query": joint_query,
            "rag_output": rag_output
        }), 200
    except ValidationError as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)