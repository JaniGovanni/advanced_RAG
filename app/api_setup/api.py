from flask import Flask, request, jsonify
from pydantic import ValidationError
import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from app.doc_processing import ProcessDocConfig, process_doc
from app.chat import ChatConfig, get_result_docs, create_RAG_output
from app.vectorstore import get_chroma_store_as_retriever, add_docs_to_store
from app.source_handling import get_stored_tags_and_files
import os
from werkzeug.utils import secure_filename
import os
import uuid
from app.source_handling import filepath_to_id
from app.doc_processing.filters.default_selections import (
    unwanted_titles_list_default,
    unwanted_categories_default
)
# python api.py

app = Flask(__name__)

@app.route('/get_default_lists', methods=['GET'])
def get_default_lists():
    try:
        default_lists = {
            "unwanted_titles_list_default": unwanted_titles_list_default,
            "unwanted_categories_default": unwanted_categories_default
        }
        return jsonify({"status": "success", "default_lists": default_lists}), 200
    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/upload_file', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({"status": "error", "message": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"status": "error", "message": "No selected file"}), 400
        if file:
            filename = secure_filename(file.filename)
            upload_dir = os.getenv('UPLOADED_FILES_PATH')
            if not upload_dir:
                return jsonify({"status": "error", "message": "UPLOADED_FILES_PATH not set"}), 500
            os.makedirs(upload_dir, exist_ok=True)
            filepath = os.path.join(upload_dir, filename)
            file.save(filepath)
            file_id = str(uuid.uuid4())  # Generate a unique identifier
            filepath_to_id(filepath, file_id)
            return jsonify({"status": "success", "message": "File uploaded successfully", "file_id": file_id}), 200
    except Exception as e:
        app.logger.error(f"Error in file upload: {str(e)}")
        return jsonify({"status": "error", "message": f"Server error: {str(e)}"}), 500
    
@app.route('/process_doc', methods=['POST'])
def api_process_doc():
    try:
        config_data = request.json
        #app.logger.info(f"Received config data: {config_data}")
        #config = ProcessDocConfigAPI(**config_data)
        doc_config = ProcessDocConfig(**config_data)
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

@app.route('/get_stored_tags_and_files', methods=['GET'])
def api_get_stored_tags_and_files():
    try:
        retriever = get_chroma_store_as_retriever()
        tag_to_files = get_stored_tags_and_files(retriever)
        return jsonify({"status": "success", "tag_to_files": tag_to_files}), 200
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
        chat_config = ChatConfig(**config_data)
    
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
        print("Error in api_get_result_docs:")
        print(e)
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8503)