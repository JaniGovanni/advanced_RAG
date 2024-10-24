# advanced_rag

Hello there and welcome to my second project hosted on github. 
it implements an advanced Retrieval-Augmented Generation (RAG) system for question answering. My goal for this project was, to implement as much as possible from the things i have learned so far, about RAG and ML in general, in this project. I also build this with the intention of modularity, because i will add more things as i go along in my learning. **I try to keep this README as actual as possible, as i implement new things.**

## Agenda

* implement image support

## Key features

The features of this project are:

1. Process, structure and store all kind of document types and also providing options to filter out unwanted document sections for efficient retrieval.
2. Combine more than one documents to build a knowledge base.
3. Implement various RAG techniques (multi query, HyDE, BM25 hybrid search, Contextual Retrieval,...) to enhance query understanding and document retrieval.
4. Ability to run the software completely locally using Ollma (llama3.2:3b, llama3.2:1b, nomic-embed-text) and ChromaDB or via API support from Groq.
5. Provide a flexible and configurable chat interface for interacting with the system.
6. Include evaluation and testing capabilities to assess the system's performance.
7. Easy deployment via Streamlit UI and containerisation with docker-compose
8. Separate frontend (Streamlit) and backend (Flask API) architecture to simplify scaling strategies and improve maintainability:
   - **Frontend**: Streamlit-based user interface for file upload, configuration, and chat interaction.
   - **Backend**: Flask API responsible for document processing, vector store operations, and RAG pipeline execution.

## Key Components

```
advanced_rag/
├── app/
│   ├── api_setup/
│   │   ├── __init__.py
│   │   ├── dataclasses/
│   │   ├── test_api.py
│   │   └── api.py
│   ├── chains/
│   │   ├── __init__.py
│   │   └── test_HistoryChain.py
│   ├── contextual_embedding/
│   │   └── __init__.py
│   ├── doc_processing/
│   │   ├── __init__.py
│   │   ├── API
│   │   ├── filters
│   │   ├── metadata/
│   │   └── late_chunking/
│   ├── llm/
│   │   └── __init__.py
│   ├── memory/
│   │   └── __init__.py
│   ├── RAG_techniques/
│   │   ├── __init__.py
│   │   ├── prompts.py
│   │   └── test_RAG_techniques.py
│   ├── source_handling/
│   │   └── __init__.py
│   ├── test/
│   │   ├── test_rag_evaluation.py
│   │   ├── grade_agent.py
│   │   ├── evaluation_results
│   │   └── test_late_chunking_evaluation.py
│   ├── utils_chat/
│   │   └── __init__.py
│   ├── source_handling/
│   │   └── __init__.py
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   ├── embeddings/
│   │   └── experimental/
│   └──  chat.py
|
```

### Key Modules and Their Functions

1. **doc_processing**: Handles document ingestion, parsing, and preprocessing.
   - Uses the unstructured library to process various document types (e.g., PDFs, HTML, xlsx, pptx,...)
   - Filters unwanted content and categories (customizable)
   - Chunks documents for efficient storage and retrieval

2. **vectorstore**: Manages the vector database for storing and retrieving document embeddings.
   - Handles creation of vectordatabase 
   - Adds documents to the vector store
   - Defines different embedding objects

3. **contextual_embedding**: Generates contextual embeddings for document chunks.
   - Implements contextual retrieval locally (see anthropics post: https://www.anthropic.com/news/contextual-retrieval)
   - Improves document representation by considering the surrounding context
   - Enhances retrieval accuracy

4. **RAG_techniques**: Implements various RAG techniques to improve query processing and document retrieval.
   - Multi-query expansion
   - Hypothetical Document Embedding (HyDE)
   - BM25 hybrid search
   - Cross-encoder reranking

5. **chains**: Defines conversation chains and query reformulation logic.
   - Implements history-aware query reformulation

6. **llm**: Provides interfaces to different language models (e.g., Ollama, Groq).
   - Abstracts LLM interactions for easy switching between models

7. **chat.py**: Implements the main chat interface and RAG pipeline.
   - Configures and manages the chat session
   - Orchestrates the RAG process from query to answer generation

8. **test**: Contains test suites for evaluating the RAG system's performance.
   - Evaluates answer relevance and correctness based on anthropics RAG evaluation dataset

9. **late_chunking**: Implements advanced document chunking for improved embeddings.
   - Generates embeddings for entire documents using Jina AI's long context embedding model
   - Enhances embedding quality by considering full document context
   - For more information about the implementation and results see: cookbooks/cookbook_late_chunking.ipynb
   - Currently not integrated in the main program, because it doesn't provide good results

10. **api_setup**: Manages the API setup for backend operations.
    - Provides endpoints for document processing and retrieval
    - Handles API requests and responses for the RAG system

11. **source_handling**: Manages file paths and source-related operations.
    - Retrieves file paths from IDs
    - Manages source metadata

12. **utils_chat**: Contains utility functions for chat operations.
    - Provides helper functions for chat configuration and management

The system is designed to be modular and configurable, allowing for easy experimentation with different RAG techniques, language models, and document processing methods. The evaluation framework helps in assessing the system's performance and identifying areas for improvement.

# Installation and Running Guidelines

### Prerequisites

1. Docker and Docker Compose installed on your system.
2. A Groq API key (optional, for using Groq's API).

### Installation Steps

1. Clone the repository:
   ```
   git clone https://github.com/JaniGovanni/advanced_RAG.git
   cd advanced_rag
   ```

2. Install ollama
Run the install_ollama.sh script to install ollama and the required models on your system. This script will only work on linux and macos. For windows you have to install ollama manually, which also shoudnt be that hard. For windows you also have to manually pull the models llama3.2, llama3.2:1b and nomic-embed-text.

3. Configure volume path in docker-compose.yml
currently the volume `advanced_rag_storage` is created in the default path of the docker installation, which is `/var/lib/docker/volumes/advanced_rag_storage/data` on linux. if you want another path on your system you have to insert it here in the docker-compose.yml file:

```
api-server:
.
.
.
   volumes:
      - path/to/your/volume:/data
```


4. Build and run the Docker container:
   ```
   export GROQ_API_KEY=your_groq_api_key
   docker-compose up --build
   ```

   This will build the Docker image and start the container. The application will be accessible at `http://localhost:8501`.


**Or simply contact me. I will launch an instance where you can test the software.😉**

## Using the Streamlit Interface

This project uses Streamlit to provide an interactive web interface for the advanced RAG system. Here's an overview of how to use the interface:

1. **Home Page (main.py)**
   - Define a source tag for your documents
   - Upload a file or provide a URL for document processing
   - View existing tags and associated files in the database
   - Select a tag to start chatting about the documents associated with that tag

2. **Document Processing Config (pages/second_page.py)**
   - Configure document processing settings
   - Define unwanted titles and categories to filter out
   - Choose whether to create contextual embeddings for contextual retrieval
   - Process the document and add it to the vector store

3. **Chat Page (pages/chat_page.py)**
   - Choose between Groq and Ollama as the LLM provider
   - Configure RAG options:
     - History awareness
     - HyDE (Hypothetical Document Embedding)
     - Multiple query formulation
     - Cross-encoder reranking
     - adding BM25 keyword search
   - You can reset your conversation with the Buttom in the sidebar
   - Interact with the chatbot using the processed documents

### Tips
- If you go to the chat page, you always chat with all files, which are associated with the selected tag. This was my intentional design decision. If you just want to chat with a single file you can simply create a new tag for this file.
- Image support is currently not supported for the RAG pipeline, so they should be filtered out. Tables work.
- You can try out different combinations of RAG techniques, but some are might not work well together.
- For example, if you use HyDE, you should not use multiple query formulation.
- I am currently trying to find out, which combinations bring the best results, for that reason i implemented the advanced_rag/app/test/test_rag_evaluation.py file. 

## Appendix

I am not sure if i have covered everything here, but you can always contact me for questions. Every Know and then I will add some stuff, as the project evolves.



