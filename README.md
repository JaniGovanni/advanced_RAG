# advanced_rag

Hello there and welcome to my second project hosted on github. 
it implements an advanced Retrieval-Augmented Generation (RAG) system for question answering. My goal for this project was, to implement as much as possible from the things i have learned so far, about RAG and ML in general, as possible in this project. I also build this with the intention of modularity, because i will add more things as i go along in my learning.

## Key features

The features of this project are:

1. Process and store all kind of document types and also providing options to filter out unwanted document sections for efficient retrieval.
2. Combine more than one documents to build a knowledge base.
3. Implement various RAG techniques (multi query, HyDE, BM25, Contextual Retrieval,...) to enhance query understanding and document retrieval.
4. Ability to run the software completely locally using Ollma and ChromaDB or via API support from Groq.
5. Provide a flexible and configurable chat interface for interacting with the system.
6. Include evaluation and testing capabilities to assess the system's performance.

## Key Components

```
advanced_rag/
├── app/
│   ├── doc_processing/
│   │   ├── __init__.py
│   │   ├── API.py
│   │   ├── filters.py
│   │   └── metadata/
│   ├── vectorstore/
│   │   └── __init__.py
│   ├── contextual_embedding/
│   │   └── __init__.py
│   ├── RAG_techniques/
│   │   └── __init__.py
│   ├── chains/
│   │   └── __init__.py
│   ├── llm/
│   │   └── __init__.py
│   ├── chat.py
│   └── test/
│       ├── test_rag_evaluation.py
│       └── test_rag_pipeline.py
└── README.md
```

### Key Modules and Their Functions

1. **doc_processing**: Handles document ingestion, parsing, and preprocessing.
   - Using unstructured library to processes various document types (e.g., PDFs, HTML, xlsx, pptx,...)
   - Filters unwanted content and categories (customizable)
   - Chunks documents section based for efficient storage and retrieval

2. **vectorstore**: Manages the vector database for storing and retrieving document embeddings.
   - Adds documents to the vector store
   - Retrieves relevant documents based on queries
   - Handles document metadata and source tracking

3. **contextual_embedding**: Generates contextual embeddings for document chunks.
   - Implements contextual retrieval locally (see anthropics post: https://www.anthropic.com/news/contextual-retrieval)
   - Improves document representation by considering the surrounding context
   - Enhances retrieval accuracy

4. **RAG_techniques**: Implements various RAG techniques to improve query processing and document retrieval.
   - Multi-query expansion
   - Hypothetical Document Embedding (HyDE)
   - Additional BM25 retrieval
   - Cross-encoder reranking

5. **chains**: Defines conversation chains and query reformulation logic.
   - Implements history-aware query reformulation

6. **llm**: Provides interfaces to different language models (e.g., Ollama, Groq).
   - Abstracts LLM interactions for easy switching between models

7. **chat.py**: Implements the main chat interface and RAG pipeline.
   - Configures and manages the chat session
   - Orchestrates the RAG process from query to answer generation

8. **test**: Contains test suites for evaluating the RAG system's performance.
   - Includes end-to-end tests for the RAG pipeline (primarily to test general functionality)
   - Evaluates answer relevance and correctness based on anthropics RAG evaluation dataset

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
   - Interact with the chatbot using the processed documents

### Tips
- If you go to the chat page, you always chat with all files, which are associated with the selected tag. This was my intentional design decision. If you just want to chat with a single file you can simply create a new tag for this file.
- Image support is currently not supported for the RAG pipeline, so they should be filtered out. Tables work.
- You can try out different combinations of RAG techniques, but some are might not work well together.
- For example, if you use HyDE, you should not use multiple query formulation.
- I am currently trying to find out, which combinations bring the best results, for that reason i implemented the advanced_rag/app/test/test_rag_evaluation.py file. 

## Appendix

I am not sure if i have covered everything here, but you can always contact me for questions. Every Know and then I will add some stuff, as the project evolves.



