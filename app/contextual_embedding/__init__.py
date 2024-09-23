import sys
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from app.llm import get_groq_llm, get_ollama_llm
import json

def load_jsonl(file_path: str):
    """Load JSONL file and return a list of dictionaries."""
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]
    
DOCUMENT_CONTEXT_PROMPT = """
<document>
{doc_content}
</document>
"""

CHUNK_CONTEXT_PROMPT = """
Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
"""

def situate_context(docs, chunks):
    llm = get_ollama_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", DOCUMENT_CONTEXT_PROMPT),
        ("human", CHUNK_CONTEXT_PROMPT),
    ])

    chain = (
        {"doc_content": lambda x: x[0], "chunk_content": lambda x: x[1]}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Process inputs in batches
    results = chain.batch(list(zip(docs, chunks)), {"max_concurrency": 5})

    # Combine original chunks with generated contexts
    combined_results = [f"{chunk}\n\n{context}" for chunk, context in zip(chunks, results)]

    return combined_results

jsonl_data = load_jsonl('/Users/jan/Desktop/advanced_rag/app/contextual_embedding/evaluation_set.jsonl')
# Example usage
doc_contents = [data['golden_documents'][0]['content'] for data in jsonl_data[:5]]  # Process first 5 documents
chunk_contents = [data['golden_chunks'][0]['content'] for data in jsonl_data[:5]]  # Process first 5 chunks

print("========SITUATED_CONTEXTS========")
responses = situate_context(doc_contents, chunk_contents)
for i, response in enumerate(responses, 1):
    print(f"--- Result {i} ---")
    print(response)
    print()