
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import json
from warnings import warn
import app.llm
import sys
import os
import json
import time
from tqdm import tqdm
import streamlit as st
from langchain_ollama import ChatOllama


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

def situate_context(doc, chunks, llm, progress_callback=None):
    prompt = ChatPromptTemplate.from_messages([
        ("system", DOCUMENT_CONTEXT_PROMPT),
        ("human", CHUNK_CONTEXT_PROMPT),
    ])

    chain = (
        {"doc_content": lambda x: x["doc_content"], "chunk_content": lambda x: x["chunk_content"]}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Prepare inputs for batch processing
    inputs = [{"doc_content": doc, "chunk_content": chunk.text} for chunk in chunks]

    total_chunks = len(inputs)
    combined_results = []
    batch_size = 3

    start_time = time.time()
    
    for i in range(0, total_chunks, batch_size):
        batch = inputs[i:i+batch_size]
        
        batch_start_time = time.time()
        # Process inputs in batches
        batch_results = chain.batch(batch)
        batch_end_time = time.time()

        # Combine original chunks with generated contexts for this batch
        batch_combined = [f"{chunk}\n\n{context}" for chunk, context in zip(chunks[i:i+batch_size], batch_results)]
        combined_results.extend(batch_combined)

        if progress_callback:
            progress = int((i + len(batch)) / total_chunks * 100)
            elapsed_time = time.time() - start_time
            batch_time = batch_end_time - batch_start_time
            progress_callback(progress, elapsed_time, batch_time)

    return combined_results



def create_contextual_embeddings(chunks, progress_callback=None):
    """
    adds context information to the chunks
    """
    # without context caching, this is simply not viable to do via API
    llm = ChatOllama(
                    model="llama3.2:1b",
                    temperature=0,
                    base_url=os.getenv('OLLAMA_BASE_URL'))

    combined_text = " ".join([chunk.text for chunk in chunks])

    # Estimate the number of tokens (roughly equal to number of chars //4)
    estimated_tokens = len(combined_text) // 4

    # llama3.2 has context window sizes over 100k tokens, so this should work
    context_window_size = 100000 # leave space for further prompt

    # Check if the estimated tokens exceed the context window size
    if estimated_tokens > context_window_size:
        warn(f"Estimated tokens ({estimated_tokens}) exceed context window size. Consider not using contextual embeddings.")

    contextualized_chunks = situate_context(combined_text, chunks, llm, progress_callback)

    return contextualized_chunks

def create_contextual_embeddings_with_progress(pdf_chunks):
    st.write("Creating contextual embeddings...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_embedding_progress(progress, elapsed_time, batch_time):
        progress_bar.progress(progress / 100)
        status_text.text(f"Progress: {progress}% | Elapsed Time: {elapsed_time:.2f}s | Last Batch Time: {batch_time:.2f}s")
    
    created_contents = create_contextual_embeddings(pdf_chunks, update_embedding_progress)
    
    progress_bar.empty()
    status_text.empty()
    st.write("Contextual embeddings created successfully!")
    
    return created_contents
    



# jsonl_data = load_jsonl('/Users/jan/Desktop/advanced_rag/app/contextual_embedding/evaluation_set.jsonl')
# # Example usage
# doc_contents = [data['golden_documents'][0]['content'] for data in jsonl_data[:5]]  # Process first 5 documents
# chunk_contents = [data['golden_chunks'][0]['content'] for data in jsonl_data[:5]]  # Process first 5 chunks

# print("========SITUATED_CONTEXTS========")
# responses = situate_context(doc_contents, chunk_contents)
# for i, response in enumerate(responses, 1):
#     print(f"--- Result {i} ---")
#     print(response)
#     print()

