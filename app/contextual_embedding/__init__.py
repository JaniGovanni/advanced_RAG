
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from warnings import warn
import os
import json
import time
import streamlit as st
from langchain_ollama import ChatOllama
from app.contextual_embedding.constants import DOCUMENT_CONTEXT_PROMPT, CHUNK_CONTEXT_PROMPT
from typing import List, Optional, Callable
from unstructured.documents.elements import Element


class ContextualEmbedder:
    def __init__(self, model: str = "llama3.2:1b", base_url: Optional[str] = None):
        self.llm = ChatOllama(
            model=model,
            temperature=0,
            base_url=base_url or os.getenv('OLLAMA_BASE_URL')
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", DOCUMENT_CONTEXT_PROMPT),
            ("human", CHUNK_CONTEXT_PROMPT),
        ])
        self.chain = (
            {"doc_content": lambda x: x["doc_content"], "chunk_content": lambda x: x["chunk_content"]}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def situate_context(self, doc: str, chunks: List[Element], progress_callback: Optional[Callable] = None) -> List[str]:
        prompt = ChatPromptTemplate.from_messages([
            ("system", DOCUMENT_CONTEXT_PROMPT),
            ("human", CHUNK_CONTEXT_PROMPT),
            ])

        chain = (
            {"doc_content": lambda x: x["doc_content"], "chunk_content": lambda x: x["chunk_content"]}
            | prompt
            | self.llm
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

    def create_contextual_embeddings(self, chunks: List[Element], progress_callback: Optional[Callable] = None) -> List[str]:
        combined_text = " ".join([chunk.text for chunk in chunks])
        estimated_tokens = len(combined_text) // 4
        context_window_size = 100000

        if estimated_tokens > context_window_size:
            warn(f"Estimated tokens ({estimated_tokens}) exceed context window size. Consider not using contextual embeddings.")

        return self.situate_context(combined_text, chunks, progress_callback)




def create_contextual_embeddings(self, chunks: List[str], progress_callback: Optional[Callable] = None) -> List[str]:
    """
    adds context information to the chunks
    """
    
    combined_text = " ".join([chunk.text for chunk in chunks])

    # Estimate the number of tokens (roughly equal to number of chars //4)
    estimated_tokens = len(combined_text) // 4

    # llama3.2 has context window sizes over 100k tokens, so this should work
    context_window_size = 100000 # leave space for further prompt

    # Check if the estimated tokens exceed the context window size
    if estimated_tokens > context_window_size:
        warn(f"Estimated tokens ({estimated_tokens}) exceed context window size. Consider not using contextual embeddings.")

    return self.situate_context(combined_text, chunks, progress_callback)

def create_contextual_embeddings_with_progress(pdf_chunks):
    st.write("Creating contextual embeddings...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_embedding_progress(progress, elapsed_time, batch_time):
        progress_bar.progress(progress / 100)
        status_text.text(f"Progress: {progress}% | Elapsed Time: {elapsed_time:.2f}s | Last Batch Time: {batch_time:.2f}s")
    
    embedder = ContextualEmbedder()  # Create an instance of ContextualEmbedder
    created_contents = embedder.create_contextual_embeddings(pdf_chunks, update_embedding_progress)
    
    progress_bar.empty()
    status_text.empty()
    st.write("Contextual embeddings created successfully!")
    
    return created_contents
