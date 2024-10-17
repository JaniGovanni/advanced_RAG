import sys
import os
from dotenv import load_dotenv

current_dir = os.getcwd()
env_path = os.path.abspath(os.path.join(current_dir, '..', 'app', '.env'))
load_dotenv(env_path)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)


def chunk_by_sentences(input_text: str, tokenizer: callable):
    """
    Split the input text into sentences using the tokenizer
    :param input_text: The text snippet to split into sentences
    :param tokenizer: The tokenizer to use
    :return: A tuple containing the list of text chunks and their corresponding token spans
    """
    inputs = tokenizer(input_text, return_tensors='pt', return_offsets_mapping=True)
    punctuation_mark_id = tokenizer.convert_tokens_to_ids('.')
    sep_id = tokenizer.convert_tokens_to_ids('[SEP]')
    token_offsets = inputs['offset_mapping'][0]
    token_ids = inputs['input_ids'][0]
    chunk_positions = [
        (i, int(start + 1))
        for i, (token_id, (start, end)) in enumerate(zip(token_ids, token_offsets))
        if token_id == punctuation_mark_id
        and (
            token_offsets[i + 1][0] - token_offsets[i][1] > 0
            or token_ids[i + 1] == sep_id
        )
    ]
    chunks = [
        input_text[x[1] : y[1]]
        for x, y in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
    ]
    span_annotations = [
        (x[0], y[0]) for (x, y) in zip([(1, 0)] + chunk_positions[:-1], chunk_positions)
    ]
    return chunks, span_annotations

from app.vectorstore.embeddings import JinaEmbeddings
# test document
input_text = "Berlin is the capital and largest city of Germany, both by area and by population. Its more than 3.85 million inhabitants make it the European Union's most populous city, as measured by population within city limits. The city is also one of the states of Germany, and is the third smallest state in the country in terms of area."
# langchain embedding class wrapper around huggingface tokenizer and model
# note: i use the small model instead of the base model because of cpu limitations
embeddings = JinaEmbeddings()
sentences, _ = chunk_by_sentences(input_text, embeddings.tokenizer)

from langchain.schema import Document
docs = [Document(page_content=sentence, metadata={"tag": 'test_berlin', "source": "berlin.txt"}) for sentence in sentences]

from app.vectorstore.experimental import get_faiss_store_as_retriever, add_docs_to_faiss_store
from app.doc_processing.late_chunking import apply_late_chunking

late_chunked_docs = apply_late_chunking(docs)
faiss_retriever = get_faiss_store_as_retriever()
add_docs_to_faiss_store(faiss_retriever, late_chunked_docs)

query = "Berlin"


faiss_results = faiss_retriever.vectorstore.similarity_search_with_score(query, k=3)

# from app.vectorstore import get_chroma_store_as_retriever, add_docs_to_store
# chroma_retriever = get_chroma_store_as_retriever(embeddings=embeddings)
# add_docs_to_store(chroma_retriever, docs)