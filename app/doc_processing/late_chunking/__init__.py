from transformers import AutoModel, AutoTokenizer
import torch
from app.vectorstore.embeddings import JinaEmbeddings, mean_pooling
import numpy as np


def late_chunking(
    model_output: 'BatchEncoding', span_annotation: list, attention_mask: torch.Tensor, max_length=None
):
    """
    Perform late chunking on model output based on span annotations.

    Args:
        model_output (BatchEncoding): The output from the transformer model.
        span_annotation (list): List of span annotations for each document.
        max_length (int, optional): Maximum length to consider for annotations.

    Returns:
        list: List of pooled embeddings for each span.
    """
    # equal to model_output.last_hidden_state
    token_embeddings = model_output[0]
    for embeddings, annotations in zip(token_embeddings, span_annotation):
        if max_length is not None:
            annotations = [
                (start, min(end, max_length - 1))
                for (start, end) in annotations
                if start < (max_length - 1)
            ]
        attention_mask = attention_mask.squeeze(0)  # Remove batch dimension
        # expand attention mask to match the shape of the embeddings
        attention_mask = attention_mask.unsqueeze(-1).expand_as(embeddings).float()
        pooled_embeddings = [
            mean_pooling(embeddings[start:end].unsqueeze(0), attention_mask[start:end].unsqueeze(0)).squeeze(0)
            for start, end in annotations
            if (end - start) >= 1
        ]
        pooled_embeddings = [
            embedding.detach().cpu().numpy() for embedding in pooled_embeddings
        ]

    return pooled_embeddings

def late_chunking_experimental(
    model_output: 'BatchEncoding', span_annotation: list, attention_mask: torch.Tensor
):
    """
    Perform late chunking on model output based on span annotations.

    Args:
        model_output (BatchEncoding): The output from the transformer model.
        span_annotation (list): List of span annotations for each document.
        max_length (int, optional): Maximum length to consider for annotations.

    Returns:
        list: List of pooled embeddings for each span.
    """
    token_embeddings = model_output.last_hidden_state
    attention_mask = attention_mask.unsqueeze(-1).expand_as(token_embeddings).float()
    pooled_embeddings = [
        mean_pooling(token_embeddings[: ,start:end, :], attention_mask[: ,start:end, :]).squeeze(0)
        for start, end in span_annotation
        if (end - start) >= 1
    ]
    pooled_embeddings = [
        embedding.detach().cpu().numpy() for embedding in pooled_embeddings
    ]

    return pooled_embeddings


def get_span_annotations(docs, tokenizer):
    """
    Generate span annotations for a list of documents.

    Args:
        docs (list): List of document objects.
        tokenizer: The tokenizer to use for tokenizing document content.

    Returns:
        list: List of (start, end) tuples representing span annotations.
    """
    span_annotations = []
    start = 0
    for doc in docs:
        doc_tokens = tokenizer(doc.page_content, return_tensors='pt', add_special_tokens=False)
        end = start + len(doc_tokens['input_ids'][0])
        span_annotations.append((start, end))
        start = end
    return span_annotations


def apply_late_chunking(docs):
    """
    Apply late chunking to a list of langchain documents.

    This function combines the content of all documents, applies a transformer model
    to generate embeddings, and then uses late chunking to create embeddings for
    each original document. The embeddings are added to each document's metadata.

    Args:
        docs (list): List of document objects to process.

    Returns:
        list: List of processed documents with embeddings added to their metadata.
    """
    complete_text = " ".join(doc.page_content for doc in docs)

    jina_embeddings = JinaEmbeddings()
    tokenizer = jina_embeddings.tokenizer
    model = jina_embeddings.model

    inputs = tokenizer(complete_text, return_tensors='pt', truncation=False)
    model_output = model(**inputs)

    span_annotations = get_span_annotations(docs, tokenizer)

    # embeddings = late_chunking(model_output, [span_annotations], inputs['attention_mask'])
    embeddings = late_chunking_experimental(model_output, span_annotations, inputs['attention_mask'])
   
    for doc, embedding in zip(docs, embeddings):
        doc.metadata['embedding'] = embedding.tolist()  # Convert numpy array to list

    return docs