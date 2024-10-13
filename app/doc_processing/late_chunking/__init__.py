from transformers import AutoModel, AutoTokenizer
import torch


def late_chunking(
    model_output: 'BatchEncoding', span_annotation: list, max_length=None
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
    token_embeddings = model_output[0]
    outputs = []
    for embeddings, annotations in zip(token_embeddings, span_annotation):
        if max_length is not None:
            annotations = [
                (start, min(end, max_length - 1))
                for (start, end) in annotations
                if start < (max_length - 1)
            ]
        pooled_embeddings = [
            embeddings[start:end].sum(dim=0) / (end - start)
            for start, end in annotations
            if (end - start) >= 1
        ]
        pooled_embeddings = [
            embedding.detach().cpu().numpy() for embedding in pooled_embeddings
        ]
        outputs.append(pooled_embeddings)

    return outputs


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
    Apply late chunking to a list of documents.

    This function combines the content of all documents, applies a transformer model
    to generate embeddings, and then uses late chunking to create embeddings for
    each original document. The embeddings are added to each document's metadata.

    Args:
        docs (list): List of document objects to process.

    Returns:
        list: List of processed documents with embeddings added to their metadata.
    """
    complete_text = " ".join(doc.page_content for doc in docs)

    tokenizer = AutoTokenizer.from_pretrained('jinaai/jina-embeddings-v2-small-en', trust_remote_code=True)
    model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-small-en', trust_remote_code=True)
    inputs = tokenizer(complete_text, return_tensors='pt', truncation=False)
    model_output = model(**inputs)

    span_annotations = get_span_annotations(docs, tokenizer)

    embeddings = late_chunking(model_output, [span_annotations])[0]

    for doc, embedding in zip(docs, embeddings):
        doc.metadata['embedding'] = embedding.tolist()  # Convert numpy array to list

    return docs