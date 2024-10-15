from langchain.embeddings.base import Embeddings
from typing import List
from transformers import AutoModel, AutoTokenizer
import torch
from langchain_community.embeddings import OllamaEmbeddings
import os

def mean_pooling(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Perform mean pooling on the input embeddings along the sequence length dimension.

    Args:
        embeddings (torch.Tensor): Input embeddings tensor of shape [batch_size, seq_len, embed_dim]

    Returns:
        torch.Tensor: Mean-pooled embeddings of shape [batch_size, embed_dim]
    """
    return embeddings.mean(dim=-2)

class JinaEmbeddings(Embeddings):
    def __init__(self, model_name: str = 'jinaai/jina-embeddings-v2-small-en'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        inputs = self.tokenizer(texts, return_tensors='pt', truncation=True, max_length=512, padding=True).to(self.device)
        with torch.no_grad():
            output = self.model(**inputs)
        embeddings = mean_pooling(output.last_hidden_state).cpu().numpy()
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def get_ollama_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text",
                            show_progress=True,
                            base_url=os.getenv('OLLAMA_BASE_URL'))