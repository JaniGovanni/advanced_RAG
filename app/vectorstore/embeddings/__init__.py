from langchain.embeddings.base import Embeddings
from typing import List
from transformers import AutoModel, AutoTokenizer
import torch
from langchain_community.embeddings import OllamaEmbeddings
import os

def mean_pooling(embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Perform mean pooling on the input embeddings along the sequence length dimension,
    taking into account the attention mask to exclude padding tokens.

    Args:
        embeddings (torch.Tensor): Input embeddings tensor of shape [batch_size, seq_len, embed_dim]
        attention_mask (torch.Tensor): Attention mask tensor of shape [batch_size, seq_len]

    Returns:
        torch.Tensor: Mean-pooled embeddings of shape [batch_size, embed_dim]
    """
    sum_embeddings = torch.sum(embeddings * attention_mask, 1)
    sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

class JinaEmbeddings(Embeddings):
    def __init__(self, model_name: str = 'jinaai/jina-embeddings-v2-small-en'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True).to(self.device)
        with torch.no_grad():
            output = self.model(**inputs)
        last_hidden_state = output.last_hidden_state
        attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden_state.size()).float()
        embeddings = mean_pooling(last_hidden_state, attention_mask).cpu().numpy()
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def get_ollama_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text",
                            show_progress=True,
                            base_url=os.getenv('OLLAMA_BASE_URL'))