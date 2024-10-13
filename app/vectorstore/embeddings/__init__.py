from langchain.embeddings.base import Embeddings
from typing import List
from transformers import AutoModel, AutoTokenizer
import torch
from langchain_community.embeddings import OllamaEmbeddings
import os

class JinaEmbeddings(Embeddings):
    def __init__(self, model_name: str = 'jinaai/jina-embeddings-v2-small-en'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512).to(self.device)
            # outputs processing is done in a different way in the late_chunking/__init__.py. But the output is the same
            with torch.no_grad():
                output = self.model(**inputs)
            embedding = output.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            embeddings.append(embedding.tolist())
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

def get_ollama_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text",
                            show_progress=True,
                            base_url=os.getenv('OLLAMA_BASE_URL'))