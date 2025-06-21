from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

class EmbeddingModel:
    def __init__(self, model_name: str):
        self.model_name = model_name