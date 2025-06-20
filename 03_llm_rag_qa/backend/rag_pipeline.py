from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_core.vectorstores.in_memory import InMemoryVectorStore


class RagPipeline:
    def __init__(self):
        embeddings = OllamaEmbeddings(model="llama3.2:1b-instruct-fp16")
        self.vectorstore = InMemoryVectorStore(embedding=embeddings)

    def load_and_index(self, file_path: str):
        documents = self.load_document(file_path)
        documents = self.split_documents(documents)
        self.calculate_embeddings(documents)

    def answer(self, query: str, top_k: int = 3):
        retrieved_docs = self.vectorstore.similarity_search(query, k=top_k)
        return [doc.page_content for doc in retrieved_docs]

    def load_document(self, file_path: str) -> list[Document]:
        if file_path.endswith(".pdf"):
            return PyPDFLoader(file_path).load()
        elif file_path.endswith(".txt"):
            return TextLoader(file_path).load()
        else:
            raise ValueError(f"{file_path} format is not supported!")
        
    def split_documents(self, documents: list[Document]) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        return text_splitter.split_documents(documents)
        
    def calculate_embeddings(self, documents: list[Document]):
        self.vectorstore.add_documents(documents)


if __name__ == "__main__":
    model = RagPipeline()
    model.load_and_index("/Users/bugesz/workspace/projects/03_llm_rag_qa/data/ci-technical-documentation-2014.pdf")
    result = model.answer("What is this document about?")
    print(result)