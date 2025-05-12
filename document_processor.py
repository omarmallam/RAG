import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def split_documents(documents, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return splitter.split_documents(documents)

def create_vector_store(chunks, embeddings, vector_store_path="vector_store"):
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(vector_store_path)
    print(f"Vector store saved to {vector_store_path}")
    return vector_store


def load_vector_store(vector_store_path, embeddings):
    if os.path.exists(os.path.join(vector_store_path, "index.faiss")):
        try:
            return FAISS.load_local(
                vector_store_path,
                embeddings,
                allow_dangerous_deserialization=True  # Required as of latest update
            )
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return None
    return None