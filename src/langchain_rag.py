# src/langchain_rag.py

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma 
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaLLM
from typing import List

# --- Core Change: Import your fast embedding function ---
from src.embeddings import generate_embedding_single
from src.config import (
    CHROMA_PERSIST_DIR, EMBEDDING_MODEL, LANGUAGE_MODEL, 
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_N_RETRIEVAL
)


# --- Core Change: Create a custom Embeddings class using your function ---
class FastOllamaEmbeddings(Embeddings):
    """
    A custom LangChain Embeddings class that uses the project's original,
    faster, one-by-one embedding function. This avoids the bottleneck from
    the standard OllamaEmbeddings class.
    """
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents using the direct ollama call."""
        # --- FIX: Call generate_embedding_single without the 'model' argument ---
        return [generate_embedding_single(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using the direct ollama call."""
        # --- FIX: Call generate_embedding_single without the 'model' argument ---
        return generate_embedding_single(text)


def create_langchain_vector_store(docs_directory="docs"):
    """
    Loads documents, splits them, and creates a persistent Chroma vector store.
    Now uses the FastOllamaEmbeddings class for high performance.
    """
    print("Loading documents...")
    txt_loader = DirectoryLoader(
        docs_directory, glob="**/*.txt", loader_cls=lambda p: TextLoader(p, encoding='utf-8'),
        show_progress=True, use_multithreading=True
    )
    pdf_loader = DirectoryLoader(
        docs_directory, glob="**/*.pdf", loader_cls=PyPDFLoader,
        show_progress=True, use_multithreading=True
    )
    
    documents = txt_loader.load() + pdf_loader.load()

    if not documents:
        print("No documents loaded. Check the 'docs' directory for .txt and .pdf files.")
        return None
    print(f"Loaded {len(documents)} document(s).")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)

    if not chunks:
        print("No chunks were created from the documents.")
        return None
    print(f"Split documents into {len(chunks)} chunks.")

    print(f"Creating/loading vector store with FAST embeddings at {CHROMA_PERSIST_DIR}...")
    
    # --- Core Change: Use the new FastOllamaEmbeddings class ---
    fast_embedder = FastOllamaEmbeddings()
    
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=fast_embedder, # Use our fast, custom embedder
        persist_directory=CHROMA_PERSIST_DIR
    )
    print(f"Vector store created. Contains {vector_store._collection.count()} entries.")
    return vector_store

def get_langchain_retriever(persist_directory=CHROMA_PERSIST_DIR):
    """
    Initializes a retriever from an existing Chroma vector store.
    """
    # --- Core Change: Use the new FastOllamaEmbeddings class ---
    fast_embedder = FastOllamaEmbeddings()

    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=fast_embedder # Use our fast, custom embedder
    )
    return vector_store.as_retriever(search_kwargs={"k": TOP_N_RETRIEVAL})


def create_langchain_rag_chain():
    """
    Creates a LangChain RetrievalQA chain.
    """
    retriever = get_langchain_retriever()
    llm = OllamaLLM(model=LANGUAGE_MODEL)

    prompt_template = """You are a helpful AI assistant.
Use ONLY the following pieces of context to answer the question. Your answer should be concise and directly address the question.
If the context does not contain the answer, state that the information is not available in the provided documents.
Do not make up any new information or answer from your general knowledge.

Context:
{context}

Question: {question}
Answer:"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain