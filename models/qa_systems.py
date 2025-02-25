import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_nvidia_ai_endpoints.embeddings import NVIDIAEmbeddings
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_mistralai import ChatMistralAI
from langchain_deepseek import ChatDeepSeek
from langchain_community.embeddings import HuggingFaceEmbeddings

@st.cache_resource
def setup_qa_system(provider, dir, files, api_key, model_name=None, is_demo=False):
    """Generic setup function for QA systems"""
    docs = [doc for file in files for doc in PyPDFLoader(f'{dir}{file}.pdf').load_and_split()]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_documents(docs)
    
    # Setup based on provider
    if provider == "nvidia":
        return setup_nvidia(chunks, api_key, is_demo)
    elif provider == "openai":
        return setup_openai(chunks, api_key, model_name)
    elif provider == "anthropic":
        return setup_anthropic(chunks, api_key, model_name)
    elif provider == "mistral":
        return setup_mistral(chunks, api_key, model_name)
    elif provider == "deepseek":
        return setup_deepseek(chunks, api_key, model_name)
    else:
        raise ValueError(f"Unknown provider: {provider}")

def setup_nvidia(chunks, api_key, is_demo=False):
    """Setup for NVIDIA models"""
    embeddings = NVIDIAEmbeddings(
        nvidia_api_key=api_key,
        model="nvidia/llama-3.2-nv-embedqa-1b-v2",
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever()
    llm = ChatNVIDIA(
        model="igenius/colosseum_355b_instruct_16k",
        temperature=0.1,
        top_p=0.8,
        max_tokens=1024 if not is_demo else 512,
        nvidia_api_key=api_key,
    )
    return RetrievalQA.from_chain_type(llm, retriever=retriever)

def setup_openai(chunks, api_key, model_name):
    """Setup for OpenAI models"""
    embeddings = OpenAIEmbeddings(api_key=api_key)
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever()
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.1,
        api_key=api_key,
    )
    return RetrievalQA.from_chain_type(llm, retriever=retriever)

def setup_anthropic(chunks, api_key, model_name):
    """Setup for Anthropic Claude"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever()
    llm = ChatAnthropic(
        model=model_name,
        temperature=0.1,
        api_key=api_key,
        max_tokens=1024,
    )
    return RetrievalQA.from_chain_type(llm, retriever=retriever)

def setup_mistral(chunks, api_key, model_name):
    """Setup for Mistral AI"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever()
    llm = ChatMistralAI(
        model=model_name,
        temperature=0.1,
        api_key=api_key,
        max_tokens=1024,
    )
    return RetrievalQA.from_chain_type(llm, retriever=retriever)

def setup_deepseek(chunks, api_key, model_name):
    """Setup for DeepSeek"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever()
    llm = ChatDeepSeek(
        model=model_name,
        temperature=0.1,
        api_key=api_key,
        max_tokens=1024,
    )
    return RetrievalQA.from_chain_type(llm, retriever=retriever)
