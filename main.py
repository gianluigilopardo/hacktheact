import streamlit as st

from dotenv import load_dotenv 

from langchain_community.document_loaders import PyPDFLoader 
from langchain.chains import RetrievalQA 
from langchain_community.vectorstores import FAISS 

from langchain_openai.embeddings import OpenAIEmbeddings 
# from langchain_nvidia_ai_endpoints.embeddings import NVIDIAEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter 

# from langchain_openai import ChatOpenAI 
from langchain_nvidia_ai_endpoints import ChatNVIDIA


load_dotenv() 
openai_api_key = st.secrets["api_keys"]["OPENAI_API_KEY"]
nvidia_api_key = st.secrets["api_keys"]["NVIDIA_API_KEY"]

# Set the title of the Streamlit app
st.title(f"Hack the Act! 🤖")
# Add a markdown description of the app
st.markdown(
    """
        **Hack the Act!** is a RAG-powered chatbot designed to demystify the [European Union AI Act](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689). 
        The underlying LLM is [Colosseum 355B](https://build.nvidia.com/igenius/colosseum_355b_instruct_16k) by [iGenius](https://www.igenius.ai/), a model tailored for regulated industries. 
"""
#        Available in five languages: English, Italian, French, German, and Spanish. 
)

# Define a function to set up the question answering system
def setup_qa_system(dir, files): 
    """Sets up a question answering system using specified PDF documents.

    This function loads and processes PDF documents from a given directory,
    creates embeddings for the text chunks, and sets up a retrieval-based
    question answering system using a pre-trained language model.

    Args:
        dir (str): The directory path where the PDF files are located.
        files (list of str): A list of file names (without extension) to be processed.

    Returns:
        RetrievalQA: A retrieval-based question answering chain ready to handle queries."""
    # Load the relevant PDFs
    docs = [doc for file in files for doc in PyPDFLoader(f'{dir}{file}.pdf').load_and_split()]

    # Split the documents into smaller chunks for processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_documents(docs)

    # Create embeddings for the document chunks
    # embeddings = NVIDIAEmbeddings()
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Create a retriever from the vector store
    retriever = vector_store.as_retriever()
    # Initialize the ChatNVIDIA model
    llm = ChatNVIDIA(
            model="igenius/colosseum_355b_instruct_16k",
            temperature=0.1,
            top_p=0.7,
            max_tokens=1024,
            nvidia_api_key=nvidia_api_key,
    )

    # Create a RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    return qa_chain 

# Define the directory and file names for the documents
# dir = "https://raw.githubusercontent.com/gianluigilopardo/hacktheact/main/docs/"
dir = "docs/"
files = ['act_en', #'act_it', 'act_es', 'act_fr', 'act_de', 
        'annex_en', #'annex_it', 'annex_es', 'annex_fr', 'annex_de', 
    ]

# Set up the question answering system
qa_chain = setup_qa_system(dir, files)

# Define a function to generate a response to a question
def generate_response(question):
    """Generates a response to a given question using a QA chain.

    Args:
        question (str): The question for which a response is to be generated.

    Returns:
        None: The function displays the result using a streamlit info message."""
    answer = qa_chain.invoke(question)
    st.info(answer['result'])
   
# Create a Streamlit form
with st.form("aiact_form"):
    # Add a text area for the user to enter their question
    text = st.text_area(
        "Ask any question about the EU AI Act:", 
        "What are the basic principles of the AI Act? "
        )
    
    # Add a submit button
    submitted = st.form_submit_button("Submit")
    # Generate and display the response if the form is submitted
    generate_response(text)
