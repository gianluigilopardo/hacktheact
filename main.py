import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_nvidia_ai_endpoints.embeddings import NVIDIAEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import time

# Set the page configuration to change the webpage title
st.set_page_config(page_title="Hack the Act!", page_icon="🤖", layout="centered")

# Set the title of the Streamlit app with an engaging emoji
st.title("🌟 Hack the Act! 🤖")

# Add a captivating description of the app
st.markdown(
    """
    **Welcome to Hack the Act!** 🚀 

    Hack the Act is your interactive guide to navigating the complexities of the [European Union AI Act](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689). 
    Powered by Retrieval-Augmented Generation (RAG) and the cutting-edge [Colosseum 355B](https://build.nvidia.com/igenius/colosseum_355b_instruct_16k) LLM by [iGenius](https://www.igenius.ai/), this chatbot is designed to provide clear, concise answers to your questions about the Act.
    """
)

# Input field for NVIDIA API key with a clear call-to-action
nvidia_api_key = st.text_input("Enter your [NVIDIA API Key](https://build.nvidia.com/) or [support the project](https://www.paypal.me/gianluigilopardo) to use the chatbot:", type="password")

# Customized donation button using Streamlit's button and HTML/CSS for styling
donate_button = st.button("❤️ Support the Project")

# Check if the donation button was clicked
if donate_button:
    st.session_state.donation_clicked = True
    st.markdown(
        """
        <a href="https://www.paypal.me/gianluigilopardo" target="_blank" style="color: #1E90FF; font-weight: bold;">Donate here</a> 🔗
        """,
        unsafe_allow_html=True
    )

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Use caching to store the QA system setup
@st.cache_resource
def setup_qa_system(dir, files, api_key):
    """Sets up a question answering system using specified PDF documents."""
    docs = [doc for file in files for doc in PyPDFLoader(f'{dir}{file}.pdf').load_and_split()]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_documents(docs)
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
        max_tokens=1024,
        nvidia_api_key=api_key,
    )
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    return qa_chain

# Define the directory and file names for the documents
dir = "docs/"
files = ['act_en', 'annex_en']

# Check if the API key is provided or donation button is clicked
if nvidia_api_key or st.session_state.get('donation_clicked'):
    # Use the provided API key or the default one if donation button is clicked
    api_key_to_use = nvidia_api_key if nvidia_api_key else st.secrets["api_keys"]["NVIDIA_API_KEY"]
    # Set up the question answering system with the appropriate API key
    qa_chain = setup_qa_system(dir, files, api_key_to_use)
else:
    st.error("Please insert a valid NVIDIA API key or [support the project](https://www.paypal.me/gianluigilopardo) to use the chatbot 🙏")
    qa_chain = None

# Define a function to generate a response to a question
def generate_response(question):
    """Generates a response to a given question using a QA chain."""
    # Simulate thinking time
    with st.spinner("Thinking..."):
        time.sleep(2)  # Simulate delay
    answer = qa_chain.invoke(question)
    return answer['result']

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input with an engaging prompt
if qa_chain and (prompt := st.chat_input("Ask away! What would you like to know about the EU AI Act?")):
    # Display user message in chat message container
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display the response
    response = generate_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

# Footer with centered text, always visible at the bottom, with smaller font size
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: black;
        text-align: center;
        padding: 10px;
        border-top: 1px solid #ddd;
        z-index: 1000;
        font-size: 0.8em; /* Smaller font size */
    }
    .main-content {
        padding-bottom: 60px; /* Height of the footer */
    }
    </style>
    <div class="main-content">
    </div>
    <div class="footer">
        Made with ❤️ by <a href="https://www.gianluigilopardo.science/" style="color: #1E90FF;">Gianluigi Lopardo</a>
    </div>
    """,
    unsafe_allow_html=True
)
