# Hack the Act!
**Hack the Act!** is a chatbot designed to demystify the [European Union AI Act](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689).

## Overview
This project employs a Retrieval-Augmented Generation (RAG) approach to deliver an interactive chatbot that answers questions about the EU AI Act. It leverages NVIDIA's AI endpoints for both the language model and embeddings:

- **Language Model:** Powered by [Colosseum 355B](https://build.nvidia.com/igenius/colosseum_355b_instruct_16k) by [iGenius](https://www.igenius.ai/), a model tailored for regulated industries.
- **Embeddings:** Utilizes [NeMo Retriever Llama3.2 embeddings](https://build.nvidia.com/nvidia/llama-3_2-nv-embedqa-1b-v2?snippet_tab=langchain) to create vector representations of document chunks based on official publications by the European Commission:
  - [EU AI Act Regulation](https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng), 13 June 2024
  - [Guidelines on Prohibited AI Practices](https://digital-strategy.ec.europa.eu/en/library/commission-publishes-guidelines-prohibited-artificial-intelligence-ai-practices-defined-ai-act), 4 February 2025

The project utilizes [Langchain](https://www.langchain.com/) and [Streamlit](https://streamlit.io/) for an easy-to-use web interface.

## Setup
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/gianluigilopardo/hacktheact.git
    cd hacktheact
    ```
2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set up environment variables:**
    Create a `.streamlit/secrets.toml` file in the root directory with the following variables:
    ```
    [api_keys]
    NVIDIA_API_KEY=your_nvidia_api_key
    ```
    Replace `your_nvidia_api_key` with your actual API keys. You can request a free NVIDIA API key at [https://build.nvidia.com/](https://build.nvidia.com/).

## Usage
Run the Streamlit app:
```
streamlit run main.py
```