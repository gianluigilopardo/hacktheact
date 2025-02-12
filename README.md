
# Hack the Act!
**Hack the Act!** is a chatbot designed to demystify the [European Union AI Act](https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689).  

## Overview
This project employs a Retrieval-Augmented Generation (RAG) approach to deliver an interactive chatbot that answers questions about the EU AI Act. Powered by [Colosseum 355B](https://build.nvidia.com/igenius/colosseum_355b_instruct_16k) by [iGenius](https://www.igenius.ai/), a model tailored for regulated industries and uses NVIDIA's AI endpoints. [OpenAI embeddings](https://platform.openai.com/docs/guides/embeddings) create vector representations of document chunks based on official publications by the European Commission: 
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

Create a `.env` file in the root directory with the following variables:

```
    OPENAI_API_KEY=your_openai_api_key
    NVIDIA_API_KEY=your_nvidia_api_key
```
Replace `your_openai_api_key` and `your_nvidia_api_key` with your actual API keys.

## Usage

Run the Streamlit app:
```
    streamlit run main.py
```
