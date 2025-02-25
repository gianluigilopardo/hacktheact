import streamlit as st

def render_header():
    st.set_page_config(page_title="Hack the Act!", page_icon="🤖", layout="centered")
    st.title("🌟 Hack the Act! 🤖")
    st.markdown(
        """
        Hack the Act! 🤖 is a RAG-based chatbot designed to navigate the complexities of the [European Union AI Act](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai).
        Powered by [NeMo Retriever Llama3.2](https://build.nvidia.com/nvidia/llama-3_2-nv-embedqa-1b-v2) embeddings from [NVIDIA](https://build.nvidia.com) and the cutting-edge [Colosseum 355B](https://build.nvidia.com/igenius/colosseum_355b_instruct_16k) LLM by [iGenius](https://www.igenius.ai/), it provides clear, concise answers to your regulatory questions.
        Check it out on [GitHub](https://github.com/gianluigilopardo/hacktheact)!
        """
    )
