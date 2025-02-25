import streamlit as st

from components.header import render_header
from components.tabs import render_tabs
from components.footer import render_footer

def main():
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "demo_clicked" not in st.session_state:
        st.session_state.demo_clicked = False
    if "question_count" not in st.session_state:
        st.session_state.question_count = 0

    # Render components
    render_header()
    render_tabs()
    render_footer()

if __name__ == "__main__":
    main()

