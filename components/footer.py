import streamlit as st

def render_footer():
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
            font-size: 0.8em;
        }
        .main-content {
            padding-bottom: 60px;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 16px;
            height: auto;
        }
        .stButton > button {
            background-color: #2e6fd1;
            color: white;
            font-weight: 500;
            border-radius: 4px;
            padding: 0.5rem 1rem;
            border: none;
        }
        .stButton > button:hover {
            background-color: #1e55a8;
        }
        [data-testid="stButton"] > button[kind="secondary"] {
            background-color: #f0ad4e;
            border-color: #eea236;
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
