import streamlit as st
from models.qa_systems import setup_qa_system

from utils.config import MODEL_CONFIGS, DIR, FILES

def render_tabs():
    tab1, tab2, tab3 = st.tabs(["NVIDIA API", "Use a different model", "Demo"])
    
    with tab1:
        render_nvidia_tab()
    
    with tab2:
        render_alternative_models_tab()
    
    with tab3:
        render_demo_tab()

    if st.session_state.get("setup_complete"):
        st.markdown("---")
        
        # Display chat messages from history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("Ask away! What would you like to know about the EU AI Act?"):
            # Check demo limit
            if not st.session_state.get('using_api_key') and st.session_state.question_count >= 10:
                st.error("Demo limit reached. Please use an API key to continue.")
            else:
                # Count questions for demo mode
                if not st.session_state.get('using_api_key'):
                    st.session_state.question_count += 1
                
                # Display user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Generate and display response
                with st.spinner("Thinking..."):
                    response = st.session_state.qa_chain.invoke(prompt)['result']
                
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
        
        # Show remaining questions for demo users
        if not st.session_state.get('using_api_key'):
            questions_remaining = 10 - st.session_state.question_count
            st.markdown(f"""
            <div style="text-align: center; font-size: 0.9em; color: #555;">
                Demo mode: {questions_remaining} question{'s' if questions_remaining != 1 else ''} remaining
            </div>
            """, unsafe_allow_html=True)

def render_nvidia_tab():
    st.markdown("### Enter your NVIDIA API key")
    nvidia_api_key = st.text_input(
        "API Key", 
        placeholder="Enter your NVIDIA API key here...",
        type="password",
        help="Your NVIDIA API key is required to access the full capabilities of the chatbot",
        key="nvidia_api_key"
    )
    
    st.markdown(f"""
    <div style="font-size: 0.9em; margin-top: -10px;">
        Need an API key? <a href="{MODEL_CONFIGS['nvidia']['api_url']}" target="_blank">Learn how to get one</a>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Submit", key="nvidia_api_submit", use_container_width=True):
        if nvidia_api_key:
            with st.spinner("Setting up the chatbot with NVIDIA models..."):
                qa_chain = setup_qa_system("nvidia", DIR, FILES, nvidia_api_key)
                st.session_state.qa_chain = qa_chain
                st.session_state.using_api_key = "nvidia"
                st.session_state.setup_complete = True
                st.success("Setup complete! You can now ask questions below.")
        else:
            st.error("Please enter a valid NVIDIA API key.")

def render_alternative_models_tab():
    st.markdown("### Choose a different AI provider")
    
    st.warning("""
    **Note:** While these models work well, we recommend using the Colosseum model
    through NVIDIA API for optimal performance with EU AI Act content.
    """)
    
    api_provider = st.selectbox(
        "Select API Provider",
        ["OpenAI", "Anthropic", "Mistral", "DeepSeek"],
        help="Select which AI provider to use"
    )
    
    provider_key = api_provider.lower()
    if provider_key in MODEL_CONFIGS:
        render_model_setup(provider_key)

def render_model_setup(provider):
    config = MODEL_CONFIGS[provider]
    
    model_name = st.selectbox(
        f"Select {config['name']} Model",
        config['models'],
        help=f"Select which {config['name']} model to use"
    )
    
    api_key = st.text_input(
        "API Key", 
        placeholder=f"Enter your {config['name']} API key here...",
        type="password",
        help=f"Your {config['name']} API key is required",
        key=f"{provider}_api_key"
    )
    
    st.markdown(f"""
    <div style="font-size: 0.9em; margin-top: -10px;">
        Need a {config['name']} API key? <a href="{config['api_url']}" target="_blank">Get one here</a>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Submit", key="alt_api_submit", use_container_width=True):
        if api_key:
            with st.spinner(f"Setting up the chatbot with {model_name}..."):
                try:
                    qa_chain = setup_qa_system(provider, DIR, FILES, api_key, model_name)
                    st.session_state.qa_chain = qa_chain
                    st.session_state.using_api_key = provider
                    st.session_state.model = model_name
                    st.session_state.setup_complete = True
                    st.success("Setup complete! You can now ask questions below.")
                except Exception as e:
                    st.error(f"Error setting up {config['name']}: {str(e)}")
        else:
            st.error(f"Please enter a valid {config['name']} API key.")

def render_demo_tab():
    st.markdown("### Try the Demo version")
    
    st.info("""
    Demo Mode provides limited functionality without requiring an API key. 
    Perfect for getting a quick overview of the chatbot's capabilities.
    
    **Limitations:**
    - Maximum of 10 questions
    - Shorter responses (reduced token limit)
    - Some features may be limited
    """)
    
    demo_col1, demo_col2, demo_col3 = st.columns([1,2,1])
    with demo_col2:
        if st.button("Start Demo", key="demo_button", use_container_width=True):
            st.session_state.demo_clicked = True
            with st.spinner("Setting up demo environment..."):
                api_key_to_use = st.secrets["api_keys"]["NVIDIA_API_KEY"]
                qa_chain = setup_qa_system("nvidia", DIR, FILES, api_key_to_use, is_demo=True)
                st.session_state.qa_chain = qa_chain
                st.session_state.using_api_key = False
                st.session_state.setup_complete = True
                st.success("Demo ready! You can now ask questions below.")
