import streamlit as st
import os

# Function to get api key from user if not already set
@st.dialog("Enter Your API Key")
def get_api():
    openai = st.text_input("OpenAI API Key", type="password", help="Your API key remains secure and is not saved.")
    st.markdown("[Create your OpenAI API Key](https://platform.openai.com/api-keys)", unsafe_allow_html=True)
    
    pine = st.text_input("Pinecone API Key", type="password", help="Your API key remains secure and is not saved.")
    st.markdown("[Create your Pinecone API Key](https://app.pinecone.io/organizations/-/projects/-/keys)", unsafe_allow_html=True)
    
    hf = st.text_input("Hugging Face Token", type="password", help="Your API key remains secure and is not saved.")
    st.markdown("[Create your Hugging Face Token](https://huggingface.co/settings/tokens)", unsafe_allow_html=True)
    

    if st.button("Submit"):
        if openai and pine and hf:
            st.session_state["OPENAI_API_KEY"] = openai
            st.session_state["PINECONE_API_KEY"] = pine
            st.session_state["HF_TOKEN"] = hf
            st.success("API key set successfully!")
            st.rerun()
        else:
            st.error("API key cannot be empty.")