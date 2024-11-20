import os
from dotenv import load_dotenv

import streamlit as st

import google.generativeai as genai

from preprocessing import get_text, get_chunks
from create_db import create_pinecone_db
from chat import get_response

# Fetching API Key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

def main():
    
    # Streamlit app

    # Title
    st.set_page_config(page_title="Chat with PDF")

    # Header
    st.title("Current Thread")

    # Initializing chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Initialize chat context for model
    if "chat_context" not in st.session_state:
        st.session_state["chat_context"] = []
    
    # Initialize vector_store
    if "db" not in st.session_state:
        st.session_state["db"] = None

    # Display chat messages from history on app rerun
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Sidebar for text input and image upload
    with st.sidebar.form(key="chat_form", clear_on_submit=True):
        st.header("File Upload")
        # Get File input from user
        file_object = st.file_uploader("Upload your PDF(s)", type=["pdf"], accept_multiple_files=True)
        # Submit Button
        submit_file = st.form_submit_button("Send")

    # Text Input
    user_question = st.chat_input(placeholder="AMA!")

    # React to user input
    if submit_file or user_question:
            # If prompt has a file attachment
            if submit_file and file_object:
                with st.spinner("Processing Files..."):
                    # Extracting text from all pdfs
                    raw_text = get_text(file_object)
                    # Breaking raw text into chunks
                    chunks = get_chunks(raw_text)
                    # Creates vector store from all pdfs
                    db = create_pinecone_db(chunks)
                    # Loading the created vector store on to session_state
                    st.session_state.db = db
                    # Display user message in chat message container
                    st.chat_message("assistant").markdown(f"Upload Successful. You can now Chat with the PDF(s).")
                    # Add user message to chat history
                    st.session_state.messages.append({"role": "assistant", "content": f"Upload Successful. You can now Chat with the PDF(s)."})
        
            # If prompt has text
            if user_question:
                # Display user message in chat message container
                st.chat_message("user").markdown(user_question)
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": user_question})
                # Add user message to chat context for model
                st.session_state.chat_context.append({"role": "user", "parts": user_question})
            
                
                # Doing similarity search to find context from our vecctor store
                context = st.session_state.db.invoke(user_question)            
                # Querying LLM for answer
                response = get_response(context, user_question, st.session_state.chat_context)
                # Display assistant response in chat message container
                st.chat_message("assistant").markdown(response)

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                # Add assistant response to chat context for model
                st.session_state.chat_context.append({"role": "model", "parts": response})

if __name__ == "__main__":
    main()