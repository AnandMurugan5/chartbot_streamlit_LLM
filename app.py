import streamlit as st
import shutil
import os
from documents_qar import VectorConvertion,QBot

st.header('Chat with your own documents')
sid = st.sidebar


UPLOAD_DIR = "upload"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

uploaded_files = sid.file_uploader("Upload Files", accept_multiple_files=True)
if uploaded_files:
    file_paths = []  # Store uploaded file paths
    for uploaded_file in uploaded_files:
        with open(os.path.join(UPLOAD_DIR, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(os.path.join(UPLOAD_DIR, uploaded_file.name))

    vector_conversion = VectorConvertion(file_paths)
    chat=vector_conversion.vector_conversion()
    st.write(chat)
    user_input = st.text_input("You:", value="")
    if st.button("Send"):
        # Process user input and get response from the chatbot
        chat_bot = QBot()
        response = chat_bot.prompt(user_input)
        st.text_area("Bot:", value=response, height=200)

