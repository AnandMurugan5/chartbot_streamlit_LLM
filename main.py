# Import necessary modules
import os
import yaml
import shutil
import pandas as pd
import streamlit as st

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, CSVLoader
from langchain.document_loaders.image import UnstructuredImageLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate

# load yaml file for get environment variables
with open('credential_config.yaml') as file:
    config = yaml.safe_load(file)

# store environment variables in local variables
OPENAI_API_TYPE = config['storage']['OPENAI_API_TYPE']
OPENAI_API_BASE = st.secrets['OPENAI_API_BASE']
OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']
OPENAI_API_VERSION = config['storage']['OPENAI_API_VERSION']
OPENAI_API_DEPLOYMENT = config['storage']['OPENAI_API_DEPLOYMENT']
OPENAI_API_MODEL = config['storage']['OPENAI_API_MODEL']


# Function to process files and extract text data
def process_files(file_paths):
    text = []
    for path in file_paths:
        file_extension = os.path.splitext(path)[1]
        loaders = {
            ".pdf": PyPDFLoader,
            ".csv": CSVLoader,
            ".docx": Docx2txtLoader,
            ".xlsx": UnstructuredExcelLoader,
            ".jpg": UnstructuredImageLoader,
            ".txt": UnstructuredFileLoader,
        }
        if file_extension in loaders:
            if file_extension == ".xlsx":
                loader = loaders[file_extension](path, mode="elements")  # Assuming 'elements' mode
            else:
                loader = loaders[file_extension](path)
            file_data = loader.load()

            text.extend(file_data)
        else:
            raise ValueError(f"Unsupported file format: {path}")
    return text



def clean_vectors(directory):
    # Delete the existing vector database directory
    if os.path.exists(directory):
        shutil.rmtree(directory)

# Function to split documents into smaller chunks
def split_docs(documents,chunk_size=2000,chunk_overlap=300):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

# Initialize Streamlit app
st.title('Chatbot Application')
sib = st.sidebar

# File upload section
uploaded_files = sib.file_uploader("Upload Files", accept_multiple_files=True)
if uploaded_files:
    file_paths = []  # Store uploaded file paths
    for uploaded_file in uploaded_files:
        with open(os.path.join("upload", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(os.path.join("upload", uploaded_file.name))
    
    # Process files to extract text data
    text = process_files(file_paths)
    docs = split_docs(text)

    # Generate embeddings for text data
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Directory to persist the vector store
    persist_directory = "chroma_db"
    clean_vectors(persist_directory)
    persist_directory = "chroma_db"

    # Create a vector database (Chroma) from the extracted text documents
    vectordb = Chroma.from_documents(
        documents=docs, embedding=embeddings, persist_directory=persist_directory
    )

    # Persist the vector database
    vectordb.persist()

    # Initialize the chat model
    llm = AzureChatOpenAI(
        openai_api_base=OPENAI_API_BASE,
        openai_api_version=OPENAI_API_VERSION,
        deployment_name=OPENAI_API_DEPLOYMENT,
        openai_api_key=OPENAI_API_KEY,
        openai_api_type=OPENAI_API_TYPE,
        model_name = OPENAI_API_MODEL
    )

    prompt_template = """
    {context}

    Question: {question}"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Set up a retrieval-based QA chain using RetrievalQA
    chain_type_kwargs = {"prompt": PROMPT}
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 10}), chain_type_kwargs=chain_type_kwargs,verbose=True)

    user_query = st.text_input("Enter your query:")
    if st.button("Ask"):
        if user_query:
            if not chain:
                st.error("The chain is not initialized. Embedding failed.")
            else:
                # Run the query through the chain and get the response
                response = chain.run(user_query)
                st.success(f"Response: {response}")
        else:
            st.error("Please enter a query.")