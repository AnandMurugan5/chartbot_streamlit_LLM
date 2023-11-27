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



    
def clean_vectors(directory):
    # Delete the existing vector database directory
    if os.path.exists(directory):
        shutil.rmtree(directory)

# Function to split documents into smaller chunks
def split_docs(documents,chunk_size=2000,chunk_overlap=300):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs


def process_files(file_paths):
    text = []
    for path in file_paths:
        try:
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
        except Exception as e:
            st.error(f"Error processing file {path}: {str(e)}")
    return text


embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

class VectorConvertion:
    def __init__(self,files) -> None:
        self.files = files
        pass
    
    def vector_conversion(self):
        persist_directory = "chroma_db"
        try:
            text = process_files(self.files)
            docs = split_docs(text)
            clean_vectors(persist_directory)
            persist_directory = "chroma_db"
            if not os.path.exists(persist_directory):
                os.makedirs(persist_directory)
            vectordb = Chroma.from_documents(
                documents=docs, embedding=embeddings, persist_directory=persist_directory
            )
            vectordb.persist()
            return "file upload complited contune for chat"
        except Exception as e:
            st.error(f"Vector conversion error: {str(e)}")
            return None

class QBot():
    def prompt(self, user_input):
        persist_directory = "./chroma_db"
        try:
            llm = AzureChatOpenAI(
                openai_api_base=OPENAI_API_BASE,
                openai_api_version=OPENAI_API_VERSION,
                deployment_name=OPENAI_API_DEPLOYMENT,
                openai_api_key=OPENAI_API_KEY,
                openai_api_type=OPENAI_API_TYPE,
                model_name=OPENAI_API_MODEL
            )
            prompt_template = """
            {context}

            Question: {question}"""
            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )

            # Set up a retrieval-based QA chain using RetrievalQA
            chain_type_kwargs = {"prompt": PROMPT}
            vectordb = Chroma(persist_directory=persist_directory,embedding_function=embeddings)
            chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectordb.as_retriever(
                    search_type="similarity", search_kwargs={"k": 10}
                ),
                chain_type_kwargs=chain_type_kwargs,
                verbose=True
            )
            if not chain:
                return "The chain is not initialized. Embedding failed."
            else:
                response = chain.run(user_input)
                print(response)
                return response
        except Exception as e:
            st.error(f"QBot error: {str(e)}")
            return None

