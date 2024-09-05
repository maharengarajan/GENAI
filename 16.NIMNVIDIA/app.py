import streamlit as st
import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings,ChatNVIDIA
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

def configure():
  load_dotenv()

configure()
NVIDIA_API_KEY=os.getenv("NVIDIA_API_KEY")
# os.environ['NVIDIA_API_KEY']=os.getenv("NVIDIA_API_KEY")

#as long as the user keeps the tab open the app's frontend maintains as active connection with backend

def vector_embeddings():
  if "vectors" not in st.session_state:
    st.session_state.embeddings=NVIDIAEmbeddings()
    st.session_state.docs=PyPDFDirectoryLoader("16.NIMNVIDIA/us_census").load() #Data ingestion
    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=700,chunk_overlap=50) #chunk creation
    st.session_state.final_docs=st.session_state.text_splitter.split_documents(st.session_state.docs[:30]) 
    st.session_state.vectors=FAISS.from_documents(st.session_state.final_docs,st.session_state.embeddings)


## Set upi the Stramlit app
st.set_page_config(page_title="Document QA",page_icon="🧮")
st.title("Document QA using NVIDIA NIM")
llm=ChatNVIDIA(model="meta/llama-3.1-8b-instruct")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)


prompt_template=st.text_input("Enter Your Question From Doduments")

if st.button(label="Start Document embedding"):
    vector_embeddings()
    st.write("vector store DB is ready")

if prompt_template:
   pass
   document_chain=create_stuff_documents_chain(llm=llm,prompt=prompt)
   retriever=st.session_state.vectors.as_retriever()
   retrieval_chain=create_retrieval_chain(retriever,document_chain)
   response=retrieval_chain.invoke({"input": prompt_template})
   st.write(response['answer'])


  