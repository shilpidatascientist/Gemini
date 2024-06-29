import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter # for diving into chunks
from langchain.chains.combine_documents import create_stuff_documents_chain # to get the doc in Q&A
from langchain_core.prompts import ChatPromptTemplate # for customizing the prompts
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS  #FAISS-vector store db created by facebook which stores vectors & perform semantic search or similarity search
from langchain_community.document_loaders import PyPDFDirectoryLoader # for reading PDF files
from langchain_google_genai import GoogleGenerativeAIEmbeddings #embeddings techniques

from dotenv import load_dotenv

load_dotenv()

## load groq & GenAI api from .env file
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

st.title("Gemma Model Document Q&A")

llm=ChatGroq(groq_api_key=groq_api_key,model_name="Gemma-7b-it")

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided content only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}
"""
)

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader=PyPDFDirectoryLoader("./us_census")  # Data Ingestion
        st.session_state.docs=st.session_state.loader.load()  # Load documents
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)#Chunk creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])#Splitting
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

prompt1= st.text_input("What you want to ask from the documents?")

if st.button("Creating Vector Store"): #on click of this button,embeddings are created
    vector_embedding()
    st.write("Vector Store DB is ready")

import time

if prompt1:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever() #return response to end user 
    retrieval_chain = create_retrieval_chain(retriever,document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({'input':prompt1})
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    #With a streamlit expander
    with st.expander("Document Similarity search"):
        #find relevant chunks
        for i,doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("----------")