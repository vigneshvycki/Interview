import os
from  dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()
groq_api=os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY']=os.getenv("GOOGLE_API_KEY")

st.title("Interview")
llm=ChatGroq(groq_api=groq_api,model='Llama3-8b-8192')
prompt=ChatPromptTemplate.from_template(
    '''
    Use only the information provided in the document to answer the question.
    Provide a clear and accurate answer.
    <context>
    {context}
    <context>
    Questions:{input}
    '''
)
def embedding():
    if "vector" not in st.session_state:
        st.session_state.embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        st.session_state.loader=PyPDFLoader('/Users/admin/Desktop/LLM/CodebasicsLLM/interview/question.py')
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
        st.session_state.final_document=st.session_state.splitter.split_documents(st.session_state.docs[:20])
        st.session_state.vector=FAISS.from_documents(st.session_state.final_document,st.session_state.embedding)
embedding()
Prompt1=st.text_input("Enter Your Questions:")

if Prompt1:
    documennt_chain=create_stuff_documents_chain(llm,prompt)
    retriver=st.session_state.vector.as_retriever()
    retreival_chain=create_retrieval_chain(retriver,documennt_chain)
    response=retreival_chain.invoke({'input':Prompt1})
    st.write(response['answer'])
        

 