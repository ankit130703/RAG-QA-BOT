from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import gradio as gr

load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings

def get_llm():
    llm= ChatGoogleGenerativeAI(
        model='gemini-1.5-flash-latest',
        temperature=0.2,
        max_output_tokens=128
    )
    return llm

def document_loader(file):
    loader = PyPDFLoader(file)
    doc=loader.load()
    return doc

def text_splitter(data):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128,
        length_function=len
    )
    chunks = text_splitter.split_documents(data)
    return chunks

def gemini_embeddings():
    embed=GoogleGenerativeAIEmbeddings(
        model='models/embedding-001',
    )

    return embed

def vector_db(chunks):
    model = gemini_embeddings()
    vectordb=Chroma.from_documents(chunks,model)
    return vectordb

def retreiver(file):
    splits=document_loader(file)
    chunks=text_splitter(splits)
    vectordb=vector_db(chunks)
    retreiver=vectordb.as_retriever()
    return retreiver

def retreiver_qa(file, query):
    llm = get_llm()
    retreiver_obj = retreiver(file)
    qa=RetrievalQA.from_chain_type(
        llm= llm,
        chain_type
        ='stuff',
        retriever =retreiver_obj,
        return_source_documents=False
    )

    response = qa.invoke(query)
    return response['result']

rag_app=gr.Interface(
    fn=retreiver_qa,
    inputs=[
        gr.File(file_count='single',file_types=['.pdf'], type="filepath",label ='Drop you file here'),
        gr.Textbox(label='Enter you query',lines=2, placeholder="Type your question here...")
    ],
    outputs=gr.Textbox(label='Result: '),
    title='RAG BOT',
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document."
)

rag_app.launch()