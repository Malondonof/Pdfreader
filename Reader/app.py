import os
import streamlit as st
from PIL import Image
import PyPDF2
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

st.title('Generación Aumentada por Recuperación (RAG) 💬')
image = Image.open("Reader/Robotpdf.jpg")
st.image(image, caption="robotomorrow")

with st.sidebar:
    st.subheader("Este Agente, te ayudará a realizar algo de análisis sobre el PDF cargado")

ke = st.text_input('Ingresa tu Clave')
os.environ['OPENAI_API_KEY'] = ke

pdfFileObj = open('Reader/Documento sin título.pdf', 'rb')
pdfReader = PdfReader(pdfFileObj)

# Carga de archivo
pdf = st.file_uploader("Carga el archivo PDF", type="pdf")

# Extracción del texto
if pdf is not None:
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # División en fragmentos
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=20, length_function=len)
    chunks = text_splitter.split_text(text)

    # Creación de embeddings
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    # Pregunta del usuario
    st.subheader("Escribe que quieres saber sobre el documento")
    user_question = st.text_area(" ")
    if user_question:
        docs = knowledge_base.similarity_search(user_question)

        llm = OpenAI(model_name="gpt-4o-mini")
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=user_question)
            print(cb)
        st.write(response)
