import streamlit as st
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline 
from langchain_community.llms import HuggingFacePipeline
import os

with st.sidebar:
    st.title('🤗💬 LLM Chat App')
    st.markdown('''
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
                
    - [LangChain](https://python.langchain.com/)
                
    - [OpenAI](https://platform.openai.com/docs/models)LLM model    
                
     ''')
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.markdown('Made with ❤️ by [Shreeya](https://github.com/shreeya07)')


def main():
    st.header("Chat With PDF😊") 

    load_dotenv()

    pdf = st.file_uploader("Upload your PDF file here")
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        #st.write(pdf_reader)

        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text = text)
        embeddings = HuggingFaceEmbeddings(
            model_name = "sentence-transformers/all-MiniLM-L6-v2")
        

        VectorStore = FAISS.from_texts(chunks, embedding = embeddings)
        store_name =pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb")as f:
                VectorStore = pickle.load(f)
            #st.write("Embeddings Loaded from the Disk")
        else:
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)

        query=st.text_input("Ask questions about your PDF file:")
        st.write(query)
        if query:
            docs = VectorStore.similarity_search(query=query, k = 3)

            st.write("Searching documents...")
            docs = VectorStore.similarity_search(query=query,k=3)
            st.write("Generating answer...")
            st.write("Loading model...")
            pipe = pipeline("text-generation", model = "google/flan-t5-small",max_new_tokens=50)
            st.write("Model loaded!!")
            llm = HuggingFacePipeline(pipeline = pipe)
            context = "\n".join([doc.page_content for doc in docs])
            prompt = f"""
            Answer the question based on the context below.
            Context:
            {context}

            Question:
            {query}

            Answer:
            """
            response = llm.invoke(prompt)
            st.write(response)
        #st.write(chunks)
        #st.write(text)
            #st.write("Embeddings Computation Completed")
if __name__ == "__main__":
    main()
    