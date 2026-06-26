# pdf-chatbot-using-langchain
# 📄 Chat With PDF – RAG-Based Question Answering System

## Overview

This project is a Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions about their content. The system retrieves the most relevant sections from the document using semantic search and generates context-aware answers.

## Features

* Upload and process PDF documents
* Extract and split text into manageable chunks
* Generate embeddings using Hugging Face models
* Store vectors using FAISS
* Perform semantic similarity search
* Generate answers from retrieved context
* Interactive Streamlit interface

## Tech Stack

* **Frontend:** Streamlit
* **Framework:** LangChain
* **Embeddings:** Hugging Face (all-MiniLM-L6-v2)
* **Vector Store:** FAISS
* **LLM:** Google FLAN-T5
* **PDF Processing:** PyPDF2

## Workflow

1. Upload PDF document
2. Extract text using PyPDF2
3. Split text into chunks
4. Generate embeddings with Hugging Face
5. Store embeddings in FAISS
6. Retrieve relevant chunks based on user query
7. Generate answers using FLAN-T5

## Learning Outcomes

Through this project, I gained hands-on experience with:

* Retrieval-Augmented Generation (RAG)
* Vector databases and semantic search
* LangChain pipelines
* Hugging Face models and embeddings
* Streamlit application development
* End-to-end NLP application design


## Author

Shreeya Suriyakumar

Aspiring Data Scientist | Python | Machine Learning | Data Analytics


