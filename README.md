# AI Agent: From Monolithic RAG to a Decoupled Service
Advanced AI Agent with FastAPI and Streamlit

This repository showcases a modern, production-ready architecture for building sophisticated AI agents. The project is currently a monolithic Streamlit application that functions as a Retrieval-Augmented Generation (RAG) chatbot. It can answer questions based on a provided set of documents. Work is in progress to upgrade this chatbot into a multi-tool reasoning agent.

The system currently runs as a single process. Streamlit provides the UI, and all the AI logic (data loading, embedding, RAG chain, LLM interaction) is executed within the same application.
![RAG_system_diagram drawio](https://github.com/user-attachments/assets/5aa7a6e4-242a-419a-9985-5b4d2d263f46)

## How to run
The project currently runs as a single Streamlit application:

      # Navigate to the project directory
      pip install -r requirements.txt

### 1. Run the Web Application (Streamlit)

      pythom main.py --ui web

### 2. Run the Command-Line Interface (Streamlit)
      pythom main.py --ui cli

## Proposed Future Architecture
To address the limitations of the current design and to demonstrate professional software engineering practices, the project's next major goal is to be refactored into a decoupled client-server architecture using FastAPI for the backend, separating the AI logic from the Streamlit UI.
