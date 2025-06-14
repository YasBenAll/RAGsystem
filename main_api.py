# main_api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# --- Import the core RAG logic from your existing project ---
from src.langchain_rag import create_langchain_rag_chain

# --- Pydantic Models for Request and Response ---
class QueryRequest(BaseModel):
    """The request model containing the user's question."""
    question: str

class QueryResponse(BaseModel):
    """The response model containing the agent's answer."""
    answer: str

# Initialize the FastAPI app
app = FastAPI(
    title="RAG Agent API",
    description="An API for interacting with a multi-tool reasoning agent.",
    version="1.0.0",
)

# --- Direct Model Loading at Module Level ---
# This code will run as soon as uvicorn imports this file.
print("INFO:     Attempting to load RAG Chain at module level...")
rag_chain = None
try:
    rag_chain = create_langchain_rag_chain()
    print("INFO:     RAG Chain Loaded Successfully.")
except Exception as e:
    # If there's any error during model loading, print it.
    print(f"FATAL:    Failed to load RAG Chain on startup: {e}")
    # You might want to re-raise the exception or handle it as needed
    rag_chain = None

@app.post("/ask_agent", response_model=QueryResponse)
def ask_agent(request: QueryRequest):
    """
    Endpoint to ask a question to the RAG chain.
    """
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG Chain is not available. Check server logs for startup errors.")
    
    print(f"INFO:     Received question: {request.question}")
    
    try:
        response_data = rag_chain.invoke({"query": request.question})
        answer = response_data.get('result', 'No answer found in the chain response.')
    except Exception as e:
        print(f"ERROR:    Error during RAG chain invocation: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing the request.")

    print(f"INFO:     RAG Chain's answer: {answer}")
    return QueryResponse(answer=answer)

# A simple root endpoint to check if the server is running
@app.get("/")
def read_root():
    """A simple endpoint to confirm the API is running."""
    return {"status": "RAG Agent API is running"}

# This block allows you to run the server directly from this script
# for easy testing.
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
