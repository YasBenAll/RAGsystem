# main_api.py
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

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

@app.post("/ask_agent", response_model=QueryResponse)
def ask_agent(request: QueryRequest):
    """
    Endpoint to ask a question to the reasoning agent.

    Takes a JSON with a "question" key and returns a JSON with an "answer" key.
    
    (Note: This is currently a mocked endpoint and does not call the real agent yet.)
    """
    # For now, we just echo back the question in the answer.
    # In the next step, we will replace this with a real call to our agent.
    mock_answer = f"You asked: '{request.question}'. The real agent is not connected yet."
    return QueryResponse(answer=mock_answer)

# A simple root endpoint to check if the server is running
@app.get("/")
def read_root():
    """A simple endpoint to confirm the API is running."""
    return {"status": "RAG Agent API is running"}

# This block allows you to run the server directly from this script
# for easy testing.
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
