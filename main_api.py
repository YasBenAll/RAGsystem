# main_api.py
from fastapi import FastAPI
import uvicorn

# Initialize the FastAPI app
app = FastAPI(
    title="RAG Agent API",
    description="An API for interacting with a multi-tool reasoning agent.",
    version="1.0.0",
)

# A simple root endpoint to check if the server is running
@app.get("/")
def read_root():
    """A simple endpoint to confirm the API is running."""
    return {"status": "RAG Agent API is running"}

# This block allows you to run the server directly from this script
# for easy testing.
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
