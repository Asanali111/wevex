import uvicorn
from app.main import app
import os
import sys

if __name__ == "__main__":
    # Get the port from environment or default to 8000
    port = int(os.environ.get("PORT", 8000))
    # Run the uvicorn server programmatically
    uvicorn.run(app, host="127.0.0.1", port=port)
