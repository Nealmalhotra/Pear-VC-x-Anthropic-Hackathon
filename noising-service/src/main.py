import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Set default port
PORT = int(os.getenv("PORT", 8000))
HOST = os.getenv("HOST", "0.0.0.0")

if __name__ == "__main__":
    print(f"Starting noising service on {HOST}:{PORT}")
    uvicorn.run("src.api:app", host=HOST, port=PORT, reload=True) 