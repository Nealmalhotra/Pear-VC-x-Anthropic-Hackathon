# Placeholder for retrieval client logic 

import asyncio
from typing import List, Dict, Any
import httpx
import os

from fastapi import HTTPException
from openai import AsyncOpenAI  # Use async client
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()
# --- Environment Variables & Configuration ---
RETRIEVAL_SERVICE_URL = os.getenv("RETRIEVAL_SERVICE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT") # For Pinecone serverless/starter, this is just the region e.g. "us-east-1"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX", "metamath-theorems")
EMBEDDING_MODEL = "text-embedding-3-small"
TOP_K_RESULTS = 5 # Number of lemmas to retrieve

# --- Client Initialization ---
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY environment variable not set.")
    openai_client = None
else:
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
    print("Warning: PINECONE_API_KEY or PINECONE_ENVIRONMENT environment variables not set.")
    pc = None
    index = None
else:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        index = pc.Index(PINECONE_INDEX_NAME)
        # # environment parameter is deprecated for serverless/starter, handled by api key
        # if PINECONE_INDEX_NAME not in pc.list_indexes():
        #     print(f"Error: Pinecone index '{PINECONE_INDEX_NAME}' not found in environment '{PINECONE_ENVIRONMENT}'. Please create it.")
        #     index = None
        # else:
        #     index = pc.Index(PINECONE_INDEX_NAME)
        #     print(f"Successfully connected to Pinecone index '{PINECONE_INDEX_NAME}'.")
    except Exception as e:
        print(f"Error initializing Pinecone: {e}")
        pc = None
        index = None

async def get_openai_embedding(text: str) -> List[float]:
    """Generates an embedding for the given text using OpenAI."""
    if not openai_client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized. Check API key.")
    try:
        response = await openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[text] # API expects a list of texts
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error calling OpenAI embedding API: {e}")
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")

def query_pinecone_sync(embedding: List[float]) -> List[Dict[str, Any]]:
    """Synchronous function to query Pinecone index."""
    if not index:
        raise HTTPException(status_code=500, detail="Pinecone index not initialized. Check API key/environment/index name.")
    try:
        results = index.query(
            vector=embedding,
            top_k=TOP_K_RESULTS,
            include_metadata=True
        )
        # Format results
        formatted_lemmas = []
        for match in results.get('matches', []):
            metadata = match.get('metadata', {})
            # Combine description and statement for the 'text' field
            description = metadata.get('description', '')
            statement = metadata.get('statement', 'No statement found')
            text = f"{description}\n{statement}" if description else statement
            formatted_lemmas.append({
                'id': match.get('id', 'N/A'),
                'text': text.strip(),
                'score': match.get('score', 0.0)
            })
        return formatted_lemmas
    except Exception as e:
        print(f"Error querying Pinecone index: {e}")
        raise HTTPException(status_code=500, detail=f"Pinecone query error: {e}")

async def get_relevant_lemmas(tokens: List[int], top_k: int = TOP_K_RESULTS) -> List[Dict[str, Any]]:
    """Retrieves relevant lemmas from Pinecone based on input tokens."""
    print(f"Attempting retrieval for tokens: {tokens[:10]}... (using Pinecone)")

    if not openai_client or not index:
        print("OpenAI or Pinecone client not initialized. Returning empty list.")
        # Or raise an error depending on desired behavior
        # raise HTTPException(status_code=503, detail="Retrieval dependencies not configured.")
        return []

    # Convert tokens to a string for embedding (simple space separation)
    # Consider if a more sophisticated token-to-text mapping is needed
    query_text = ' '.join(map(str, tokens))

    try:
        print(f"Generating embedding for query text (first 50 chars): {query_text[:50]}...")
        embedding = await get_openai_embedding(query_text)
        print(f"Generated embedding vector (dim: {len(embedding)}). Querying Pinecone...")
        
        # Run synchronous Pinecone query in a thread pool
        loop = asyncio.get_running_loop()
        lemmas = await loop.run_in_executor(
            None,  # Use default executor
            query_pinecone_sync, 
            embedding
        )
        
        print(f"Retrieved {len(lemmas)} lemmas from Pinecone.")
        return lemmas

    except HTTPException as e:
        # Re-raise HTTP exceptions from underlying calls
        raise e
    except Exception as e:
        # Catch any other unexpected errors during the process
        print(f"Unexpected error during retrieval pipeline: {e}")
        raise HTTPException(status_code=500, detail="Internal error during retrieval pipeline")

    # TODO: Replace with actual httpx call to RETRIEVAL_SERVICE_URL /retrieve
    # try:
    #     async with httpx.AsyncClient() as client:
    #         # Need to decide how to represent 'tokens' as a query parameter
    #         # Assuming a simple comma-separated string for now
    #         query_param = ','.join(map(str, tokens))
    #         url = f"{RETRIEVAL_SERVICE_URL}/retrieve?query={query_param}"
    #         response = await client.get(url, timeout=5.0) # Add timeout
    #         response.raise_for_status() # Raise exception for 4xx/5xx errors
    #         lemmas = response.json() # Assuming the service returns JSON
    #         print(f"Retrieved {len(lemmas)} lemmas.")
    #         return lemmas
    # except httpx.HTTPStatusError as e:
    #     print(f"HTTP error calling retrieval service: {e.response.status_code} - {e.response.text}")
    #     # Re-raise or return empty list/error indicator based on desired handling
    #     raise HTTPException(status_code=503, detail="Retrieval service error")
    # except httpx.RequestError as e:
    #     print(f"Network error calling retrieval service: {e}")
    #     raise HTTPException(status_code=503, detail="Retrieval service unavailable")
    # except Exception as e:
    #     print(f"Unexpected error during retrieval: {e}")
    #     raise HTTPException(status_code=500, detail="Internal error during retrieval")

    # --- Dummy Data --- 
    await asyncio.sleep(0.05) # Simulate network latency
    dummy_lemmas = [
        {"id": "lemma_1", "text": "∀x, P(x) → Q(x)", "score": 0.95},
        {"id": "lemma_2", "text": "∃y, R(y)", "score": 0.88},
    ]
    print(f"Returning {len(dummy_lemmas)} dummy lemmas.")
    return dummy_lemmas 