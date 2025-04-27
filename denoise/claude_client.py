# Client for interacting with the Anthropic Claude API 
import os
import anthropic
from fastapi import HTTPException
import time 

# Retrieve API key from environment variable
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
MAX_CLAUDE_RETRIES = int(os.getenv("MAX_CLAUDE_RETRIES", 3))
RETRY_DELAY_SECONDS = float(os.getenv("RETRY_DELAY_SECONDS", 1.0))

# Initialize the Anthropic client
# Handle potential missing API key
if not CLAUDE_API_KEY:
    print("Warning: CLAUDE_API_KEY environment variable not set.")
    # You might want to raise an error or use a default behavior
    # For now, we'll allow it to proceed but client calls will fail
    client = None
else:
    try:
        client = anthropic.AsyncAnthropic(
            # Defaults to os.environ.get("ANTHROPIC_API_KEY")
            # Pass explicitly if using a different env var name like CLAUDE_API_KEY
            api_key=CLAUDE_API_KEY
        )
    except Exception as e:
        print(f"Error initializing Anthropic client: {e}")
        client = None

# Define default parameters for the Claude API call
DEFAULT_MODEL = "claude-3-5-sonnet-20241022" # Or another suitable model
MAX_TOKENS_TO_SAMPLE = 512 # Adjust as needed
TEMPERATURE = 0.1 # Low temperature for more deterministic output (as per PRD FR2)

# Remove cache decorator
async def call_claude_api(prompt: str) -> str:
    """
    Calls the Claude API with the given prompt and returns the text response.
    Includes basic retry logic and rate limit handling.
    # Caching removed due to incompatibility with async def
    """
    if not client:
        print("Error: Anthropic client is not initialized (check API key).")
        raise HTTPException(status_code=500, detail="Claude API client not configured")

    last_exception = None
    for attempt in range(MAX_CLAUDE_RETRIES):
        try:
            # Find where the Assistant part starts to separate user/system prompt
            assistant_marker = "\nAssistant:"
            if assistant_marker in prompt:
                user_prompt_content = prompt[:prompt.find(assistant_marker)].strip()
            else:
                user_prompt_content = prompt 

            print(f"Claude API Call - Attempt {attempt + 1}/{MAX_CLAUDE_RETRIES}")
            message = await client.messages.create(
                model=DEFAULT_MODEL,
                max_tokens=MAX_TOKENS_TO_SAMPLE,
                temperature=TEMPERATURE,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt_content
                    }
                ]
            )
            
            if message.content and isinstance(message.content, list) and hasattr(message.content[0], 'text'):
                 response_text = message.content[0].text
                 print(f"Claude API call successful. Response length: {len(response_text)}")
                 return response_text # Success, return immediately
            else:
                print(f"Claude API returned unexpected response structure: {message.content}")
                # Treat unexpected structure as a retryable error for now
                last_exception = HTTPException(status_code=500, detail="Claude API returned unexpected response format")

        except anthropic.RateLimitError as e:
            print(f"Claude API rate limit exceeded on attempt {attempt + 1}: {e}")
            last_exception = HTTPException(status_code=429, detail="Claude API rate limit exceeded")
            # For rate limits, often best to break and return 429 immediately, but we'll retry here per simple retry logic
            # In production, consider exponential backoff or breaking the loop.
        except anthropic.APIConnectionError as e:
            print(f"Claude API connection error on attempt {attempt + 1}: {e}")
            last_exception = HTTPException(status_code=503, detail=f"Could not connect to Claude API: {e}")
        except anthropic.APIStatusError as e:
            # Retry on 5xx errors, raise immediately for others (like 4xx auth errors)
            if e.status_code >= 500:
                 print(f"Claude API status error {e.status_code} on attempt {attempt + 1}: {e.message}")
                 last_exception = HTTPException(status_code=e.status_code, detail=f"Claude API server error: {e.message}")
            else:
                 print(f"Claude API non-retryable status error: {e.status_code} - {e.message}")
                 # Raise immediately for non-retryable errors (e.g., 400, 401)
                 raise HTTPException(status_code=e.status_code, detail=f"Claude API error: {e.message}") from e
        except Exception as e:
            # Catch other potential errors during the attempt
            print(f"Unexpected error during Claude API attempt {attempt + 1}: {e}")
            last_exception = HTTPException(status_code=500, detail=f"Internal error interacting with Claude API: {e}")
        
        # If we haven't returned yet, wait before retrying (unless it's the last attempt)
        if attempt < MAX_CLAUDE_RETRIES - 1:
            print(f"Waiting {RETRY_DELAY_SECONDS}s before next attempt...")
            time.sleep(RETRY_DELAY_SECONDS)

    # If all retries failed, raise the last encountered exception
    print(f"Claude API call failed after {MAX_CLAUDE_RETRIES} attempts.")
    if last_exception:
        raise last_exception
    else:
        # Should not happen if loop runs, but as a fallback
        raise HTTPException(status_code=500, detail="Claude API call failed after multiple retries for an unknown reason") 