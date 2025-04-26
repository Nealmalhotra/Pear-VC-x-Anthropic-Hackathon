from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from tokenizer import MathTokenizer

app = FastAPI(
    title="Mathematical Expression Tokenizer",
    description="A service that tokenizes mathematical expressions using Byte-Pair Encoding with math-symbol extensions",
    version="1.0.0"
)

# Initialize tokenizer
tokenizer = MathTokenizer()

class TokenizeRequest(BaseModel):
    text: str
    add_special_tokens: Optional[bool] = True

class TokenizeResponse(BaseModel):
    token_ids: List[int]
    tokens: List[str]
    decoded: str

@app.post("/tokenize", response_model=TokenizeResponse)
async def tokenize(request: TokenizeRequest):
    """
    Tokenize a mathematical expression or prompt.
    
    Example requests:
    - "prove that 2 plus 2 equals 4"
    - "solve the equation x^2 + 2x + 1 = 0"
    - "what is the derivative of sin(x)?"
    """
    try:
        # Tokenize the input text
        token_ids = tokenizer.tokenize(
            request.text,
            add_special_tokens=request.add_special_tokens
        )
        
        # Get the tokens
        tokens = [tokenizer.reverse_vocab.get(token_id, tokenizer.unk_token) for token_id in token_ids]
        
        # Decode back to text
        decoded = tokenizer.decode(token_ids)
        
        return TokenizeResponse(
            token_ids=token_ids,
            tokens=tokens,
            decoded=decoded
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "vocab_size": tokenizer.get_vocab_size()
    }

@app.post("/train")
async def train_tokenizer(texts: List[str], num_merges: int = 1000):
    """
    Train the tokenizer on a corpus of mathematical expressions.
    
    Args:
        texts: List of mathematical expressions to train on
        num_merges: Number of BPE merges to perform
        
    Returns:
        Training status
    """
    try:
        tokenizer.train(texts, num_merges)
        return {
            "status": "success",
            "vocab_size": tokenizer.get_vocab_size(),
            "num_merges": num_merges
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save")
async def save_tokenizer(path: str):
    """
    Save the tokenizer's vocabulary and merges.
    
    Args:
        path: Path to save the tokenizer
        
    Returns:
        Save status
    """
    try:
        tokenizer.save(path)
        return {"status": "success", "path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load")
async def load_tokenizer(path: str):
    """
    Load the tokenizer's vocabulary and merges.
    
    Args:
        path: Path to load the tokenizer from
        
    Returns:
        Load status
    """
    try:
        tokenizer.load(path)
        return {
            "status": "success",
            "path": path,
            "vocab_size": tokenizer.get_vocab_size()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tokenize")
async def tokenize_text(request: TokenizeRequest):
    try:
        # Tokenize the input text
        token_ids = tokenizer.tokenize(request.text)
        
        # Convert token IDs back to tokens for demonstration
        tokens = [tokenizer.reverse_vocab.get(token_id, "[UNK]") for token_id in token_ids]
        
        return {
            "token_ids": token_ids,
            "tokens": tokens,
            "decoded": tokenizer.decode(token_ids)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 