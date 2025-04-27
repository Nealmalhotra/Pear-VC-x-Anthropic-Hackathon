import sys
import os
import unittest
from fastapi.testclient import TestClient
import json
import tempfile

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import api
from tokenizer import MathBPETokenizer

class TestAPI(unittest.TestCase):
    def setUp(self):
        # Create a test client
        self.client = TestClient(api.app)
        
        # Mock the tokenizer for testing
        with tempfile.NamedTemporaryFile(suffix='.json') as tmp:
            tokenizer = MathBPETokenizer(vocab_size=1000, min_frequency=1)
            
            # Sample texts for training
            sample_texts = [
                "Let x be a real number.",
                "For all epsilon > 0, there exists delta > 0 such that...",
                "The integral of f(x) from a to b is...",
                "\\sum_{i=1}^{n} i^2 = \\frac{n(n+1)(2n+1)}{6}",
                "Prove that there are infinitely many prime numbers."
            ]
            
            # Train the tokenizer
            tokenizer.train(sample_texts, save_path=tmp.name)
            
            # Set the tokenizer in the API
            api.tokenizer = tokenizer
    
    def test_health_endpoint(self):
        """Test the health check endpoint."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["tokenizer_loaded"])
    
    def test_tokenize_endpoint_basic(self):
        """Test the tokenize endpoint with basic text."""
        payload = {"text": "Let x be a real number."}
        response = self.client.post("/tokenize", json=payload)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("tokens", data)
        self.assertIsInstance(data["tokens"], list)
        self.assertGreater(len(data["tokens"]), 0)
        
        # tokens_text should be None when not requested
        self.assertIsNone(data.get("tokens_text"))
    
    def test_tokenize_endpoint_with_tokens_text(self):
        """Test the tokenize endpoint with token text included."""
        payload = {"text": "Let x be a real number.", "return_text_tokens": True}
        response = self.client.post("/tokenize", json=payload)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("tokens", data)
        self.assertIn("tokens_text", data)
        self.assertIsInstance(data["tokens_text"], list)
        self.assertEqual(len(data["tokens"]), len(data["tokens_text"]))
    
    def test_tokenize_endpoint_with_math(self):
        """Test the tokenize endpoint with mathematical notation."""
        payload = {"text": "The sum is \\sum_{i=1}^{n} i^2 = \\frac{n(n+1)(2n+1)}{6}"}
        response = self.client.post("/tokenize", json=payload)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("tokens", data)
        self.assertGreater(len(data["tokens"]), 0)
    
    def test_tokenize_endpoint_with_proof(self):
        """Test the tokenize endpoint with a mathematical proof."""
        proof_text = """
        Prove that there are infinitely many prime numbers.
        
        Suppose, for contradiction, that there are only finitely many primes: p₁, p₂, ..., pₙ.
        Let P = p₁ × p₂ × ... × pₙ + 1.
        P is not divisible by any pᵢ (it leaves remainder 1).
        Therefore, P is either prime or divisible by a prime not in our list.
        Either way, our assumption was false.
        Therefore, there are infinitely many primes.
        """
        
        payload = {"text": proof_text, "return_text_tokens": True}
        response = self.client.post("/tokenize", json=payload)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("tokens", data)
        self.assertIn("tokens_text", data)
        self.assertGreater(len(data["tokens"]), 0)
        
        # Check that some key tokens are present
        tokens_text = data["tokens_text"]
        self.assertTrue(any("prime" in token for token in tokens_text))
    
    def test_invalid_request(self):
        """Test the tokenize endpoint with an invalid request."""
        # Missing 'text' field
        payload = {"return_text_tokens": True}
        response = self.client.post("/tokenize", json=payload)
        
        self.assertEqual(response.status_code, 422)  # Unprocessable Entity

if __name__ == "__main__":
    unittest.main() 