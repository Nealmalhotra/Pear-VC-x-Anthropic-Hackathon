import sys
import os
import unittest
import json
import tempfile

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from tokenizer import MathBPETokenizer

class TestMathBPETokenizer(unittest.TestCase):
    def setUp(self):
        # Create a tokenizer with a small vocab size for faster tests
        self.tokenizer = MathBPETokenizer(vocab_size=1000, min_frequency=1)
        
        # Sample texts for training
        self.sample_texts = [
            "Let x be a real number.",
            "For all epsilon > 0, there exists delta > 0 such that...",
            "The integral of f(x) from a to b is...",
            "\\sum_{i=1}^{n} i^2 = \\frac{n(n+1)(2n+1)}{6}",
            "If A and B are sets, then A \\cup B = \\{x | x \\in A \\text{ or } x \\in B\\}",
            "Prove that there are infinitely many prime numbers.",
            "Let P(n) be the statement that n^2 + n + 41 is prime for all natural numbers n.",
            "Consider a sequence defined by a_n = a_{n-1} + a_{n-2} with a_0 = 0 and a_1 = 1."
        ]
        
        # Train the tokenizer
        with tempfile.NamedTemporaryFile(suffix='.json') as tmp:
            self.tokenizer.train(self.sample_texts, save_path=tmp.name)
    
    def test_basic_tokenization(self):
        """Test tokenization of basic mathematical text."""
        text = "Let x be a real number."
        tokens = self.tokenizer.encode(text)
        self.assertIsInstance(tokens, dict)
        self.assertIn("tokens", tokens)
        self.assertIn("tokens_text", tokens)
        self.assertGreater(len(tokens["tokens"]), 0)
        
        # Ensure we can decode back to text
        decoded = self.tokenizer.decode(tokens["tokens"])
        self.assertIsInstance(decoded, str)
    
    def test_greek_symbols(self):
        """Test tokenization of Greek symbols."""
        text = "\\alpha, \\beta, \\gamma, \\delta"
        tokens = self.tokenizer.encode(text)
        self.assertGreater(len(tokens["tokens"]), 0)
        
        # Check individual tokens are present
        self.assertIn("\\", tokens["tokens_text"])
        self.assertIn("alpha", tokens["tokens_text"])
    
    def test_math_expressions(self):
        """Test tokenization of math expressions."""
        text = "\\frac{1}{2} + \\frac{1}{3} = \\frac{5}{6}"
        tokens = self.tokenizer.encode(text)
        self.assertGreater(len(tokens["tokens"]), 0)
        
        # Check individual tokens
        self.assertIn("frac", tokens["tokens_text"])
    
    def test_nested_math(self):
        """Test tokenization of nested math structures."""
        text = "\\begin{align} \\sum_{i=1}^{n} i &= \\frac{n(n+1)}{2} \\\\ \\end{align}"
        tokens = self.tokenizer.encode(text)
        
        # Check environment components
        self.assertIn("begin", tokens["tokens_text"])
        self.assertIn("align", tokens["tokens_text"])
        self.assertIn("end", tokens["tokens_text"])
    
    def test_complex_proof(self):
        """Test tokenization of a complex mathematical proof."""
        text = """
        \\begin{proof}
        Let $S = \\{2, 3, 5, \\ldots, p\\}$ be the set of all primes up to $p$.
        Consider the number $q = (\\prod_{p \\in S} p) + 1$.
        
        By construction, $q > p$ and when $q$ is divided by any prime in $S$, the remainder is 1.
        Therefore, $q$ is not divisible by any prime in $S$.
        
        Now, either $q$ is itself prime, or it is divisible by some prime $p' > p$.
        In either case, we have found a prime larger than $p$.
        
        Since $p$ was arbitrary, we conclude that there are infinitely many primes.
        \\end{proof}
        """
        tokens = self.tokenizer.encode(text)
        
        # Check environment components
        self.assertIn("begin", tokens["tokens_text"])
        self.assertIn("proof", tokens["tokens_text"])
        self.assertIn("end", tokens["tokens_text"])
        
        # Decode and check that the text contains key phrases
        decoded = self.tokenizer.decode(tokens["tokens"])
        self.assertIn("infinitely", decoded)
        self.assertIn("prime", decoded)
    
    def test_edge_case_long_formula(self):
        """Test tokenization of an extremely long formula."""
        # Generate a long formula with repeated elements
        repeated = "\\sum_{i=1}^{n} " * 10 + "i^2"
        text = f"Consider the following sum: {repeated}"
        
        tokens = self.tokenizer.encode(text)
        self.assertGreater(len(tokens["tokens"]), 0)
        
        # Ensure we can decode it and it contains key terms
        decoded = self.tokenizer.decode(tokens["tokens"])
        self.assertIn("sum", decoded)
        self.assertIn("i^2", decoded)
    
    def test_edge_case_unicode_math(self):
        """Test tokenization of Unicode math symbols (outside of LaTeX)."""
        text = "The equation α² + β² = γ² is a form of the Pythagorean theorem."
        tokens = self.tokenizer.encode(text)
        
        # Decode and check for key terms
        decoded = self.tokenizer.decode(tokens["tokens"])
        self.assertIn("equation", decoded)
        self.assertIn("Pythagorean", decoded)
    
    def test_edge_case_mixed_notation(self):
        """Test tokenization of mixed Markdown and LaTeX notation."""
        text = "In **markdown** with math $\\sqrt{x^2 + y^2}$ and more text."
        tokens = self.tokenizer.encode(text)
        
        # Check for key tokens
        decoded = self.tokenizer.decode(tokens["tokens"])
        self.assertIn("markdown", decoded)
        self.assertIn("sqrt", decoded)

if __name__ == "__main__":
    unittest.main() 