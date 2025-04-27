from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors
import json
import os
from typing import List, Dict, Union, Optional
import re

class MathBPETokenizer:
    """
    A Byte-Pair Encoding tokenizer with math-symbol extensions.
    Handles LaTeX and Markdown math notation effectively.
    """
    def __init__(
        self, 
        vocab_size: int = 30000, 
        min_frequency: int = 2,
        special_tokens: List[str] = ["<|endoftext|>", "<|pad|>", "<|unk|>"],
        pre_trained_path: Optional[str] = None
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens
        
        # Initialize the tokenizer
        if pre_trained_path and os.path.exists(pre_trained_path):
            self.tokenizer = Tokenizer.from_file(pre_trained_path)
        else:
            # Start with a byte-level BPE model
            self.tokenizer = Tokenizer(models.BPE())
            
            # Set up the pre-tokenizer to split on whitespace and punctuation
            self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
                pre_tokenizers.WhitespaceSplit(),
                pre_tokenizers.Punctuation(),
                # Add a custom pattern-based pre-tokenizer for LaTeX commands
                pre_tokenizers.Split(
                    pattern=r"(\\[a-zA-Z]+|\\[^a-zA-Z]|[\{\}\(\)\[\]])", 
                    behavior="isolated"
                )
            ])
            
            # Set up the decoder
            self.tokenizer.decoder = decoders.ByteLevel()
        
        # Configure the post-processor for special tokens
        self.tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{self.special_tokens[0]} $A {self.special_tokens[0]}",
            special_tokens=[(token, self.tokenizer.token_to_id(token) if self.tokenizer.token_to_id(token) else i) 
                           for i, token in enumerate(special_tokens)]
        )
        
        # Common LaTeX math symbols and commands
        self.math_symbols = [
            # Greek letters
            "\\alpha", "\\beta", "\\gamma", "\\delta", "\\epsilon", "\\zeta", "\\eta", "\\theta",
            "\\iota", "\\kappa", "\\lambda", "\\mu", "\\nu", "\\xi", "\\pi", "\\rho", "\\sigma",
            "\\tau", "\\upsilon", "\\phi", "\\chi", "\\psi", "\\omega",
            "\\Gamma", "\\Delta", "\\Theta", "\\Lambda", "\\Xi", "\\Pi", "\\Sigma", "\\Upsilon", 
            "\\Phi", "\\Psi", "\\Omega",
            
            # Math operators
            "\\sum", "\\prod", "\\lim", "\\int", "\\oint", "\\partial", "\\nabla", "\\infty",
            "\\forall", "\\exists", "\\nexists", "\\in", "\\notin", "\\subset", "\\supset",
            "\\cup", "\\cap", "\\emptyset", "\\mathbb{R}", "\\mathbb{Z}", "\\mathbb{N}", "\\mathbb{Q}",
            
            # Math symbols
            "\\equiv", "\\approx", "\\cong", "\\neq", "\\sim", "\\simeq", "\\propto",
            "\\leq", "\\geq", "\\ll", "\\gg", "\\prec", "\\succ", "\\preceq", "\\succeq",
            
            # Structural elements
            "\\frac", "\\sqrt", "\\overline", "\\underline", "\\overbrace", "\\underbrace",
            "\\begin{equation}", "\\end{equation}", "\\begin{align}", "\\end{align}",
            "\\begin{proof}", "\\end{proof}", "\\begin{theorem}", "\\end{theorem}",
            "\\begin{lemma}", "\\end{lemma}", "\\begin{corollary}", "\\end{corollary}",
            
            # Popular math environments
            "\\begin{matrix}", "\\end{matrix}", "\\begin{cases}", "\\end{cases}",
            "\\begin{array}", "\\end{array}", "\\begin{bmatrix}", "\\end{bmatrix}",
            
            # Common in proofs
            "\\implies", "\\iff", "\\therefore", "\\because", "\\to", "\\mapsto",
            "\\hfill\\square", "\\square", "\\blacksquare", "Q.E.D."
        ]
        
    def train(self, texts: List[str], save_path: str = "math_tokenizer.json"):
        """Train the tokenizer on a list of texts."""
        # Add math symbols to the training data to ensure they're in the vocabulary
        augmented_texts = texts + self.math_symbols
        
        # Create a trainer
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
            show_progress=True
        )
        
        # Train the tokenizer
        self.tokenizer.train_from_iterator(augmented_texts, trainer=trainer)
        
        # Save the trained tokenizer
        self.tokenizer.save(save_path)
        return self
    
    def encode(self, text: str) -> Dict[str, List[int]]:
        """Encode text to token IDs."""
        encoded = self.tokenizer.encode(text)
        return {
            "tokens": encoded.ids,
            "tokens_text": encoded.tokens
        }
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        return self.tokenizer.decode(token_ids)
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into tokens without converting to IDs."""
        encoded = self.tokenizer.encode(text)
        return encoded.tokens
        
    def save(self, path: str):
        """Save the tokenizer to a file."""
        self.tokenizer.save(path)
        
    @classmethod
    def from_file(cls, path: str):
        """Load a tokenizer from a file."""
        tokenizer = cls(pre_trained_path=path)
        return tokenizer 