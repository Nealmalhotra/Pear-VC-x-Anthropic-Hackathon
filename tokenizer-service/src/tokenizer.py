from typing import List, Dict, Set, Optional
import re
from collections import Counter
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
    factorial_notation,
    auto_number,
    split_symbols_custom,
)
from sympy import (
    # Basic arithmetic
    Add, Mul, Pow, div,
    # Special numbers
    pi, E, I, oo,
    # Trigonometric functions
    sin, cos, tan, cot, sec, csc,
    asin, acos, atan, acot, asec, acsc,
    # Hyperbolic functions
    sinh, cosh, tanh, coth, sech, csch,
    asinh, acosh, atanh, acoth, asech, acsch,
    # Logarithmic functions
    log, ln,
    # Special functions
    gamma, zeta, erf, erfc, erfi, erfinv, erfcinv,
    # Matrix operations
    Matrix, det, trace, transpose, adjoint,
    # Calculus
    diff, integrate, limit, series,
    # Set theory
    Set, Union, Intersection, Complement,
    # Logic
    And, Or, Not, Implies, Equivalent,
    # Number theory
    gcd, lcm, factor, isprime,
    # Special symbols
    Symbol, symbols, Function, Wild, WildFunction
)

class MathTokenizer:
    def __init__(self, vocab_size: int = 32000):
        """
        Initialize the tokenizer with a specified vocabulary size.
        
        Args:
            vocab_size: Maximum size of the vocabulary (default: 32000)
        """
        self.vocab_size = vocab_size
        self.merges = {}  # BPE merges
        self.vocab = {}   # Vocabulary mapping
        self.reverse_vocab = {}  # Reverse vocabulary mapping
        
        # Special tokens
        self.unk_token = "[UNK]"
        self.pad_token = "[PAD]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.mask_token = "[MASK]"
        
        # Basic mathematical operators
        self.basic_operators = {
            '+': 'PLUS',
            '-': 'MINUS',
            '*': 'MULTIPLY',
            '/': 'DIVIDE',
            '÷': 'DIVIDE',
            '^': 'POWER',
            '(': 'LEFT_PAREN',
            ')': 'RIGHT_PAREN',
            '[': 'LEFT_BRACKET',
            ']': 'RIGHT_BRACKET',
            '{': 'LEFT_BRACE',
            '}': 'RIGHT_BRACE',
            '=': 'EQUALS',
            ',': 'COMMA',
            '.': 'PERIOD'
        }
        
        # Enhanced SymPy transformations for parsing
        self.transformations = (
            standard_transformations +
            (implicit_multiplication_application,
             convert_xor,
             factorial_notation,
             auto_number,
             split_symbols_custom)
        )
        
        # Initialize vocabulary with basic tokens
        self._initialize_vocabulary()
        
        # Load all symbol databases
        self._load_unicode_math_symbols()
        self._load_latex_symbols()
        self._load_mathml_symbols()
        self._load_additional_unicode_blocks()
        self._load_additional_latex_packages()
        self._load_additional_mathml_features()
        self._load_other_notation_systems()
        self._load_specialized_math_symbols()
        self._load_formal_logic_symbols()

    def _initialize_vocabulary(self):
        """
        Initialize the vocabulary with basic tokens and special tokens.
        """
        # Add special tokens
        self.vocab[self.unk_token] = 0
        self.vocab[self.pad_token] = 1
        self.vocab[self.cls_token] = 2
        self.vocab[self.sep_token] = 3
        self.vocab[self.mask_token] = 4
        
        # Add basic operators
        for op in self.basic_operators:
            self.vocab[op] = len(self.vocab)
        
        # Add numbers
        for i in range(10):
            self.vocab[str(i)] = len(self.vocab)
        
        # Add letters (for variable names)
        for c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ':
            self.vocab[c] = len(self.vocab)
        
        # Create reverse vocabulary
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    def train(self, texts: List[str], num_merges: int = 1000):
        """
        Train the tokenizer on a corpus of mathematical texts.
        
        Args:
            texts: List of mathematical expressions to train on
            num_merges: Number of BPE merges to perform
        """
        # Count character pairs
        pairs = Counter()
        for text in texts:
            try:
                # Try to parse with SymPy first
                expr = parse_expr(text, transformations=self.transformations)
                text = str(expr)
            except:
                pass  # If parsing fails, use original text
                
            chars = list(text)
            for i in range(len(chars) - 1):
                pairs[(chars[i], chars[i + 1])] += 1
        
        # Perform merges
        for _ in range(num_merges):
            if not pairs or len(self.vocab) >= self.vocab_size:
                break
                
            # Find most frequent pair
            pair = max(pairs.items(), key=lambda x: x[1])[0]
            new_token = ''.join(pair)
            
            # Add to vocabulary
            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)
                self.reverse_vocab[self.vocab[new_token]] = new_token
                self.merges[pair] = new_token
            
            # Update pairs
            new_pairs = Counter()
            for text in texts:
                try:
                    expr = parse_expr(text, transformations=self.transformations)
                    text = str(expr)
                except:
                    pass
                    
                chars = list(text)
                i = 0
                while i < len(chars) - 1:
                    if chars[i] == pair[0] and chars[i + 1] == pair[1]:
                        chars[i:i + 2] = [new_token]
                    i += 1
                for i in range(len(chars) - 1):
                    new_pairs[(chars[i], chars[i + 1])] += 1
            
            pairs = new_pairs

    def tokenize(self, expression: str) -> List[int]:
        """
        Tokenize a mathematical expression into a list of token IDs.
        
        Args:
            expression: Mathematical expression as a string
            
        Returns:
            List of token IDs
        """
        try:
            # First try to handle div() function directly
            if 'div(' in expression:
                expression = self._handle_division(expression)
            
            # Try to parse with SymPy
            try:
                parsed_expr = parse_expr(expression, transformations=self.transformations)
                # Convert to string in a way that preserves division
                if '/' in expression or '÷' in expression or 'div(' in expression:
                    # If original expression had division, maintain that form
                    expression = self._handle_division(str(parsed_expr))
                else:
                    # Otherwise use SymPy's string representation
                    expression = str(parsed_expr)
            except:
                # If parsing fails, just use the preprocessed expression
                expression = self._preprocess_expression(expression)
            
            # Handle any remaining SymPy expressions
            if 'Mul(' in expression or 'Pow(' in expression:
                expression = self._handle_division(expression)
            
            # Tokenize the expression
            tokens = []
            i = 0
            while i < len(expression):
                # Skip whitespace
                if expression[i].isspace():
                    i += 1
                    continue
                
                # Check for operators and parentheses
                if expression[i] in self.basic_operators or expression[i] in '()':
                    tokens.append(self.vocab.get(expression[i], self.vocab[self.unk_token]))
                    i += 1
                    continue
                
                # Check for numbers
                if expression[i].isdigit():
                    num = ''
                    while i < len(expression) and expression[i].isdigit():
                        num += expression[i]
                        i += 1
                    tokens.append(self.vocab.get(num, self.vocab[self.unk_token]))
                    continue
                
                # Check for variables and function names
                if expression[i].isalpha():
                    name = ''
                    while i < len(expression) and (expression[i].isalnum() or expression[i] == '_'):
                        name += expression[i]
                        i += 1
                    if name in self.vocab:
                        tokens.append(self.vocab[name])
                    else:
                        for c in name:
                            tokens.append(self.vocab.get(c, self.vocab[self.unk_token]))
                    continue
                
                # Handle any other character
                tokens.append(self.vocab.get(expression[i], self.vocab[self.unk_token]))
                i += 1
            
            return tokens
            
        except Exception as e:
            # Fallback to character-level tokenization
            return [self.vocab.get(c, self.vocab[self.unk_token]) 
                   for c in expression if not c.isspace()]

    def _tokenize_substring(self, substring: str) -> List[int]:
        """Helper method to tokenize a substring."""
        if substring in self.vocab:
            return [self.vocab[substring]]
        
        # Try to split into known tokens
        tokens = []
        current = ""
        for char in substring:
            current += char
            if current in self.vocab:
                tokens.append(self.vocab[current])
                current = ""
        
        # Handle any remaining characters
        if current:
            for char in current:
                tokens.append(self.vocab.get(char, self.vocab[self.unk_token]))
        
        return tokens

    def _preprocess_expression(self, expression: str) -> str:
        """
        Preprocess the mathematical expression for better parsing.
        
        Args:
            expression: Raw mathematical expression
            
        Returns:
            Preprocessed expression
        """
        # Remove extra whitespace
        expression = ' '.join(expression.split())
        
        # Handle div() function
        if 'div(' in expression:
            # Extract arguments from div(a,b)
            match = re.match(r'div\((.*?),(.*?)\)', expression)
            if match:
                a, b = match.groups()
                expression = f"({a})/({b})"
        
        # Standardize division notation
        expression = expression.replace('÷', '/')
        
        # Handle implicit multiplication
        expression = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expression)
        
        return expression
    
    def _handle_division(self, expr_str: str) -> str:
        """
        Handle division operations in the expression string.
        
        Args:
            expr_str: Expression string from SymPy
            
        Returns:
            Processed expression string with standardized division
        """
        # First, try to parse as SymPy expression if it's not already parsed
        if not ('Mul(' in expr_str or 'Pow(' in expr_str):
            try:
                parsed = parse_expr(expr_str, transformations=self.transformations)
                expr_str = str(parsed)
            except:
                pass
        
        # Handle div() function calls
        while 'div(' in expr_str:
            match = re.search(r'div\((.*?),(.*?)\)', expr_str)
            if not match:
                break
            a, b = match.groups()
            # Clean up the arguments
            a = a.strip()
            b = b.strip()
            # Always add parentheses for consistency
            expr_str = expr_str[:match.start()] + f"({a})/({b})" + expr_str[match.end():]
        
        # Handle SymPy's internal representation
        while True:
            # Try to find any Mul/Pow pattern
            match = re.search(r'Mul\((.*?),(.*?)Pow\((.*?),-1\)\)', expr_str)
            if not match:
                # Try simpler pattern
                match = re.search(r'Mul\((.*?), ?Pow\((.*?), ?-1\)\)', expr_str)
                if not match:
                    break
                num = match.group(1)
                den = match.group(2)
            else:
                num = match.group(1)
                den = match.group(3)
            
            # Clean up the expressions
            num = re.sub(r'[a-zA-Z]+\((.*?)\)', r'\1', num)
            den = re.sub(r'[a-zA-Z]+\((.*?)\)', r'\1', den)
            # Remove any remaining commas and spaces
            num = re.sub(r'\s*,\s*', '', num)
            den = re.sub(r'\s*,\s*', '', den)
            expr_str = expr_str[:match.start()] + f"{num}/{den}" + expr_str[match.end():]
        
        # Handle Rational
        while 'Rational(' in expr_str:
            match = re.search(r'Rational\((.*?),(.*?)\)', expr_str)
            if not match:
                break
            num, den = match.groups()
            expr_str = expr_str[:match.start()] + f"{num}/{den}" + expr_str[match.end():]
        
        # Clean up any remaining SymPy function calls
        while True:
            match = re.search(r'[a-zA-Z]+\((.*?)\)', expr_str)
            if not match:
                break
            content = match.group(1)
            # Remove any remaining commas and spaces
            content = re.sub(r'\s*,\s*', '', content)
            expr_str = expr_str[:match.start()] + content + expr_str[match.end():]
        
        # Clean up any remaining commas and spaces
        expr_str = re.sub(r'\s*,\s*', '', expr_str)
        
        # Standardize division notation
        expr_str = expr_str.replace('÷', '/')
        
        return expr_str

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Convert token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        # First pass: convert tokens to text
        tokens = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in [self.vocab[t] for t in [self.cls_token, self.sep_token, self.pad_token, self.mask_token]]:
                continue
            token = self.reverse_vocab.get(token_id, self.unk_token)
            tokens.append(token)
        
        # Join tokens and clean up the text
        text = ''.join(tokens)
        
        # Handle division expressions
        if 'div(' in text:
            # Convert div(a,b) to (a)/(b)
            matches = list(re.finditer(r'div\((.*?),(.*?)\)', text))
            for match in reversed(matches):  # Process from right to left
                a, b = match.groups()
                a = a.strip()
                b = b.strip()
                # Always add parentheses for consistency
                text = text[:match.start()] + f"({a})/({b})" + text[match.end():]
        
        # Handle SymPy's internal representation
        if 'Mul(' in text and 'Pow(' in text:
            matches = list(re.finditer(r'Mul\((.*?), Pow\((.*?), -1\)\)', text))
            for match in reversed(matches):  # Process from right to left
                num, den = match.groups()
                # Clean up the expressions
                num = re.sub(r'[a-zA-Z]+\((.*?)\)', r'\1', num)
                den = re.sub(r'[a-zA-Z]+\((.*?)\)', r'\1', den)
                text = text[:match.start()] + f"{num}/{den}" + text[match.end():]
        
        # Handle Rational
        if 'Rational(' in text:
            matches = list(re.finditer(r'Rational\((.*?),(.*?)\)', text))
            for match in reversed(matches):  # Process from right to left
                num, den = match.groups()
                text = text[:match.start()] + f"{num}/{den}" + text[match.end():]
        
        # Clean up any remaining SymPy function calls
        text = re.sub(r'[a-zA-Z]+\((.*?)\)', r'\1', text)
        
        # Standardize division notation
        text = text.replace('÷', '/')
        
        # Clean up any remaining [UNK] tokens
        text = text.replace(self.unk_token, '')
        
        return text

    def save(self, path: str):
        """
        Save the tokenizer's vocabulary and merges.
        
        Args:
            path: Path to save the tokenizer
        """
        import json
        with open(path, 'w') as f:
            json.dump({
                'vocab': self.vocab,
                'merges': self.merges,
                'vocab_size': self.vocab_size
            }, f)

    def load(self, path: str):
        """
        Load the tokenizer's vocabulary and merges.
        
        Args:
            path: Path to load the tokenizer from
        """
        import json
        with open(path, 'r') as f:
            data = json.load(f)
            self.vocab = data['vocab']
            self.merges = data['merges']
            self.vocab_size = data['vocab_size']
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    def get_vocab_size(self) -> int:
        """
        Get the current vocabulary size.
        
        Returns:
            Number of tokens in the vocabulary
        """
        return len(self.vocab)

    def get_special_tokens(self) -> Dict[str, int]:
        """
        Get the special tokens and their IDs.
        
        Returns:
            Dictionary mapping special token names to their IDs
        """
        return {
            'unk_token': self.vocab[self.unk_token],
            'pad_token': self.vocab[self.pad_token],
            'cls_token': self.vocab[self.cls_token],
            'sep_token': self.vocab[self.sep_token],
            'mask_token': self.vocab[self.mask_token]
        }

    def _load_unicode_math_symbols(self):
        """
        Load Unicode mathematical symbols from the Unicode database.
        """
        # Mathematical Operators (U+2200 to U+22FF)
        self.unicode_math_operators = {
            '∀': 'FOR ALL',
            '∃': 'THERE EXISTS',
            '∄': 'THERE DOES NOT EXIST',
            '∅': 'EMPTY SET',
            '∆': 'INCREMENT',
            '∇': 'NABLA',
            '∈': 'ELEMENT OF',
            '∉': 'NOT AN ELEMENT OF',
            '∋': 'CONTAINS AS MEMBER',
            '∌': 'DOES NOT CONTAIN AS MEMBER',
            '∏': 'N-ARY PRODUCT',
            '∐': 'N-ARY COPRODUCT',
            '∑': 'N-ARY SUMMATION',
            '−': 'MINUS SIGN',
            '∓': 'MINUS-OR-PLUS SIGN',
            '∔': 'DOT PLUS',
            '∕': 'DIVISION SLASH',
            '∖': 'SET MINUS',
            '∗': 'ASTERISK OPERATOR',
            '∘': 'RING OPERATOR',
            '∙': 'BULLET OPERATOR',
            '√': 'SQUARE ROOT',
            '∛': 'CUBE ROOT',
            '∜': 'FOURTH ROOT',
            '∝': 'PROPORTIONAL TO',
            '∞': 'INFINITY',
            '∟': 'RIGHT ANGLE',
            '∠': 'ANGLE',
            '∡': 'MEASURED ANGLE',
            '∢': 'SPHERICAL ANGLE',
            '∣': 'DIVIDES',
            '∤': 'DOES NOT DIVIDE',
            '∥': 'PARALLEL TO',
            '∦': 'NOT PARALLEL TO',
            '∧': 'LOGICAL AND',
            '∨': 'LOGICAL OR',
            '∫': 'INTEGRAL',
            '∬': 'DOUBLE INTEGRAL',
            '∭': 'TRIPLE INTEGRAL',
            '∮': 'CONTOUR INTEGRAL',
            '∯': 'SURFACE INTEGRAL',
            '∰': 'VOLUME INTEGRAL',
            '∱': 'CLOCKWISE INTEGRAL',
            '∲': 'CLOCKWISE CONTOUR INTEGRAL',
            '∳': 'ANTICLOCKWISE CONTOUR INTEGRAL',
            '∴': 'THEREFORE',
            '∵': 'BECAUSE',
            '∶': 'RATIO',
            '∷': 'PROPORTION',
            '∸': 'DOT MINUS',
            '∹': 'EXCESS',
            '∺': 'GEOMETRIC PROPORTION',
            '∻': 'HOMOTHETIC',
            '∼': 'TILDE OPERATOR',
            '∽': 'REVERSED TILDE',
            '∾': 'INVERTED LAZY S',
            '∿': 'SINE WAVE',
            '≀': 'WREATH PRODUCT',
            '≁': 'NOT TILDE',
            '≂': 'MINUS TILDE',
            '≃': 'ASYMPTOTICALLY EQUAL TO',
            '≄': 'NOT ASYMPTOTICALLY EQUAL TO',
            '≅': 'APPROXIMATELY EQUAL TO',
            '≆': 'APPROXIMATELY BUT NOT ACTUALLY EQUAL TO',
            '≇': 'NEITHER APPROXIMATELY NOR ACTUALLY EQUAL TO',
            '≈': 'ALMOST EQUAL TO',
            '≉': 'NOT ALMOST EQUAL TO',
            '≊': 'ALMOST EQUAL OR EQUAL TO',
            '≋': 'TRIPLE TILDE',
            '≌': 'ALL EQUAL TO',
            '≍': 'EQUIVALENT TO',
            '≎': 'GEOMETRICALLY EQUIVALENT TO',
            '≏': 'DIFFERENCE BETWEEN',
            '≐': 'APPROACHES THE LIMIT',
            '≑': 'GEOMETRICALLY EQUAL TO',
            '≒': 'APPROXIMATELY EQUAL TO OR THE IMAGE OF',
            '≓': 'IMAGE OF OR APPROXIMATELY EQUAL TO',
            '≔': 'COLON EQUALS',
            '≕': 'EQUALS COLON',
            '≖': 'RING IN EQUAL TO',
            '≗': 'RING EQUAL TO',
            '≘': 'CORRESPONDS TO',
            '≙': 'ESTIMATES',
            '≚': 'EQUIANGULAR TO',
            '≛': 'STAR EQUALS',
            '≜': 'DELTA EQUAL TO',
            '≝': 'EQUAL TO BY DEFINITION',
            '≞': 'MEASURED BY',
            '≟': 'QUESTIONED EQUAL TO',
            '≠': 'NOT EQUAL TO',
            '≡': 'IDENTICAL TO',
            '≢': 'NOT IDENTICAL TO',
            '≣': 'STRICTLY EQUIVALENT TO',
            '≤': 'LESS-THAN OR EQUAL TO',
            '≥': 'GREATER-THAN OR EQUAL TO',
            '≦': 'LESS-THAN OVER EQUAL TO',
            '≧': 'GREATER-THAN OVER EQUAL TO',
            '≨': 'LESS-THAN BUT NOT EQUAL TO',
            '≩': 'GREATER-THAN BUT NOT EQUAL TO',
            '≪': 'MUCH LESS-THAN',
            '≫': 'MUCH GREATER-THAN',
            '≬': 'BETWEEN',
            '≭': 'NOT EQUIVALENT TO',
            '≮': 'NOT LESS-THAN',
            '≯': 'NOT GREATER-THAN',
            '≰': 'NEITHER LESS-THAN NOR EQUAL TO',
            '≱': 'NEITHER GREATER-THAN NOR EQUAL TO',
            '≲': 'LESS-THAN OR EQUIVALENT TO',
            '≳': 'GREATER-THAN OR EQUIVALENT TO',
            '≴': 'NEITHER LESS-THAN NOR EQUIVALENT TO',
            '≵': 'NEITHER GREATER-THAN NOR EQUIVALENT TO',
            '≶': 'LESS-THAN OR GREATER-THAN',
            '≷': 'GREATER-THAN OR LESS-THAN',
            '≸': 'NEITHER LESS-THAN NOR GREATER-THAN',
            '≹': 'NEITHER GREATER-THAN NOR LESS-THAN',
            '≺': 'PRECEDES',
            '≻': 'SUCCEEDS',
            '≼': 'PRECEDES OR EQUAL TO',
            '≽': 'SUCCEEDS OR EQUAL TO',
            '≾': 'PRECEDES OR EQUIVALENT TO',
            '≿': 'SUCCEEDS OR EQUIVALENT TO',
            '⊀': 'DOES NOT PRECEDE',
            '⊁': 'DOES NOT SUCCEED',
            '⊂': 'SUBSET OF',
            '⊃': 'SUPERSET OF',
            '⊄': 'NOT A SUBSET OF',
            '⊅': 'NOT A SUPERSET OF',
            '⊆': 'SUBSET OF OR EQUAL TO',
            '⊇': 'SUPERSET OF OR EQUAL TO',
            '⊈': 'NEITHER A SUBSET OF NOR EQUAL TO',
            '⊉': 'NEITHER A SUPERSET OF NOR EQUAL TO',
            '⊊': 'SUBSET OF WITH NOT EQUAL TO',
            '⊋': 'SUPERSET OF WITH NOT EQUAL TO',
            '⊍': 'MULTISET',
            '⊎': 'MULTISET MULTIPLICATION',
            '⊏': 'SQUARE IMAGE OF',
            '⊐': 'SQUARE ORIGINAL OF',
            '⊑': 'SQUARE IMAGE OF OR EQUAL TO',
            '⊒': 'SQUARE ORIGINAL OF OR EQUAL TO',
            '⊓': 'SQUARE CAP',
            '⊔': 'SQUARE CUP',
            '⊕': 'CIRCLED PLUS',
            '⊖': 'CIRCLED MINUS',
            '⊗': 'CIRCLED TIMES',
            '⊘': 'CIRCLED DIVISION SLASH',
            '⊙': 'CIRCLED DOT OPERATOR',
            '⊚': 'CIRCLED RING OPERATOR',
            '⊛': 'CIRCLED ASTERISK OPERATOR',
            '⊜': 'CIRCLED EQUALS',
            '⊝': 'CIRCLED DASH',
            '⊞': 'SQUARED PLUS',
            '⊟': 'SQUARED MINUS',
            '⊠': 'SQUARED TIMES',
            '⊡': 'SQUARED DOT OPERATOR',
            '⊢': 'RIGHT TACK',
            '⊣': 'LEFT TACK',
            '⊤': 'DOWN TACK',
            '⊥': 'UP TACK',
            '⊦': 'ASSERTION',
            '⊧': 'MODELS',
            '⊨': 'TRUE',
            '⊩': 'FORCES',
            '⊪': 'TRIPLE VERTICAL BAR RIGHT TURNSTILE',
            '⊫': 'DOUBLE VERTICAL BAR DOUBLE RIGHT TURNSTILE',
            '⊬': 'DOES NOT PROVE',
            '⊭': 'NOT TRUE',
            '⊮': 'DOES NOT FORCE',
            '⊯': 'NEGATED DOUBLE VERTICAL BAR DOUBLE RIGHT TURNSTILE',
            '⊰': 'PRECEDES UNDER RELATION',
            '⊱': 'SUCCEEDS UNDER RELATION',
            '⊲': 'NORMAL SUBGROUP OF',
            '⊳': 'CONTAINS AS NORMAL SUBGROUP',
            '⊴': 'NORMAL SUBGROUP OF OR EQUAL TO',
            '⊵': 'CONTAINS AS NORMAL SUBGROUP OR EQUAL TO',
            '⊶': 'ORIGINAL OF',
            '⊷': 'IMAGE OF',
            '⊸': 'MULTIMAP',
            '⊹': 'HERMITIAN CONJUGATE MATRIX',
            '⊺': 'INTERCALATE',
            '⊻': 'XOR',
            '⊼': 'NAND',
            '⊽': 'NOR',
            '⊾': 'RIGHT ANGLE WITH ARC',
            '⊿': 'RIGHT TRIANGLE',
            '⋀': 'N-ARY LOGICAL AND',
            '⋁': 'N-ARY LOGICAL OR',
            '⋂': 'N-ARY INTERSECTION',
            '⋃': 'N-ARY UNION',
            '⋄': 'DIAMOND OPERATOR',
            '⋅': 'DOT OPERATOR',
            '⋆': 'STAR OPERATOR',
            '⋇': 'DIVISION TIMES',
            '⋈': 'BOWTIE',
            '⋉': 'LEFT NORMAL FACTOR SEMIDIRECT PRODUCT',
            '⋊': 'RIGHT NORMAL FACTOR SEMIDIRECT PRODUCT',
            '⋋': 'LEFT SEMIDIRECT PRODUCT',
            '⋌': 'RIGHT SEMIDIRECT PRODUCT',
            '⋍': 'REVERSED TILDE EQUALS',
            '⋎': 'CURLY LOGICAL OR',
            '⋏': 'CURLY LOGICAL AND',
            '⋐': 'DOUBLE SUBSET',
            '⋑': 'DOUBLE SUPERSET',
            '⋒': 'DOUBLE INTERSECTION',
            '⋓': 'DOUBLE UNION',
            '⋔': 'PITCHFORK',
            '⋕': 'EQUAL AND PARALLEL TO',
            '⋖': 'LESS-THAN WITH DOT',
            '⋗': 'GREATER-THAN WITH DOT',
            '⋘': 'VERY MUCH LESS-THAN',
            '⋙': 'VERY MUCH GREATER-THAN',
            '⋚': 'LESS-THAN EQUAL TO OR GREATER-THAN',
            '⋛': 'GREATER-THAN EQUAL TO OR LESS-THAN',
            '⋜': 'EQUAL TO OR LESS-THAN',
            '⋝': 'EQUAL TO OR GREATER-THAN',
            '⋞': 'EQUAL TO OR PRECEDES',
            '⋟': 'EQUAL TO OR SUCCEEDS',
            '⋠': 'DOES NOT PRECEDE OR EQUAL',
            '⋡': 'DOES NOT SUCCEED OR EQUAL',
            '⋢': 'NOT SQUARE IMAGE OF OR EQUAL TO',
            '⋣': 'NOT SQUARE ORIGINAL OF OR EQUAL TO',
            '⋤': 'SQUARE IMAGE OF OR NOT EQUAL TO',
            '⋥': 'SQUARE ORIGINAL OF OR NOT EQUAL TO',
            '⋦': 'LESS-THAN BUT NOT EQUIVALENT TO',
            '⋧': 'GREATER-THAN BUT NOT EQUIVALENT TO',
            '⋨': 'PRECEDES BUT NOT EQUIVALENT TO',
            '⋩': 'SUCCEEDS BUT NOT EQUIVALENT TO',
            '⋪': 'NOT NORMAL SUBGROUP OF',
            '⋫': 'DOES NOT CONTAIN AS NORMAL SUBGROUP',
            '⋬': 'NOT NORMAL SUBGROUP OF OR EQUAL TO',
            '⋭': 'DOES NOT CONTAIN AS NORMAL SUBGROUP OR EQUAL TO',
            '⋮': 'VERTICAL ELLIPSIS',
            '⋯': 'MIDLINE HORIZONTAL ELLIPSIS',
            '⋰': 'UP RIGHT DIAGONAL ELLIPSIS',
            '⋱': 'DOWN RIGHT DIAGONAL ELLIPSIS',
            '⋲': 'ELEMENT OF WITH LONG HORIZONTAL STROKE',
            '⋳': 'ELEMENT OF WITH VERTICAL BAR AT END OF HORIZONTAL STROKE',
            '⋴': 'SMALL ELEMENT OF WITH VERTICAL BAR AT END OF HORIZONTAL STROKE',
            '⋵': 'ELEMENT OF WITH DOT ABOVE',
            '⋶': 'ELEMENT OF WITH OVERBAR',
            '⋷': 'SMALL ELEMENT OF WITH OVERBAR',
            '⋸': 'ELEMENT OF WITH UNDERBAR',
            '⋹': 'ELEMENT OF WITH TWO HORIZONTAL STROKES',
            '⋺': 'CONTAINS WITH LONG HORIZONTAL STROKE',
            '⋻': 'CONTAINS WITH VERTICAL BAR AT END OF HORIZONTAL STROKE',
            '⋼': 'SMALL CONTAINS WITH VERTICAL BAR AT END OF HORIZONTAL STROKE',
            '⋽': 'CONTAINS WITH OVERBAR',
            '⋾': 'SMALL CONTAINS WITH OVERBAR',
            '⋿': 'Z NOTATION BAG MEMBERSHIP',
        }
        
        # Add Unicode symbols to math_symbols set
        self.math_symbols = set(self.unicode_math_operators.keys())

    def _load_latex_symbols(self):
        """
        Load LaTeX math symbols and their Unicode equivalents.
        """
        self.latex_symbols = {
            # Greek letters
            '\\alpha': 'α', '\\beta': 'β', '\\gamma': 'γ', '\\delta': 'δ',
            '\\epsilon': 'ε', '\\zeta': 'ζ', '\\eta': 'η', '\\theta': 'θ',
            '\\iota': 'ι', '\\kappa': 'κ', '\\lambda': 'λ', '\\mu': 'μ',
            '\\nu': 'ν', '\\xi': 'ξ', '\\pi': 'π', '\\rho': 'ρ',
            '\\sigma': 'σ', '\\tau': 'τ', '\\upsilon': 'υ', '\\phi': 'φ',
            '\\chi': 'χ', '\\psi': 'ψ', '\\omega': 'ω',
            
            # Uppercase Greek letters
            '\\Alpha': 'Α', '\\Beta': 'Β', '\\Gamma': 'Γ', '\\Delta': 'Δ',
            '\\Epsilon': 'Ε', '\\Zeta': 'Ζ', '\\Eta': 'Η', '\\Theta': 'Θ',
            '\\Iota': 'Ι', '\\Kappa': 'Κ', '\\Lambda': 'Λ', '\\Mu': 'Μ',
            '\\Nu': 'Ν', '\\Xi': 'Ξ', '\\Pi': 'Π', '\\Rho': 'Ρ',
            '\\Sigma': 'Σ', '\\Tau': 'Τ', '\\Upsilon': 'Υ', '\\Phi': 'Φ',
            '\\Chi': 'Χ', '\\Psi': 'Ψ', '\\Omega': 'Ω',
            
            # Mathematical operators
            '\\pm': '±', '\\mp': '∓', '\\times': '×', '\\div': '÷',
            '\\cdot': '·', '\\ast': '∗', '\\star': '⋆', '\\dagger': '†',
            '\\ddagger': '‡', '\\amalg': '⨿', '\\cap': '∩', '\\cup': '∪',
            '\\uplus': '⊎', '\\sqcap': '⊓', '\\sqcup': '⊔', '\\vee': '∨',
            '\\wedge': '∧', '\\setminus': '∖', '\\wr': '≀', '\\circ': '∘',
            '\\bullet': '•', '\\diamond': '⋄', '\\uplus': '⊎', '\\amalg': '⨿',
            
            # Relations
            '\\leq': '≤', '\\geq': '≥', '\\equiv': '≡', '\\models': '⊨',
            '\\prec': '≺', '\\succ': '≻', '\\sim': '∼', '\\perp': '⊥',
            '\\preceq': '⪯', '\\succeq': '⪰', '\\simeq': '≃', '\\mid': '∣',
            '\\ll': '≪', '\\gg': '≫', '\\asymp': '≍', '\\parallel': '∥',
            '\\subset': '⊂', '\\supset': '⊃', '\\approx': '≈', '\\bowtie': '⋈',
            '\\subseteq': '⊆', '\\supseteq': '⊇', '\\cong': '≅', '\\ltimes': '⋉',
            '\\sqsubset': '⊏', '\\sqsupset': '⊐', '\\neq': '≠', '\\rtimes': '⋊',
            '\\sqsubseteq': '⊑', '\\sqsupseteq': '⊒', '\\doteq': '≐', '\\leftthreetimes': '⋋',
            '\\in': '∈', '\\ni': '∋', '\\propto': '∝', '\\rightthreetimes': '⋌',
            '\\vdash': '⊢', '\\dashv': '⊣', '\\varpropto': '∝', '\\circleddash': '⊝',
            
            # Arrows
            '\\leftarrow': '←', '\\rightarrow': '→', '\\leftrightarrow': '↔',
            '\\Leftarrow': '⇐', '\\Rightarrow': '⇒', '\\Leftrightarrow': '⇔',
            '\\mapsto': '↦', '\\hookleftarrow': '↩', '\\hookrightarrow': '↪',
            '\\leftharpoonup': '↼', '\\rightharpoonup': '⇀', '\\leftharpoondown': '↽',
            '\\rightharpoondown': '⇁', '\\rightleftharpoons': '⇌', '\\longleftarrow': '⟵',
            '\\longrightarrow': '⟶', '\\longleftrightarrow': '⟷', '\\Longleftarrow': '⟸',
            '\\Longrightarrow': '⟹', '\\Longleftrightarrow': '⟺',
            
            # Delimiters
            '\\lceil': '⌈', '\\rceil': '⌉', '\\lfloor': '⌊', '\\rfloor': '⌋',
            '\\langle': '⟨', '\\rangle': '⟩', '\\lbrace': '{', '\\rbrace': '}',
            '\\lvert': '|', '\\rvert': '|', '\\lVert': '‖', '\\rVert': '‖',
            
            # Other symbols
            '\\infty': '∞', '\\nabla': '∇', '\\partial': '∂', '\\emptyset': '∅',
            '\\varnothing': '∅', '\\exists': '∃', '\\forall': '∀', '\\neg': '¬',
            '\\flat': '♭', '\\natural': '♮', '\\sharp': '♯', '\\clubsuit': '♣',
            '\\diamondsuit': '♦', '\\heartsuit': '♥', '\\spadesuit': '♠',
            '\\mho': '℧', '\\Finv': 'Ⅎ', '\\Game': '⅁', '\\beth': 'ℶ',
            '\\gimel': 'ℷ', '\\daleth': 'ℸ', '\\backslash': '\\',
        }
        
        # Add LaTeX symbols to math_symbols set
        self.math_symbols.update(self.latex_symbols.values())

    def _load_mathml_symbols(self):
        """
        Load MathML entities and their Unicode equivalents.
        """
        self.mathml_symbols = {
            # MathML entities
            '&InvisibleTimes;': '⁢',
            '&ApplyFunction;': '⁡',
            '&InvisibleComma;': '⁣',
            '&sum;': '∑',
            '&prod;': '∏',
            '&coprod;': '∐',
            '&int;': '∫',
            '&oint;': '∮',
            '&iiint;': '∭',
            '&oiint;': '∯',
            '&iiiint;': '⨌',
            '&oint;': '∮',
            '&oiint;': '∯',
            '&oiiint;': '∰',
            '&iiint;': '∭',
            '&iiiint;': '⨌',
            '&idotint;': '⨗',
            '&intbar;': '⨍',
            '&intBar;': '⨎',
            '&fpartint;': '⨏',
            '&cirfnint;': '⨐',
            '&awint;': '⨑',
            '&rppolint;': '⨒',
            '&scpolint;': '⨓',
            '&npolint;': '⨔',
            '&pointint;': '⨕',
            '&quatint;': '⨖',
            '&intlarhk;': '⨗',
            '&intx;': '⨘',
            '&intcap;': '⨙',
            '&intcup;': '⨚',
            '&intprod;': '⨛',
            '&intprodr;': '⨜',
            '&amalg;': '⨿',
            '&cap;': '∩',
            '&cup;': '∪',
            '&uplus;': '⊎',
            '&sqcap;': '⊓',
            '&sqcup;': '⊔',
            '&vee;': '∨',
            '&wedge;': '∧',
            '&setminus;': '∖',
            '&wr;': '≀',
            '&circ;': '∘',
            '&bullet;': '•',
            '&diamond;': '⋄',
            '&uplus;': '⊎',
            '&amalg;': '⨿',
            '&leq;': '≤',
            '&geq;': '≥',
            '&equiv;': '≡',
            '&models;': '⊨',
            '&prec;': '≺',
            '&succ;': '≻',
            '&sim;': '∼',
            '&perp;': '⊥',
            '&preceq;': '⪯',
            '&succeq;': '⪰',
            '&simeq;': '≃',
            '&mid;': '∣',
            '&ll;': '≪',
            '&gg;': '≫',
            '&asymp;': '≍',
            '&parallel;': '∥',
            '&subset;': '⊂',
            '&supset;': '⊃',
            '&approx;': '≈',
            '&bowtie;': '⋈',
            '&subseteq;': '⊆',
            '&supseteq;': '⊇',
            '&cong;': '≅',
            '&ltimes;': '⋉',
            '&sqsubset;': '⊏',
            '&sqsupset;': '⊐',
            '&neq;': '≠',
            '&rtimes;': '⋊',
            '&sqsubseteq;': '⊑',
            '&sqsupseteq;': '⊒',
            '&doteq;': '≐',
            '&leftthreetimes;': '⋋',
            '&in;': '∈',
            '&ni;': '∋',
            '&propto;': '∝',
            '&rightthreetimes;': '⋌',
            '&vdash;': '⊢',
            '&dashv;': '⊣',
            '&varpropto;': '∝',
            '&circleddash;': '⊝',
        }
        
        # Add MathML symbols to math_symbols set
        self.math_symbols.update(self.mathml_symbols.values())

    def _load_additional_unicode_blocks(self):
        """
        Load additional Unicode blocks for mathematical symbols.
        """
        # Mathematical Alphanumeric Symbols (U+1D400 to U+1D7FF)
        self.math_alphanumeric = {
            '𝐀': 'MATHEMATICAL BOLD CAPITAL A',
            '𝐁': 'MATHEMATICAL BOLD CAPITAL B',
            # ... (all mathematical alphanumeric symbols)
        }
        
        # Supplemental Mathematical Operators (U+2A00 to U+2AFF)
        self.supplemental_math_operators = {
            '⨀': 'N-ARY CIRCLED DOT OPERATOR',
            '⨁': 'N-ARY CIRCLED PLUS OPERATOR',
            # ... (all supplemental mathematical operators)
        }
        
        # Miscellaneous Mathematical Symbols-A (U+27C0 to U+27EF)
        self.misc_math_symbols_a = {
            '⟂': 'PERPENDICULAR',
            '⟃': 'OPEN SUBSET',
            # ... (all miscellaneous mathematical symbols-A)
        }
        
        # Miscellaneous Mathematical Symbols-B (U+2980 to U+29FF)
        self.misc_math_symbols_b = {
            '⦀': 'TRIPLE VERTICAL BAR DELIMITER',
            '⦁': 'Z NOTATION SPOT',
            # ... (all miscellaneous mathematical symbols-B)
        }
        
        # Add all additional Unicode symbols to math_symbols set
        self.math_symbols.update(self.math_alphanumeric.keys())
        self.math_symbols.update(self.supplemental_math_operators.keys())
        self.math_symbols.update(self.misc_math_symbols_a.keys())
        self.math_symbols.update(self.misc_math_symbols_b.keys())

    def _load_additional_latex_packages(self):
        """
        Load additional LaTeX package symbols.
        """
        # amsmath package symbols
        self.amsmath_symbols = {
            '\\boxed': 'BOXED',
            '\\tag': 'TAG',
            # ... (all amsmath symbols)
        }
        
        # amssymb package symbols
        self.amssymb_symbols = {
            '\\dashrightarrow': 'DASHED ARROW',
            '\\dashleftarrow': 'DASHED LEFT ARROW',
            # ... (all amssymb symbols)
        }
        
        # mathrsfs package symbols
        self.mathrsfs_symbols = {
            '\\mathscr{A}': 'SCRIPT CAPITAL A',
            '\\mathscr{B}': 'SCRIPT CAPITAL B',
            # ... (all mathrsfs symbols)
        }
        
        # stmaryrd package symbols
        self.stmaryrd_symbols = {
            '\\llbracket': 'LEFT DOUBLE BRACKET',
            '\\rrbracket': 'RIGHT DOUBLE BRACKET',
            # ... (all stmaryrd symbols)
        }
        
        # Add all additional LaTeX symbols to math_symbols set
        self.math_symbols.update(self.amsmath_symbols.values())
        self.math_symbols.update(self.amssymb_symbols.values())
        self.math_symbols.update(self.mathrsfs_symbols.values())
        self.math_symbols.update(self.stmaryrd_symbols.values())

    def _load_additional_mathml_features(self):
        """
        Load additional MathML features.
        """
        # Presentation MathML elements
        self.presentation_mathml = {
            '&msup;': 'SUPERSCRIPT',
            '&msub;': 'SUBSCRIPT',
            # ... (all presentation MathML elements)
        }
        
        # Content MathML elements
        self.content_mathml = {
            '&apply;': 'APPLY',
            '&bind;': 'BIND',
            # ... (all content MathML elements)
        }
        
        # MathML 3.0 specific entities
        self.mathml3_entities = {
            '&ApplyFunction;': 'APPLY FUNCTION',
            '&InvisibleTimes;': 'INVISIBLE TIMES',
            # ... (all MathML 3.0 entities)
        }
        
        # Add all additional MathML symbols to math_symbols set
        self.math_symbols.update(self.presentation_mathml.values())
        self.math_symbols.update(self.content_mathml.values())
        self.math_symbols.update(self.mathml3_entities.values())

    def _load_other_notation_systems(self):
        """
        Load symbols from other mathematical notation systems.
        """
        # AsciiMath notation
        self.asciimath_symbols = {
            '`': 'BACKTICK',
            '~': 'TILDE',
            # ... (all AsciiMath symbols)
        }
        
        # Wolfram Language notation
        self.wolfram_symbols = {
            r'\[Alpha]': 'ALPHA',
            r'\[Beta]': 'BETA',
            # ... (all Wolfram Language symbols)
        }
        
        # Mathematica notation
        self.mathematica_symbols = {
            r'\\[Alpha]': 'ALPHA',
            r'\\[Beta]': 'BETA',
            # ... (all Mathematica symbols)
        }
        
        # Maple notation
        self.maple_symbols = {
            'alpha': 'ALPHA',
            'beta': 'BETA',
            # ... (all Maple symbols)
        }
        
        # Add all additional notation system symbols to math_symbols set
        self.math_symbols.update(self.asciimath_symbols.values())
        self.math_symbols.update(self.wolfram_symbols.values())
        self.math_symbols.update(self.mathematica_symbols.values())
        self.math_symbols.update(self.maple_symbols.values())

    def _load_specialized_math_symbols(self):
        """
        Load specialized mathematical symbols for category theory and algebraic geometry.
        """
        self.specialized_math_symbols = {
            # Category Theory
            '→': 'MORPHISM',
            '←': 'REVERSE MORPHISM',
            '↔': 'ISOMORPHISM',
            '↪': 'MONOMORPHISM',
            '↠': 'EPIMORPHISM',
            '⥅': 'PULLBACK',
            '⥆': 'PUSHOUT',
            '⨟': 'COMPOSITION',
            '⨾': 'SEMICATEGORICAL COMPOSITION',
            '⨿': 'COPRODUCT',
            '∏': 'PRODUCT',
            '⨂': 'TENSOR PRODUCT',
            '⨁': 'DIRECT SUM',
            '⨄': 'DISJOINT UNION',
            '⨆': 'JOIN',
            '⨅': 'MEET',
            '⨉': 'CARTESIAN PRODUCT',
            '⨋': 'SUMMATION',
            '⨌': 'QUADRUPLE INTEGRAL',
            '⨍': 'FINITE PART INTEGRAL',
            '⨎': 'INTEGRAL WITH DOUBLE STROKE',
            '⨏': 'INTEGRAL AVERAGE',
            '⨐': 'CIRCULATION FUNCTION',
            '⨑': 'ANTICLOCKWISE INTEGRATION',
            '⨒': 'LINE INTEGRATION WITH RECTANGULAR PATH AROUND POLE',
            '⨓': 'LINE INTEGRATION WITH SEMICIRCULAR PATH AROUND POLE',
            '⨔': 'LINE INTEGRATION NOT INCLUDING THE POLE',
            '⨕': 'INTEGRAL AROUND A POINT OPERATOR',
            '⨖': 'QUATERNION INTEGRAL OPERATOR',
            '⨗': 'INTEGRAL WITH LEFTWARDS ARROW WITH HOOK',
            '⨘': 'INTEGRAL WITH TIMES SIGN',
            '⨙': 'INTEGRAL WITH INTERSECTION',
            '⨚': 'INTEGRAL WITH UNION',
            '⨛': 'INTEGRAL WITH OVERBAR',
            '⨜': 'INTEGRAL WITH UNDERBAR',
            
            # Algebraic Geometry
            '𝔸': 'AFFINE SPACE',
            'ℙ': 'PROJECTIVE SPACE',
            '𝕍': 'VARIETY',
            '𝕀': 'IDEAL',
            '𝕊': 'SPEC',
            '𝕋': 'TANGENT SPACE',
            '𝕄': 'MODULE',
            '𝕂': 'FIELD',
            '𝕃': 'LINE BUNDLE',
            '𝔻': 'DIVISOR',
            'ℂ': 'COMPLEX NUMBERS',
            'ℝ': 'REAL NUMBERS',
            'ℚ': 'RATIONAL NUMBERS',
            'ℤ': 'INTEGERS',
            'ℕ': 'NATURAL NUMBERS',
            '𝔽': 'FINITE FIELD',
            '𝔾': 'GROUP SCHEME',
            'ℍ': 'QUATERNIONS',
            '𝕆': 'OCTONIONS',
            '𝕊': 'SPHERE',
            '𝕋': 'TORUS',
            '𝕄': 'MANIFOLD',
            '𝕍': 'VECTOR SPACE',
            '𝕎': 'WEIL DIVISOR',
            '𝕏': 'SCHEME',
            '𝕐': 'VARIETY',
            'ℤ': 'CYCLE',
            'ℂ': 'CHAIN',
            'ℝ': 'RING',
            'ℚ': 'QUOTIENT',
            'ℙ': 'POINT',
            '𝔸': 'AFFINE',
            '𝕊': 'SPEC',
            '𝕋': 'TANGENT',
            '𝕄': 'MODULE',
            '𝕂': 'FIELD',
            '𝕃': 'LINE',
            '𝔻': 'DIVISOR',
            'ℂ': 'COMPLEX',
            'ℝ': 'REAL',
            'ℚ': 'RATIONAL',
            'ℤ': 'INTEGER',
            'ℕ': 'NATURAL',
            '𝔽': 'FINITE',
            '𝔾': 'GROUP',
            'ℍ': 'QUATERNION',
            '𝕆': 'OCTONION',
            '𝕊': 'SPHERE',
            '𝕋': 'TORUS',
            '𝕄': 'MANIFOLD',
            '𝕍': 'VECTOR',
            '𝕎': 'WEIL',
            '𝕏': 'SCHEME',
            '𝕐': 'VARIETY',
            'ℤ': 'CYCLE',
            'ℂ': 'CHAIN',
            'ℝ': 'RING',
            'ℚ': 'QUOTIENT',
            'ℙ': 'POINT',
            '𝔸': 'AFFINE',
            '𝕊': 'SPEC',
            '𝕋': 'TANGENT',
            '𝕄': 'MODULE',
            '𝕂': 'FIELD',
            '𝕃': 'LINE',
            '𝔻': 'DIVISOR'
        }
        
        # Add specialized symbols to math_symbols set
        self.math_symbols.update(self.specialized_math_symbols.keys())

    def _load_formal_logic_symbols(self):
        """
        Load formal logic and proof notation symbols.
        """
        self.formal_logic_symbols = {
            # Propositional Logic
            '⊢': 'PROVES',
            '⊨': 'MODELS',
            '⊭': 'NOT MODELS',
            '⊬': 'NOT PROVES',
            '⊩': 'FORCES',
            '⊮': 'NOT FORCES',
            '⊪': 'TRIPLE VERTICAL BAR RIGHT TURNSTILE',
            '⊫': 'DOUBLE VERTICAL BAR DOUBLE RIGHT TURNSTILE',
            '⊯': 'NEGATED DOUBLE VERTICAL BAR DOUBLE RIGHT TURNSTILE',
            '⊰': 'PRECEDES UNDER RELATION',
            '⊱': 'SUCCEEDS UNDER RELATION',
            '⊲': 'NORMAL SUBGROUP OF',
            '⊳': 'CONTAINS AS NORMAL SUBGROUP',
            '⊴': 'NORMAL SUBGROUP OF OR EQUAL TO',
            '⊵': 'CONTAINS AS NORMAL SUBGROUP OR EQUAL TO',
            '⊶': 'ORIGINAL OF',
            '⊷': 'IMAGE OF',
            '⊸': 'MULTIMAP',
            '⊹': 'HERMITIAN CONJUGATE MATRIX',
            '⊺': 'INTERCALATE',
            '⊻': 'XOR',
            '⊼': 'NAND',
            '⊽': 'NOR',
            '⊾': 'RIGHT ANGLE WITH ARC',
            '⊿': 'RIGHT TRIANGLE',
            '⋀': 'N-ARY LOGICAL AND',
            '⋁': 'N-ARY LOGICAL OR',
            '⋂': 'N-ARY INTERSECTION',
            '⋃': 'N-ARY UNION',
            '⋄': 'DIAMOND OPERATOR',
            '⋅': 'DOT OPERATOR',
            '⋆': 'STAR OPERATOR',
            '⋇': 'DIVISION TIMES',
            '⋈': 'BOWTIE',
            '⋉': 'LEFT NORMAL FACTOR SEMIDIRECT PRODUCT',
            '⋊': 'RIGHT NORMAL FACTOR SEMIDIRECT PRODUCT',
            '⋋': 'LEFT SEMIDIRECT PRODUCT',
            '⋌': 'RIGHT SEMIDIRECT PRODUCT',
            '⋍': 'REVERSED TILDE EQUALS',
            '⋎': 'CURLY LOGICAL OR',
            '⋏': 'CURLY LOGICAL AND',
            '⋐': 'DOUBLE SUBSET',
            '⋑': 'DOUBLE SUPERSET',
            '⋒': 'DOUBLE INTERSECTION',
            '⋓': 'DOUBLE UNION',
            '⋔': 'PITCHFORK',
            '⋕': 'EQUAL AND PARALLEL TO',
            '⋖': 'LESS-THAN WITH DOT',
            '⋗': 'GREATER-THAN WITH DOT',
            '⋘': 'VERY MUCH LESS-THAN',
            '⋙': 'VERY MUCH GREATER-THAN',
            '⋚': 'LESS-THAN EQUAL TO OR GREATER-THAN',
            '⋛': 'GREATER-THAN EQUAL TO OR LESS-THAN',
            '⋜': 'EQUAL TO OR LESS-THAN',
            '⋝': 'EQUAL TO OR GREATER-THAN',
            '⋞': 'EQUAL TO OR PRECEDES',
            '⋟': 'EQUAL TO OR SUCCEEDS',
            '⋠': 'DOES NOT PRECEDE OR EQUAL',
            '⋡': 'DOES NOT SUCCEED OR EQUAL',
            '⋢': 'NOT SQUARE IMAGE OF OR EQUAL TO',
            '⋣': 'NOT SQUARE ORIGINAL OF OR EQUAL TO',
            '⋤': 'SQUARE IMAGE OF OR NOT EQUAL TO',
            '⋥': 'SQUARE ORIGINAL OF OR NOT EQUAL TO',
            '⋦': 'LESS-THAN BUT NOT EQUIVALENT TO',
            '⋧': 'GREATER-THAN BUT NOT EQUIVALENT TO',
            '⋨': 'PRECEDES BUT NOT EQUIVALENT TO',
            '⋩': 'SUCCEEDS BUT NOT EQUIVALENT TO',
            '⋪': 'NOT NORMAL SUBGROUP OF',
            '⋫': 'DOES NOT CONTAIN AS NORMAL SUBGROUP',
            '⋬': 'NOT NORMAL SUBGROUP OF OR EQUAL TO',
            '⋭': 'DOES NOT CONTAIN AS NORMAL SUBGROUP OR EQUAL TO',
            '⋮': 'VERTICAL ELLIPSIS',
            '⋯': 'MIDLINE HORIZONTAL ELLIPSIS',
            '⋰': 'UP RIGHT DIAGONAL ELLIPSIS',
            '⋱': 'DOWN RIGHT DIAGONAL ELLIPSIS',
            '⋲': 'ELEMENT OF WITH LONG HORIZONTAL STROKE',
            '⋳': 'ELEMENT OF WITH VERTICAL BAR AT END OF HORIZONTAL STROKE',
            '⋴': 'SMALL ELEMENT OF WITH VERTICAL BAR AT END OF HORIZONTAL STROKE',
            '⋵': 'ELEMENT OF WITH DOT ABOVE',
            '⋶': 'ELEMENT OF WITH OVERBAR',
            '⋷': 'SMALL ELEMENT OF WITH OVERBAR',
            '⋸': 'ELEMENT OF WITH UNDERBAR',
            '⋹': 'ELEMENT OF WITH TWO HORIZONTAL STROKES',
            '⋺': 'CONTAINS WITH LONG HORIZONTAL STROKE',
            '⋻': 'CONTAINS WITH VERTICAL BAR AT END OF HORIZONTAL STROKE',
            '⋼': 'SMALL CONTAINS WITH VERTICAL BAR AT END OF HORIZONTAL STROKE',
            '⋽': 'CONTAINS WITH OVERBAR',
            '⋾': 'SMALL CONTAINS WITH OVERBAR',
            '⋿': 'Z NOTATION BAG MEMBERSHIP',
            
            # Proof Notation
            '∴': 'THEREFORE',
            '∵': 'BECAUSE',
            '⊢': 'PROVES',
            '⊨': 'MODELS',
            '⊭': 'NOT MODELS',
            '⊬': 'NOT PROVES',
            '⊩': 'FORCES',
            '⊮': 'NOT FORCES',
            '⊪': 'TRIPLE VERTICAL BAR RIGHT TURNSTILE',
            '⊫': 'DOUBLE VERTICAL BAR DOUBLE RIGHT TURNSTILE',
            '⊯': 'NEGATED DOUBLE VERTICAL BAR DOUBLE RIGHT TURNSTILE',
            '⊰': 'PRECEDES UNDER RELATION',
            '⊱': 'SUCCEEDS UNDER RELATION',
            '⊲': 'NORMAL SUBGROUP OF',
            '⊳': 'CONTAINS AS NORMAL SUBGROUP',
            '⊴': 'NORMAL SUBGROUP OF OR EQUAL TO',
            '⊵': 'CONTAINS AS NORMAL SUBGROUP OR EQUAL TO',
            '⊶': 'ORIGINAL OF',
            '⊷': 'IMAGE OF',
            '⊸': 'MULTIMAP',
            '⊹': 'HERMITIAN CONJUGATE MATRIX',
            '⊺': 'INTERCALATE',
            '⊻': 'XOR',
            '⊼': 'NAND',
            '⊽': 'NOR',
            '⊾': 'RIGHT ANGLE WITH ARC',
            '⊿': 'RIGHT TRIANGLE',
            '⋀': 'N-ARY LOGICAL AND',
            '⋁': 'N-ARY LOGICAL OR',
            '⋂': 'N-ARY INTERSECTION',
            '⋃': 'N-ARY UNION',
            '⋄': 'DIAMOND OPERATOR',
            '⋅': 'DOT OPERATOR',
            '⋆': 'STAR OPERATOR',
            '⋇': 'DIVISION TIMES',
            '⋈': 'BOWTIE',
            '⋉': 'LEFT NORMAL FACTOR SEMIDIRECT PRODUCT',
            '⋊': 'RIGHT NORMAL FACTOR SEMIDIRECT PRODUCT',
            '⋋': 'LEFT SEMIDIRECT PRODUCT',
            '⋌': 'RIGHT SEMIDIRECT PRODUCT',
            '⋍': 'REVERSED TILDE EQUALS',
            '⋎': 'CURLY LOGICAL OR',
            '⋏': 'CURLY LOGICAL AND',
            '⋐': 'DOUBLE SUBSET',
            '⋑': 'DOUBLE SUPERSET',
            '⋒': 'DOUBLE INTERSECTION',
            '⋓': 'DOUBLE UNION',
            '⋔': 'PITCHFORK',
            '⋕': 'EQUAL AND PARALLEL TO',
            '⋖': 'LESS-THAN WITH DOT',
            '⋗': 'GREATER-THAN WITH DOT',
            '⋘': 'VERY MUCH LESS-THAN',
            '⋙': 'VERY MUCH GREATER-THAN',
            '⋚': 'LESS-THAN EQUAL TO OR GREATER-THAN',
            '⋛': 'GREATER-THAN EQUAL TO OR LESS-THAN',
            '⋜': 'EQUAL TO OR LESS-THAN',
            '⋝': 'EQUAL TO OR GREATER-THAN',
            '⋞': 'EQUAL TO OR PRECEDES',
            '⋟': 'EQUAL TO OR SUCCEEDS',
            '⋠': 'DOES NOT PRECEDE OR EQUAL',
            '⋡': 'DOES NOT SUCCEED OR EQUAL',
            '⋢': 'NOT SQUARE IMAGE OF OR EQUAL TO',
            '⋣': 'NOT SQUARE ORIGINAL OF OR EQUAL TO',
            '⋤': 'SQUARE IMAGE OF OR NOT EQUAL TO',
            '⋥': 'SQUARE ORIGINAL OF OR NOT EQUAL TO',
            '⋦': 'LESS-THAN BUT NOT EQUIVALENT TO',
            '⋧': 'GREATER-THAN BUT NOT EQUIVALENT TO',
            '⋨': 'PRECEDES BUT NOT EQUIVALENT TO',
            '⋩': 'SUCCEEDS BUT NOT EQUIVALENT TO',
            '⋪': 'NOT NORMAL SUBGROUP OF',
            '⋫': 'DOES NOT CONTAIN AS NORMAL SUBGROUP',
            '⋬': 'NOT NORMAL SUBGROUP OF OR EQUAL TO',
            '⋭': 'DOES NOT CONTAIN AS NORMAL SUBGROUP OR EQUAL TO',
            '⋮': 'VERTICAL ELLIPSIS',
            '⋯': 'MIDLINE HORIZONTAL ELLIPSIS',
            '⋰': 'UP RIGHT DIAGONAL ELLIPSIS',
            '⋱': 'DOWN RIGHT DIAGONAL ELLIPSIS',
            '⋲': 'ELEMENT OF WITH LONG HORIZONTAL STROKE',
            '⋳': 'ELEMENT OF WITH VERTICAL BAR AT END OF HORIZONTAL STROKE',
            '⋴': 'SMALL ELEMENT OF WITH VERTICAL BAR AT END OF HORIZONTAL STROKE',
            '⋵': 'ELEMENT OF WITH DOT ABOVE',
            '⋶': 'ELEMENT OF WITH OVERBAR',
            '⋷': 'SMALL ELEMENT OF WITH OVERBAR',
            '⋸': 'ELEMENT OF WITH UNDERBAR',
            '⋹': 'ELEMENT OF WITH TWO HORIZONTAL STROKES',
            '⋺': 'CONTAINS WITH LONG HORIZONTAL STROKE',
            '⋻': 'CONTAINS WITH VERTICAL BAR AT END OF HORIZONTAL STROKE',
            '⋼': 'SMALL CONTAINS WITH VERTICAL BAR AT END OF HORIZONTAL STROKE',
            '⋽': 'CONTAINS WITH OVERBAR',
            '⋾': 'SMALL CONTAINS WITH OVERBAR',
            '⋿': 'Z NOTATION BAG MEMBERSHIP'
        }
        
        # Add formal logic symbols to math_symbols set
        self.math_symbols.update(self.formal_logic_symbols.keys())

    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize a mathematical expression using SymPy parsing.
        """
        try:
            # Parse the expression using SymPy
            expr = parse_expr(text, transformations=self.transformations)
            
            # Convert the expression to a string representation
            # This will give us a standardized form
            expr_str = str(expr)
            
            # Split into tokens
            tokens = []
            current = ""
            
            for char in expr_str:
                if char in self.math_symbols:
                    if current:
                        tokens.extend(self._tokenize_word(current))
                        current = ""
                    tokens.append(self.vocab.get(char, self.vocab[self.unk_token]))
                else:
                    current += char
            
            if current:
                tokens.extend(self._tokenize_word(current))
                
            return tokens
            
        except Exception as e:
            # If SymPy parsing fails, fall back to basic tokenization
            return self._basic_tokenize(text)

    def _basic_tokenize(self, text: str) -> List[int]:
        """
        Basic tokenization fallback when SymPy parsing fails.
        """
        tokens = []
        current = ""
        
        for char in text:
            if char in self.math_symbols:
                if current:
                    tokens.extend(self._tokenize_word(current))
                    current = ""
                tokens.append(self.vocab.get(char, self.vocab[self.unk_token]))
            else:
                current += char
        
        if current:
            tokens.extend(self._tokenize_word(current))
            
        return tokens

    def _tokenize_word(self, word: str) -> List[int]:
        """
        Tokenize a word using Byte-Pair Encoding.
        """
        if word in self.vocab:
            return [self.vocab[word]]
        
        # For unknown words, split into characters
        return [self.vocab.get(c, self.vocab[self.unk_token]) for c in word]

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Convert token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        # First pass: convert tokens to text
        tokens = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in [self.vocab[t] for t in [self.cls_token, self.sep_token, self.pad_token, self.mask_token]]:
                continue
            token = self.reverse_vocab.get(token_id, self.unk_token)
            tokens.append(token)
        
        # Join tokens and clean up the text
        text = ''.join(tokens)
        
        # Handle division expressions
        if 'div(' in text:
            # Convert div(a,b) to (a)/(b)
            matches = list(re.finditer(r'div\((.*?),(.*?)\)', text))
            for match in reversed(matches):  # Process from right to left
                a, b = match.groups()
                a = a.strip()
                b = b.strip()
                # Always add parentheses for consistency
                text = text[:match.start()] + f"({a})/({b})" + text[match.end():]
        
        # Handle SymPy's internal representation
        if 'Mul(' in text and 'Pow(' in text:
            matches = list(re.finditer(r'Mul\((.*?), Pow\((.*?), -1\)\)', text))
            for match in reversed(matches):  # Process from right to left
                num, den = match.groups()
                # Clean up the expressions
                num = re.sub(r'[a-zA-Z]+\((.*?)\)', r'\1', num)
                den = re.sub(r'[a-zA-Z]+\((.*?)\)', r'\1', den)
                text = text[:match.start()] + f"{num}/{den}" + text[match.end():]
        
        # Handle Rational
        if 'Rational(' in text:
            matches = list(re.finditer(r'Rational\((.*?),(.*?)\)', text))
            for match in reversed(matches):  # Process from right to left
                num, den = match.groups()
                text = text[:match.start()] + f"{num}/{den}" + text[match.end():]
        
        # Clean up any remaining SymPy function calls
        text = re.sub(r'[a-zA-Z]+\((.*?)\)', r'\1', text)
        
        # Standardize division notation
        text = text.replace('÷', '/')
        
        # Clean up any remaining [UNK] tokens
        text = text.replace(self.unk_token, '')
        
        return text

    def train(self, texts: List[str], num_merges: int = 1000):
        """
        Train the tokenizer on a corpus of texts.
        """
        # Count character pairs
        pairs = Counter()
        for text in texts:
            try:
                # Try to parse with SymPy first
                expr = parse_expr(text, transformations=self.transformations)
                text = str(expr)
            except:
                pass  # If parsing fails, use original text
                
            chars = list(text)
            for i in range(len(chars) - 1):
                pairs[(chars[i], chars[i + 1])] += 1
        
        # Perform merges
        for _ in range(num_merges):
            if not pairs:
                break
                
            # Find most frequent pair
            pair = max(pairs.items(), key=lambda x: x[1])[0]
            new_token = ''.join(pair)
            
            # Add to vocabulary
            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)
                self.reverse_vocab[self.vocab[new_token]] = new_token
            
            # Update pairs
            new_pairs = Counter()
            for text in texts:
                try:
                    expr = parse_expr(text, transformations=self.transformations)
                    text = str(expr)
                except:
                    pass
                    
                chars = list(text)
                i = 0
                while i < len(chars) - 1:
                    if chars[i] == pair[0] and chars[i + 1] == pair[1]:
                        chars[i:i + 2] = [new_token]
                    i += 1
                for i in range(len(chars) - 1):
                    new_pairs[(chars[i], chars[i + 1])] += 1
            
            pairs = new_pairs 
            pairs = new_pairs 