import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tokenizer import MathTokenizer
import unittest

class TestDivisionHandling(unittest.TestCase):
    def setUp(self):
        self.tokenizer = MathTokenizer()

    def test_simple_division(self):
        """Test basic division operations"""
        expressions = [
            "1/2",
            "a/b",
            "(x+1)/(y-1)",
            "div(3,4)",
            "3÷4"
        ]
        
        for expr in expressions:
            tokens = self.tokenizer.tokenize(expr)
            decoded = self.tokenizer.decode(tokens)
            # Remove spaces for comparison
            decoded = ''.join(decoded.split())
            expr_normalized = ''.join(expr.split())
            # Replace ÷ with / for normalization
            expr_normalized = expr_normalized.replace('÷', '/')
            # Replace div(a,b) with (a)/(b)
            if 'div(' in expr_normalized:
                a, b = expr_normalized[4:-1].split(',')
                expr_normalized = f"({a})/({b})"
            
            self.assertEqual(decoded, expr_normalized, 
                           f"Failed to correctly handle division in expression: {expr}")

    def test_complex_division(self):
        """Test more complex division scenarios"""
        expressions = [
            "1/(2+x)",
            "(a+b)/(c+d)",
            "1/2/3",  # Multiple divisions
            "(1/2)/(3/4)",  # Nested divisions
            "x/(y/z)"  # Mixed divisions
        ]
        
        for expr in expressions:
            tokens = self.tokenizer.tokenize(expr)
            decoded = self.tokenizer.decode(tokens)
            # Remove spaces for comparison
            decoded = ''.join(decoded.split())
            expr_normalized = ''.join(expr.split())
            
            self.assertEqual(decoded, expr_normalized, 
                           f"Failed to correctly handle division in expression: {expr}")

    def test_sympy_division(self):
        """Test SymPy's internal division representation"""
        expressions = [
            "Mul(x, Pow(y, -1))",  # Should become x/y
            "Rational(1,2)",       # Should become 1/2
            "div(a,b)"            # Should become (a)/(b)
        ]
        
        expected = [
            "x/y",
            "1/2",
            "(a)/(b)"
        ]
        
        for expr, expect in zip(expressions, expected):
            tokens = self.tokenizer.tokenize(expr)
            decoded = ''.join(self.tokenizer.decode(tokens).split())
            expect_normalized = ''.join(expect.split())
            
            self.assertEqual(decoded, expect_normalized, 
                           f"Failed to handle SymPy division in expression: {expr}")

if __name__ == '__main__':
    unittest.main() 