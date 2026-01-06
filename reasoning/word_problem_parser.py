#!/usr/bin/env python3
"""
ECH0-PRIME Advanced Word Problem Parser
NLP-powered mathematical word problem understanding and solving.
"""

import re
import nltk
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
import spacy
import numpy as np

# Try to load spaCy model, fallback if not available
try:
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except:
    nlp = None
    SPACY_AVAILABLE = False

@dataclass
class Quantity:
    """Represents a quantity mentioned in a word problem"""
    value: float
    unit: str = ""
    entity: str = ""  # What the quantity refers to (apples, dollars, etc.)
    is_unknown: bool = False  # True if this represents the unknown we're solving for

@dataclass
class MathOperation:
    """Represents a mathematical operation extracted from text"""
    operation: str  # 'add', 'subtract', 'multiply', 'divide', 'equals'
    operands: List[Quantity]
    result: Optional[Quantity] = None

@dataclass
class WordProblem:
    """Complete representation of a parsed word problem"""
    original_text: str
    quantities: List[Quantity]
    operations: List[MathOperation]
    question: str
    unknown_variable: str = "x"

class AdvancedWordProblemParser:
    """
    Advanced NLP-powered word problem parser using spaCy and custom rules.
    """

    def __init__(self):
        # Number word mappings
        self.number_words = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
            'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
            'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
            'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000
        }

        # Operation keywords
        self.operation_keywords = {
            'add': ['plus', 'and', 'added to', 'combined with', 'together', 'total'],
            'subtract': ['minus', 'subtract', 'take away', 'less', 'fewer', 'gives', 'gave', 'lost'],
            'multiply': ['times', 'multiplied by', 'of', 'product', 'each', 'per'],
            'divide': ['divided by', 'split', 'shared', 'each gets', 'per person'],
            'equals': ['equals', 'is', 'are', 'was', 'were', 'has', 'have', 'left with']
        }

        # Question patterns
        self.question_patterns = [
            r'how many.*\?',
            r'what is.*\?',
            r'find.*\?',
            r'calculate.*\?',
            r'what.*total.*\?',
            r'how much.*\?'
        ]

    def parse_word_problem(self, text: str) -> WordProblem:
        """
        Parse a complete word problem into mathematical components.
        """
        # Clean and normalize text
        text = self._preprocess_text(text)

        # Extract question
        question = self._extract_question(text)

        # Extract quantities
        quantities = self._extract_quantities(text)

        # Extract operations and relationships
        operations = self._extract_operations(text, quantities)

        # Build relationships between quantities
        operations = self._build_relationships(quantities, operations, text)

        return WordProblem(
            original_text=text,
            quantities=quantities,
            operations=operations,
            question=question
        )

    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text for processing."""
        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Normalize number words to digits where possible
        for word, num in self.number_words.items():
            text = re.sub(rf'\b{word}\b', str(num), text)

        return text

    def _extract_question(self, text: str) -> str:
        """Extract the question part of the word problem."""
        sentences = re.split(r'[.!?]+', text)

        for sentence in sentences:
            sentence = sentence.strip()
            if any(re.search(pattern, sentence) for pattern in self.question_patterns):
                return sentence

        # Fallback: last sentence is usually the question
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences[-1] if sentences else ""

    def _extract_quantities(self, text: str) -> List[Quantity]:
        """Extract all quantities mentioned in the text."""
        quantities = []

        # Use spaCy if available for better entity recognition
        if SPACY_AVAILABLE and nlp:
            doc = nlp(text)

            # Extract numeric entities
            for ent in doc.ents:
                if ent.label_ in ['CARDINAL', 'QUANTITY', 'MONEY']:
                    value = self._parse_numeric_value(ent.text)
                    if value is not None:
                        # Get context for what this quantity refers to
                        entity = self._extract_entity_context(ent, doc)
                        quantities.append(Quantity(
                            value=value,
                            entity=entity,
                            unit=self._extract_unit(ent.text)
                        ))

        # Fallback: regex-based extraction
        if not quantities:
            quantities = self._extract_quantities_regex(text)

        # Check for unknown quantities (what we're solving for)
        question_words = ['how many', 'how much', 'what is']
        for quantity in quantities:
            if any(word in quantity.entity.lower() for word in question_words):
                quantity.is_unknown = True

        return quantities

    def _extract_quantities_regex(self, text: str) -> List[Quantity]:
        """Fallback quantity extraction using regex."""
        quantities = []

        # Pattern for numbers with optional units
        number_pattern = r'(\d+(?:\.\d+)?)\s*([a-zA-Z]+(?:\s+[a-zA-Z]+)*)?'
        matches = re.findall(number_pattern, text)

        for match in matches:
            value_str, unit = match
            try:
                value = float(value_str)
                quantities.append(Quantity(
                    value=value,
                    unit=unit.strip() if unit else "",
                    entity=self._infer_entity_from_context(text, value)
                ))
            except ValueError:
                continue

        return quantities

    def _parse_numeric_value(self, text: str) -> Optional[float]:
        """Parse numeric value from text."""
        # Remove commas and handle decimals
        text = text.replace(',', '')

        # Try direct float conversion
        try:
            return float(text)
        except ValueError:
            pass

        # Handle fractions like "1/2"
        fraction_match = re.search(r'(\d+)/(\d+)', text)
        if fraction_match:
            num, den = map(int, fraction_match.groups())
            return num / den

        return None

    def _extract_entity_context(self, ent, doc) -> str:
        """Extract what a quantity refers to using spaCy."""
        # Look at surrounding words for context
        start_idx = max(0, ent.start - 3)
        end_idx = min(len(doc), ent.end + 3)

        context_words = []
        for token in doc[start_idx:end_idx]:
            if token.pos_ in ['NOUN', 'PROPN'] and token.text != ent.text:
                context_words.append(token.text)

        return ' '.join(context_words) if context_words else 'unknown'

    def _extract_unit(self, text: str) -> str:
        """Extract unit from quantity text."""
        # Common units
        units = ['apples', 'oranges', 'dollars', 'cents', 'meters', 'feet', 'inches',
                'pounds', 'ounces', 'hours', 'minutes', 'seconds', 'people', 'students']

        for unit in units:
            if unit in text.lower():
                return unit

        return ""

    def _infer_entity_from_context(self, text: str, value: float) -> str:
        """Infer what a quantity refers to from surrounding context."""
        # Look for nouns near the number
        words = text.split()
        try:
            num_idx = words.index(str(int(value)))
            # Look at words after the number
            for i in range(num_idx + 1, min(num_idx + 4, len(words))):
                word = words[i].lower()
                if word in ['apples', 'oranges', 'balls', 'books', 'people', 'dollars']:
                    return word
        except ValueError:
            pass

        return "items"

    def _extract_operations(self, text: str, quantities: List[Quantity]) -> List[MathOperation]:
        """Extract mathematical operations from text."""
        operations = []

        # Look for operation keywords
        for op_type, keywords in self.operation_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    # Find quantities involved in this operation
                    involved_quantities = self._find_quantities_for_operation(text, keyword, quantities)
                    if len(involved_quantities) >= 2:
                        operations.append(MathOperation(
                            operation=op_type,
                            operands=involved_quantities
                        ))

        # Special handling for multiplication patterns like "each", "per", "at $X each"
        if 'each' in text or 'per' in text or 'at $' in text:
            mult_quantities = self._find_multiplication_quantities(text, quantities)
            if len(mult_quantities) >= 2:
                operations.append(MathOperation(
                    operation='multiply',
                    operands=mult_quantities
                ))

        return operations

    def _find_quantities_for_operation(self, text: str, keyword: str, quantities: List[Quantity]) -> List[Quantity]:
        """Find quantities involved in a specific operation."""
        # Simple heuristic: quantities mentioned around the keyword
        keyword_pos = text.find(keyword)
        if keyword_pos == -1:
            return []

        # Get quantities within a window around the keyword
        window_size = 50
        start_pos = max(0, keyword_pos - window_size)
        end_pos = min(len(text), keyword_pos + window_size)
        window_text = text[start_pos:end_pos]

        involved = []
        for quantity in quantities:
            if str(int(quantity.value)) in window_text:
                involved.append(quantity)

        return involved[:2]  # Most operations involve 2 quantities

    def _find_multiplication_quantities(self, text: str, quantities: List[Quantity]) -> List[Quantity]:
        """Find quantities involved in multiplication (each, per, etc.)"""
        # Look for patterns like "X items for Y each" or "Y per X"
        mult_quantities = []

        # Sort quantities by their position in text to maintain order
        text_lower = text.lower()

        # Simple heuristic: if "each" or "per" is present, multiply the two quantities
        if ('each' in text_lower or 'per' in text_lower or 'at $' in text_lower) and len(quantities) >= 2:
            # Return the first two quantities found
            return quantities[:2]

        return mult_quantities

    def _build_relationships(self, quantities: List[Quantity], operations: List[MathOperation], text: str) -> List[MathOperation]:
        """Build relationships between quantities and operations."""
        # Enhanced relationship building would go here
        # For now, return operations as-is
        return operations

    def solve_word_problem(self, problem: WordProblem) -> Dict[str, Any]:
        """
        Solve a parsed word problem.
        """
        try:
            # Simple solving logic for common patterns
            if not problem.quantities:
                return {"solution": "No quantities found", "confidence": 0.0}

            # Pattern 1: Simple subtraction (giving/taking away)
            if any(op.operation == 'subtract' for op in problem.operations):
                subtract_ops = [op for op in problem.operations if op.operation == 'subtract']
                if subtract_ops and len(subtract_ops[0].operands) == 2:
                    op1, op2 = subtract_ops[0].operands
                    result = op1.value - op2.value
                    return {
                        "solution": str(int(result) if result.is_integer() else result),
                        "method": "subtraction",
                        "confidence": 0.8
                    }

            # Pattern 2: Simple addition (combining totals)
            if any(op.operation == 'add' for op in problem.operations):
                add_ops = [op for op in problem.operations if op.operation == 'add']
                if add_ops and len(add_ops[0].operands) == 2:
                    op1, op2 = add_ops[0].operands
                    result = op1.value + op2.value
                    return {
                        "solution": str(int(result) if result.is_integer() else result),
                        "method": "addition",
                        "confidence": 0.8
                    }

            # Pattern 3: Simple multiplication (each, per, etc.)
            if any(op.operation == 'multiply' for op in problem.operations):
                mult_ops = [op for op in problem.operations if op.operation == 'multiply']
                if mult_ops and len(mult_ops[0].operands) == 2:
                    op1, op2 = mult_ops[0].operands
                    result = op1.value * op2.value
                    return {
                        "solution": str(int(result) if result.is_integer() else result),
                        "method": "multiplication",
                        "confidence": 0.8
                    }

            # Pattern 3: If we have unknowns, set up equations
            unknowns = [q for q in problem.quantities if q.is_unknown]
            if unknowns:
                # Try to solve for unknown
                return self._solve_for_unknown(problem)

            # Default: return largest quantity or sum
            if len(problem.quantities) >= 2:
                total = sum(q.value for q in problem.quantities)
                return {
                    "solution": str(int(total) if total.is_integer() else total),
                    "method": "sum_all_quantities",
                    "confidence": 0.5
                }
            elif problem.quantities:
                return {
                    "solution": str(int(problem.quantities[0].value) if problem.quantities[0].value.is_integer() else problem.quantities[0].value),
                    "method": "single_quantity",
                    "confidence": 0.6
                }

            return {
                "solution": "Unable to solve word problem",
                "method": "no_solution",
                "confidence": 0.0
            }

        except Exception as e:
            return {
                "solution": f"Error solving word problem: {str(e)}",
                "method": "error",
                "confidence": 0.0
            }

    def _solve_for_unknown(self, problem: WordProblem) -> Dict[str, Any]:
        """Solve for unknown quantities in word problems."""
        # This would implement more advanced equation solving
        # For now, return a placeholder
        return {
            "solution": "Advanced equation solving not yet implemented",
            "method": "unknown_variable",
            "confidence": 0.3
        }

def create_word_problem_parser() -> AdvancedWordProblemParser:
    """Factory function for word problem parser."""
    return AdvancedWordProblemParser()

def get_word_problem_solver():
    """Get a word problem solver instance."""
    return AdvancedWordProblemParser()

# Test function
def test_word_problem_parser():
    """Test the word problem parser."""
    parser = AdvancedWordProblemParser()

    test_problems = [
        "John has 5 apples and gives 2 to Mary. How many does he have left?",
        "A store has 15 red balls and 10 blue balls. How many balls total?",
        "Sarah bought 3 books for $12 each. How much did she spend?"
    ]

    for problem_text in test_problems:
        print(f"\nProblem: {problem_text}")

        try:
            # Parse the problem
            problem = parser.parse_word_problem(problem_text)
            print(f"Quantities found: {len(problem.quantities)}")
            for q in problem.quantities:
                print(f"  - {q.value} {q.entity}")

            # Try to solve
            solution = parser.solve_word_problem(problem)
            print(f"Solution: {solution['solution']} (confidence: {solution['confidence']:.2f})")

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    test_word_problem_parser()