"""
Text processing utilities for RAG query system.

Adapted from plans/lightrag-code/utils/text_processing.py
"""

import html
import logging
import re
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class TokenizerInterface(Protocol):
    """Defines the interface for a tokenizer, requiring encode and decode methods."""

    def encode(self, content: str) -> list[int]:
        """Encodes a string into a list of tokens."""
        ...

    def decode(self, tokens: list[int]) -> str:
        """Decodes a list of tokens into a string."""
        ...


class Tokenizer:
    """A wrapper around a tokenizer to provide a consistent interface for encoding and decoding."""

    def __init__(self, model_name: str, tokenizer: TokenizerInterface):
        """
        Initializes the Tokenizer with a tokenizer model name and a tokenizer instance.

        Args:
            model_name: The associated model name for the tokenizer.
            tokenizer: An instance of a class implementing the TokenizerInterface.
        """
        self.model_name: str = model_name
        self.tokenizer: TokenizerInterface = tokenizer

    def encode(self, content: str) -> list[int]:
        """
        Encodes a string into a list of tokens using the underlying tokenizer.

        Args:
            content: The string to encode.

        Returns:
            A list of integer tokens.
        """
        return self.tokenizer.encode(content)

    def decode(self, tokens: list[int]) -> str:
        """
        Decodes a list of tokens into a string using the underlying tokenizer.

        Args:
            tokens: A list of integer tokens to decode.

        Returns:
            The decoded string.
        """
        return self.tokenizer.decode(tokens)


class TiktokenTokenizer(Tokenizer):
    """A Tokenizer implementation using the tiktoken library."""

    def __init__(self, model_name: str = "gpt-4o-mini"):
        """
        Initializes the TiktokenTokenizer with a specified model name.

        Args:
            model_name: The model name for the tiktoken tokenizer to use.  Defaults to "gpt-4o-mini".

        Raises:
            ImportError: If tiktoken is not installed.
            ValueError: If the model_name is invalid.
        """
        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "tiktoken is not installed. Please install it with `pip install tiktoken` "
                "or define custom `tokenizer_func`."
            ) from None

        try:
            tokenizer = tiktoken.encoding_for_model(model_name)
            super().__init__(model_name=model_name, tokenizer=tokenizer)
        except KeyError:
            raise ValueError(f"Invalid model_name: {model_name}.") from None


def truncate_list_by_token_size(
    list_data: list[Any],
    key: Callable[[Any], str],
    max_token_size: int,
    tokenizer: Tokenizer,
) -> list[Any]:
    """Truncate a list of data by token size

    Args:
        list_data: List of data items to truncate
        key: Function to extract string from data item
        max_token_size: Maximum number of tokens
        tokenizer: Tokenizer instance for counting tokens

    Returns:
        Truncated list within token limit
    """
    if max_token_size <= 0:
        return []
    tokens = 0
    for i, data in enumerate(list_data):
        tokens += len(tokenizer.encode(key(data)))
        if tokens > max_token_size:
            return list_data[:i]
    return list_data


def cosine_similarity(v1, v2):
    """Calculate cosine similarity between two vectors

    Args:
        v1: First vector
        v2: Second vector

    Returns:
        Cosine similarity score
    """
    dot_product = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return dot_product / (norm1 * norm2)


def sanitize_text_for_encoding(text: str, replacement_char: str = "") -> str:
    """Sanitize text to ensure safe UTF-8 encoding by removing or replacing problematic characters.

    This function handles:
    - Surrogate characters (the main cause of encoding errors)
    - Other invalid Unicode sequences
    - Control characters that might cause issues
    - Unescape HTML escapes
    - Remove control characters
    - Whitespace trimming

    Args:
        text: Input text to sanitize
        replacement_char: Character to use for replacing invalid sequences

    Returns:
        Sanitized text that can be safely encoded as UTF-8

    Raises:
        ValueError: When text contains uncleanable encoding issues that cannot be safely processed
    """
    if not text:
        return text

    try:
        # First, strip whitespace
        text = text.strip()

        # Early return if text is empty after basic cleaning
        if not text:
            return text

        # Try to encode/decode to catch any encoding issues early
        text.encode("utf-8")

        # Remove or replace surrogate characters (U+D800 to U+DFFF)
        # These are the main cause of the encoding error
        sanitized = ""
        for char in text:
            code_point = ord(char)
            # Check for surrogate characters
            if 0xD800 <= code_point <= 0xDFFF:
                # Replace surrogate with replacement character
                sanitized += replacement_char
                continue
            # Check for other problematic characters
            elif code_point == 0xFFFE or code_point == 0xFFFF:
                # These are non-characters in Unicode
                sanitized += replacement_char
                continue
            else:
                sanitized += char

        # Additional cleanup: remove null bytes and other control characters that might cause issues
        # (but preserve common whitespace like \t, \n, \r)
        sanitized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", replacement_char, sanitized)

        # Test final encoding to ensure it's safe
        sanitized.encode("utf-8")

        # Unescape HTML escapes
        sanitized = html.unescape(sanitized)

        # Remove control characters but preserve common whitespace (\t, \n, \r)
        sanitized = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", "", sanitized)

        return sanitized.strip()

    except UnicodeEncodeError as e:
        # Critical change: Don't return placeholder, raise exception for caller to handle
        error_msg = f"Text contains uncleanable UTF-8 encoding issues: {str(e)[:100]}"
        logger.error(f"Text sanitization failed: {error_msg}")
        raise ValueError(error_msg) from e

    except Exception as e:
        logger.error(f"Text sanitization: Unexpected error: {str(e)}")
        # For other exceptions, if no encoding issues detected, return original text
        try:
            text.encode("utf-8")
            return text
        except UnicodeEncodeError:
            raise ValueError(f"Text sanitization failed with unexpected error: {str(e)}") from e


def remove_html_tags(text: str) -> str:
    """
    Remove HTML tags using BeautifulSoup for proper text extraction.

    This properly handles spacing between words when removing tags.
    Example: '<p>BRCA1</p><br/>gene' -> 'BRCA1 gene' (with space)

    Args:
        text: Text that may contain HTML tags

    Returns:
        Text with HTML tags removed and proper spacing preserved
    """
    if not text or "<" not in text:
        return text

    # Use BeautifulSoup to parse HTML and extract text
    # separator=' ' ensures words are separated by spaces
    soup = BeautifulSoup(text, "html.parser")
    cleaned = soup.get_text(separator=" ", strip=True)

    # Normalize multiple spaces to single space
    cleaned = re.sub(r"\s+", " ", cleaned)

    return cleaned.strip()


def normalize_extracted_info(name: str, remove_inner_quotes=False) -> str:
    """Normalize entity/relation names and description with the following rules:

    - Clean HTML tags (using BeautifulSoup for proper spacing)
    - Convert Chinese symbols to English symbols
    - Remove spaces between Chinese characters
    - Remove spaces between Chinese characters and English letters/numbers
    - Preserve spaces within English text and numbers
    - Replace Chinese parentheses with English parentheses
    - Replace Chinese dash with English dash
    - Remove English quotation marks from the beginning and end of the text
    - Remove English quotation marks in and around chinese
    - Remove Chinese quotation marks
    - Filter out short numeric-only text (length < 3 and only digits/dots)
    - remove_inner_quotes = True
        remove Chinese quotes
        remove English queotes in and around chinese
        Convert non-breaking spaces to regular spaces
        Convert narrow non-breaking spaces after non-digits to regular spaces

    Args:
        name: Entity name to normalize
        remove_inner_quotes: Whether to remove inner quotes

    Returns:
        Normalized entity name
    """
    # Clean HTML tags using BeautifulSoup (preserves word boundaries)
    name = remove_html_tags(name)

    # Chinese full-width letters to half-width (A-Z, a-z)
    name = name.translate(
        str.maketrans(
            "ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ",
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
        )
    )

    # Chinese full-width numbers to half-width
    name = name.translate(str.maketrans("０１２３４５６７８９", "0123456789"))

    # Chinese full-width symbols to half-width
    name = name.replace("－", "-")  # Chinese minus
    name = name.replace("＋", "+")  # Chinese plus
    name = name.replace("／", "/")  # Chinese slash
    name = name.replace("＊", "*")  # Chinese asterisk

    # Replace Chinese parentheses with English parentheses
    name = name.replace("（", "(").replace("）", ")")

    # Replace Chinese dash with English dash (additional patterns)
    name = name.replace("—", "-").replace("－", "-")

    # Chinese full-width space to regular space (after other replacements)
    name = name.replace("　", " ")

    # Use regex to remove spaces between Chinese characters
    # Regex explanation:
    # (?<=[\u4e00-\u9fa5]): Positive lookbehind for Chinese character
    # \s+: One or more whitespace characters
    # (?=[\u4e00-\u9fa5]): Positive lookahead for Chinese character
    name = re.sub(r"(?<=[\u4e00-\u9fa5])\s+(?=[\u4e00-\u9fa5])", "", name)

    # Remove spaces between Chinese and English/numbers/symbols
    name = re.sub(r"(?<=[\u4e00-\u9fa5])\s+(?=[a-zA-Z0-9\(\)\[\]@#$%!&\*\-=+_])", "", name)
    name = re.sub(r"(?<=[a-zA-Z0-9\(\)\[\]@#$%!&\*\-=+_])\s+(?=[\u4e00-\u9fa5])", "", name)

    # Remove outer quotes
    if len(name) >= 2:
        # Handle double quotes
        if name.startswith('"') and name.endswith('"'):
            inner_content = name[1:-1]
            if '"' not in inner_content:  # No double quotes inside
                name = inner_content

        # Handle single quotes
        if name.startswith("'") and name.endswith("'"):
            inner_content = name[1:-1]
            if "'" not in inner_content:  # No single quotes inside
                name = inner_content

        # Handle Chinese-style double quotes
        if name.startswith(""") and name.endswith("""):
            inner_content = name[1:-1]
            if """ not in inner_content and """ not in inner_content:
                name = inner_content
        if name.startswith("'") and name.endswith("'"):
            inner_content = name[1:-1]
            if "'" not in inner_content and "'" not in inner_content:
                name = inner_content

        # Handle Chinese-style book title mark
        if name.startswith("《") and name.endswith("》"):
            inner_content = name[1:-1]
            if "《" not in inner_content and "》" not in inner_content:
                name = inner_content

    if remove_inner_quotes:
        # Remove Chinese quotes
        name = name.replace(""", "").replace(""", "").replace("'", "").replace("'", "")
        # Remove English queotes in and around chinese
        name = re.sub(r"['\"]+(?=[\u4e00-\u9fa5])", "", name)
        name = re.sub(r"(?<=[\u4e00-\u9fa5])['\"]+", "", name)
        # Convert non-breaking space to regular space
        name = name.replace("\u00a0", " ")
        # Convert narrow non-breaking space to regular space when after non-digits
        name = re.sub(r"(?<=[^\d])\u202F", " ", name)

    # Remove spaces from the beginning and end of the text
    name = name.strip()

    # Filter out pure numeric content with length < 3
    if len(name) < 3 and re.match(r"^[0-9]+$", name):
        return ""

    def should_filter_by_dots(text):
        """
        Check if the string consists only of dots and digits, with at least one dot
        Filter cases include: 1.2.3, 12.3, .123, 123., 12.3., .1.23 etc.
        """
        return all(c.isdigit() or c == "." for c in text) and "." in text

    if len(name) < 6 and should_filter_by_dots(name):
        # Filter out mixed numeric and dot content with length < 6
        return ""

    return name


def sanitize_and_normalize_extracted_text(input_text: str, remove_inner_quotes=False) -> str:
    """Santitize and normalize extracted text

    Args:
        input_text: text string to be processed
        remove_inner_quotes: whether to remove inner quotes

    Returns:
        Santitized and normalized text string
    """
    safe_input_text = sanitize_text_for_encoding(input_text)
    if safe_input_text:
        normalized_text = normalize_extracted_info(safe_input_text, remove_inner_quotes=remove_inner_quotes)
        return normalized_text
    return ""
