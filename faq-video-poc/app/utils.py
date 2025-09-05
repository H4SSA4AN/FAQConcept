"""
Utility functions for FAQ Video POC.
"""

import os
import re
import pandas as pd
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger


def clean_text(text: str) -> str:
    """
    Clean and normalize text for better embeddings.

    Args:
        text: Input text to clean

    Returns:
        Cleaned text
    """
    if not text:
        return ""

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text.strip())

    # Preserve separators like slashes by spacing them, so tokens don't merge
    # e.g., "placements/have" -> "placements / have"
    text = text.replace('/', ' / ')

    # Relaxed cleaning: keep common punctuation that conveys structure
    # Keep: . , ! ? - / & ( ) : '
    text = re.sub(r"[^\w\s\.,!\?/&():'-]", '', text)

    # Collapse any extra spaces introduced
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def validate_csv_format(csv_path: str) -> pd.DataFrame:
    """
    Validate FAQ CSV file format and return DataFrame.

    Args:
        csv_path: Path to CSV file

    Returns:
        Validated DataFrame

    Raises:
        ValueError: If CSV format is invalid
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    try:
        # Use UTF-8 encoding to preserve punctuation and special characters
        df = pd.read_csv(csv_path, encoding='utf-8')

        # Check required columns
        required_columns = ['id', 'question', 'answer']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Validate data types
        if not pd.api.types.is_integer_dtype(df['id']):
            raise ValueError("Column 'id' must contain integers")

        # Check for empty values in required columns
        for col in required_columns:
            if df[col].isnull().any():
                raise ValueError(f"Column '{col}' contains empty values")

        # Clean text columns (lightweight - preserves separators like '/')
        df['question'] = df['question'].apply(clean_text)
        df['answer'] = df['answer'].apply(clean_text)

        if 'category' in df.columns:
            df['category'] = df['category'].fillna('General')

        logger.info(f"Validated CSV with {len(df)} rows")
        return df

    except Exception as e:
        logger.error(f"Failed to validate CSV: {e}")
        raise


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    logger.remove()  # Remove default handler

    # Console handler
    logger.add(
        lambda msg: print(msg, end=""),
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    # File handler if specified
    if log_file:
        logger.add(
            log_file,
            level=log_level,
            rotation="10 MB",
            retention="1 week",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
        )


def ensure_directory(path: Path) -> Path:
    """
    Ensure directory exists, create if necessary.

    Args:
        path: Directory path

    Returns:
        Directory path
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_size_mb(file_path: Path) -> float:
    """
    Get file size in megabytes.

    Args:
        file_path: Path to file

    Returns:
        File size in MB
    """
    if not file_path.exists():
        return 0.0

    size_bytes = file_path.stat().st_size
    return size_bytes / (1024 * 1024)


def format_search_results(results: List[Dict[str, Any]], max_answer_length: int = 200) -> str:
    """
    Format search results for display.

    Args:
        results: List of search result dictionaries
        max_answer_length: Maximum answer length to display

    Returns:
        Formatted results string
    """
    if not results:
        return "No results found."

    formatted = []

    for i, result in enumerate(results, 1):
        question = result.get('question', 'N/A')
        answer = result.get('answer', 'N/A')
        category = result.get('category', 'General')
        score = result.get('score', 0.0)
        source = result.get('source', 'unknown')

        # Truncate answer if too long
        if len(answer) > max_answer_length:
            answer = answer[:max_answer_length] + "..."

        formatted_result = f"""{i}. Question: {question}
   Answer: {answer}
   Category: {category}
   Score: {score:.3f} (Source: {source})"""

        formatted.append(formatted_result)

    return "\n\n".join(formatted)


def calculate_similarity_stats(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate statistics from search results.

    Args:
        results: List of search result dictionaries

    Returns:
        Dictionary with statistics
    """
    if not results:
        return {
            'count': 0,
            'avg_score': 0.0,
            'max_score': 0.0,
            'min_score': 0.0
        }

    scores = [r.get('score', 0.0) for r in results]

    return {
        'count': len(results),
        'avg_score': sum(scores) / len(scores),
        'max_score': max(scores),
        'min_score': min(scores)
    }


def load_env_file(env_path: Optional[str] = None) -> bool:
    """
    Load environment variables from .env file.

    Args:
        env_path: Path to .env file

    Returns:
        True if file was loaded successfully, False otherwise
    """
    if env_path is None:
        env_path = Path(".env")

    if not env_path.exists():
        logger.warning(f"Environment file not found: {env_path}")
        return False

    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
        logger.info(f"Loaded environment variables from: {env_path}")
        return True
    except ImportError:
        logger.warning("python-dotenv not installed, skipping .env file loading")
        return False
    except Exception as e:
        logger.error(f"Failed to load .env file: {e}")
        return False


def log_answered_question(user_question: str, matched_question: str, accuracy_score: float,
                         csv_path: str = "data/answered_questions.csv") -> bool:
    """
    Log an answered question to CSV file.

    Args:
        user_question: The user's original question
        matched_question: The matched question from the FAQ database
        accuracy_score: The similarity/accuracy score
        csv_path: Path to the CSV file

    Returns:
        True if logged successfully, False otherwise
    """
    try:
        # Ensure data directory exists
        csv_file = Path(csv_path)
        csv_file.parent.mkdir(parents=True, exist_ok=True)

        # Check if file exists to determine if we need headers
        file_exists = csv_file.exists()

        # Prepare the data
        timestamp = datetime.now().isoformat()
        data = {
            'timestamp': timestamp,
            'user_question': user_question,
            'matched_question': matched_question,
            'accuracy_score': round(accuracy_score, 4)
        }

        # Write to CSV
        with open(csv_file, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['timestamp', 'user_question', 'matched_question', 'accuracy_score']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header if file is new
            if not file_exists:
                writer.writeheader()

            # Write the answered question
            writer.writerow(data)

        logger.info(f"✅ Logged answered question: '{user_question}' -> '{matched_question}' (score: {accuracy_score:.4f})")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to log answered question: {e}")
        return False
