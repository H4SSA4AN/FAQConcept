"""
FAQ Video POC Application

A proof-of-concept application for FAQ search and retrieval
using vector databases and embeddings with real-time video capabilities.
"""

__version__ = "1.0.0"
__author__ = "FAQ Video POC Team"

from .settings import Settings
from .search import FAQSearch

__all__ = ["Settings", "FAQSearch"]
