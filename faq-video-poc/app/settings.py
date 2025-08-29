"""
Application settings and configuration management.
"""

import os
from typing import Optional
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    chroma_persist_dir: str = "chroma_db"


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    max_seq_length: int = 512


@dataclass
class DataConfig:
    """Data and file paths configuration."""
    faq_data_path: str = "data/faq.csv"
    demo_questions_path: str = "scripts/demo_questions.txt"


@dataclass
class WebRTCConfig:
    """WebRTC streaming configuration."""
    ice_servers: list = None
    video_width: int = 640
    video_height: int = 480
    video_fps: int = 30

    def __post_init__(self):
        if self.ice_servers is None:
            self.ice_servers = ["stun:stun.l.google.com:19302"]


@dataclass
class AppConfig:
    """Main application configuration."""
    log_level: str = "INFO"
    max_results: int = 5
    similarity_threshold: float = 0.7


class Settings:
    """Centralized settings management."""

    def __init__(self):
        self.database = DatabaseConfig(
            chroma_host=os.getenv("CHROMA_HOST", "localhost"),
            chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
            qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
            qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
            chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
        )

        self.embedding = EmbeddingConfig(
            model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            dimension=int(os.getenv("EMBEDDING_DIMENSION", "384")),
            max_seq_length=int(os.getenv("MAX_SEQ_LENGTH", "512"))
        )

        self.data = DataConfig(
            faq_data_path=os.getenv("FAQ_DATA_PATH", "data/faq.csv"),
            demo_questions_path=os.getenv("DEMO_QUESTIONS_PATH", "scripts/demo_questions.txt")
        )

        self.webrtc = WebRTCConfig(
            ice_servers=os.getenv("WEBRTC_ICE_SERVERS", "stun:stun.l.google.com:19302").split(","),
            video_width=int(os.getenv("VIDEO_WIDTH", "640")),
            video_height=int(os.getenv("VIDEO_HEIGHT", "480")),
            video_fps=int(os.getenv("VIDEO_FPS", "30"))
        )

        self.app = AppConfig(
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            max_results=int(os.getenv("MAX_RESULTS", "5")),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
        )

    @property
    def project_root(self) -> Path:
        """Get the project root directory."""
        return Path(__file__).parent.parent

    @property
    def faq_data_path(self) -> Path:
        """Get the full path to FAQ data file."""
        return self.project_root / self.data.faq_data_path

    @property
    def chroma_persist_dir(self) -> Path:
        """Get the full path to Chroma persistence directory."""
        return self.project_root / self.database.chroma_persist_dir

    @property
    def demo_questions_path(self) -> Path:
        """Get the full path to demo questions file."""
        return self.project_root / self.data.demo_questions_path


# Global settings instance
settings = Settings()
