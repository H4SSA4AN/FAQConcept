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
    chroma_persist_dir: str = "chroma_db"


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model_name: str = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
    dimension: int = 384
    max_seq_length: int = 512


@dataclass
class DataConfig:
    """Data and file paths configuration."""
    faq_data_path: str = "data/faq.csv"


# Note: WebRTC is not used in the current web app; removed redundant config


@dataclass
class SpeechConfig:
    """Speech-to-text configuration."""
    provider: str = "whisper"  # Options: "whisper" (local), "openai" (API)
    model_name: str = "turbo"  # Options: "tiny", "base", "small", "medium", "large", "turbo"
    language: str = "en"  # Language code, None for auto-detection
    sample_rate: int = 16000  # Audio sample rate
    chunk_duration: float = 0.1  # Duration of audio chunks for processing (smaller = more responsive)
    silence_threshold: float = 0.8  # Silence duration to stop recording (seconds)
    max_recording_time: int = 30  # Maximum recording time in seconds
    device_index: Optional[int] = None  # Audio device index, None for default
    energy_threshold: int = 300  # Legacy energy threshold (for compatibility)

    # Advanced VAD (Voice Activity Detection) settings
    vad_min_recording_duration: float = 1.0  # Minimum recording duration to keep (seconds)
    vad_pre_roll_duration: float = 0.2  # Amount of pre-roll audio to include (seconds)
    vad_noise_floor: float = 0.001  # Minimum noise floor for adaptive threshold
    vad_min_speech_frames: int = 3  # Min consecutive speech frames to start recording
    vad_min_silence_frames: int = 8  # Min consecutive silence frames to stop recording

    # OpenAI API settings
    openai_api_key: Optional[str] = None
    openai_api_base: Optional[str] = None
    openai_model: str = "gpt-4o-transcribe"  # default STT model name for OpenAI API


@dataclass
class AppConfig:
    """Main application configuration."""
    log_level: str = "INFO"
    max_results: int = 5
    similarity_threshold: float = 0.3


class Settings:
    """Centralized settings management."""

    def __init__(self):
        self.database = DatabaseConfig(
            chroma_host=os.getenv("CHROMA_HOST", "localhost"),
            chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
            chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
        )

        self.embedding = EmbeddingConfig(
            model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"),
            dimension=int(os.getenv("EMBEDDING_DIMENSION", "384")),
            max_seq_length=int(os.getenv("MAX_SEQ_LENGTH", "512"))
        )

        self.data = DataConfig(
            faq_data_path=os.getenv("FAQ_DATA_PATH", "data/faq.csv")
        )

        # WebRTC configuration removed (unused)

        self.speech = SpeechConfig(
            provider=os.getenv("SPEECH_PROVIDER", "openai" if os.getenv("OPENAI_API_KEY") else "whisper"),
            model_name=os.getenv("WHISPER_MODEL", "turbo"),
            language=os.getenv("WHISPER_LANGUAGE", "en"),
            sample_rate=int(os.getenv("AUDIO_SAMPLE_RATE", "16000")),
            chunk_duration=float(os.getenv("AUDIO_CHUNK_DURATION", "0.1")),
            silence_threshold=float(os.getenv("SILENCE_THRESHOLD", "0.8")),
            max_recording_time=int(os.getenv("MAX_RECORDING_TIME", "30")),
            device_index=int(os.getenv("AUDIO_DEVICE_INDEX")) if os.getenv("AUDIO_DEVICE_INDEX") else None,
            energy_threshold=int(os.getenv("ENERGY_THRESHOLD", "300")),
            # Advanced VAD settings
            vad_min_recording_duration=float(os.getenv("VAD_MIN_RECORDING_DURATION", "1.0")),
            vad_pre_roll_duration=float(os.getenv("VAD_PRE_ROLL_DURATION", "0.2")),
            vad_noise_floor=float(os.getenv("VAD_NOISE_FLOOR", "0.001")),
            vad_min_speech_frames=int(os.getenv("VAD_MIN_SPEECH_FRAMES", "3")),
            vad_min_silence_frames=int(os.getenv("VAD_MIN_SILENCE_FRAMES", "8")),
            # OpenAI API
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_api_base=os.getenv("OPENAI_API_BASE"),
            openai_model=os.getenv("OPENAI_STT_MODEL", "gpt-4o-transcribe")
        )

        self.app = AppConfig(
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            max_results=int(os.getenv("MAX_RESULTS", "5")),
            similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
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
        """Deprecated: demo questions path removed."""
        return self.project_root


# Global settings instance
settings = Settings()
