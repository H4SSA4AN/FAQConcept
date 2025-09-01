"""
Speech-to-text functionality using OpenAI Whisper v3 Turbo.

This module provides real-time speech-to-text capabilities for the FAQ system,
allowing users to speak their questions instead of typing them.
"""

import io
import time
import numpy as np
import sounddevice as sd
from typing import Optional, Callable, List
from loguru import logger
import scipy.io.wavfile as wav
from pathlib import Path

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logger.warning("Whisper not available. Install with: pip install openai-whisper")


class SpeechToText:
    """Speech-to-text engine using Whisper."""

    def __init__(self, model_name: str = "turbo", language: str = "en",
                 sample_rate: int = 16000, device_index: Optional[int] = None,
                 energy_threshold: int = 300):
        """
        Initialize the speech-to-text engine.

        Args:
            model_name: Whisper model name ("tiny", "base", "small", "medium", "large", "turbo")
            language: Language code for transcription
            sample_rate: Audio sample rate
            device_index: Audio device index (None for default)
            energy_threshold: Energy threshold for voice activity detection
        """
        if not WHISPER_AVAILABLE:
            raise ImportError("Whisper is not installed. Run: pip install openai-whisper")

        self.model_name = model_name
        self.language = language
        self.sample_rate = sample_rate
        self.device_index = device_index
        self.energy_threshold = energy_threshold

        self.model = None
        self._is_recording = False

        # Load the model
        self._load_model()

    def _load_model(self):
        """Load the Whisper model."""
        try:
            logger.info(f"Loading Whisper model: {self.model_name}")
            self.model = whisper.load_model(self.model_name)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def list_audio_devices(self) -> List[str]:
        """List available audio input devices."""
        try:
            devices = sd.query_devices()
            input_devices = []

            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    input_devices.append(f"{i}: {device['name']}")

            return input_devices
        except Exception as e:
            logger.error(f"Failed to list audio devices: {e}")
            return []

    def _calculate_audio_energy(self, audio_data: np.ndarray) -> float:
        """Calculate RMS audio energy for voice activity detection."""
        if len(audio_data) == 0:
            return 0.0
        # RMS (Root Mean Square) energy - better than simple mean for audio
        return np.sqrt(np.mean(audio_data**2))

    def _calculate_adaptive_threshold(self, audio_history: List[float], noise_floor: float = 0.001) -> float:
        """Calculate adaptive threshold based on recent audio history."""
        if not audio_history:
            return noise_floor

        # Use the 10th percentile of recent energy levels as threshold
        # This adapts to background noise levels
        sorted_history = sorted(audio_history[-50:])  # Use last 50 samples
        if len(sorted_history) < 10:
            return noise_floor

        adaptive_threshold = sorted_history[len(sorted_history) // 10]  # 10th percentile
        # Ensure minimum threshold to prevent too much sensitivity
        return max(adaptive_threshold, noise_floor)

    def _is_speech_detected(self, energy: float, threshold: float,
                           speech_frames: int, min_speech_frames: int = 3) -> bool:
        """Determine if speech is detected with hysteresis."""
        # Require multiple consecutive speech frames to start recording
        # This prevents brief noise spikes from triggering recording
        return energy > threshold and speech_frames >= min_speech_frames

    def _is_silence_detected(self, energy: float, threshold: float,
                           silence_frames: int, min_silence_frames: int = 5) -> bool:
        """Determine if silence is detected with hysteresis."""
        # Require multiple consecutive silence frames to stop recording
        # This prevents brief pauses from stopping recording
        silence_threshold = threshold * 0.3  # Lower threshold for stopping (more sensitive)
        return energy < silence_threshold and silence_frames >= min_silence_frames

    def record_audio(self, max_duration: int = 30, silence_threshold: float = 0.8,
                    chunk_duration: float = 0.1, callback: Optional[Callable] = None,
                    min_recording_duration: float = 1.0, pre_roll_duration: float = 0.2) -> Optional[np.ndarray]:
        """
        Record audio from microphone with advanced voice activity detection.

        Args:
            max_duration: Maximum recording duration in seconds
            silence_threshold: Silence duration to stop recording (seconds)
            chunk_duration: Duration of audio chunks for processing (seconds)
            callback: Optional callback function for real-time updates
            min_recording_duration: Minimum recording duration to keep (seconds)
            pre_roll_duration: Amount of pre-roll audio to include (seconds)

        Returns:
            Recorded audio as numpy array, or None if recording failed
        """
        try:
            self._is_recording = True
            logger.info("🎤 Starting voice activity detection...")

            # Calculate buffer sizes
            chunk_samples = int(self.sample_rate * chunk_duration)
            silence_frames_needed = int(silence_threshold / chunk_duration)
            pre_roll_frames = int(pre_roll_duration / chunk_duration)
            min_recording_frames = int(min_recording_duration / chunk_duration)

            # State variables
            audio_buffer = []  # Rolling buffer for pre-roll
            recorded_audio = []  # Final recorded audio
            energy_history = []  # For adaptive thresholding
            consecutive_speech_frames = 0
            consecutive_silence_frames = 0
            recording_started = False
            start_time = time.time()

            def audio_callback(indata, frames, time_info, status):
                """Audio callback for real-time processing."""
                if status:
                    logger.warning(f"Audio callback status: {status}")

                nonlocal consecutive_speech_frames, consecutive_silence_frames
                nonlocal recording_started, audio_buffer, recorded_audio, energy_history

                audio_chunk = indata.flatten()

                # Calculate energy and update history
                energy = self._calculate_audio_energy(audio_chunk)
                energy_history.append(energy)

                # Calculate adaptive threshold
                adaptive_threshold = self._calculate_adaptive_threshold(energy_history)

                # Determine if this chunk contains speech
                is_speech = energy > adaptive_threshold

                # Update consecutive frame counters
                if is_speech:
                    consecutive_speech_frames += 1
                    consecutive_silence_frames = 0
                else:
                    consecutive_speech_frames = 0
                    consecutive_silence_frames += 1

                # Maintain pre-roll buffer
                audio_buffer.append(audio_chunk)
                if len(audio_buffer) > pre_roll_frames:
                    audio_buffer.pop(0)

                # Check if we should start recording
                if not recording_started and self._is_speech_detected(
                    energy, adaptive_threshold, consecutive_speech_frames):
                    logger.info("🎙️ Speech detected! Starting recording...")
                    recording_started = True
                    # Include pre-roll audio
                    recorded_audio.extend(audio_buffer[:-1])  # Exclude current chunk (already added below)

                # Record if we're actively recording
                if recording_started:
                    recorded_audio.append(audio_chunk)

                # Call user callback if provided
                if callback:
                    callback(indata, frames, {
                        'energy': energy,
                        'threshold': adaptive_threshold,
                        'is_speech': is_speech,
                        'recording_started': recording_started,
                        'speech_frames': consecutive_speech_frames,
                        'silence_frames': consecutive_silence_frames
                    })

            # Start listening
            with sd.InputStream(samplerate=self.sample_rate,
                              channels=1,
                              dtype=np.float32,
                              device=self.device_index,
                              callback=audio_callback,
                              blocksize=chunk_samples):

                logger.info("👂 Listening for speech... (say something)")

                while self._is_recording:
                    time.sleep(0.05)  # Small delay to prevent busy waiting
                    elapsed_time = time.time() - start_time

                    # Check timeout
                    if elapsed_time >= max_duration:
                        logger.info(f"⏰ Maximum recording time reached ({max_duration}s)")
                        break

                    # Check if we should stop recording due to silence
                    if (recording_started and
                        self._is_silence_detected(energy_history[-1] if energy_history else 0,
                                                self._calculate_adaptive_threshold(energy_history),
                                                consecutive_silence_frames,
                                                silence_frames_needed)):
                        logger.info("🤫 Silence detected, stopping recording")
                        break

                self._is_recording = False

            # Process recorded audio
            if recorded_audio:
                audio_data = np.concatenate(recorded_audio, axis=0).flatten()
                recording_duration = len(audio_data) / self.sample_rate

                # Check minimum recording duration
                if recording_duration < min_recording_duration:
                    logger.warning(f"⚠️ Recording too short ({recording_duration:.1f}s < {min_recording_duration}s)")
                    return None

                logger.info(f"✅ Recording complete. Duration: {recording_duration:.1f}s")
                return audio_data
            else:
                logger.warning("⚠️ No speech detected")
                return None

        except Exception as e:
            logger.error(f"❌ Recording failed: {e}")
            return None
        finally:
            self._is_recording = False

    def transcribe_audio(self, audio_data: np.ndarray) -> Optional[str]:
        """
        Transcribe audio data to text using Whisper.

        Args:
            audio_data: Audio data as numpy array

        Returns:
            Transcribed text, or None if transcription failed
        """
        try:
            if audio_data is None or len(audio_data) == 0:
                logger.warning("No audio data to transcribe")
                return None

            # Convert to the format expected by Whisper
            audio_float32 = audio_data.astype(np.float32)

            # Ensure proper range for Whisper
            if audio_float32.max() > 1.0 or audio_float32.min() < -1.0:
                audio_float32 = audio_float32 / np.max(np.abs(audio_float32))

            logger.info("🎯 Transcribing audio...")

            # Transcribe using Whisper
            result = self.model.transcribe(
                audio_float32,
                language=self.language,
                fp16=False,  # Use FP32 for better compatibility
                verbose=False
            )

            transcribed_text = result["text"].strip()

            if transcribed_text:
                logger.info(f"📝 Transcribed: '{transcribed_text}'")
                return transcribed_text
            else:
                logger.warning("⚠️ No speech detected in audio")
                return None

        except Exception as e:
            logger.error(f"❌ Transcription failed: {e}")
            return None

    def save_audio_to_file(self, audio_data: np.ndarray, filename: str) -> bool:
        """
        Save audio data to WAV file.

        Args:
            audio_data: Audio data as numpy array
            filename: Output filename

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Convert to proper format for WAV
            audio_int16 = (audio_data * 32767).astype(np.int16)

            wav.write(filename, self.sample_rate, audio_int16)
            logger.info(f"💾 Audio saved to: {filename}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to save audio: {e}")
            return False

    def listen_and_transcribe(self, max_duration: int = 30, silence_threshold: float = 0.8,
                             save_audio: bool = False, audio_filename: str = "recorded_audio.wav",
                             min_recording_duration: float = 1.0, pre_roll_duration: float = 0.2) -> Optional[str]:
        """
        Complete speech-to-text pipeline with advanced VAD: record and transcribe.

        Args:
            max_duration: Maximum recording duration in seconds
            silence_threshold: Silence duration to stop recording (seconds)
            save_audio: Whether to save recorded audio to file
            audio_filename: Filename for saved audio
            min_recording_duration: Minimum recording duration to keep (seconds)
            pre_roll_duration: Amount of pre-roll audio to include (seconds)

        Returns:
            Transcribed text, or None if failed
        """
        # Record audio with advanced VAD
        audio_data = self.record_audio(
            max_duration=max_duration,
            silence_threshold=silence_threshold,
            min_recording_duration=min_recording_duration,
            pre_roll_duration=pre_roll_duration
        )

        if audio_data is None:
            return None

        # Save audio if requested
        if save_audio:
            self.save_audio_to_file(audio_data, audio_filename)

        # Transcribe
        return self.transcribe_audio(audio_data)

    def stop_recording(self):
        """Stop the current recording."""
        self._is_recording = False
        logger.info("🛑 Recording stopped")

    def record_audio_manual(self, max_duration: int = 30, chunk_duration: float = 0.1,
                           callback: Optional[Callable] = None) -> Optional[np.ndarray]:
        """
        Record audio with manual start/stop control (press Enter to start/stop).

        Args:
            max_duration: Maximum recording duration in seconds
            chunk_duration: Duration of audio chunks for processing
            callback: Optional callback function for real-time updates

        Returns:
            Recorded audio as numpy array, or None if recording failed
        """
        try:
            # Calculate buffer size
            chunk_samples = int(self.sample_rate * chunk_duration)

            logger.info("🔴 Press Enter to START recording...")
            input()  # Wait for Enter to start

            self._is_recording = True
            logger.info("🎤 Recording started! Press Enter to STOP...")

            recorded_audio = []
            start_time = time.time()

            def audio_callback(indata, frames, time_info, status):
                """Audio callback for real-time processing."""
                if status:
                    logger.warning(f"Audio callback status: {status}")

                recorded_audio.append(indata.copy())

                # Call user callback if provided
                if callback:
                    callback(indata, frames)

            # Start recording
            with sd.InputStream(samplerate=self.sample_rate,
                              channels=1,
                              dtype=np.float32,
                              device=self.device_index,
                              callback=audio_callback,
                              blocksize=chunk_samples):

                # Use a separate thread to wait for Enter key press
                import threading
                stop_event = threading.Event()

                def wait_for_enter():
                    input()  # Wait for Enter to stop
                    stop_event.set()

                # Start the input thread
                input_thread = threading.Thread(target=wait_for_enter, daemon=True)
                input_thread.start()

                while self._is_recording and not stop_event.is_set():
                    time.sleep(0.05)  # Small delay to prevent busy waiting

                    elapsed_time = time.time() - start_time

                    # Check timeout
                    if elapsed_time >= max_duration:
                        logger.info(f"⏰ Maximum recording time reached ({max_duration}s)")
                        break

                self._is_recording = False
                logger.info("🛑 Recording stopped by user")

            # Combine recorded chunks
            if recorded_audio:
                audio_data = np.concatenate(recorded_audio, axis=0).flatten()
                logger.info(f"✅ Recording complete. Duration: {len(audio_data)/self.sample_rate:.1f}s")
                return audio_data
            else:
                logger.warning("⚠️ No audio recorded")
                return None

        except KeyboardInterrupt:
            logger.info("🛑 Recording interrupted by user")
            return None
        except Exception as e:
            logger.error(f"❌ Recording failed: {e}")
            return None
        finally:
            self._is_recording = False

    def record_and_transcribe_manual(self, max_duration: int = 30, save_audio: bool = True,
                                    audio_filename: str = "recorded_audio.wav") -> Optional[str]:
        """
        Complete manual recording and transcription pipeline.
        Press Enter to start, Enter again to stop, save, and transcribe.

        Args:
            max_duration: Maximum recording duration in seconds
            save_audio: Whether to save recorded audio to file
            audio_filename: Filename for saved audio

        Returns:
            Transcribed text, or None if failed
        """
        # Record audio manually
        audio_data = self.record_audio_manual(max_duration=max_duration)

        if audio_data is None:
            return None

        # Save audio if requested
        if save_audio:
            self.save_audio_to_file(audio_data, audio_filename)

        # Transcribe
        return self.transcribe_audio(audio_data)


def create_speech_engine(model_name: str = "turbo", language: str = "en",
                        sample_rate: int = 16000, device_index: Optional[int] = None,
                        energy_threshold: int = 300) -> SpeechToText:
    """
    Factory function to create a SpeechToText instance.

    Args:
        model_name: Whisper model name
        language: Language code
        sample_rate: Audio sample rate
        device_index: Audio device index
        energy_threshold: Energy threshold for VAD

    Returns:
        Configured SpeechToText instance
    """
    return SpeechToText(
        model_name=model_name,
        language=language,
        sample_rate=sample_rate,
        device_index=device_index,
        energy_threshold=energy_threshold
    )
