# FAQ Video POC

A proof-of-concept application for FAQ search and retrieval using vector databases and embeddings.

## Features

- Vector-based FAQ search using Chroma
- Text embeddings for semantic search
- **🤖 Advanced Speech-to-text with Whisper v3 Turbo** (default mode!)
- **🎤 Voice Activity Detection** - completely hands-free operation
- CLI interface for easy interaction
- Real-time video streaming capabilities
- Continuous conversation loop with voice commands

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables in `.env` file
4. Seed the Chroma database:
   ```bash
   python -m app.cli seed
   ```

## Usage

### CLI Commands

- Search FAQs: `python -m app.cli search "your question"`
- **Speech search**: `python -m app.cli speech` (automatic voice detection)
- **Manual recording**: `python -m app.cli record` (press Enter to start/stop)
- List audio devices: `python -m app.cli devices`
- Seed Chroma database: `python -m app.cli seed`
- Clear Chroma database: `python -m app.cli clear`
- Show database statistics: `python -m app.cli stats`

### Interactive Script

Run the interactive FAQ search script from the scripts directory:

```bash
cd scripts
python interactive_faq.py
```

Or from the project root:
```bash
python scripts/interactive_faq.py
```

This will start an interactive session where you can:
- Ask questions and get the top 5 most relevant answers
- See similarity scores for each answer
- Type 'quit', 'exit', or 'q' to end the session

#### Speech Mode (Default)

Speech mode is now the **default** - just run the script for hands-free interaction!

```bash
# From scripts directory (speech mode by default)
python interactive_faq.py

# From project root (speech mode by default)
python scripts/interactive_faq.py

# Force text mode if needed
python interactive_faq.py --text
```

In speech mode:
- **🎤 Default Mode** - No flags needed, just run the script!
- **🤖 Advanced AI-powered Voice Activity Detection** - completely hands-free!
- **👂 Automatic speech detection** - starts recording when you speak (no button pressing!)
- **⏹️  Smart silence detection** - stops recording automatically when you finish
- **🎙️  Pre-roll audio capture** - includes the beginning of your speech
- **🔄 Continuous conversation loop** - runs until Ctrl+C or you say "stop"
- **🗣️  Voice commands** - say "quit", "exit", or "stop" to end the session
- **💡 Natural interaction** - speak as you normally would in conversation

#### Manual Recording Mode

For precise control over recording start/stop:

```bash
# CLI manual recording
python -m app.cli record

# Or with custom duration
python -m app.cli record --max-duration 60
```

Manual mode workflow:
1. **Press Enter** to START recording
2. Speak your message/question
3. **Press Enter again** to STOP recording
4. Audio is automatically saved and transcribed
5. Transcription is displayed

This mode gives you complete control over when recording begins and ends, without relying on automatic silence detection.

### Database Management

The interactive FAQ script automatically handles database seeding and updates:

- **First run:** Automatically seeds the database with FAQ data
- **Subsequent runs:** Automatically detects and applies updates when FAQ data changes
- **No manual seeding required** - just run the interactive script!

### Python API

```python
from app.search import FAQSearch

search = FAQSearch()
results = search.search("How do I reset my password?")
```

## Project Structure

```
faq-video-poc/
├── README.md
├── requirements.txt
├── .env
├── data/
│   └── faq.csv
├── app/
│   ├── __init__.py
│   ├── settings.py
│   ├── embed.py
│   ├── index_chroma.py
│   ├── search.py
│   ├── cli.py
│   └── utils.py
└── scripts/
    ├── seed_chroma.py
    ├── interactive_faq.py
    └── demo_questions.txt
```

## Configuration

Configure the application through the `.env` file:

- Database connection settings
- Embedding model parameters
- API keys and endpoints
- **Speech settings:**
  - `WHISPER_MODEL`: Model size ("tiny", "base", "small", "medium", "large", "turbo")
  - `WHISPER_LANGUAGE`: Language code (default: "en")
  - `AUDIO_SAMPLE_RATE`: Sample rate for audio (default: 16000)
  - `SILENCE_THRESHOLD`: Seconds of silence to stop recording (default: 0.8)
  - `MAX_RECORDING_TIME`: Maximum recording duration (default: 30)
  - `ENERGY_THRESHOLD`: Legacy energy threshold (for compatibility)
  - **Advanced VAD settings:**
    - `VAD_MIN_RECORDING_DURATION`: Minimum recording duration to keep (default: 1.0s)
    - `VAD_PRE_ROLL_DURATION`: Pre-roll audio to include (default: 0.2s)
    - `VAD_NOISE_FLOOR`: Minimum noise floor for adaptive threshold (default: 0.001)
    - `VAD_MIN_SPEECH_FRAMES`: Min consecutive speech frames to start (default: 3)
    - `VAD_MIN_SILENCE_FRAMES`: Min consecutive silence frames to stop (default: 8)

### Speech Configuration Examples

```bash
# Use smaller model for faster processing
WHISPER_MODEL=base

# Support multiple languages
WHISPER_LANGUAGE=en

# Adjust for noisy environments (automatic mode)
ENERGY_THRESHOLD=500

# Use specific audio device
AUDIO_DEVICE_INDEX=1

# Manual recording settings
MAX_RECORDING_TIME=60  # Maximum recording time in seconds

# Advanced VAD fine-tuning
VAD_MIN_RECORDING_DURATION=0.5  # Shorter minimum for quick questions
VAD_PRE_ROLL_DURATION=0.3       # More pre-roll for better context
VAD_NOISE_FLOOR=0.002          # Higher noise floor for noisy environments
VAD_MIN_SPEECH_FRAMES=5        # More frames needed to start (less sensitive)
VAD_MIN_SILENCE_FRAMES=10      # More frames needed to stop (more stable)
```

### Recording Modes

The system supports two recording modes:

1. **Automatic Mode** (`speech` command) - Advanced VAD:
   - **Smart voice activity detection** with adaptive thresholding
   - **Automatically detects speech start** - no button pressing needed
   - **Pre-roll audio capture** - includes audio before speech detection
   - **Adaptive to background noise** - adjusts sensitivity automatically
   - **Hysteresis protection** - prevents false starts/stops
   - **Minimum recording duration** - ensures meaningful recordings
   - Best for: Hands-free, conversational interactions

2. **Manual Mode** (`record` command) - User Controlled:
   - User-controlled start/stop via Enter key
   - Precise control over recording duration
   - No automatic detection - complete manual control
   - Best for: Structured recording, testing, or when you want exact control

### Advanced VAD Features

The automatic mode includes sophisticated voice activity detection:

- **Adaptive Thresholding**: Adjusts to background noise levels
- **RMS Energy Calculation**: More accurate than simple amplitude
- **Hysteresis**: Requires multiple consecutive frames for stable detection
- **Pre-roll Buffer**: Captures audio before speech detection triggers
- **Noise Floor Protection**: Prevents overly sensitive triggering
- **Minimum Duration Enforcement**: Filters out accidental short recordings
