# FAQ Text-to-Speech Generator

This script generates audio files from FAQ answers using OpenAI's Text-to-Speech API.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set your OpenAI API key as an environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

   Or create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

## Usage

Run the script from the project root:
```bash
python generate_tts.py
```

The script will:
- Read answers from `data/faq.csv`
- Generate audio files using OpenAI's TTS API
- Save audio files in the `audio/` folder
- Name files as `audio_{id}.mp3` where `{id}` is the FAQ ID

## Output

Audio files will be saved in the `audio/` folder with the following naming convention:
- `audio_1.mp3`
- `audio_2.mp3`
- etc.

## API Configuration

The script uses:
- Model: `tts-1` (fast and cost-effective)
- Voice: `alloy` (you can change this to: echo, fable, onyx, nova, shimmer)

## Cost Estimation

Each audio file costs approximately $0.015 per 1K characters. Monitor your OpenAI usage dashboard for actual costs.


