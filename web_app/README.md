# Voice FAQ Web Application

A web-based voice FAQ assistant that transcribes speech, searches for relevant answers, and plays corresponding videos.

## 🐛 Recent Fix (Audio Format Issue)

**Fixed**: Audio recording from browser now works correctly. The issue was that browsers record audio in WebM format, but the backend expected WAV format. The fix includes:

- ✅ Automatic audio format detection and conversion
- ✅ Support for WebM, MP4, and WAV audio formats
- ✅ Proper MIME type handling in JavaScript
- ✅ Robust error handling for audio processing

## Features

- 🎬 **Dual Video System**: Two video layers prevent white flash during transitions
- 🎬 **Idle Video Loop**: IdleVideo.mp4 plays continuously in background (muted, looping)
- 🎬 **Answer Videos**: Full-screen answer videos with smooth fade transitions (unmuted)
- 🎤 **Voice Input**: Record audio directly from your browser's microphone
- 🔍 **Smart Search**: Uses AI-powered FAQ search with similarity matching
- 📱 **Responsive Design**: Optimized for desktop and mobile devices
- 💬 **Collapsible Chat**: Show/hide answer interface with "Show Chat" button
- 📝 **Text Mode**: Alternative text input for manual typing
- ⚡ **Instant Loading**: Videos load in background before switching
- 🌊 **Smooth Transitions**: 0.3s fade effects between videos
- 🔄 **Auto Transitions**: Seamlessly switches between idle and answer videos
- 💯 **Confidence Scores**: Shows how well the answer matches your question

## Prerequisites

Make sure you have the main FAQ system set up:

1. Install the main project dependencies:
   ```bash
   cd ../faq-video-poc
   pip install -r requirements.txt
   ```

2. Install web application dependencies:
   ```bash
   python install_deps.py
   ```

3. Make sure the ChromaDB database is initialized:
   ```bash
   python scripts/seed_chroma.py
   ```

## Usage

1. Start the web application:
   ```bash
   python start.py
   ```
   Or directly:
   ```bash
   python app.py
   ```

2. Open your browser and go to: http://localhost:5000

3. **Watch the Background Video**: The IdleVideo.mp4 will play on loop in full-screen

4. **Ask a Question**:
   - **Voice Mode** (default): Click "Start Recording" and speak your question
   - **Text Mode**: Click "Text Mode" button and type your question

5. **Experience Seamless Video Transitions**:
   - **Idle video** plays continuously in background (muted, looping)
   - **Answer video loads** in background before appearing (no flash)
   - **Smooth fade transition** (0.3s) when switching videos
   - **Answer video** appears full-screen when ready (unmuted)
   - When answer video ends, it **smoothly fades back** to idle video
   - "Show Chat" button appears to view text answer

6. **View Text Answer**:
   - Click "Show Chat" to see the answer details
   - Click "Hide Chat" to return to full-screen video experience

7. The application will:
   - Transcribe your speech (voice mode) or process your text
   - Search for the most relevant FAQ answer
   - **Instantly switch** between idle and answer videos (no loading delay)
   - Display text answer in collapsible chat panel

## API Endpoints

- `GET /` - Main web interface
- `POST /api/process_audio` - Process audio file and return FAQ answer
- `POST /api/search_text` - Process text query and return FAQ answer
- `GET /api/health` - Health check
- `GET /videos/<filename>` - Serve video files

## Troubleshooting

### Step 1: Check Browser Console
Open your browser's developer console (F12) to see detailed error messages and debug information.

### Audio Recording Issues
1. **Browser Permissions**: Make sure your browser allows microphone access
2. **HTTPS Required**: Some browsers require HTTPS for microphone access (try http://localhost)
3. **Browser Compatibility**: Try Chrome or Firefox first
4. **Microphone Check**: Verify your microphone works in other applications

### Audio Processing Errors
1. **Dependencies Missing**:
   ```bash
   python check_deps.py  # Check what dependencies are available
   python install_deps.py  # Install missing dependencies
   ```

2. **FFmpeg Not Found**: If pydub fails, install ffmpeg:
   - Windows: Download from https://ffmpeg.org/download.html and add to PATH
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt install ffmpeg`

3. **File Format Issues**: The app automatically converts WebM/MP4 to WAV
   - Check browser console for conversion errors
   - Verify that temporary files are being created/cleaned up

### Video Playback Issues
- Videos must be in MP4 format
- Check that video files exist in the `../faq-video-poc/videos/` directory
- Ensure video filenames match the `answer__url` column in `faq.csv`

### Search Issues
- Make sure ChromaDB is properly initialized
- Check that `../faq-video-poc/data/faq.csv` exists and is properly formatted
- Verify that the FAQ search engine can find relevant matches

### Debug Tools
- **Browser Console**: Check for JavaScript errors (F12)
- **Server Logs**: Look at Flask console output for backend errors
- **Network Tab**: Check if API requests are successful

## Browser Compatibility

- ✅ Chrome (recommended)
- ✅ Firefox
- ✅ Safari
- ✅ Edge

**Important**: Allow microphone access when prompted by your browser.
