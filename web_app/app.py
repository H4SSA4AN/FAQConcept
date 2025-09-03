"""
Flask web application for FAQ voice search with video playback.
"""

import os
import tempfile
import json
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
import sys

# Add the faq-video-poc directory to the Python path
project_root = Path(__file__).parent.parent / "faq-video-poc"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "app"))

try:
    from app.search import FAQSearch
    from app.settings import settings
    from app.speech import SpeechToText
except ImportError as e:
    print(f"❌ Failed to import required modules: {e}")
    print("Make sure you're running this from the web_app directory.")
    sys.exit(1)

app = Flask(__name__)

# Initialize components
faq_search = None
speech_engine = None

def initialize_components():
    """Initialize the FAQ search and speech engines."""
    global faq_search, speech_engine

    try:
        print("Initializing FAQ search engine...")
        faq_search = FAQSearch(use_chroma=True)
        print("✅ FAQ search engine initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize FAQ search engine: {e}")
        return False

    try:
        print("🎤 Initializing speech-to-text engine...")
        speech_engine = SpeechToText(
            model_name=settings.speech.model_name,
            language=settings.speech.language,
            sample_rate=settings.speech.sample_rate,
            device_index=settings.speech.device_index,
            energy_threshold=settings.speech.energy_threshold
        )
        print("✅ Speech-to-text engine initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize speech engine: {e}")
        return False

    return True

@app.route('/')
def index():
    """Serve the main web page."""
    return render_template('index.html')

@app.route('/test')
def test_page():
    """Serve the audio test page."""
    return render_template('test.html')

@app.route('/videos/<path:filename>')
def serve_video(filename):
    """Serve video files."""
    video_dir = project_root / "videos"
    return send_from_directory(video_dir, filename)

@app.route('/api/process_audio', methods=['POST'])
def process_audio():
    """Process uploaded audio file and return FAQ answer."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'error': 'No audio file selected'}), 400

    # Get format information from form data
    audio_format = request.form.get('format', 'webm')

    # Save uploaded audio to temporary file
    extension = '.webm' if audio_format == 'webm' else '.wav'
    with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as temp_file:
        audio_file.save(temp_file.name)
        temp_audio_path = temp_file.name

    converted_file_path = None

    try:
        # Convert audio to WAV format if needed
        import scipy.io.wavfile as wav
        import numpy as np

        print(f"🎤 Processing audio file (format: {audio_format})...")

        # Convert to WAV if not already in WAV format
        if audio_format != 'wav':
            try:
                from pydub import AudioSegment
                print(f"📁 Converting {audio_format} to WAV...")

                # Load audio file with pydub
                audio = AudioSegment.from_file(temp_audio_path, format=audio_format)

                # Export as WAV
                converted_file_path = temp_audio_path + '_converted.wav'
                audio.export(converted_file_path, format="wav")
                temp_audio_path = converted_file_path
                print("✅ Audio conversion completed")
            except ImportError as e:
                print(f"⚠️  Pydub not available: {e}")
                # Try alternative approach without pydub
                try:
                    import subprocess
                    converted_file_path = temp_audio_path + '_converted.wav'
                    # Try using ffmpeg directly if available
                    result = subprocess.run([
                        'ffmpeg', '-i', temp_audio_path,
                        '-acodec', 'pcm_s16le', '-ar', '16000',
                        converted_file_path
                    ], capture_output=True, text=True, timeout=30)

                    if result.returncode == 0:
                        temp_audio_path = converted_file_path
                        print("✅ Audio conversion completed (using ffmpeg)")
                    else:
                        return jsonify({'error': 'Audio conversion failed. Please install pydub or ffmpeg.'}), 500
                except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError) as e:
                    print(f"⚠️  FFmpeg conversion failed: {e}")
                    return jsonify({'error': 'Audio processing requires pydub or ffmpeg. Please install dependencies.'}), 500
            except Exception as e:
                print(f"⚠️  Audio conversion failed: {e}")
                return jsonify({'error': f'Failed to convert audio: {str(e)}'}), 400

        print("🎤 Loading audio file...")
        sample_rate, audio_data = wav.read(temp_audio_path)

        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)

        # Transcribe audio to text
        print("🎤 Transcribing audio...")
        transcribed_text = speech_engine.transcribe_audio(audio_data.astype(np.float32))

        if not transcribed_text:
            return jsonify({'error': 'Could not transcribe audio'}), 400

        print(f"📝 Transcribed: '{transcribed_text}'")

        # Search for FAQ answers
        print("🔍 Searching for answers...")
        results = faq_search.search(transcribed_text, limit=1, threshold=0.0)

        if not results:
            return jsonify({
                'transcription': transcribed_text,
                'error': 'No relevant answers found'
            }), 404

        # Get the best result
        best_result = results[0]

        # Check if there's a video URL in metadata
        video_url = None
        if best_result.metadata and 'answer__url' in best_result.metadata:
            video_filename = best_result.metadata['answer__url']
            if video_filename and video_filename.strip():
                video_url = f'/videos/{video_filename}'

        return jsonify({
            'transcription': transcribed_text,
            'question': best_result.question,
            'answer': best_result.answer,
            'category': best_result.category,
            'confidence': round(best_result.score * 100, 1),
            'video_url': video_url
        })

    except Exception as e:
        print(f"❌ Error processing audio: {e}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

    finally:
        # Clean up temporary files
        try:
            os.unlink(temp_audio_path)
            # Also clean up converted file if it exists
            if converted_file_path and os.path.exists(converted_file_path):
                try:
                    os.unlink(converted_file_path)
                except:
                    pass
        except:
            pass

@app.route('/api/search_text', methods=['POST'])
def search_text():
    """Process text query and return FAQ answer."""
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({'error': 'No query provided'}), 400

    query = data['query'].strip()
    if not query:
        return jsonify({'error': 'Empty query'}), 400

    try:
        # Search for FAQ answers
        print(f"🔍 Searching for: '{query}'")
        results = faq_search.search(query, limit=1, threshold=0.0)

        if not results:
            return jsonify({
                'query': query,
                'error': 'No relevant answers found'
            }), 404

        # Get the best result
        best_result = results[0]

        # Check if there's a video URL in metadata
        video_url = None
        if best_result.metadata and 'answer__url' in best_result.metadata:
            video_filename = best_result.metadata['answer__url']
            if video_filename and video_filename.strip():
                video_url = f'/videos/{video_filename}'

        return jsonify({
            'query': query,
            'question': best_result.question,
            'answer': best_result.answer,
            'category': best_result.category,
            'confidence': round(best_result.score * 100, 1),
            'video_url': video_url
        })

    except Exception as e:
        print(f"❌ Error processing query: {e}")
        return jsonify({'error': f'Search failed: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})



if __name__ == '__main__':
    if not initialize_components():
        print("❌ Failed to initialize components. Exiting.")
        sys.exit(1)

    print("🚀 Starting Flask web application...")
    print("🌐 Open your browser and go to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
