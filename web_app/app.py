"""
Flask web application for FAQ voice search with video playback.
"""

import os
import tempfile
import json
import csv
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from datetime import datetime
import sys
import time
import math

# Add the faq-video-poc directory to the Python path
project_root = Path(__file__).parent.parent / "faq-video-poc"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "app"))

try:
    from app.search import FAQSearch
    from app.settings import settings
    from app.speech import SpeechToText
    from app.retrieval import answer as retrieve_answer
    from app.utils import log_answered_question
except ImportError as e:
    print(f"❌ Failed to import required modules: {e}")
    print("Make sure you're running this from the web_app directory.")
    sys.exit(1)

app = Flask(__name__)

# Initialize components
faq_search = None
speech_engine_whisper = None
speech_engine_openai = None

def initialize_components():
    """Initialize the FAQ search and speech engines."""
    global faq_search, speech_engine_whisper, speech_engine_openai

    try:
        print("Initializing FAQ search engine...")
        faq_search = FAQSearch(use_chroma=True)
        print("✅ FAQ search engine initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize FAQ search engine: {e}")
        return False

    # Lazily initialize speech engines to reduce startup time; pre-init default if whisper
    try:
        default_provider = settings.speech.provider
        if default_provider == 'whisper':
            print("🎤 Preloading Whisper speech-to-text engine (default provider)...")
            speech_engine_whisper = SpeechToText(
                model_name=settings.speech.model_name,
                language=settings.speech.language,
                sample_rate=settings.speech.sample_rate,
                device_index=settings.speech.device_index,
                energy_threshold=settings.speech.energy_threshold,
                provider='whisper'
            )
            print("✅ Whisper engine loaded")
        else:
            print("🎤 Using OpenAI as default provider; engines will be loaded on demand")
    except Exception as e:
        print(f"❌ Failed to prepare speech engines: {e}")
        return False

    return True


def get_speech_engine(provider: str):
    """Return a cached speech engine for the given provider, creating it if necessary."""
    global speech_engine_whisper, speech_engine_openai

    provider = provider.lower()
    if provider == 'whisper':
        if speech_engine_whisper is None:
            print("🎤 Loading Whisper engine (on demand)...")
            speech_engine_whisper = SpeechToText(
                model_name=settings.speech.model_name,
                language=settings.speech.language,
                sample_rate=settings.speech.sample_rate,
                device_index=settings.speech.device_index,
                energy_threshold=settings.speech.energy_threshold,
                provider='whisper'
            )
            print("✅ Whisper engine loaded")
        return speech_engine_whisper

    if provider == 'openai':
        if not settings.speech.openai_api_key:
            raise RuntimeError('OPENAI_API_KEY not set on server')
        if speech_engine_openai is None:
            print("🎤 Initializing OpenAI STT client (on demand)...")
            speech_engine_openai = SpeechToText(
                model_name=settings.speech.model_name,
                language=settings.speech.language,
                sample_rate=settings.speech.sample_rate,
                device_index=settings.speech.device_index,
                energy_threshold=settings.speech.energy_threshold,
                provider='openai',
                openai_api_key=settings.speech.openai_api_key,
                openai_api_base=settings.speech.openai_api_base,
                openai_model=settings.speech.openai_model
            )
            print("✅ OpenAI STT ready")
        return speech_engine_openai

    raise ValueError(f"Unknown provider: {provider}")


def save_unanswered_question(question, source="voice"):
    """Save unanswered question to questions.csv file."""
    questions_file = project_root / "data" / "questions.csv"

    # Create the data directory if it doesn't exist
    questions_file.parent.mkdir(exist_ok=True)

    # Check if file exists to determine if we need headers
    file_exists = questions_file.exists()

    try:
        with open(questions_file, 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['timestamp', 'question', 'source']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header if file is new
            if not file_exists:
                writer.writeheader()

            # Write the unanswered question
            writer.writerow({
                'timestamp': datetime.now().isoformat(),
                'question': question,
                'source': source
            })

        print(f"📝 Saved unanswered question: '{question}'")
    except Exception as e:
        print(f"❌ Error saving unanswered question: {e}")

def find_video_url_for_question(question_text: str):
    """Find video URL for an exact question match from faq.csv."""
    try:
        faq_csv = project_root / "data" / "faq.csv"
        if not faq_csv.exists():
            return None
        with open(faq_csv, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (row.get('question') or '').strip() == (question_text or '').strip():
                    fname = (row.get('answer__url') or '').strip()
                    if fname:
                        return f"/videos/{fname}"
        return None
    except Exception:
        return None

@app.route('/')
def index():
    """Serve the main web page."""
    return render_template('index.html', default_provider=settings.speech.provider)


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

    # Get format and provider information from form data
    audio_format = request.form.get('format', 'webm')
    requested_provider = request.form.get('provider', settings.speech.provider)
    provider = requested_provider if requested_provider in ['openai', 'whisper'] else settings.speech.provider

    # Save uploaded audio to temporary file
    extension = '.webm' if audio_format == 'webm' else '.wav'
    with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as temp_file:
        audio_file.save(temp_file.name)
        temp_audio_path = temp_file.name

    converted_file_path = None

    try:
        # Start elapsed time measurement as soon as we begin processing
        start_time = time.perf_counter()
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

        # Transcribe audio to text using selected provider (cached engine)
        print(f"🎤 Transcribing audio using provider: {provider}...")
        try:
            engine = get_speech_engine(provider)
        except RuntimeError as key_err:
            return jsonify({'error': str(key_err), 'provider': provider}), 400
        except Exception as engine_err:
            print(f"❌ Failed to get speech engine: {engine_err}")
            return jsonify({'error': f'Failed to initialize STT engine: {engine_err}', 'provider': provider}), 500

        transcribed_text = engine.transcribe_audio(audio_data.astype(np.float32))

        if not transcribed_text:
            return jsonify({'error': 'Could not transcribe audio'}), 400

        print(f"📝 Transcribed: '{transcribed_text}'")

        # Retrieve with cross-encoder rerank
        print("🔍 Retrieving answers...")
        ret = retrieve_answer(transcribed_text, k=10)
        elapsed_ms = int((time.perf_counter() - start_time) * 1000)

        # Derive top candidate and confidence regardless of original mode
        top_q = None
        top_a = None
        top_score = None
        if ret.get('mode') == 'answer':
            top_q = ret.get('question')
            top_a = ret.get('answer')
            top_score = float(ret.get('score') or 0.0)
        else:
            cands0 = ret.get('candidates') or []
            if cands0:
                top_q = cands0[0].get('question')
                top_a = cands0[0].get('answer')
                top_score = float(cands0[0].get('score') or 0.0)

        top_conf = (1.0 / (1.0 + math.exp(-top_score))) * 100.0 if top_score is not None else 0.0

        top_conf = round(top_conf, 2)

        if top_conf > 60.0 and top_q and top_a:
            # Prefer direct mapping via returned id if present
            top_id = ret.get('id') if ret.get('mode') == 'answer' else (ret.get('candidates') or [{}])[0].get('id')
            video_url = None
            if top_id:
                try:
                    video_url = f"/videos/answer_{int(top_id)}.mp4"
                except Exception:
                    video_url = None
            if not video_url:
                video_url = find_video_url_for_question(top_q)
            log_answered_question(
                user_question=transcribed_text,
                matched_question=top_q,
                accuracy_score=(top_conf / 100.0),
                csv_path=str(project_root / "data" / "answered_questions.csv")
            )
            return jsonify({
                'transcription': transcribed_text,
                'question': top_q,
                'answer': top_a,
                'category': 'General',
                'confidence': top_conf,
                'video_url': video_url,
                'provider': provider,
                'elapsed_ms': elapsed_ms,
                'mode': 'answer'
            })

        # Prepare suggestions list with confidences
        cands = ret.get('candidates') or []
        for c in cands:
            c_score = float(c.get('score') or 0.0)
            c['confidence'] = round(((1.0 / (1.0 + math.exp(-c_score))) * 100.0), 2)

        if 30.0 <= top_conf <= 60.0:
            # Log unanswered question (low confidence suggest mode)
            try:
                save_unanswered_question(transcribed_text, source="voice")
            except Exception:
                pass
            return jsonify({
                'transcription': transcribed_text,
                'mode': 'suggest',
                'candidates': cands[:3],
                'provider': provider,
                'elapsed_ms': elapsed_ms,
                'video_url': '/videos/audio_noAns.mp4'
            })
        else:
            # Log unanswered question (very low confidence)
            try:
                save_unanswered_question(transcribed_text, source="voice")
            except Exception:
                pass
            return jsonify({
                'transcription': transcribed_text,
                'mode': 'suggest',
                'candidates': [],
                'message': "Sorry, I can't answer that",
                'provider': provider,
                'elapsed_ms': elapsed_ms,
                'video_url': '/videos/audio_noAns.mp4'
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
        # Retrieve with cross-encoder rerank
        print(f"🔍 Retrieving for: '{query}'")
        ret = retrieve_answer(query, k=10)

        # Derive top candidate and confidence regardless of original mode
        top_q = None
        top_a = None
        top_score = None
        if ret.get('mode') == 'answer':
            top_q = ret.get('question')
            top_a = ret.get('answer')
            top_score = float(ret.get('score') or 0.0)
        else:
            cands0 = ret.get('candidates') or []
            if cands0:
                top_q = cands0[0].get('question')
                top_a = cands0[0].get('answer')
                top_score = float(cands0[0].get('score') or 0.0)

        top_conf = (1.0 / (1.0 + math.exp(-top_score))) * 100.0 if top_score is not None else 0.0

        top_conf = round(top_conf, 2)

        if top_conf > 60.0 and top_q and top_a:
            top_id = ret.get('id') if ret.get('mode') == 'answer' else (ret.get('candidates') or [{}])[0].get('id')
            video_url = None
            if top_id:
                try:
                    video_url = f"/videos/answer_{int(top_id)}.mp4"
                except Exception:
                    video_url = None
            if not video_url:
                video_url = find_video_url_for_question(top_q)
            log_answered_question(
                user_question=query,
                matched_question=top_q,
                accuracy_score=(top_conf / 100.0),
                csv_path=str(project_root / "data" / "answered_questions.csv")
            )
            return jsonify({
                'query': query,
                'question': top_q,
                'answer': top_a,
                'category': 'General',
                'confidence': top_conf,
                'video_url': video_url,
                'mode': 'answer'
            })

        # Prepare suggestions list with confidences
        cands = ret.get('candidates') or []
        for c in cands:
            c_score = float(c.get('score') or 0.0)
            c['confidence'] = round(((1.0 / (1.0 + math.exp(-c_score))) * 100.0), 2)

        if 30.0 <= top_conf <= 60.0:
            # Log unanswered question (low confidence suggest mode)
            try:
                save_unanswered_question(query, source="text")
            except Exception:
                pass
            return jsonify({
                'query': query,
                'mode': 'suggest',
                'candidates': cands[:3],
                'video_url': '/videos/audio_noAns.mp4'
            })
        else:
            # Log unanswered question (very low confidence)
            try:
                save_unanswered_question(query, source="text")
            except Exception:
                pass
            return jsonify({
                'query': query,
                'mode': 'suggest',
                'candidates': [],
                'message': "Sorry, I can't answer that",
                'video_url': '/videos/audio_noAns.mp4'
            })

    except Exception as e:
        print(f"❌ Error processing query: {e}")
        return jsonify({'error': f'Search failed: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})




if __name__ == '__main__':
    # Avoid double initialization/boot logs when Flask reloader is active
    from os import environ
    is_reloader_child = environ.get('WERKZEUG_RUN_MAIN') == 'true'

    if not app.debug or is_reloader_child:
        if not initialize_components():
            print("❌ Failed to initialize components. Exiting.")
            sys.exit(1)

        print("🚀 Starting Flask web application...")
        print("🌐 Open your browser and go to: http://localhost:5000")

    app.run(debug=True, host='0.0.0.0', port=5000)
