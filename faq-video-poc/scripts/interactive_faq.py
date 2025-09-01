#!/usr/bin/env python3
"""
Interactive FAQ Search Script

A simple interactive script that allows users to ask questions
and get the top k most relevant FAQ answers with their similarity scores.
"""

import sys
import os
from pathlib import Path

# Add the app directory to the Python path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
app_dir = project_root / "app"

# Add both the app directory and the project root to sys.path for flexibility
sys.path.insert(0, str(app_dir))
sys.path.insert(0, str(project_root))

try:
    from app.search import FAQSearch
    from app.settings import settings
    from app.speech import SpeechToText
except ImportError as e:
    # Fallback: try different import paths
    try:
        import app.search as search_module
        import app.settings as settings_module
        import app.speech as speech_module
        FAQSearch = search_module.FAQSearch
        settings = settings_module.settings
        SpeechToText = speech_module.SpeechToText
    except ImportError:
        print(f"❌ Failed to import required modules: {e}")
        print("Make sure you're running this script from the scripts directory.")
        sys.exit(1)


def main(speech_mode=True):
    """Main interactive FAQ search loop.

    Args:
        speech_mode: Whether to use speech input instead of text (default: True)
    """
    if speech_mode:
        print("🎤 Interactive FAQ Search (Speech Mode - Default)")
        print("🤖 Advanced AI-powered voice activity detection activated!")
    else:
        print("⌨️ Interactive FAQ Search (Text Mode)")
    print("=" * 60)

    # Debug information
    print(f"Script location: {Path(__file__).resolve()}")
    print(f"Project root: {project_root}")
    print(f"App directory: {app_dir}")
    print()

    try:
        print(f"Using model: {settings.embedding.model_name}")
        print(f"Similarity threshold: {settings.app.similarity_threshold}")
        print(f"FAQ data path: {settings.faq_data_path}")
        print(f"Chroma persist dir: {settings.chroma_persist_dir}")
    except Exception as e:
        print(f"❌ Failed to load settings: {e}")
        print(f"Error details: {type(e).__name__}: {str(e)}")
        return

    # Check if required files exist
    if not settings.faq_data_path.exists():
        print(f"❌ FAQ data file not found at {settings.faq_data_path}")
        print("Please ensure the data/faq.csv file exists.")
        return

    # Initialize the search engine
    try:
        print("Initializing search engine...")
        search_engine = FAQSearch(use_chroma=True)
        print("✅ Search engine initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize search engine: {e}")
        print(f"Error details: {type(e).__name__}: {str(e)}")
        return

    # Initialize speech engine if speech mode is enabled
    speech_engine = None
    if speech_mode:
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
            print("Falling back to text mode...")
            speech_mode = False

    # Auto-update database if needed
    try:
        stats = search_engine.get_stats()
        db_count = stats.get('chroma', {}).get('count', 0)

        # Count CSV rows (minus header)
        import pandas as pd
        csv_count = len(pd.read_csv(settings.faq_data_path))

        if db_count == 0:
            print("🔄 First-time setup: Seeding database...")
            search_engine.add_faqs_from_csv()
            print("✅ Database seeded successfully!")
        elif db_count != csv_count:
            print(f"🔄 Updating database: {db_count} → {csv_count} entries")
            # Clear and re-seed for simplicity
            try:
                search_engine.chroma_indexer.delete_collection()
            except Exception:
                pass  # Collection might not exist, that's okay
            search_engine.add_faqs_from_csv()
            print("✅ Database updated successfully!")
        else:
            print(f"✅ Database is up to date with {db_count} FAQ entries.")
    except Exception as e:
        print(f"⚠️  Could not check/update database: {e}")
        print("This is likely the first run. Attempting to seed database...")

        # Try to seed the database if stats failed
        try:
            search_engine.add_faqs_from_csv()
            print("✅ Database seeded successfully!")
        except Exception as seed_error:
            print(f"❌ Failed to seed database: {seed_error}")
            print("Please check your ChromaDB setup and try again.")
            return

    if speech_mode:
        print("🎤 Advanced Speech Mode - Continuous Loop")
        print("🤖 AI-powered voice activity detection activated!")
        print("👂 Just speak your questions - I'll listen automatically")
        print("🔄 Continuous conversation - ask as many questions as you want")
        print("🛑 To exit: Press Ctrl+C or say 'stop', 'quit', or 'exit'")
        print("💡 Completely hands-free - no typing required!\n")
    else:
        print("⌨️  Text Mode Activated")
        print("Type your question and press Enter.")
        print("Type 'quit', 'exit', or 'q' to end the session.\n")

    while True:
        try:
            if speech_mode and speech_engine:
                # Speech input mode with advanced VAD
                print("👂 Ready! Speak your question naturally...")
                print("   (Recording will start automatically when speech is detected)")

                # Attempt to capture speech with advanced VAD
                transcribed_text = speech_engine.listen_and_transcribe(
                    max_duration=settings.speech.max_recording_time,
                    silence_threshold=settings.speech.silence_threshold,
                    min_recording_duration=settings.speech.vad_min_recording_duration,
                    pre_roll_duration=settings.speech.vad_pre_roll_duration
                )

                if not transcribed_text:
                    print("❌ No speech detected or transcription failed.")
                    print("💡 Try speaking louder, closer to the microphone, or in a quieter environment.")
                    print("🔄 Ready for your next question...\n")
                    continue

                query = transcribed_text.strip()
                print(f"🎯 Heard: '{query}'")

                # Check for speech-based exit commands
                if any(exit_word in query.lower() for exit_word in ['quit', 'exit', 'stop', 'bye', 'goodbye']):
                    print("👋 Got it! Ending our conversation...")
                    print("🎤 Thanks for using the voice-powered FAQ assistant!")
                    break

            else:
                # Text input mode
                query = input("❓ Ask a question: ").strip()

                # Check for exit commands
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye! Thanks for using the FAQ assistant!")
                    break

            # Skip empty queries
            if not query:
                continue

            # Search for answers (show search message if not already shown for speech mode)
            if not speech_mode:
                print(f"🔍 Searching for: '{query}'")
                print("-" * 40)

            results = search_engine.search(query, limit=5, threshold=0.0)

            if not results:
                if speech_mode:
                    print("🤔 I couldn't find a good answer for that question.")
                    print("💡 Try rephrasing it or ask about something else!")
                else:
                    print("❌ No relevant answers found. Try rephrasing your question.")
            else:
                if speech_mode:
                    print(f"✅ Found {len(results)} relevant answer(s) for you:\n")
                else:
                    print(f"📋 Found {len(results)} relevant answer(s):\n")

                for i, result in enumerate(results, 1):
                    if speech_mode:
                        print(f"{i}. 🎯 {result.question}")
                        print(f"   💡 {result.answer}")
                        print(f"   📊 (Confidence: {result.score:.1f}%, Category: {result.category})")
                    else:
                        print(f"{i}. 📖 Question: {result.question}")
                        print(f"   💡 Answer: {result.answer}")
                        print(f"   🏷️  Category: {result.category}")
                        print(".3f")
                    print()

            # Add a separator and ready message for speech mode
            if speech_mode and speech_engine:
                print("-" * 50)
                print("👂 Ready for your next question! Speak when you're ready...\n")

        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted! Thanks for using the FAQ assistant!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Interactive FAQ Search with Advanced Voice Activity Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Speech mode (default - completely hands-free!)
  %(prog)s --text             # Text mode for manual typing
  %(prog)s -t                 # Short form for text mode

Speech Mode Features:
• 🤖 Advanced voice activity detection - no button pressing needed
• 🎤 Automatically detects when you start/stop speaking
• 🔄 Continuous conversation loop - runs until Ctrl+C or 'stop'
• 🎙️ Includes pre-roll audio for complete capture
• 🔊 Adapts to background noise levels
• 💡 Completely hands-free operation

To exit: Press Ctrl+C or say 'stop', 'quit', or 'exit'
        """
    )
    parser.add_argument("--text", "-t", action="store_true",
                       help="Force text mode instead of default speech mode")
    parser.add_argument("--speech", "-s", action="store_true",
                       help="Explicitly enable speech mode (default behavior)")

    args = parser.parse_args()

    # Determine mode: speech is default, text only if --text is used
    speech_mode = not args.text  # Speech is default unless --text is specified

    main(speech_mode=speech_mode)
