"""
Command Line Interface for FAQ Video POC.
"""

import click
import pandas as pd
from pathlib import Path
from loguru import logger
from typing import Optional

from .settings import settings
from .search import FAQSearch
from .index_chroma import ChromaIndexer
from .speech import SpeechToText


@click.group()
@click.option('--log-level', default=settings.app.log_level,
              type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              help='Set logging level')
@click.pass_context
def cli(ctx, log_level):
    """FAQ Video POC - Command Line Interface"""
    # Configure logging
    logger.remove()
    logger.add(lambda msg: click.echo(msg, err=True),
               level=log_level, format="{time} {level} {message}")

    # Store settings in context
    ctx.ensure_object(dict)
    ctx.obj['settings'] = settings


@cli.command()
@click.argument('query')
@click.option('--limit', '-n', default=settings.app.max_results,
              help='Maximum number of results to return')
@click.option('--threshold', '-t', default=settings.app.similarity_threshold,
              help='Minimum similarity threshold')
@click.pass_context
def search(ctx, query, limit, threshold):
    """Search for FAQs matching the query."""
    try:
        # Initialize search engine
        search_engine = FAQSearch(use_chroma=True)

        # Perform search
        results = search_engine.search(query, limit=limit, threshold=threshold)

        if not results:
            click.echo(f"No results found for query: '{query}'")
            return

        # Display results
        click.echo(f"\nFound {len(results)} results for query: '{query}'\n")

        for i, result in enumerate(results, 1):
            click.echo(f"{i}. Question: {result.question}")
            click.echo(f"   Answer: {result.answer}")
            click.echo(f"   Category: {result.category}")
            click.echo(f"   Score: {result.score:.3f} (Source: {result.source})")
            click.echo()

    except Exception as e:
        logger.error(f"Search failed: {e}")
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--max-duration', '-d', default=settings.speech.max_recording_time,
              help='Maximum recording duration in seconds')
@click.option('--silence-threshold', '-s', default=settings.speech.silence_threshold,
              help='Silence threshold in seconds to stop recording')
@click.option('--limit', '-n', default=settings.app.max_results,
              help='Maximum number of results to return')
@click.option('--threshold', '-t', default=settings.app.similarity_threshold,
              help='Minimum similarity threshold')
@click.option('--save-audio', is_flag=True, help='Save recorded audio to file')
@click.option('--audio-file', default='recorded_audio.wav', help='Filename for saved audio')
@click.pass_context
def speech(ctx, max_duration, silence_threshold, limit, threshold, save_audio, audio_file):
    """Search FAQs using voice input with speech-to-text."""
    try:
        click.echo("🎤 Initializing speech-to-text engine...")
        click.echo(f"Using Whisper model: {settings.speech.model_name}")
        click.echo(f"Language: {settings.speech.language}")
        click.echo()

        # Initialize speech engine
        speech_engine = SpeechToText(
            model_name=settings.speech.model_name,
            language=settings.speech.language,
            sample_rate=settings.speech.sample_rate,
            device_index=settings.speech.device_index,
            energy_threshold=settings.speech.energy_threshold
        )

        # Initialize search engine
        search_engine = FAQSearch(use_chroma=True)

        # Record and transcribe with advanced VAD
        click.echo("🎙️ Advanced Voice Activity Detection enabled!")
        click.echo("Say your question clearly - recording starts automatically when speech is detected")
        click.echo("(Recording will stop automatically when you finish speaking)")
        click.echo()

        transcribed_text = speech_engine.listen_and_transcribe(
            max_duration=max_duration,
            silence_threshold=silence_threshold,
            save_audio=save_audio,
            audio_filename=audio_file,
            min_recording_duration=settings.speech.vad_min_recording_duration,
            pre_roll_duration=settings.speech.vad_pre_roll_duration
        )

        if not transcribed_text:
            click.echo("❌ Failed to transcribe speech. Please try again.", err=True)
            return

        click.echo(f"📝 You said: '{transcribed_text}'")
        click.echo()

        # Search for answers
        click.echo("🔍 Searching for relevant answers...")
        results = search_engine.search(transcribed_text, limit=limit, threshold=threshold)

        if not results:
            click.echo(f"No results found for: '{transcribed_text}'")
            return

        # Display results
        click.echo(f"\n📋 Found {len(results)} relevant answer(s):\n")

        for i, result in enumerate(results, 1):
            click.echo(f"{i}. 📖 Question: {result.question}")
            click.echo(f"   💡 Answer: {result.answer}")
            click.echo(f"   🏷️  Category: {result.category}")
            click.echo(f"   📊 Score: {result.score:.3f} (Source: {result.source})")
            click.echo()

        if save_audio:
            click.echo(f"💾 Audio saved as: {audio_file}")

    except Exception as e:
        logger.error(f"Speech search failed: {e}")
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.pass_context
def devices(ctx):
    """List available audio input devices."""
    try:
        speech_engine = SpeechToText()
        devices = speech_engine.list_audio_devices()

        if not devices:
            click.echo("❌ No audio input devices found")
            return

        click.echo("🎤 Available audio input devices:\n")

        for device in devices:
            click.echo(f"  {device}")

        click.echo("\nTo use a specific device, set AUDIO_DEVICE_INDEX environment variable")
        click.echo("Example: export AUDIO_DEVICE_INDEX=1")

    except Exception as e:
        logger.error(f"Failed to list devices: {e}")
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--max-duration', '-d', default=settings.speech.max_recording_time,
              help='Maximum recording duration in seconds')
@click.option('--save-audio', is_flag=True, default=True, help='Save recorded audio to file')
@click.option('--audio-file', default='recorded_audio.wav', help='Filename for saved audio')
@click.pass_context
def record(ctx, max_duration, save_audio, audio_file):
    """Record audio manually and transcribe it (press Enter to start/stop)."""
    try:
        click.echo("🎤 Initializing speech-to-text engine...")
        click.echo(f"Using Whisper model: {settings.speech.model_name}")
        click.echo(f"Language: {settings.speech.language}")
        click.echo()

        # Initialize speech engine
        speech_engine = SpeechToText(
            model_name=settings.speech.model_name,
            language=settings.speech.language,
            sample_rate=settings.speech.sample_rate,
            device_index=settings.speech.device_index,
            energy_threshold=settings.speech.energy_threshold
        )

        # Record and transcribe manually
        click.echo("🎙️ Manual recording mode:")
        click.echo("  1. Press Enter to START recording")
        click.echo("  2. Speak your message")
        click.echo("  3. Press Enter again to STOP recording")
        click.echo("  4. Audio will be saved and transcribed")
        click.echo()

        transcribed_text = speech_engine.record_and_transcribe_manual(
            max_duration=max_duration,
            save_audio=save_audio,
            audio_filename=audio_file
        )

        if not transcribed_text:
            click.echo("❌ Failed to transcribe audio.", err=True)
            return

        click.echo(f"\n📝 Transcription: '{transcribed_text}'")

        if save_audio:
            click.echo(f"💾 Audio saved as: {audio_file}")

    except Exception as e:
        logger.error(f"Recording failed: {e}")
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--csv-path', help='Path to FAQ CSV file')
@click.pass_context
def seed(ctx, csv_path):
    """Seed the Chroma database with FAQ data."""
    try:
        if csv_path is None:
            csv_path = str(settings.faq_data_path)

        # Validate CSV file exists
        if not Path(csv_path).exists():
            click.echo(f"Error: CSV file not found: {csv_path}", err=True)
            raise click.Abort()

        # Load CSV data
        click.echo(f"Loading FAQ data from: {csv_path}")
        faqs_df = pd.read_csv(csv_path)
        click.echo(f"Loaded {len(faqs_df)} FAQs")

        # Seed Chroma database
        click.echo("Seeding Chroma database...")
        chroma_indexer = ChromaIndexer()
        chroma_indexer.add_faqs(faqs_df)
        click.echo("✓ Chroma database seeded successfully")

    except Exception as e:
        logger.error(f"Seeding failed: {e}")
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.pass_context
def clear(ctx):
    """Clear the Chroma database."""
    try:
        click.echo("Clearing Chroma database...")
        chroma_indexer = ChromaIndexer()
        chroma_indexer.delete_collection()
        click.echo("✓ Chroma database cleared")

    except Exception as e:
        logger.error(f"Clearing failed: {e}")
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.pass_context
def stats(ctx):
    """Show Chroma database statistics."""
    try:
        search_engine = FAQSearch(use_chroma=True)
        stats = search_engine.get_stats()

        click.echo("\nDatabase Statistics:\n")

        if 'chroma' in stats:
            chroma_stats = stats['chroma']
            click.echo("Chroma Database:")
            click.echo(f"  Collection: {chroma_stats['name']}")
            click.echo(f"  Documents: {chroma_stats['count']}")
            click.echo(f"  Embedding Dimension: {chroma_stats['embedding_dimension']}")
            click.echo()

    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--questions-file', help='Path to questions file')
@click.pass_context
def demo(ctx, questions_file):
    """Run demo search with sample questions."""
    try:
        if questions_file is None:
            questions_file = str(settings.demo_questions_path)

        # Load demo questions
        if not Path(questions_file).exists():
            click.echo(f"Error: Questions file not found: {questions_file}", err=True)
            raise click.Abort()

        with open(questions_file, 'r') as f:
            questions = [line.strip() for line in f if line.strip()]

        if not questions:
            click.echo("No questions found in the demo file", err=True)
            raise click.Abort()

        # Initialize search engine
        search_engine = FAQSearch()

        click.echo(f"\nRunning demo with {len(questions)} questions...\n")

        for i, question in enumerate(questions, 1):
            click.echo(f"Demo Question {i}: {question}")
            click.echo("-" * 50)

            results = search_engine.search(question, limit=3)

            if results:
                for j, result in enumerate(results, 1):
                    click.echo(f"{j}. {result.question}")
                    click.echo(f"   → {result.answer}")
                    click.echo(f"   Score: {result.score:.3f}")
                    click.echo()
            else:
                click.echo("No results found.")
                click.echo()

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        click.echo(f"Error: {e}", err=True)
        raise click.Abort()


if __name__ == '__main__':
    cli()
