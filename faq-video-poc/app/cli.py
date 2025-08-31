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
