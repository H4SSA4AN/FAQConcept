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
except ImportError as e:
    # Fallback: try different import paths
    try:
        import app.search as search_module
        import app.settings as settings_module
        FAQSearch = search_module.FAQSearch
        settings = settings_module.settings
    except ImportError:
        print(f"❌ Failed to import required modules: {e}")
        print("Make sure you're running this script from the scripts directory.")
        sys.exit(1)


def main():
    """Main interactive FAQ search loop."""
    print("🤖 Interactive FAQ Search")
    print("=" * 50)

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

    print("Type 'quit', 'exit', or 'q' to end the session.\n")

    while True:
        try:
            # Get user input
            query = input("❓ Ask a question: ").strip()

            # Check for exit commands
            if query.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break

            # Skip empty queries
            if not query:
                continue

            # Search for answers
            print(f"\n🔍 Searching for: '{query}'")
            print("-" * 40)

            results = search_engine.search(query, limit=5, threshold=0.0)

            if not results:
                print("❌ No relevant answers found. Try rephrasing your question.")
            else:
                print(f"📋 Found {len(results)} relevant answer(s):\n")

                for i, result in enumerate(results, 1):
                    print(f"{i}. 📖 Question: {result.question}")
                    print(f"   💡 Answer: {result.answer}")
                    print(f"   🏷️  Category: {result.category}")
                    print(".3f")
                    print()

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! (Interrupted)")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            continue


if __name__ == "__main__":
    main()
