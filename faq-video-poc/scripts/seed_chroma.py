#!/usr/bin/env python3
"""
Script to seed Chroma database with FAQ data.
"""

import sys
import pandas as pd
from pathlib import Path

# Add app directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.settings import settings
from app.index_chroma import ChromaIndexer
from app.utils import validate_csv_format, setup_logging


def main():
    """Main seeding function."""
    setup_logging(settings.app.log_level)

    try:
        # Load and validate FAQ data
        csv_path = settings.faq_data_path
        print(f"Loading FAQ data from: {csv_path}")

        if not csv_path.exists():
            print(f"Error: FAQ CSV file not found at {csv_path}")
            sys.exit(1)

        faqs_df = validate_csv_format(str(csv_path))
        print(f"Loaded {len(faqs_df)} FAQs from CSV")

        # Initialize Chroma indexer
        print("Initializing Chroma indexer...")
        indexer = ChromaIndexer()

        # Clear existing data and recreate collection
        print("Clearing existing Chroma collection to update video URLs...")
        try:
            indexer.delete_collection()
            print("✓ Cleared existing collection")
        except Exception as e:
            print(f"⚠️  Warning: Failed to clear collection: {e}")

        # Force reinitialization of the indexer to create a fresh collection
        print("Creating fresh Chroma indexer...")
        try:
            # Create a new indexer instance to ensure clean state
            indexer = ChromaIndexer()
            print("✓ Created fresh indexer")
        except Exception as e:
            print(f"❌ Failed to create fresh indexer: {e}")
            return

        # Add FAQ data
        print("Adding FAQs to Chroma database...")
        indexer.add_faqs(faqs_df)

        # Get collection info
        info = indexer.get_collection_info()
        print("✓ Successfully seeded Chroma database!")
        print(f"  Collection: {info['name']}")
        print(f"  Documents: {info['count']}")
        print(f"  Embedding Dimension: {info['embedding_dimension']}")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
