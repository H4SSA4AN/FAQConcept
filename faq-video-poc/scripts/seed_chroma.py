#!/usr/bin/env python3
"""
Script to seed Chroma database with FAQ data.
"""

import sys
import shutil
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
        # 0) Clean up old Chroma data so only the newest collection remains
        persist_path = settings.chroma_persist_dir
        try:
            if persist_path.exists():
                print(f"Cleaning Chroma persist dir: {persist_path}")
                shutil.rmtree(persist_path, ignore_errors=True)
            persist_path.mkdir(parents=True, exist_ok=True)
            print("✓ Reset Chroma persist directory")
        except Exception as e:
            print(f"⚠️  Warning: Failed to fully reset persist dir: {e}")

        # Load and validate FAQ data
        csv_path = settings.faq_data_path
        print(f"Loading FAQ data from: {csv_path}")

        if not csv_path.exists():
            print(f"Error: FAQ CSV file not found at {csv_path}")
            sys.exit(1)

        faqs_df = validate_csv_format(str(csv_path))
        print(f"Loaded {len(faqs_df)} FAQs from CSV")

        # Initialize Chroma indexer (will create a fresh collection in the clean dir)
        print("Initializing Chroma indexer...")
        indexer = ChromaIndexer()

        # No need to delete collection here since the persist dir was reset

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
