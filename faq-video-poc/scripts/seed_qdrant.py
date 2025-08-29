#!/usr/bin/env python3
"""
Script to seed Qdrant database with FAQ data.
"""

import sys
import pandas as pd
from pathlib import Path

# Add app directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.settings import settings
from app.index_qdrant import QdrantIndexer
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

        # Initialize Qdrant indexer
        print("Initializing Qdrant indexer...")
        indexer = QdrantIndexer()

        # Clear existing data (optional)
        user_input = input("Clear existing Qdrant collection? (y/N): ")
        if user_input.lower() in ['y', 'yes']:
            try:
                indexer.delete_collection()
                print("✓ Cleared existing collection")
            except Exception as e:
                print(f"Warning: Failed to clear collection: {e}")

        # Add FAQ data
        print("Adding FAQs to Qdrant database...")
        indexer.add_faqs(faqs_df)

        # Get collection info
        info = indexer.get_collection_info()
        print("✓ Successfully seeded Qdrant database!"        print(f"  Collection: {info['name']}")
        print(f"  Points: {info.get('points_count', 'N/A')}")
        print(f"  Status: {info.get('status', 'N/A')}")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
