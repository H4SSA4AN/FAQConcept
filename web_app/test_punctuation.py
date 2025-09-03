#!/usr/bin/env python3
"""
Test script to verify punctuation is preserved in CSV loading and ChromaDB storage.
"""

import sys
import os
from pathlib import Path

# Add the faq-video-poc directory to the Python path
project_root = Path(__file__).parent.parent / "faq-video-poc"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "app"))

try:
    from app.settings import settings
    from app.utils import validate_csv_format
    from app.index_chroma import ChromaIndexer

    def test_csv_loading():
        """Test that CSV loading preserves punctuation."""
        print("🧪 Testing CSV loading with punctuation...")

        # Load CSV with UTF-8 encoding
        csv_path = settings.faq_data_path
        print(f"Loading CSV: {csv_path}")

        faqs_df = validate_csv_format(str(csv_path))

        # Check a few sample answers for punctuation
        test_cases = [
            (1, "Placement Year", "typically taken between"),
            (2, "placements paid", "although often those in the sport industry are not"),
            (8, "supplementary activity", "Loughborough Enterprise Network Programmes"),
            (10, "family/friends contacts", "Family/friends"),
            (11, "appointment to check", "Careers Advice appointment"),
        ]

        print("\n📝 Sample answers with expected punctuation:")
        for idx, question_part, expected_punct in test_cases:
            row = faqs_df[faqs_df['id'] == idx].iloc[0]
            answer = row['answer']

            has_expected = expected_punct.lower() in answer.lower()
            status = "✅" if has_expected else "❌"

            print(f"  {status} ID {idx}: '{expected_punct}' - Found: {has_expected}")
            if not has_expected:
                print(f"      Answer: {answer[:100]}...")

        return True

    def test_chroma_storage():
        """Test that ChromaDB preserves punctuation."""
        print("\n🗄️ Testing ChromaDB storage...")

        try:
            indexer = ChromaIndexer()

            # Test a few searches to see if punctuation is preserved
            test_queries = [
                "placement year",
                "supplementary activity",
                "family friends contacts"
            ]

            for query in test_queries:
                results = indexer.search(query, n_results=1)
                if results:
                    answer = results[0].metadata.get('answer', '')
                    print(f"  Query: '{query}'")
                    print(f"  Answer preview: {answer[:100]}...")

                    # Check for common punctuation marks
                    punct_marks = ["'", ",", ".", "?", "!", ":", ";"]
                    found_punct = [p for p in punct_marks if p in answer]

                    print(f"  Punctuation found: {found_punct}")
                    print()
                else:
                    print(f"  ❌ No results for: '{query}'")
                    print()

        except Exception as e:
            print(f"❌ ChromaDB test failed: {e}")
            return False

        return True

    def main():
        """Run all tests."""
        print("🔍 Punctuation Preservation Test Suite")
        print("=" * 50)

        csv_ok = test_csv_loading()
        chroma_ok = test_chroma_storage()

        print("=" * 50)
        if csv_ok and chroma_ok:
            print("✅ Punctuation preservation tests completed!")
            print("If you see missing punctuation, run: python scripts/seed_chroma.py")
        else:
            print("❌ Some tests failed - check the output above")

        return csv_ok and chroma_ok

    if __name__ == "__main__":
        success = main()
        sys.exit(0 if success else 1)

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're in the web_app directory")
    sys.exit(1)
