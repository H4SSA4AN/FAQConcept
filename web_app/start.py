#!/usr/bin/env python3
"""
Simple startup script for the Voice FAQ Web Application.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Start the Flask web application."""
    # Change to the web_app directory
    web_app_dir = Path(__file__).parent
    os.chdir(web_app_dir)

    print("Starting Voice FAQ Web Application...")
    print("=" * 50)

    # Check if we're in the right directory
    if not Path("app.py").exists():
        print("Error: app.py not found. Please run this script from the web_app directory.")
        sys.exit(1)

    # Check if the faq-video-poc directory exists
    faq_project_dir = web_app_dir.parent / "faq-video-poc"
    if not faq_project_dir.exists():
        print("Error: faq-video-poc directory not found. Please ensure the main project is in the parent directory.")
        sys.exit(1)

    print("Web app directory:", web_app_dir)
    print("FAQ project directory:", faq_project_dir)
    print()

    # Check if Chroma database exists
    chroma_db_path = faq_project_dir / "chroma_db"
    if not chroma_db_path.exists() or not any(chroma_db_path.iterdir()):
        print("Chroma database not found. Please run the Chroma seeder first:")
        print(f"   cd {faq_project_dir}")
        print("   python scripts/seed_chroma.py")
        print()
        sys.exit(1)

    print("\nStarting web server...")
    print("Open your browser and go to: http://localhost:5000")
    print("Click 'Start Recording' to ask questions")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    print()

    try:
        # Run the Flask app
        subprocess.run([sys.executable, "app.py"], check=True)
    except KeyboardInterrupt:
        print("\nApplication stopped by user.")
    except subprocess.CalledProcessError as e:
        print(f"Application failed to start: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check that all dependencies are installed: python check_deps.py")
        print("2. Install missing dependencies: python install_deps.py")
        print("3. Make sure ChromaDB is initialized in the main project")
        sys.exit(1)

if __name__ == "__main__":
    main()
