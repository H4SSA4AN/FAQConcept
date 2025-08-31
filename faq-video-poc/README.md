# FAQ Video POC

A proof-of-concept application for FAQ search and retrieval using vector databases and embeddings.

## Features

- Vector-based FAQ search using Chroma
- Text embeddings for semantic search
- CLI interface for easy interaction
- Real-time video streaming capabilities

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables in `.env` file
4. Seed the Chroma database:
   ```bash
   python -m app.cli seed
   ```

## Usage

### CLI Commands

- Search FAQs: `python -m app.cli search "your question"`
- Seed Chroma database: `python -m app.cli seed`
- Clear Chroma database: `python -m app.cli clear`
- Show database statistics: `python -m app.cli stats`

### Interactive Script

Run the interactive FAQ search script from the scripts directory:

```bash
cd scripts
python interactive_faq.py
```

Or from the project root:
```bash
python scripts/interactive_faq.py
```

This will start an interactive session where you can:
- Ask questions and get the top 5 most relevant answers
- See similarity scores for each answer
- Type 'quit', 'exit', or 'q' to end the session

### Database Management

The interactive FAQ script automatically handles database seeding and updates:

- **First run:** Automatically seeds the database with FAQ data
- **Subsequent runs:** Automatically detects and applies updates when FAQ data changes
- **No manual seeding required** - just run the interactive script!

### Python API

```python
from app.search import FAQSearch

search = FAQSearch()
results = search.search("How do I reset my password?")
```

## Project Structure

```
faq-video-poc/
├── README.md
├── requirements.txt
├── .env
├── data/
│   └── faq.csv
├── app/
│   ├── __init__.py
│   ├── settings.py
│   ├── embed.py
│   ├── index_chroma.py
│   ├── search.py
│   ├── cli.py
│   └── utils.py
└── scripts/
    ├── seed_chroma.py
    ├── interactive_faq.py
    └── demo_questions.txt
```

## Configuration

Configure the application through the `.env` file:

- Database connection settings
- Embedding model parameters
- API keys and endpoints
