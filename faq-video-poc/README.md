# FAQ Video POC

A proof-of-concept application for FAQ search and retrieval using vector databases and embeddings.

## Features

- Vector-based FAQ search using Chroma and Qdrant
- Text embeddings for semantic search
- CLI interface for easy interaction
- Support for multiple vector databases
- Real-time video streaming capabilities

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables in `.env` file
4. Seed the vector databases:
   ```bash
   python -m app.cli seed-chroma
   python -m app.cli seed-qdrant
   ```

## Usage

### CLI Commands

- Search FAQs: `python -m app.cli search "your question"`
- Seed Chroma database: `python -m app.cli seed-chroma`
- Seed Qdrant database: `python -m app.cli seed-qdrant`

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
│   ├── index_qdrant.py
│   ├── search.py
│   ├── cli.py
│   └── utils.py
└── scripts/
    ├── seed_chroma.py
    ├── seed_qdrant.py
    └── demo_questions.txt
```

## Configuration

Configure the application through the `.env` file:

- Database connection settings
- Embedding model parameters
- API keys and endpoints
