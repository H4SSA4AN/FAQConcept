

## Prerequisites

- Tested using python 3.13.5
- FFMPEG installed and is on PATH
  ```bash
  ffmepg -version #In cmd to check
  

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/H4SSA4AN/FAQConcept.git
   cd FAQConcept
   pip install -r requirements.txt
   cd faq-video-poc
   python scripts/seed_chroma.py
   cd ..
   cd web_app
   python start.py
