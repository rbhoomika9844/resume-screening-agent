# Resume Screening Agent

**Submitted by:** Raghunathachar Bhoomika

**For:** AI Agent Development Challenge — Deadline: 29 Nov 2025

---

# Resume Screening Agent — Demo

## Overview
This demo ranks candidate resumes against a provided Job Description using OpenAI embeddings and a small LLM for explanations.

## What's included
- `app.py` — Streamlit app that accepts a Job Description and multiple resumes (.pdf, .docx, .txt) and returns a ranked list.
- `requirements.txt` — Python dependencies.
- `run_local.sh` — helper script to run locally.
- `sample_data/jd.txt` and `sample_data/resumes/` — sample job description and sample resumes.

## How to run locally (Linux / macOS)
1. Install Python 3.10+.
2. Clone repo or unzip project.
3. Create virtualenv:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Add your OpenAI API key to the environment:
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```
   (Windows PowerShell: `$env:OPENAI_API_KEY = "sk-..."`)
6. Run the app:
   ```bash
   streamlit run app.py
   ```

## Notes & configuration
- The app uses `text-embedding-3-small` and `gpt-4o-mini` in examples. Replace with models you have access to.
- For many resumes / production use, integrate a vector DB (Pinecone, Chroma) to avoid re-embedding every run.
- Resume parsing is basic; consider using NER to extract structured fields.

## License
MIT
