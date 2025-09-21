# AI Resume & Job Description Matcher

Simple Streamlit app that:
- Extracts keywords from a Job Description using an LLM
- Computes an ATS-like match score against a resume
- Generates an optimized Summary & Skills section
- Allows download as DOCX

## Setup
1. Create venv & install:
```bash
python -m venv venv
source venv/bin/activate   # mac/linux
venv\Scripts\activate      # windows
pip install -r requirements.txt
