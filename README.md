# DuRAG

1. Create venv: `python -m venv .venv`
2. Activate venv: `source .venv/bin/activate`
3. Update pip: `pip install --upgrade pip`
4. Install DuRAG: `pip install -e .`
5. Add .env file at project root

# Unit tests

`python -m unittest discover -s tests`

# Roadmap

- "grid search" for types of retrievers
  - Variables: query expansion; types of search models [ft, sm, hybrid]; types of postprocessing [sentence window, auto merging, hybrid];
- Filtering:
  - 2-step filtering:
    - full-text search against 128 chunks then aggregate pdf_document_name and use top 5-10 pdf_document_name as filters
    - full-text search against entire pdf content and take top 5-10 pdf_document_name and filter on those
  - 1-step filtering:
    - NER based filtering -> take the NERs from the query to filter from NER's in chunks
    - NER based filtering -> take the NERs from the query to filter from the chunk text
