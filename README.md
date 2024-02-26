# DuRAG

1. Create venv: `python -m venv .venv`
2. Activate venv: `source .venv/bin/activate`
3. Update pip: `pip install --upgrade pip`
4. Install DuRAG: `pip install -e .`
5. Add .env file at project root

# Issues

- SWR uses pages instead of the whole document. is this okay?
- AMR chunks in rds has uuid for the 2048 chunks but i did not intend this. it just points to the uuid for the pdf itself. not an issue but can be cleaned up...
- need help to write the amr retriever logic if using SQL
- need to write generator.py (chiayu)
- need to write query_expander.py (chiayu - try something super simple first like just getting some ners from the qn to filter down)
- need to write eval scripts in eval/ (vic)
- need to include all useful experiments in experiments/ (dylan + everyone) include the code to run UNI-NER also
- need to run UNI-NER again on 512_recursive_dhanush im so sorry (dylan)
- rename 512_recursive_dhanush to something more apt
- thoughts on how to move all of this to cloud? (dylan)
