from vertexai.preview.generative_models import GenerativeModel, GenerationConfig
import google.generativeai as genai
import os

RAG_PROMPT = """
You are an expert in the field of finance and legal reasoning. 
Context information from multiple sources is below.
---------------------
{}
---------------------
Given the information from multiple sources and not prior knowledge, answer the query.\
If you are unsure, just say there is not enough information to answer the question.
Query: {}
"""

QUERY_EXPANSION_PROMPT = """
You are an expert in the field of finance and legal reasoning for listed companies in Singapore \
(which may include multinational companies). Based on the following query, please provide a \
a more comprehensive query that can be used for semantic search.
Original Query: {}
"""

QUERY_ENTITY_PROMPT = """
You are an expert in the field of finance and legal reasoning for listed companies in Singapore \
(which may include multinational companies). Based on the following query, please provide a \
list of entities that can be used for filtering documents. Look out for entities such as \
company names, people, locations, dates, events, and financial keywords.
Query: {}
"""

QUERY_DECOMPOSITION_PROMPT = """
You are an expert in the field of finance and legal reasoning for listed companies in Singapore \
(which may include multinational companies). Based on the following query, please break it down to 3 to 5 \
smaller and more specific queries.

---------------------
For example, "What was the financial performance of company X in 2020?" can be broken down into:
1. What was the revenue of Company X in 2020?
2. What was the net income of Company X in 2020?
3. What the debt of Company X in 2020?
---------------------

Query: {}

"""


class Generator:
    def __init__(self, use_google_api=True):
        if use_google_api:
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.generation_config = GenerationConfig(
            temperature=0.2, max_output_tokens=2048, top_p=0.8, top_k=40
        )

        self.gemini = GenerativeModel(
            "gemini-pro", generation_config=self.generation_config
        )
        self.palm_bison = GenerativeModel(
            "chat-bison", generation_config=self.generation_config
        )

    def response_synthesis(
        self, retrieved_text: list, query: str, use_gemini=True
    ) -> str:
        if use_gemini:
            out = self._gemini_generation(retrieved_text, query)
        else:
            out = self._palm_generation(retrieved_text, query)

        return out

    def _gemini_generation(self, retrieved_text: list, query) -> str:
        context_str = ""
        for i in range(len(retrieved_text)):
            chunk = str(i + 1) + " " + retrieved_text[i]
            context_str += chunk

        prompt = RAG_PROMPT.format(retrieved_text, query)

        response = self.gemini.generate_content(prompt, stream=True)
        out = []
        print("Generated Answer: ")
        for r in response:
            out.append(r.text)
            print(r.text, end="")
        print("\n")
        return " ".join(out)

    def _palm_generation(self, retrieved_text: list, query) -> str:
        context_str = ""
        for i in range(len(retrieved_text)):
            chunk = str(i + 1) + " " + retrieved_text[i]
            context_str += chunk

        prompt = RAG_PROMPT.format(retrieved_text, query)

        response = self.palm_bison.generate_content(prompt, stream=True)
        out = []

        for r in response:
            out.append(r)
            print(r.text, end="")
        return " ".join(out)

    # def query_expansion(self, query: str) -> str:
    #     prompt = QUERY_EXPANSION_PROMPT.format(query)
    #     response = self.llm.generate_content(prompt)
    #     return response[0].text
