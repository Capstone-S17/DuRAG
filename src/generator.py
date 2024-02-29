from vertexai.preview.generative_models import GenerativeModel, GenerationConfig
from vertexai.preview.language_models import TextGenerationModel, ChatModel
import google.generativeai as genai
import os
from typing import List, Generator


class Generator:
    def __init__(self, use_google_api=True):
        self.prompt = """Context information from multiple sources is below.
            ---------------------
            {}
            ---------------------
            Given the information from multiple sources and not prior knowledge, answer the query.
            Query: {}
            """
        """
        maybe should modify the prompt to say:
            - you are an expert in the field of finance..
            - if you are unsure, just say there is not enough information to answer the question

        """
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

        return out[0]

    def _gemini_generation(self, retrieved_text: list, query) -> list[str]:
        context_str = ""
        for i in range(len(retrieved_text)):
            chunk = str(i + 1) + " " + retrieved_text[i]
            context_str += chunk

        prompt = self.prompt.format(retrieved_text, query)

        response = self.gemini.generate_content(prompt, stream=True)
        out = []
        for r in response:
            out.append(r)
            print(r.text, end="")
        return out

    def _palm_generation(self, retrieved_text: List, query) -> List:
        context_str = ""
        for i in range(len(retrieved_text)):
            chunk = str(i + 1) + " " + retrieved_text[i]
            context_str += chunk

        prompt = self.prompt.format(retrieved_text, query)

        response = self.palm_bison.generate_content(prompt, stream=True)
        out = []

        for r in response:
            out.append(r)
            print(r.text, end="")
        return out
