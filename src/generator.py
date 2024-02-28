from vertexai.preview.generative_models import GenerativeModel, GenerationConfig
from vertexai.preview.language_models import TextGenerationModel, ChatModel
import google.generativeai as genai
import os


class Generator: 

    def response_synthesis(self, retrieved_text, query) -> str:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


        context_str = ""
        for i in len(retrieved_text):
            chunk = str(i) + " " + retrieved_text[i]
            context_str += chunk
        
        prompt = f"""Context information from multiple sources is below.
            ---------------------
            {context_str}
            ---------------------
            Given the information from multiple sources and not prior knowledge, answer the query.
            Query: {query}
            """

        return text_bison(prompt)

    def gemini_generation(self, prompt) -> str:
        model = GenerativeModel("gemini-pro", generation_config = GenerationConfig(temperature=0.2, max_output_tokens=2048))
        return model.generate_content(prompt, stream=True)
    
    
    def text_bison(prompt)-> str:
        textmodel = TextGenerationModel.from_pretrained("text-bison")
        parameters = {
            "temperature": 0.1,  # Temperature controls the degree of randomness in token selection.
            "max_output_tokens": 256,  # Token limit determines the maximum amount of text output.
            "top_p": 0.8,  # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
            "top_k": 40,  # A top_k of 1 means the selected token is the most probable among all tokens.
        }

        responses = textmodel.predict_streaming(prompt=prompt, **parameters)

#    def count_tokens(self, prompt: str) -> int:
#        