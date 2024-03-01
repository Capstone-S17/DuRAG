import numpy as np
from trulens_eval.feedback.provider.litellm import LiteLLM
from trulens_eval.feedback.provider.litellm import LiteLLM as LiteLLMProvider
from typing import Optional,List
from trulens_eval import TruLlama
from trulens_eval import FeedbackMode
from trulens_eval.feedback import Groundedness
import numpy as np
from trulens_eval import Feedback
from trulens_eval import Tru
import logging


class RAGeval:

    def __init__(self):

        ## Just default to gemini-pro, can change later
        litellm_provider = LiteLLM(model_engine='gemini-pro')
        grounded = Groundedness(groundedness_provider=litellm_provider)
        self.f_groundedness = (
                    Feedback(grounded.groundedness_measure_with_cot_reasons,
                             name="Groundedness"
                            )
                    .on_input_output()
                    .aggregate(grounded.grounded_statements_aggregator)
                )
        self.f_qs_relevance = (
                    Feedback(litellm_provider.qs_relevance_with_cot_reasons,
                             name="Context Relevance")
                    .on_input_output()
                    .aggregate(np.mean)
                )
        
        self.f_qa_relevance = (
                    Feedback(litellm_provider.relevance_with_cot_reasons, name = "Answer Relevance")
                    .on_input_output()
                )
        
    def get_groundness_score(self, source: str, statement: str):

        score_dict, reason_dict = self.f_groundedness(source,statement)

        ## Idk why groundedness function returns a different format compared to other 2 eval func
        score_dict = score_dict['statement_0']
        return score_dict

    def get_answer_relevance_score(self, source: str, statement: str):

        score, reasoning = self.f_qa_relevance(source,statement)
        return score
    def get_context_relevance_score(self, source: str, statement: str):

        score, reasoning = self.f_qs_relevance(source,statement)

        return score
        
