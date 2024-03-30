import logging
from typing import List, Optional

import numpy as np
from trulens_eval import Feedback, FeedbackMode, Tru, TruLlama
from trulens_eval.feedback import Groundedness
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
import litellm



class RAGeval:
    def __init__(self):

        
        self.litellm_provider = LiteLLM(model_engine='gemini-pro')
        grounded = Groundedness(groundedness_provider=self.litellm_provider)
        self.f_groundedness = (
            Feedback(
                grounded.groundedness_measure_with_cot_reasons, name="Groundedness"
            )
            .on_input_output()
            .aggregate(grounded.grounded_statements_aggregator)
        )
        self.f_qs_relevance = (
                    Feedback(self.litellm_provider.qs_relevance_with_cot_reasons,
                             name="Context Relevance")
                    .on_input_output()
                    .aggregate(np.mean)
                )
        
        self.f_qa_relevance = (
                    Feedback(self.litellm_provider.relevance_with_cot_reasons, name = "Answer Relevance")
                    .on_input_output()
                )
    def _get_answer_correctness_score(self, question: str, generated_ans: str, ground_truth: str):
        template = f'''You are given a question, ground truth answer and a generated answer. Your task is to score this answer base on how accurate is it from a scale of 0 to 1.
QUESTION:{question}
GROUND TRUTH ANSWER: {ground_truth}
GENERATED ANSWER: {generated_ans}
SCORE: '''
        
        response = litellm.completion(model="gemini-pro", messages=[{"role": "user", "content": template}])
        try:
            output = response.choices[0].message.content
            if output.isnumeric():
                output = float(output)
            else:
                output = 0
        except:
            output = 0
        return output
        
        
    def _get_groundness_score(self, source: str, statement: str):
        score_dict, reason_dict = self.f_groundedness(source, statement)
        ## Idk why groundedness function returns a different format compared to other 2 eval func
        #print(score_dict, reason_dict)
        score_dict = score_dict['statement_0']
        return score_dict

    def _get_answer_relevance_score(self, source: str, statement: str):
        score, reasoning = self.f_qa_relevance(source, statement)
        return score
    
    def _get_context_relevance_score(self, source: str, statement: str):
        score, reasoning = self.f_qs_relevance(source,statement)
        return score

    def _document_retrieval_accuracy(self, groundtruth: str, retrieved: list):
        for r in retrieved:
            if groundtruth == r[0]:
                return 1

    def _filter_accuracy(self, groundtruth: str, filtered: list):
        if groundtruth in filtered:
            return 1
    
    def assess_single_retrieval(self, question: str, context: str, generated: str, ground_truth: str):
        if generated == '':
            return {
            "groundness_score": 0,
            "answer_relevance_score": 0,
            "context_relevance_score": 0,
            "answer_correctness_score": 0
        }
        groundness_score = self._get_groundness_score(context, generated)
        answer_relevance_score = self._get_answer_relevance_score(question, generated)
        context_relevance_score = self._get_context_relevance_score(question, context)
        answer_correctness_score = self._get_answer_correctness_score(question, generated, ground_truth)
        
    
        return {
            "groundness_score": groundness_score,
            "answer_relevance_score": answer_relevance_score,
            "context_relevance_score": context_relevance_score,
            "answer_correctness_score": answer_correctness_score
        }
