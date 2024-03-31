import os
import json
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd
from eval.rag_eval import RAGeval
import numpy as np
load_dotenv()


with open('eval_retrieved500.json','r') as f:
    data = [json.loads(line) for line in f]

def main():
   
   


    eval_list = []
    
    rag_eval = RAGeval()
    
    for i in tqdm(range(100)):
        question = data[i]['question']
        ground_truth = data[i]['ground_truth']
        generated_answer = data[i]['answer']
        context = '\n\n'.join(data[i]['contexts'])
        try:
            out = rag_eval.assess_single_retrieval(question=question, context=context , generated=generated_answer, ground_truth=ground_truth)
            
           
        except:
            out= {
            "groundness_score": 0,
            "answer_relevance_score": 0,
            "context_relevance_score": 0,
            "answer_correctness_score": 0
        }
        eval_list.append(out)
       
    df = pd.DataFrame(eval_list)
    question_list = [data[:100][i]['question'] for i in range(100)]
    context_list = [data[:100][i]['contexts'] for i in range(100)]
    generated_ans = [data[:100][i]['answer'] for i in range(100)]
    ground_truth = [data[:100][i]['ground_truth'] for i in range(100)]
    df['ground_truth'] = ground_truth
    df['generated_ans'] = generated_ans
    df['context_list'] = context_list
    df['question_list'] = question_list
    df.to_csv('eval_output.csv')
    print(f"Groundness score {df.groundness_score.mean()}")
    print(f"Answer relevance score {df.answer_relevance_score.mean()}")
    print(f"Answer correctness score {df.answer_correctness_score.mean()}")
    print(f"Context relevance score {df.context_relevance_score.mean()}")
   