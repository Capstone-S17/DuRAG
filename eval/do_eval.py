from ragas import evaluate
from datasets import Dataset
import time
import pandas as pd
from datasets import Dataset
import os
import json
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

from ragas.metrics import (
    context_precision,
    answer_relevancy,  # AnswerRelevancy
    faithfulness,
    context_recall,
    answer_correctness
)


def main():
    with open('eval.json','r') as f:
        data = [json.loads(line) for line in f]
    dataset = convert_json_to_dataset(data)
    batch = 32 
    df = pd.DataFrame()
    
    for i in tqdm(range(1+len(dataset)//batch)):
        start, end = batch*i, batch*(i+1)
        score = evaluate(dataset.select(range(start,end)),metrics=metrics
            )
        time.sleep(20)
        df = pd.concat([df,score.to_pandas()])
        
    df.to_csv('output.csv',index=False)
