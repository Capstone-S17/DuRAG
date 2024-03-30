import argparse
import json
from tqdm import tqdm

from dotenv import load_dotenv

from DuRAG.eval.rag_eval import RAGeval
from DuRAG.pipelines.rag_amr import amr_pipeline
from DuRAG.pipelines.rag_swr import swr_pipeline
from DuRAG.rds import db

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", type=str, required=True)
    parser.add_argument("--output_file_name",type=str,required=True) 



def write_json(data, path):
    f = open(path, mode='a', encoding='utf-8')
    json.dump(data, f, ensure_ascii=True)
    f.write('\n')
    f.close()
if __name__ == "__main__":
    
    args = parse_args()
    
    with open("eval_set_500.json", "r") as f:
        data = [json.loads(line) for line in f]
    output_file = args.output_file_name
    
    if args.retrieval == "amr":
        pipeline = amr_pipeline
    elif args.retrieval == "swr":
        pipeline = swr_pipeline
    else:
        print("Invalid retrieval method")
        exit(1)
    
    with open('eval_set_500.json','r') as f:
        data = [json.loads(line) for line in f]
    if len(data)>0:
        start = len(data)
        print(f"RESUMING FROM {start}")
    else:
        start = 0
        
    for i in tqdm(range(start,len(data))):
        query = data[i]["question"]
        filters = [data[i]["pdf_name"]]
        print("Question: " + query)
        print("-" * 100)

        response, retrieved = pipeline(query, filters)
        
        retrieved_text = [''.join([retrieved[i][0][2] for i in range(len(retrieved))])]
        
        eval_object = {
            "question": query,
            'contexts':retrieved_text,
            "answer":response,
            "ground_truth":data[i]['answer']
            
            
        }
       
        write_json(eval_object,f"{output_dir}.json")
      
