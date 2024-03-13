from rag_swr import swr_pipeline
from rag_amr import amr_pipeline
from src.eval.rag_eval import RAGeval
from src.rds import db
import argparse 
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", type=str, required=True)
    return parser.parse_args()



if __name__ == "__main__":
    args=parse_args()
    evaluation = RAGeval()
    retrieval = args.retrieval
    with db.get_cursor() as cur:
        cur.execute("""SELECT question, pdf_document_name FROM "QUESTION_BANK" """)
        questions = cur.fetchall()
        if retrieval == "swr":
            pipeline = swr_pipeline
            eval_table = "chunked_128_sentence_window"
        elif retrieval == "amr":
            pipeline = amr_pipeline #pipeline not up
            eval_table = "amr_node"
        for i,(query, pdf_name) in enumerate(questions):
            eval_dict = {
                "groundness_score": 0,
                "answer_relevance_score": 0,
                "context_relevance_score": 0,
            }
            print("Question: "+ q[0])
            print("-" * 100)
            response, retrieved = pipeline(q[0])
            retrieved_text = ''.join([retrieved[i][0][2] for i in range(len(retrieved))])
            eval_object = {'question':query,
               'context':retrieved_text,
               'generated':response }
            # docs = []
            # for r in retrieved:
            #     print(r[0][0])
            #     cur.execute("""SELECT pdf_document_name FROM chunked_128_sentence_window WHERE chunk_id = %s""", (str(r[0][0]),))
            #     doc = cur.fetchall()
            #     docs.extend(doc)
            print("Evaluation: ")
            print(evaluation.assess_single_retrieval(q[0], response))


from rag_swr import swr_pipeline
from rag_amr import amr_pipeline
from src.eval.rag_eval import RAGeval
from src.rds import db
import argparse 
import os
from dotenv import load_dotenv
load_dotenv()
#os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'arched-forest-409209-854f83e41ed5.json'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", type=str, required=True)
    return parser.parse_args()




evaluation = RAGeval()

with db.get_cursor() as cur:
    cur.execute("""SELECT question, pdf_document_name FROM "QUESTION_BANK" """)
    questions = cur.fetchall()
   
    pipeline = swr_pipeline
    eval_table = "chunked_128_sentence_window"
    eval_dict = {
            "groundness_score": 0,
            "answer_relevance_score": 0,
            "context_relevance_score": 0,
        }
    
    for i,(query, pdf_name) in enumerate(questions):
        print("Question: "+ query)
        print("-" * 100)
        response, retrieved = pipeline(query)
        retrieved_text = ''.join([retrieved[i][0][2] for i in range(len(retrieved))])
        # docs = []
        # for r in retrieved:
        #     print(r[0][0])
        #     cur.execute("""SELECT pdf_document_name FROM chunked_128_sentence_window WHERE chunk_id = %s""", (str(r[0][0]),))
        #     doc = cur.fetchall()
        #     docs.extend(doc)
        
        print("Evaluation: ")
        eval_object = {'question':query,
               'context':retrieved_text,
               'generated':response }

        out = evaluation.assess_single_retrieval(**eval_object)
        eval_dict['groundness_score'] += out['groundness_score']
        eval_dict['answer_relevance_score']+= out['answer_relevance_score']
        eval_dict['context_relevance_score']+= out['context_relevance_score']

        print('context_relevance_score',eval_dict['context_relevance_score']/(i+1))
        print('answer_relevance_score',eval_dict['answer_relevance_score']/(i+1))
        print('groundness_score',eval_dict['groundness_score']/(i+1))
        
                


        

