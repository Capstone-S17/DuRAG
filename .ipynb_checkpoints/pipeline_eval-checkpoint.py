from rag_swr import swr_pipeline
#from rag_amr import amr_pipeline
from src.eval.rag_eval import RAGeval
from src.rds import db
import argparse

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
        for q in questions:
            response, retrieved = pipeline(q[0])
            docs = []
            for r in retrieved:
                print(r)
                cur.execute("""SELECT pdf_document_name FROM "%s" WHERE chunk_id = %s""", (eval_table, r[1]))
                doc = cur.fetchall()
                docs.extend(doc)
            evaluation.assess_single_retrieval(q[0], response, docs)

                


        

