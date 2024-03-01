from rag_swr import swr_pipeline
from src.eval.rag_eval import RAGeval
from src.rds import db
import argparse

if __name__ == "__main__":
    #retrieval = argparse.ArgumentParser(description="Choose retrieval method")
    with db.get_cursor() as cur:
        cur.execute("""SELECT * FROM "QUESTION_BANK" """)
        questions = cur.fetchall()
        if retrieval == "swr":
            pipeline = swr_pipeline
            eval_table = "chunked_128_sentence_window"
        elif retrieval == "amr":
            pipeline = amr_pipeline #pipeline not up
            eval_table = "amr_node"
        for q in questions:
            print(q)
                


        

