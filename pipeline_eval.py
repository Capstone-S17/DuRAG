from rag_swr import swr_pipeline
# from rag_amr import amr_pipeline
from DuRAG.eval.rag_eval import RAGeval
from DuRAG.rds import db
import argparse
from dotenv import load_dotenv
import json

load_dotenv()


# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--retrieval", type=str, required=True)
#     parser.add_argument("--extract_keyword", action="store_true", default=False)

#     return parser.parse_args()


if __name__ == "__main__":
    evaluation = RAGeval()
    with open('generated_question.json','r') as f:
        data = [json.loads(line) for line in f]
    # args = parse_args()
    # with db.get_cursor() as cur:
    #     cur.execute("""SELECT question, pdf_document_name FROM "QUESTION_BANK" """)
    #     questions = cur.fetchall()

    eval_dict = {
        "groundness_score": 0,
        "answer_relevance_score": 0,
        "context_relevance_score": 0,
    }

    for i in range(len(data)):
        query = data[i]["question"]
        filters = [data[i]["pdf_name"]]
        print("Question: " + query)
        print("-" * 100)

        response, retrieved = swr_pipeline(query, filters)
        
        retrieved_text = "".join(
            [retrieved[i][0][2] for i in range(len(retrieved))]
        )
        # print(retrieved_text)
        # docs = []
        # for r in retrieved:
        #     print(r[0][0])
        #     cur.execute("""SELECT pdf_document_name FROM chunked_128_sentence_window WHERE chunk_id = %s""", (str(r[0][0]),))
        #     doc = cur.fetchall()
        #     docs.extend(doc)

        print("Evaluation: ")
        eval_object = {
            "question": query,
            "context": retrieved_text,
            "generated": response,
        }

        out = evaluation.assess_single_retrieval(**eval_object)
        eval_dict["groundness_score"] += out["groundness_score"]
        eval_dict["answer_relevance_score"] += out["answer_relevance_score"]
        eval_dict["context_relevance_score"] += out["context_relevance_score"]

        print(
            "context_relevance_score",
            eval_dict["context_relevance_score"] / (i + 1),
        )
        print(
            "answer_relevance_score", eval_dict["answer_relevance_score"] / (i + 1)
        )
        print("groundness_score", eval_dict["groundness_score"] / (i + 1))
