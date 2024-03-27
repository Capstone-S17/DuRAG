from DuRAG import Reranker, AutoMergingRetriever, Generator
from DuRAG.rds import db

import weaviate


def amr_pipeline(query: str, filters):

    # Process results
    # print("Sentence Window response: \n\n")
    # for result in reranked_results:
    #     print("-" * 100)
    #     print(result)

    # reranked_context = [reranked_results[i][0][2] for i in range(len(reranked_results))]
    # response = generator.response_synthesis(reranked_context, query)

    # return response, reranked_results

    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
    client = weaviate.connect_to_local()
    reranker = Reranker()
    bge_query = BGE_QUERY_PREFIX + query
    with db.get_cursor() as rds_cursor:
        amr_engine = AutoMergingRetriever(client, rds_cursor)
        filter_params = amr_engine._get_filter_param(filters, mode="or", property_name="pdf_name")
        retrieval_response = amr_engine.hybrid_search(bge_query, filter_params=filter_params, limit=10)
        # need to do reranking but also keep a lot of information
        # reranked_results = reranker.rerank_top_k(reranke)
        # retrieval_response = amr_engine.full_text_search(query, filters=None, limit=10)
        # retrieval_response = amr_engine.semantic_search(query, filters=None, limit=10)

        rerank_format = amr_engine.get_rerank_format(query, retrieval_response.objects)
        reranked_chunks = reranker.rerank_top_k(rerank_format, 5)
        # print(reranked_results)
        merged_chunks = amr_engine.retrieve(retrieval_response.objects)
        
        f = lambda i:lambda x: x.uuid == i[0][0]
        reranked_results = []
        for i in reranked_chunks:
            merged = list(filter(f(i), merged_chunks))[0]
            print("-" * 100)
            print("original text: ", i[0][2], "\n")
            print("merged text: ", merged.properties["content"])
            l = list(i[0])
            l[2] = merged.properties["content"]
            t0 = tuple(l)
            t = (t0, i[1])
            reranked_results.append(t)

            # rds_cursor.execute(
            #     """SELECT pdf_page_id FROM "amr_nodes" WHERE chunk_id = %s """, (str(t[0][0]),)
            # )
            # pdf_page_id = rds_cursor.fetchall()
            # print(pdf_page_id)
            # rds_cursor.execute(
            #     """SELECT page_num FROM "EXTRACTED_PDF_PAGE" WHERE id = %s """, (str(pdf_page_id[0][0]),)
            # )
            # page_num = rds_cursor.fetchall()
            # print(page_num)
        
        generator = Generator()
        reranked_context = [reranked_results[i][0][2] for i in range(len(reranked_results))]
        response = generator.response_synthesis(reranked_context, query)

        

        return response, merged_chunks


if __name__ == "__main__":
    amr_pipeline(
        "What is the deadline for submitting questions to the company prior to the AGM?", ['SG230712OTHRO2AC_Pan Hong Holdings Group Limited_20230712000140_00_AR_4Q_20230331.1.pdf']
    )
