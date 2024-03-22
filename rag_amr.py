from src.reranker import Reranker
from src.retriever.amr.amr_retriever import AutoMergingRetriever
from src.generator import Generator
from src.rds import db

import weaviate

def amr_pipeline(query: str):
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
    client = weaviate.connect_to_local()
    reranker = Reranker()
    query = (
        BGE_QUERY_PREFIX
        + query
    )
    with db.get_cursor() as rds_cursor:
        amr_engine = AutoMergingRetriever(client, rds_cursor)
        retrieval_response = amr_engine.hybrid_search(query, filters=None, limit=10)
        # need to do reranking but also keep a lot of information
        # reranked_results = reranker.rerank_top_k(reranke)
        # retrieval_response = amr_engine.full_text_search(query, filters=None, limit=10)
        # retrieval_response = amr_engine.semantic_search(query, filters=None, limit=10)

        merged_chunks = amr_engine.retrieve(retrieval_response.objects)

        # Process results
        for result in merged_chunks:
            print("-" * 100)
            print(result)
        generator = Generator()
        response = generator.response_synthesis(merged_chunks, query)

        return response, merged_chunks


if __name__ == "__main__":
    amr_pipeline("How long from financial year-end before Stamford Land Corporation annual financial statements are released?")