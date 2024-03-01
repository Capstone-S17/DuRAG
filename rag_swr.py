from reranker import Reranker
from swr.swr_retriever import SentenceWindowRetriever
from src.generator import Generator

import weaviate


def swr_pipeline(query: str):
    BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
    client = weaviate.connect_to_local()
    reranker = Reranker()
    swr_engine = SentenceWindowRetriever(client)
    query = (
        BGE_QUERY_PREFIX
        + query
    )
    # retrieval_response = swr_engine.query_collection(query, filters=None, limit=10)
    results = swr_engine.get_rerank_format(query)
    reranked_results = reranker.rerank_top_k(results, 3)

    # Process results
    for result[1] in reranked_results:
        print("-" * 100)
        print(result)

    generator = Generator();
    response = generator.response_synthesis(reranked_results, query) 

    return response, reranked_results

    # i think we should coalese if there is overlap before passing to generator

    
if __name__ == "__main__":
    swr_pipeline("How long from financial year-end before Stamford Land Corporation annual financial statements are released?")
    