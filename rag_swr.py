from reranker import Reranker
from swr.swr_retriever import SentenceWindowRetriever

import weaviate

if __name__ == "__main__":
    client = weaviate.connect_to_local()
    reranker = Reranker()
    swr_engine = SentenceWindowRetriever(client)
    query = "How long from financial year-end before Stamford Land Corporation annual financial statements are released?"
    results = swr_engine.get_rerank_format(query)
    reranked_results = reranker.rerank_top_k(results, 3)

    # Process results
    for result in reranked_results:
        print("-" * 100)
        print(result)

    # i think we should coalese if there is overlap before passing to generator
