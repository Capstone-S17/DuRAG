from reranker import Reranker
from retriever.swr.swr_retriever import SentenceWindowRetriever
from src.generator import Generator

import weaviate

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

if __name__ == "__main__":
    client = weaviate.connect_to_local()
    reranker = Reranker()
    swr_engine = SentenceWindowRetriever(client)
    query = (
        BGE_QUERY_PREFIX
        + "How long from financial year-end before Stamford Land Corporation annual financial statements are released?"
    )
    retrieval_response = swr_engine.semantic_search(query, filters=None, limit=10)
    sentence_windows = swr_engine.get_sentence_windows(retrieval_response.objects)

    results = swr_engine.get_rerank_format(query)
    reranked_results = reranker.rerank_top_k(results, 3)

    # Process results
    for result in reranked_results:
        print("-" * 100)
        print(result)

    generator = Generator()
    print(generator.response_synthesis(reranked_results, query))

    # i think we should coalese if there is overlap before passing to generator
