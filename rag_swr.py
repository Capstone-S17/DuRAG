from typing import Optional
from dotenv import load_dotenv
from DuRAG import Reranker, SentenceWindowRetriever, Generator
import weaviate.classes as wvc
import weaviate

load_dotenv()

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def swr_pipeline(query: str, filters):

    generator = Generator()
    client = weaviate.connect_to_local()
    reranker = Reranker()
    swr_engine = SentenceWindowRetriever(client)
    bge_query = BGE_QUERY_PREFIX + query
    filter_params = swr_engine._get_filter_param(filters, mode="or", property_name="pdf_name")
    retrieval_response = swr_engine.hybrid_search(bge_query, limit=10, filter_params=filter_params)
    sentence_windows = swr_engine.get_sentence_windows(retrieval_response.objects)
    results = swr_engine.get_rerank_format(query, sentence_windows)
    reranked_results = reranker.rerank_top_k(results, 5)

    # Process results
    print("Sentence Window response: \n\n")
    for result in reranked_results:
        print("-" * 100)
        print(result)

    reranked_context = [reranked_results[i][0][2] for i in range(len(reranked_results))]
    response = generator.response_synthesis(reranked_context, query)

    return response, reranked_results

    # i think we should coalese if there is overlap before passing to generator


if __name__ == "__main__":
    swr_pipeline(
        "How long from financial year-end before Stamford Land Corporation annual financial statements are released?",
    )
