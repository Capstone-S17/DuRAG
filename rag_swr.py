from src.reranker import Reranker
from retriever.swr.swr_retriever import SentenceWindowRetriever
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
    filters = "Stamford Land Corporation"
    # filters = "monkey"
    retrieval_response = swr_engine.hybrid_search(query, filters=filters, limit=10)
    print("*" * 100)
    print("Retrieval response: \n\n")
    print(swr_engine.chunk_text_joiner_response(retrieval_response.objects))
    print("*" * 100)
    # retrieval_response = swr_engine.full_text_search(query, filters=None, limit=10)
    # retrieval_response = swr_engine.semantic_search(query, filters=None, limit=10)
    sentence_windows = swr_engine.get_sentence_windows(retrieval_response.objects)
    print("*" * 100)
    print("Sentence Windows: \n\n")
    print(sentence_windows)
    print("*" * 100)
    results = swr_engine.get_rerank_format(query, sentence_windows)
    reranked_results = reranker.rerank_top_k(results, 3)

    # Process results
    print("Sentence Window response: \n\n")
    for result in reranked_results:
        print("-" * 100)
        print(result)

    generator = Generator();
    reranked_context = [reranked_results[i][0][2] for i in range(len(reranked_results))]
    response = generator.response_synthesis(reranked_context, query) 

    return response, reranked_results

    # i think we should coalese if there is overlap before passing to generator

    
if __name__ == "__main__":
    swr_pipeline("How long from financial year-end before Stamford Land Corporation annual financial statements are released?")
    