from DuRAG import Reranker, SentenceWindowRetriever, RetrievalObject
import weaviate.classes as wvc
import logging
from DuRAG.logger import logger
import weaviate

logging.getLogger("DuRAG").setLevel(logging.DEBUG)

# Create a console handler and set the log level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Add the console handler to the logger
logging.getLogger("DuRAG").addHandler(console_handler)

filters = ["SG230712OTHRNR1F_Old Chang Kee Ltd._20230712174222_00_AR_4Q_20230331.2.pdf"]
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
query = "what was the revenue of the company in the last quarter of 2023?"
client = weaviate.connect_to_local()
collection = client.collections.get("SWR_chunks")
reranker = Reranker()
swr_engine = SentenceWindowRetriever(client)
bge_query = BGE_QUERY_PREFIX + query
filter_params = swr_engine._get_filter_param(
    filters, mode="or", property_name="pdf_name"
)


chunk_count = collection.query.fetch_objects(
    filters=(
        wvc.query.Filter.by_property("pdf_name").equal(
            "SG230907OTHR98Q9_Totm Technologies Limited_20230907225023_00_AR_4Q_20230531.1.pdf"
        )
    )
)

print(f"{len(chunk_count.objects)=}")

retrieval_response = swr_engine.hybrid_search(
    bge_query, limit=10, filter_params=filter_params
)
print(f"{retrieval_response.objects=}")
print(f"{len(retrieval_response.objects)=}")
sentence_windows = swr_engine.get_sentence_windows(retrieval_response.objects)
retrieval_objects = [
    RetrievalObject(
        uuid=str(window_obj.center_uuid),
        query=bge_query,
        chunk=window_obj.joined_text(),
        pdf_name=window_obj.pdf_name,
    )
    for window_obj in sentence_windows
]
client.close()

reranked_objects = reranker.rerank_top_k(retrieval_objects, 5)
print("reranked_objects: ", reranked_objects)
