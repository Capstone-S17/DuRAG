from DuRAG import Reranker, AutoMergingRetriever, Generator, RetrievalObject
from DuRAG.rds import db
import weaviate
from dotenv import load_dotenv
import logging
from pydantic import BaseModel
from DuRAG.logger import logger


load_dotenv()

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class QueryObj(BaseModel):
    query: str
    filters: list[int]  # pdf_ids


class RagResponse(BaseModel):
    message: str
    chunks: list[RetrievalObject]


logging.getLogger("DuRAG").setLevel(logging.DEBUG)

# Create a console handler and set the log level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Add the console handler to the logger
logging.getLogger("DuRAG").addHandler(console_handler)


def execute_sql(cursor, query, params):
    try:
        cursor.execute(query, params)
        return cursor.fetchall()
    except Exception as e:
        logger.error(f"An error occurred executing SQL: {e}")
        raise


def get_pdf_names(cursor, filters):
    return execute_sql(
        cursor,
        """SELECT pdf_document_name FROM "EXTRACTED_PDF" WHERE id IN %s""",
        (tuple(filters),),
    )


def amr_pipeline(query_obj: QueryObj):
    with db.get_cursor() as rds_cursor:
        names = [name for (name,) in get_pdf_names(rds_cursor, query_obj.filters)]
        logger.info(names)

        pdf_name_to_id_map = dict(zip(names, query_obj.filters))

        client = weaviate.connect_to_local()

        amr_retriever = AutoMergingRetriever(client, rds_cursor)
        filter_params = amr_retriever._get_filter_param(
            names, mode="or", property_name="pdf_name"
        )
        logger.debug(f"Filter params: {filter_params}")

        bge_query = BGE_QUERY_PREFIX + query_obj.query
        retrieval_response = amr_retriever.hybrid_search(
            bge_query, filter_params=filter_params, limit=100
        )
        logger.debug(f"Retrieval response: {retrieval_response.objects}")

        retrieval_objects = [
            RetrievalObject(
                uuid=str(chunk.uuid),
                query=query_obj.query,
                chunk=chunk.properties["content"],
                pdf_name=chunk.properties["pdf_name"],
            )
            for chunk in retrieval_response.objects
        ]

        # aggregate chunks by 1 level
        first_level_aggregation = amr_retriever.aggregate_chunks(retrieval_objects)

        # rerank after first level aggregation
        reranker = Reranker()
        reranked_objects = reranker.rerank_top_k(
            first_level_aggregation, len(first_level_aggregation)
        )

        # add more context to the chunks
        second_level_aggregation = amr_retriever.aggregate_chunks(reranked_objects)
        client.close()

        # Prepare the context for the generator model by removing the query
        generator_context = [obj.chunk for obj in reranked_objects]
        generator = Generator()
        response = generator.response_synthesis(generator_context, query_obj.query)

        return RagResponse(message=response, chunks=second_level_aggregation)


if __name__ == "__main__":
    test_query = QueryObj(query="What is ascent bridge", filters=[247, 305, 205])
    amr_pipeline(test_query)
