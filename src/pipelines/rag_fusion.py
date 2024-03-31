import logging

import weaviate

from DuRAG.generator import Generator
from DuRAG.logger import logger
from DuRAG.rds import db
from DuRAG.reranker import Reranker
from DuRAG.retriever.amr.amr_retriever import AutoMergingRetriever
from DuRAG.retriever.data_models import QueryObj, RagResponse, RetrievalObject
from DuRAG.retriever.swr.swr_retriever import SentenceWindowRetriever

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


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


def get_page_details(cursor, uuid):
    return execute_sql(
        cursor,
        """SELECT p.pdf_page_id, e.page_num FROM "chunked_128_sentence_window" AS p
           JOIN "EXTRACTED_PDF_PAGE" AS e ON p.pdf_page_id = e.id WHERE p.chunk_id = %s""",
        (uuid,),
    )


def fusion_pipeline(query_obj: QueryObj):
    with db.get_cursor() as cur:
        names = [name for (name,) in get_pdf_names(cur, query_obj.filters)]
        logger.info(names)

        pdf_name_to_id_map = dict(zip(names, query_obj.filters))

        client = weaviate.connect_to_local()
        reranker = Reranker()

        # SWR Retrieval
        logger.debug("SWR Retrieval")
        swr_retriever = SentenceWindowRetriever(client)
        filter_params = swr_retriever._get_filter_param(
            names, mode="or", property_name="pdf_name"
        )
        logger.debug(f"{filter_params=}")

        bge_query = BGE_QUERY_PREFIX + query_obj.query
        swr_retrieval_response = swr_retriever.hybrid_search(
            bge_query, limit=100, filter_params=filter_params
        )
        logger.debug("retrieval_response: ", swr_retrieval_response.objects)

        sentence_windows = swr_retriever.get_sentence_windows(
            swr_retrieval_response.objects
        )
        logger.debug("Sentence Windows: ", sentence_windows)

        swr_retrieval_objects = [
            RetrievalObject(
                uuid=str(window_obj.center_uuid),
                query=query_obj.query,
                chunk=window_obj.joined_text(),
                pdf_name=window_obj.pdf_name,
            )
            for window_obj in sentence_windows
        ]

        # AMR Retrieval
        logger.debug("AMR Retrieval")
        amr_retriever = AutoMergingRetriever(client, cur)
        filter_params = amr_retriever._get_filter_param(
            names, mode="or", property_name="pdf_name"
        )
        logger.debug(f"Filter params: {filter_params}")

        amr_retrieval_response = amr_retriever.hybrid_search(
            bge_query, filter_params=filter_params, limit=100
        )
        logger.debug(f"Retrieval response: {amr_retrieval_response.objects}")

        amr_retrieval_objects = [
            RetrievalObject(
                uuid=str(chunk.uuid),
                query=query_obj.query,
                chunk=chunk.properties["content"],
                pdf_name=chunk.properties["pdf_name"],
            )
            for chunk in amr_retrieval_response.objects
        ]

        # aggregate chunks by 1 level
        first_level_aggregation = amr_retriever.aggregate_chunks(amr_retrieval_objects)

        # Rerank the retrieved results
        reranked_objects = reranker.rerank_top_k(
            swr_retrieval_objects + amr_retrieval_objects, 5
        )

        # Prepare the context for the generator model by removing the query
        generator = Generator()
        generator_context = [obj.chunk for obj in reranked_objects]
        response = generator.response_synthesis(generator_context, query_obj.query)

        return RagResponse(message=response, chunks=first_level_aggregation)
