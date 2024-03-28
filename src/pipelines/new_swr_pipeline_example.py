import weaviate
from dotenv import load_dotenv
from DuRAG.rds import db
from DuRAG import Reranker, SentenceWindowRetriever, Generator, RetrievalObject
import logging
from DuRAG.logger import logger
from pydantic import BaseModel

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


def get_page_details(cursor, uuid):
    return execute_sql(
        cursor,
        """SELECT p.pdf_page_id, e.page_num FROM "chunked_128_sentence_window" AS p
           JOIN "EXTRACTED_PDF_PAGE" AS e ON p.pdf_page_id = e.id WHERE p.chunk_id = %s""",
        (uuid,),
    )


def pipeline(query_obj: QueryObj):
    with db.get_cursor() as cur:
        names = [name for (name,) in get_pdf_names(cur, query_obj.filters)]
        logger.info(names)

        pdf_name_to_id_map = dict(zip(names, query_obj.filters))

        client = weaviate.connect_to_local()
        reranker = Reranker()
        swr_retriever = SentenceWindowRetriever(client)
        filter_params = swr_retriever._get_filter_param(
            names, mode="or", property_name="pdf_name"
        )
        logging.debug(f"{filter_params=}")

        generator = Generator()
        logging.debug(
            generator.query_summary(
                "What is the percentage of shares held by the largest shareholder of Samurai 2K Aerosol Limited as of 27 June 2023?"
            )
        )
        bge_query = BGE_QUERY_PREFIX + query_obj.query
        retrieval_response = swr_retriever.hybrid_search(
            bge_query, limit=10, filter_params=filter_params
        )
        print("retrieval_response: ", retrieval_response.objects)

        sentence_windows = swr_retriever.get_sentence_windows(
            retrieval_response.objects
        )
        print("Sentence Windows: ", sentence_windows)

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

        # Rerank the retrieved results
        reranked_objects = reranker.rerank_top_k(retrieval_objects, 5)

        # Prepare the context for the generator model by removing the query
        generator_context = [obj.chunk for obj in reranked_objects]
        response = generator.response_synthesis(generator_context, query_obj.query)

        print("Sentence Window response: \n\n")
        for result in retrieval_objects:
            print("-" * 100)

            pdf_page_id, page_num = get_page_details(cur, result.uuid)[0]
            result.pdf_page_id = pdf_page_id
            result.pdf_page_num = page_num
            result.pdf_id = pdf_name_to_id_map[result.pdf_name]

            print(result)

        return RagResponse(message=response, chunks=retrieval_objects)


if __name__ == "__main__":
    test_query = QueryObj(query="What is ascent bridge", filters=[247, 305, 205])
    pipeline(test_query)
