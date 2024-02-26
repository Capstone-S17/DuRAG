import weaviate
import weaviate.classes as wvc
from weaviate import WeaviateClient
from weaviate.collections import Collection
from tqdm import tqdm
from typing import Any
from rds import db


weaviate_client = weaviate.connect_to_local()


def create_collection(client: WeaviateClient):
    client.collections.create(
        name="SWR_chunks",
        description="chunks meant to be used for SWR 512 tokens",
        vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_transformers(),
        properties=[
            wvc.config.Property(
                data_type=wvc.config.DataType.TEXT,
                description="full name of the pdf document",
                name="pdf_name",
                vectorize_property_name=False,
                skip_vectorization=True,
                index_filterable=True,
                index_searchable=True,
            ),
            wvc.config.Property(
                data_type=wvc.config.DataType.TEXT,
                description="content of the chunk",
                name="content",
                vectorize_property_name=False,
                skip_vectorization=False,
                index_filterable=True,
                index_searchable=True,
            ),
            wvc.config.Property(
                data_type=wvc.config.DataType.TEXT_ARRAY,
                description="NERs of the chunk",
                name="NER",
                skip_vectorization=True,
                index_filterable=True,
                vectorize_property_name=False,
                index_searchable=False,
            ),
        ],
    )


def write_to_collection(collection: Collection, data: list[tuple[Any, ...]]):
    with collection.batch.fixed_size(batch_size=100) as batch:
        for chunk_id, chunk_text, pdf_name in tqdm(data):
            properties = {
                "pdf_name": pdf_name,
                "content": chunk_text,
            }
            batch.add_object(properties=properties, uuid=chunk_id)


if __name__ == "__main__":
    with db.get_cursor() as cur:
        cur.execute(
            """SELECT c.chunk_id, c.chunk_text, epp.pdf_document_name FROM chunked_512_recursive_dhanush c JOIN "EXTRACTED_PDF_PAGE" epp ON c.pdf_page_id = epp.id; """
        )
        pages = cur.fetchall()

        if weaviate_client.collections.exists(
            "SWR_chunks"
        ):  # be careful, do no delete if you dont intend to!
            weaviate_client.collections.delete("SWR_chunks")
        create_collection(weaviate_client)
        collection = weaviate_client.collections.get("SWR_chunks")
        write_to_collection(collection, pages)

        weaviate_client.close()
