import weaviate
import weaviate.classes as wvc
from weaviate import WeaviateClient
from weaviate.collections import Collection
from tqdm import tqdm
from typing import Any
from DuRAG.rds import db


weaviate_client = weaviate.connect_to_local()


def create_collection(client: WeaviateClient):
    client.collections.create(
        name="AMR_chunks",
        description="chunks meant to be used for AMR 128 tokens",
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
            """SELECT chunk_id, chunk_text, pdf_document_name FROM amr_nodes WHERE chunk_size  = 128;"""
        )
        pages = cur.fetchall()

        if weaviate_client.collections.exists(
            "AMR_chunks"
        ):  # be careful, do no delete if you dont intend to!
            weaviate_client.collections.delete("AMR_chunks")
        create_collection(weaviate_client)
        collection = weaviate_client.collections.get("AMR_chunks")
        write_to_collection(collection, pages)

        weaviate_client.close()
