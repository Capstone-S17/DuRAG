import weaviate.classes as wvc
from typing import Optional


class Retriever:
    def __init__(self, weaviate_client, collection_name: str):
        self.client = weaviate_client
        self.collection = self.client.query.get(class_name=collection_name)

    def _get_filter_param(self, filters: Optional[str]):
        return (
            wvc.query.Filter().by_property("content").like(filters) if filters else None
        )

    def semantic_search(self, query: str, filters: Optional[str] = None, limit=10):
        filter_param = self._get_filter_param(filters)
        return self.collection.query.near_text(
            query=query,
            # include_vector=True,
            filters=filter_param,
            limit=limit,
        )

    def full_text_search(self, query: str, filters: Optional[str] = None, limit=10):
        filter_param = self._get_filter_param(filters)
        return self.collection.query.bm25(
            query=query,
            filters=filter_param,
            limit=limit,
        )

    def hybrid_search(self, query: str, filters: Optional[str] = None, limit=10):
        filter_param = self._get_filter_param(filters)
        return self.collection.query.hybrid(
            query=query,
            # include_vector=True,
            filters=filter_param,
            limit=limit,
        )

    @classmethod
    def chunk_text_joiner(cls, chunks: list[str]):
        """
        Joins the chunks into a single string and enumerates them. Meant to be used in RAG prompt.

        NOTE: not sure if this method should work on list[str] or the response obj format...
        currently assuming that the user should clean up whatever they pass in
        """
        return "\n".join(f"{i+1}. {chunk}" for i, chunk in enumerate(chunks))