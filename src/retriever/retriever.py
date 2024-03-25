import weaviate.classes as wvc
from functools import reduce
from typing import Optional, Any, Literal


class Retriever:
    def __init__(self, weaviate_client, collection_name: str):
        self.client = weaviate_client
        self.collection = self.client.collections.get(collection_name)

    def _get_filter_param(
        self,
        filters: Optional[list[str]],
        mode: Literal["and", "or"] = "and",
        property_name: Literal["content", "pdf_name", "NER"] = "content",
    ):
        """
        For pdf_name - use mode = "and" and do exact match (.equal)
        For content/NER - use mode = "or" and do partial match (like)
        """
        if filters:
            list_of_filters = [
                wvc.query.Filter.by_property(property_name).like(keyword)
                for keyword in filters
            ]
            # Dynamically combine the filters using the pipe or ampersand operator
            if mode == "and":
                combined_filter = reduce(lambda a, b: a | b, list_of_filters)
            else:
                combined_filter = reduce(lambda a, b: a & b, list_of_filters)
            return combined_filter
        return None

    def semantic_search(
        self, query: str, filters: Optional[list[str]] = None, limit=10
    ):
        filter_param = self._get_filter_param(filters)
        return self.collection.query.near_text(
            query=query,
            # include_vector=True,
            filters=filter_param,
            limit=limit,
        )

    def full_text_search(
        self, query: str, filters: Optional[list[str]] = None, limit=10
    ):
        filter_param = self._get_filter_param(filters)
        return self.collection.query.bm25(
            query=query,
            filters=filter_param,
            limit=limit,
        )

    def hybrid_search(self, query: str, filter_params: list, limit=10):
        """
        need to change filters to be a list of filters and use _get_filter_param to create the filters
        """
        return self.collection.query.hybrid(
            query=query,
            # include_vector=True,
            filters=filter_params,
            limit=limit,
        )

    @classmethod
    def chunk_text_joiner(cls, chunks: list[str]):
        """
        Joins the chunks into a single string and enumerates them. Meant to be used in RAG prompt.
        """
        return "\n".join(f"Chunk {i+1}: {chunk}" for i, chunk in enumerate(chunks))

    @classmethod
    def chunk_text_joiner_response(cls, chunks: list[Any]):
        """
        Joins the chunks into a single string and enumerates them. Works directly with the response from the retriever.
        """
        return "\n".join(
            f"Chunk {i+1}: {chunk.properties['content']}"
            for i, chunk in enumerate(chunks)
        )
