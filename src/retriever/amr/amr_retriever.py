import weaviate.classes as wvc
from rds import db
from typing import Optional


class AutoMergingRetriever:
    def __init__(self, weaviate_client, rds_cursor):
        self.weaviate_client = weaviate_client
        self.rds_cursor = rds_cursor
        self.collection = self.weaviate_client.collections.get("AMR_chunks")

    def query_collection(self, query: str, filters: Optional[str] = None, limit=10):
        filter_param = (
            wvc.query.Filter.by_property("content").like(filters) if filters else None
        )
        return self.collection.query.near_text(
            query=query,
            # include_vector=True,
            filters=filter_param,
            limit=limit,
        )

    def retrieve_by_uuid_from_weaviate(self, uuid):
        return self.collection.query.fetch_object_by_id(uuid)

    def retrieve_by_uuid_from_rds(self, uuid):
        self.rds_cursor.execute(
            """SELECT * FROM "amr_nodes" WHERE chunk_id = %s""", uuid
        )
        return self.rds_cursor.fetchall()

    def aggregate_chunks(self, chunks):
        parents = {}
        for chunk in chunks:
            # Retrieve the parent id for the given chunk
            parent_id = self.retrieve_by_uuid_from_rds(chunk["id"])
            if parent_id:
                # If two or more chunks have the same parent, we keep the parent in the aggregation
                parent_uuid = parent_id[0][
                    0
                ]  # Assuming the first column is the parent_uuid
                if parent_uuid in parents:
                    parents[parent_uuid].append(chunk)
                else:
                    parents[parent_uuid] = [chunk]
            else:
                parents[chunk["id"]] = [chunk]

        # Now, replace the smaller chunks with their parent chunk
        aggregated_chunks = []
        for parent_uuid, chunks in parents.items():
            # If the parent has more than one chunk, we only add the parent
            if len(chunks) > 1:
                aggregated_chunks.append(
                    self.retrieve_by_uuid_from_weaviate(parent_uuid)
                )
            else:
                # If the parent has only one chunk, we keep the original chunk
                aggregated_chunks.extend(chunks)

        return aggregated_chunks

    def retrieve(self, query, filters=None, limit=10):
        """
        from all the retrieved chunks, iterate through them and first
        find the parent chunks. If 2 or more chunks have the same parent,
        replace the smaller chunk with the parent chunk. Check again if the
        parent chunks have the same parent, if so, replace the smaller chunk
        with the parent chunk.
        """
        response = self.query_collection(query, filters, limit)

        # first aggregation
        first_aggregation = self.aggregate_chunks(response)

        # second aggregation
        second_aggregation = self.aggregate_chunks(first_aggregation)

        return second_aggregation

    def chunk_text_joiner(self, chunks):
        return " ".join(chunks)
