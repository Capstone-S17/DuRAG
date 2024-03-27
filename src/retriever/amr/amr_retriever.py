from DuRAG.retriever.retriever import Retriever


class AutoMergingRetriever(Retriever):
    def __init__(self, weaviate_client, rds_cursor):
        super().__init__(weaviate_client, "AMR_chunks")
        self.weaviate_client = weaviate_client
        self.rds_cursor = rds_cursor

    def retrieve_by_uuid_from_weaviate(self, uuid):
        return self.collection.query.fetch_object_by_id(uuid)

    def retrieve_by_uuid_from_rds(self, uuid: str):
        self.rds_cursor.execute(
            f"""SELECT * FROM "amr_nodes" WHERE chunk_id = '{uuid}'"""
        )
        # print(self.rds_cursor.fetchall())
        return self.rds_cursor.fetchall()

    def aggregate_chunks(self, chunks):
        parents = {}
        for chunk in chunks:
            # Retrieve the parent id for the given chunk
            parent_id = self.retrieve_by_uuid_from_rds(str(chunk.uuid))
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
                parents[chunk.uuid] = [chunk]

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

    def retrieve(self, response):
        """
        from all the retrieved chunks, iterate through them and first
        find the parent chunks. If 2 or more chunks have the same parent,
        replace the smaller chunk with the parent chunk. Check again if the
        parent chunks have the same parent, if so, replace the smaller chunk
        with the parent chunk.
        """
        # first aggregation
        first_aggregation = self.aggregate_chunks(response)
        # second aggregation
        second_aggregation = self.aggregate_chunks(first_aggregation)
        return second_aggregation

    # @classmethod
    # def text_joiner(cls, node: list[tuple[str, str]]) -> str:
    #     return " ".join(i[1] for i in node)

    @classmethod
    def get_rerank_format(
        cls, query: str, response_ojects
    ):
        leaf_chunks = []
        for i in response_ojects:
            leaf_chunks.append(
                (i.uuid, query, i.properties["content"], i.properties["pdf_name"])
            )
        return leaf_chunks