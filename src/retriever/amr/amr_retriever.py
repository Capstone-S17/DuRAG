from collections import defaultdict

from DuRAG.logger import logger
from DuRAG.retriever.data_models import RetrievalObject
from DuRAG.retriever.retriever import Retriever


class AutoMergingRetriever(Retriever):
    def __init__(self, weaviate_client, db):
        super().__init__(weaviate_client, "AMR_chunks")
        self.weaviate_client = weaviate_client
        self.db = db
        #self.rds_cursor = rds_cursor

    def retrieve_by_uuid_from_weaviate(self, uuid):
        
        return self.collection.query.fetch_object_by_id(uuid)

    def retrieve_by_uuid_from_rds(self, uuid: str, rds_cursor):
        
        rds_cursor.execute(
                f"""SELECT * FROM "amr_nodes" WHERE chunk_id = '{uuid}'"""
            )
        return rds_cursor.fetchall()

    def aggregate_chunks(self, retrieved_chunks: list[RetrievalObject]):
        with self.db.get_cursor() as rds_cursor:
            parents = defaultdict(list)
            logger.info(f"AMR aggregate chunks received {len(retrieved_chunks)} chunks")
            for chunk in retrieved_chunks:
                # Retrieve the parent id for the given chunk
                rds_row_info = self.retrieve_by_uuid_from_rds(str(chunk.uuid),rds_cursor)
                # If two or more chunks have the same parent, we keep the parent in the aggregation
                parent_uuid = rds_row_info[0][1]
                parents[parent_uuid].append(chunk)
    
            # Now, replace the smaller chunks with their parent chunk
            aggregated_chunks = []
            for parent_uuid, chunks in parents.items():
                # If the parent has more than one chunk, we only add the parent
                if len(chunks) > 1:
                    logger.info(f"AMR aggregate chunks replacing {len(chunks)} with parent")
                    retrieved_parent = self.retrieve_by_uuid_from_rds(parent_uuid,rds_cursor)
                    parent = RetrievalObject(
                        uuid=parent_uuid,
                        query=chunks[0].query,
                        chunk=retrieved_parent[0][3],
                        pdf_name=retrieved_parent[0][2],
                    )
                    aggregated_chunks.append(parent)
                else:
                    # If the parent has only one chunk, we keep the original chunk
                    aggregated_chunks.extend(chunks)

        return aggregated_chunks
