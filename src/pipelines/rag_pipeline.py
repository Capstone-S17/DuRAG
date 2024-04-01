
import weaviate
from DuRAG.generator import Generator
from DuRAG.logger import logger
from DuRAG.rds import db
from DuRAG.reranker import Reranker
from DuRAG.retriever.swr.swr_retriever import SentenceWindowRetriever
import logging
from DuRAG.retriever.amr.amr_retriever import AutoMergingRetriever
from DuRAG.retriever.data_models import QueryObj, RagResponse, RetrievalObject
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


    
class RAGpipeline:


    def __init__(self):
        
        self.reranker = Reranker()
        self.client = weaviate.connect_to_local()
        self.generator = Generator()
        self.amr_retriever = AutoMergingRetriever(self.client,db)
        self.swr_retriever = SentenceWindowRetriever(self.client)
        self.client.close()
        
        
        
    
    def swr_retrieval(self,query_obj, alpha = 0.5,limit = 100):
        
        with db.get_cursor() as rds_cursor:
            names = [name for (name,) in get_pdf_names(rds_cursor, query_obj.filters)]
            filter_params = self.swr_retriever._get_filter_param(
                names, mode="or", property_name="pdf_name"
            )
            logger.debug(f"{filter_params=}")
            pdf_name_to_id_map = dict(zip(names, query_obj.filters))
    
            
            bge_query = BGE_QUERY_PREFIX + query_obj.query
            retrieval_response = self.swr_retriever.hybrid_search(
                bge_query, 
                limit=limit,
                alpha=alpha,
                filter_params=filter_params
            )
            logger.debug("retrieval_response: ", retrieval_response.objects)
    
            sentence_windows = self.swr_retriever.get_sentence_windows(
                retrieval_response.objects
            )
            logger.debug("Sentence Windows: ", sentence_windows)
    
            retrieval_objects = [
                RetrievalObject(
                    uuid=str(window_obj.center_uuid),
                    query=query_obj.query,
                    chunk=window_obj.joined_text(),
                    pdf_name=window_obj.pdf_name,
                    pdf_id = pdf_name_to_id_map[window_obj.pdf_name]
                )
                for window_obj in sentence_windows
            ]
       

            # Rerank the retrieved results
            reranked_objects = self.reranker.rerank_top_k(retrieval_objects, 10)

     
            print("Sentence Window response: \n\n")
            for result in reranked_objects:
                print("-" * 100)
    
                pdf_page_id, page_num = get_page_details(rds_cursor, result.uuid)[0]
                result.pdf_page_id = pdf_page_id
                result.pdf_page_num = page_num
                result.pdf_id = pdf_name_to_id_map[result.pdf_name]

        return reranked_objects
        
    def amr_retrieval(self,query_obj, alpha = 0.5,limit = 100):
        # aggregate chunks by 1 level
        with db.get_cursor() as rds_cursor:
            names = [name for (name,) in get_pdf_names(rds_cursor, query_obj.filters)]
            logger.info(names)
    
            pdf_name_to_id_map = dict(zip(names, query_obj.filters))
            ## Doesn't matter which retriever we using as the hybrid search is inherited from base retriever
            filter_params = self.amr_retriever._get_filter_param(
                names, mode="or", property_name="pdf_name"
            )
            
            logger.debug(f"Filter params: {filter_params}")
            
            bge_query = BGE_QUERY_PREFIX + query_obj.query
            retrieval_response = self.amr_retriever.hybrid_search(
                bge_query, 
                filter_params=filter_params,
                limit=limit,
                alpha=alpha
            )
           
            logger.debug(f"Retrieval response: {retrieval_response.objects}")
            retrieval_objects = [
                    RetrievalObject(
                        uuid=str(chunk.uuid),
                        query=query_obj.query,
                        chunk=chunk.properties["content"],
                        pdf_name=chunk.properties["pdf_name"],
                        pdf_id = pdf_name_to_id_map[chunk.properties["pdf_name"]]
                    )
                    for chunk in retrieval_response.objects
                ]
            first_level_aggregation = self.amr_retriever.aggregate_chunks(retrieval_objects)
                
             # rerank after first level aggregation
               
            reranked_objects = self.reranker.rerank_top_k(
                    first_level_aggregation,10
                )
                
            # add more context to the chunks
            reranked_objects = self.amr_retriever.aggregate_chunks(reranked_objects) 
            for result in reranked_objects:
                print("-" * 100)
    
                pdf_page_id, page_num = get_page_details(rds_cursor, result.uuid)[0]
                result.pdf_page_id = pdf_page_id
                result.pdf_page_num = page_num
                result.pdf_id = pdf_name_to_id_map[result.pdf_name]
                
        return reranked_objects

        
    def pipeline(self,query_obj, alpha = 0.5,limit = 100,retrieve_type = 'swr'):
        
        self.client.connect()
        with db.get_cursor() as rds_cursor:
            names = [name for (name,) in get_pdf_names(rds_cursor, query_obj.filters)]
            logger.info(names)
    
            pdf_name_to_id_map = dict(zip(names, query_obj.filters))
            ## Doesn't matter which retriever we using as the hybrid search is inherited from base retriever
            filter_params = self.amr_retriever._get_filter_param(
                names, mode="or", property_name="pdf_name"
            )
            
            logger.debug(f"Filter params: {filter_params}")
            
            bge_query = BGE_QUERY_PREFIX + query_obj.query
            retrieval_response = self.amr_retriever.hybrid_search(
                bge_query, 
                filter_params=filter_params,
                limit=limit,
                alpha=alpha
            )
           
            logger.debug(f"Retrieval response: {retrieval_response.objects}")
            
            
            ## Do aggregation if retrieval type is amr
            if retrieve_type == "amr":
                
                reranked_objects = self.amr_retrieval(query_obj,alpha,limit)
                
                               
            ## Do sentence window aggregation if retrieval type is swr
            elif retrieve_type == "swr":
                reranked_objects = self.swr_retrieval(query_obj,alpha,limit)
                
            elif retrieve_type == 'fusion':
                ## Get SWR objects
                reranked_object_amr = self.amr_retrieval(query_obj,alpha,limit)
                reranked_object_swr = self.swr_retrieval(query_obj,alpha,limit)
                
            
                reranked_objects = self.reranker.rerank_top_k(
                    reranked_object_swr+reranked_object_amr,10
                )
                
                
          
            generator_context = [obj.chunk for obj in reranked_objects]
            response = self.generator.response_synthesis(generator_context, query_obj.query)
           
        self.client.close()
        
        return RagResponse(message=response, chunks=reranked_objects)
    
if __name__ == "__main__":
    test_query = QueryObj(query="What is ascent bridge", filters=[247, 305, 205])
    rag = RAGpipeline()
    rag.pipeline(test_query,alpha=0.5,limit=100,retrieve_type='swr')
