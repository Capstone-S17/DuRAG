from typing import Optional

from pydantic import BaseModel


class RetrievalObject(BaseModel):
    """
    A data model representing a retrieval object. Meant to be used to restructure data
    after retrieval and before reranking.
    """

    uuid: str
    """
    Chunk uuid usually form weaviate but also could be from RDS if AMR is used.
    """

    query: str
    """
    The original user query. BGE preamble is not included.
    """

    chunk: str
    """
    The text content of the retrieved chunk.
    """

    pdf_name: str
    """
    The name of the PDF file from which the chunk was retrieved.
    """

    score: float = 0.0
    """
    The relevance score associated with the retrieved chunk.
    Defaults to 0.0 if not provided.
    """

    pdf_page_id: Optional[int] = None
    """
    id of the extracted_pdf_page table. 
    Can be None if not applicable. Used in SWR
    """

    pdf_page_num: Optional[int] = None
    """
    The page number within the PDF file from which the chunk was retrieved.
    Can be None if not applicable. Used in SWR
    """

    pdf_id: Optional[int] = None
    """
    id of the extracted_pdf table. used to get the pdf_document_name.
    Can be None if not applicable.
    """

    amr_parent_id: Optional[str] = None
    """
    uuid of the parent node in the AMR tree. Parent UUIDs are ONLY found in RDS.
    Can be None if not applicable.
    """


class QueryObj(BaseModel):
    """
    QueryObj is a data model that stores the user query and the selected pdfs to search.
    """

    query: str
    filters: list[int]
    """
    list of extracted_pdf ids
    """


class RagResponse(BaseModel):
    message: str
    chunks: list[RetrievalObject]
