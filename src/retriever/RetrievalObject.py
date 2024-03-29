from pydantic import BaseModel
from typing import Optional


class RetrievalObject(BaseModel):
    uuid: str
    query: str
    chunk: str
    pdf_name: str
    score: float = 0.0
    pdf_page_id: Optional[int] = None
    pdf_page_num: Optional[int] = None
    pdf_id: Optional[int] = None
    amr_parent_id: Optional[str] = None
