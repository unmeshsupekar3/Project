from pydantic import BaseModel

class MunichWiki(BaseModel):
    question: str
    page_head: str