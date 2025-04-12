from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str
    course_id: int
