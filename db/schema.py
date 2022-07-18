from pydantic import BaseModel

class CreateUsers(BaseModel):
    title: str
    description: str
