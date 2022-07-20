from pydantic import BaseModel

class CreateUsers(BaseModel):
    username: str
    password: str

    class Config:
        orm_mode = True
