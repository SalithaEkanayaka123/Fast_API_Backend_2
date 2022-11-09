from pydantic import BaseModel, Field
from typing import Optional, Generic, TypeVar
from pydantic.generics import GenericModel

T = TypeVar('T')

class CreateUsers(BaseModel):
    username: str
    password: str

    class Config:
        orm_mode = True

class Request(GenericModel, Generic[T]):
    parameter: Optional[T] = Field(...)

class RequestUser(CreateUsers):
    id: int

class Response(GenericModel, Generic[T]):
    code: str
    status: str
    message: str
    result: Optional[T]

class Firebaseurl(BaseModel):
    url: str

class Request(BaseModel):
    image_url: str