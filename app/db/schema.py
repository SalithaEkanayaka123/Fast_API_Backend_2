from typing import TypeVar, Generic, Optional
from pydantic import BaseModel, Field
from pydantic.generics import GenericModel

T = TypeVar('T')


# User model base class.
class CreateUsers(BaseModel):
    username: str
    password: str

    class Config:
        orm_mode = True


# Schema class for classification history.
class CreateClassification(BaseModel):
    user_id: int
    classification_category: str
    classification_filename: str
    classification_label: str
    confidence_value: str
    date: str

    class Config:
        orm_mode = True


# Create classification request.
class ReqeustClassificationHistory(CreateClassification):
    parameter: CreateClassification = Field(...)


# Response schema for the classification.
class ResponseClassificationHistory(GenericModel, Generic[T]):
    status: str
    code: str
    date: str
    message: str
    result: Optional[T]

class Request(GenericModel, Generic[T]):
    parameter: Optional[T] = Field(...)

class Request(BaseModel):
    image_url: str
