from pydantic import BaseModel

class CreateUsers(BaseModel):
    username: str
    password: str

    class Config:
        orm_mode = True

# Schema class for classification history.
class CreateClassification(BaseModel):
    user_id: int
    classification_category: str
    classification_name: str
    classification_label: str
    confidence_value: str
    date: str

