from datetime import datetime

from sqlalchemy.orm import Session
from db.model import User, Classification
from db.schema import CreateUsers, CreateClassification, ReqeustClassificationHistory


# CRUD Method to insert users.
def insert_user(details: CreateUsers, db: Session):
    to_create = User(
        name=details.username,
        hash_password=details.password
    )
    db.add(to_create)
    db.commit()
    return {
        "success": True,
        "create_id": to_create.id
    }


# CRUD Method to get all the users.
def get_all_users(db: Session, skin: int = 0, limit: int = 100):
    return db.query(User).offset(skin).limit(limit).all()


# CRUD method to get a single user by name.
def get_by_name(name: str, db: Session):
    return db.query(User).filter(User.name == name).first()


# CRUD method to get a single user by id.
def get_by_id(id: int, db: Session):
    return db.query(User).filter(User.id == id).first()


# CRUD method to insert classification data.
def insert_classification(details: ReqeustClassificationHistory, db: Session):
    create_classification = Classification(
        category=details.classification_category,
        filename=details.classification_filename,
        label=details.classification_label,
        confidence=details.confidence_value,
        date=datetime.now(),
        user_id=details.user_id,
    )
    db.add(create_classification)
    db.commit()
    return {
        "success": True,
        "create_id": create_classification.id
    }


# CRUD method to get classification data.
def get_classification_details(uid: int, db: Session):
    return db.query(Classification).filter(Classification.user_id == uid).all()
