from sqlalchemy.orm import Session
from db.model import User
from db.schema import CreateUsers


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
def get_all_users(db: Session):
    return db.query(User).offset(0).limit(100).all()
