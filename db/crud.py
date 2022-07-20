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
def get_all_users(db: Session, skin: int = 0, limit: int = 100):
    return db.query(User).offset(skin).limit(limit).all()


# CRUD method to get a single user by name.
def get_by_name(name: str, db: Session):
    return db.query(User).filter(User.name == name).first()

# CRUD method to get a single user by id.
def get_by_id(id: int, db: Session):
    return db.query(User).filter(User.id == id).first()
