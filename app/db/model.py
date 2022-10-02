from sqlalchemy import Integer, String, Date
from sqlalchemy.orm import relationship
from sqlalchemy.sql.schema import Column, ForeignKey
from app.db.database import Base


class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    hash_password = Column(String, nullable=False)
    classifications = relationship("Classification", back_populates='user')

    # Add other necessary parameters for the user.


class Classification(Base):
    __tablename__ = 'classifications'

    id = Column(Integer, primary_key=True)
    category = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    label = Column(String, nullable=False)
    confidence = Column(String, nullable=True)
    date = Column(Date, nullable=True)
    user_id = Column(Integer, ForeignKey('users.id'))

    # Relationships
    user = relationship("User", back_populates='classifications')

    # Refactor the columns appropriately.

