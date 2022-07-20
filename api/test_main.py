from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.database import get_db, Base
from main import app


SQLALCHEMY_DATABASE_URL = "postgresql://akash:76FTjP5KuSwP5aA@ml-backend-postgresql.postgres.database.azure.com/postgres?sslmode=require"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


Base.metadata.create_all(bind=engine)


def override_get_db():
    try:
        db = TestingSessionLocal()
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)


def test_method():
    response = client.post(
        "/get-users-name/?name=Akash",

    )
    data = response.json()
    print(data)
