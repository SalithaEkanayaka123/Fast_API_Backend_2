from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Database credentials and users, just for the reference - (NOT USED)
# ⚠️Add to the external resource file when deploying to the Azure.

username = 'akash@ml-backend-postgresql.postgres.database.azure.com'
password = '76FTjP5KuSwP5aA'
host = 'ml-backend-postgresql.postgres.database.azure.com'
database = 'postgres'

SQLALCHEMY_DATABASE_URL = "postgresql://akash:76FTjP5KuSwP5aA@ml-backend-postgresql.postgres.database.azure.com/postgres?sslmode=require"

engine = create_engine(SQLALCHEMY_DATABASE_URL, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    except:
        db.close()
