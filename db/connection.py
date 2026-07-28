import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

load_dotenv()

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        url = (
            f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
            f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        )
        _engine = create_engine(
            url,
            pool_size=5,
            max_overflow=10,
            echo=False
        )
    return _engine


def get_session():
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def test_connection():
    try:
        with get_engine().connect() as conn:
            result = conn.execute(text("SELECT version();"))
            print(f"Connected: {result.fetchone()[0]}")
        return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False


if __name__ == "__main__":
    test_connection()