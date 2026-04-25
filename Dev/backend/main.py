# uvicorn main:app --reload
from fastapi import FastAPI
from core.database import engine, Base
from api.chat import router as chat_router
from core.logging import setup_logging
from dotenv import load_dotenv
load_dotenv()

setup_logging()

app = FastAPI()

Base.metadata.create_all(bind=engine)

app.include_router(chat_router)


@app.get("/")
def root():
    return {"message": "API running"}