from fastapi import FastAPI
from core.database import engine, Base
from api.chat import router as chat_router

app = FastAPI()

Base.metadata.create_all(bind=engine)

app.include_router(chat_router)


@app.get("/")
def root():
    return {"message": "API running"}