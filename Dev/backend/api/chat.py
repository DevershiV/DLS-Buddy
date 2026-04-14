from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from core.database import SessionLocal
from schemas.chat import ChatRequest, ChatResponse
from services.chat_service import handle_chat

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    response, conversation_id = handle_chat(
        db, request.message, request.conversation_id
    )

    return ChatResponse(
        response=response,
        conversation_id=conversation_id
    )