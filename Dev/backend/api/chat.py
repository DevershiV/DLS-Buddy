from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from core.database import SessionLocal
from schemas.chat import ChatRequest, ChatResponse
from services.chat_service import handle_chat
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    logger.info(f"Incoming message: {request.message}")

    try:
        response, conversation_id = await handle_chat(
            db, request.message, request.conversation_id
        )
        logger.info(f"Response generated for conversation {conversation_id}")
        return ChatResponse(
            response=response,
            conversation_id=conversation_id
        )
    except Exception as e:
        logger.error(f"Chat failed: {str(e)}")
        raise
