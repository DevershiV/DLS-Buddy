from sqlalchemy.orm import Session
from models.chat import Conversation, Message


def handle_chat(db: Session, message: str, conversation_id: int | None):

    # Create conversation if needed
    if not conversation_id:
        convo = Conversation()
        db.add(convo)
        db.commit()
        db.refresh(convo)
        conversation_id = convo.id

    # Save user message
    user_msg = Message(
        conversation_id=conversation_id,
        role="user",
        content=message
    )
    db.add(user_msg)

    # Generate response (dummy for now)
    response_text = f"Echo: {message}"

    # Save assistant message
    bot_msg = Message(
        conversation_id=conversation_id,
        role="assistant",
        content=response_text
    )
    db.add(bot_msg)

    db.commit()

    return response_text, conversation_id