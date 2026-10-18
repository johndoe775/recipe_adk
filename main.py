import logging
import os
import sys

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configure logging before importing the ADK agent so import-time errors are visible.
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("recipe_assistant")
logger.info("Starting Recipe Assistant API imports")

try:
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types
    from root_agent.agent import root_agent
except Exception:
    logger.exception("Failed while importing ADK dependencies or root_agent.agent")
    raise

logger.info("Imports completed successfully")

APP_NAME = root_agent.name
app = FastAPI(title="Recipe Assistant API")

# Replace YOUR_USERNAME with the GitHub username/organization hosting the Pages site.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://johndoe775.github.io/recipe_adk/",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

logger.info("Creating ADK session service")
session_service = InMemorySessionService()

logger.info("Initializing ADK Runner for agent %s", APP_NAME)
try:
    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )
except Exception:
    logger.exception("Failed to initialize ADK Runner")
    raise

logger.info("ADK Runner initialized; creating API routes")


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=100)
    session_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=5000)


class ChatResponse(BaseModel):
    answer: str


@app.get("/")
async def root():
    return {"status": "ok", "service": "recipe-assistant"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    logger.info(
        "Received chat request user_id=%s session_id=%s message_length=%d",
        request.user_id,
        request.session_id,
        len(request.message),
    )
    try:
        session = await session_service.get_session(
            app_name=APP_NAME,
            user_id=request.user_id,
            session_id=request.session_id,
        )

        if session is None:
            logger.info("Creating new ADK session")
            await session_service.create_session(
                app_name=APP_NAME,
                user_id=request.user_id,
                session_id=request.session_id,
            )

        message = types.Content(
            role="user",
            parts=[types.Part(text=request.message)],
        )

        answer_parts = []
        async for event in runner.run_async(
            user_id=request.user_id,
            session_id=request.session_id,
            new_message=message,
        ):
            if event.is_final_response() and event.content:
                answer_parts.extend(
                    part.text
                    for part in (event.content.parts or [])
                    if part.text
                )

        answer = "\n".join(answer_parts).strip()
        logger.info("Chat request completed; answer_length=%d", len(answer))
        return ChatResponse(answer=answer)

    except Exception as exc:
        logger.exception("ADK chat request failed")
        raise HTTPException(
            status_code=500,
            detail="Failed to process the recipe request. Check server logs.",
        ) from exc


if __name__ == "__main__":
    # Render sets PORT automatically. 10000 is only a local fallback.
    port = int(os.environ.get("PORT", "10000"))
    logger.info("Starting Uvicorn on 0.0.0.0:%d", port)
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        access_log=True,
    )
