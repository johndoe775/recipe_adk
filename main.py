
import logging
import os

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from root_agent.agent import root_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_port() -> int:
    raw_port = os.environ.get("PORT", "8000")
    try:
        port = int(raw_port)
    except (TypeError, ValueError):
        return 8000

    return max(1, min(port, 65535))


APP_NAME = root_agent.name

app = FastAPI(title="Recipe Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://YOUR_USERNAME.github.io",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# Explicitly initialize ADK session storage and runner.
session_service = InMemorySessionService()

runner = Runner(
    agent=root_agent,
    app_name=APP_NAME,
    session_service=session_service,
)


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
    try:
        session = await session_service.get_session(
            app_name=APP_NAME,
            user_id=request.user_id,
            session_id=request.session_id,
        )

        if session is None:
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

        return ChatResponse(answer="\n".join(answer_parts))

    except Exception as exc:
        logger.exception("ADK chat request failed")
        raise HTTPException(
            status_code=500,
            detail="Failed to process the recipe request.",
        ) from exc


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=get_port(),
    )
