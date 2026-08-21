from typing import Any

from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from src.agent.graph import graph


app = FastAPI(title="Customer Support Agent")


class ChatRequest(BaseModel):
    message: str
    thread_id: str


class ChatResponse(BaseModel):
    response: str | None = None
    approval_required: bool = False
    interrupt: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="healthy")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    config = {"configurable": {"thread_id": request.thread_id}}
    result = graph.invoke(
        {"messages": [HumanMessage(content=request.message)]},
        config=config,
    )

    if result.get("__interrupt__"):
        interrupt_payload = result["__interrupt__"][0].value
        return ChatResponse(
            approval_required=True,
            interrupt=interrupt_payload,
        )

    newest_response = result["messages"][-1].content
    return ChatResponse(response=str(newest_response))
