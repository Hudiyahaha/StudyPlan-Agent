import json
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from app.arena import execute
from app.config import ROOT, settings
from app.llm import configured_models
from app.models import ArenaRequest, ArenaResponse, ChatRequest

log = logging.getLogger("studyplan.api")
router = APIRouter()


@router.get("/")
def index():
    return FileResponse(ROOT / "app/static/index.html")


@router.get("/health")
def health():
    return {
        "status": "ok",
        "implementation": "studyplan",
        "agent_name": "StudyPlan Agent",
        "domain": "Study-Plan Builder",
        "model_provider": settings.model_provider,
    }


@router.get("/arena/manifest")
def manifest():
    return json.loads((ROOT / "arena_manifest.json").read_text(encoding="utf-8"))


@router.get("/models")
def models():
    # Expose only configured, allowed model identifiers — never keys.
    return {"models": configured_models()}


@router.post("/arena/run", response_model=ArenaResponse)
async def arena_run(payload: ArenaRequest):
    model = settings.model_name or configured_models()[0]
    if settings.model_provider.lower() in {"", "unconfigured"}:
        model = "studyplan-mock-v1"
    elif settings.model_provider.lower() == "mock":
        model = "studyplan-mock-v1"
    return await execute(payload, model=model)


@router.post("/chat", response_model=ArenaResponse)
async def chat(payload: ChatRequest, request: Request):
    allowed = models()["models"]
    if payload.model not in allowed:
        raise HTTPException(400, "Model is not enabled")
    busy = request.app.state.busy
    if payload.session_id in busy:
        raise HTTPException(409, "This chat is already running")
    busy.add(payload.session_id)
    memory = request.app.state.memory
    try:
        result = await execute(payload, memory.get(payload.session_id), payload.model)
        memory.add(payload.session_id, payload.task, result.final_response)
        return result
    finally:
        busy.discard(payload.session_id)


@router.delete("/chat/{session_id}")
def reset(session_id: str, request: Request):
    if session_id in request.app.state.busy:
        raise HTTPException(409, "Chat is running")
    request.app.state.memory.clear(session_id)
    return {"status": "cleared"}
