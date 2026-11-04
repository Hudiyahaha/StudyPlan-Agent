import json
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from app.arena import execute
from app.config import ROOT, settings
from app.llm import configured_models, default_run_model, live_credentials_available
from app.models import ArenaRequest, ArenaResponse, ChatRequest

log = logging.getLogger("studyplan.api")
router = APIRouter()


@router.get("/")
def index():
    return FileResponse(ROOT / "app/static/index.html")


@router.get("/health")
def health():
    provider = settings.model_provider
    live = live_credentials_available()
    return {
        "status": "ok",
        "implementation": "studyplan",
        "agent_name": "StudyPlan Agent",
        "domain": "Study-Plan Builder",
        "model_provider": provider,
        "live_llm": live,
        "active_model": default_run_model(),
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
    return await execute(payload, model=default_run_model())


@router.post("/chat", response_model=ArenaResponse)
async def chat(payload: ChatRequest, request: Request):
    model = default_run_model() if payload.model == "unconfigured" else payload.model
    if model != default_run_model() and model not in models()["models"]:
        raise HTTPException(400, "Model is not enabled")
    busy = request.app.state.busy
    if payload.session_id in busy:
        raise HTTPException(409, "This chat is already running")
    busy.add(payload.session_id)
    memory = request.app.state.memory
    try:
        result = await execute(payload, memory.get(payload.session_id), model)
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
