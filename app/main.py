from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api import router
from app.config import ROOT
from app.memory import Memory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.memory = Memory()
    app.state.busy = set()
    yield


app = FastAPI(title="StudyPlan Agent — Agent Arena", lifespan=lifespan)
app.include_router(router)
app.mount("/static", StaticFiles(directory=ROOT / "app/static"), name="static")
