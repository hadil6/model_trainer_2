"""FastAPI app factory."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import chat, datasets, eval, gpu, notifications, reports, runs, training_images


def create_app() -> FastAPI:
    app = FastAPI(title="model_trainer API", version="2.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat.router)
    app.include_router(runs.router)
    app.include_router(datasets.router)
    app.include_router(eval.router)
    app.include_router(reports.router)
    app.include_router(training_images.router)
    app.include_router(gpu.router)
    app.include_router(notifications.router)

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    return app
