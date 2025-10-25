from contextlib import asynccontextmanager

import uvicorn
from dishka import AsyncContainer
from dishka.integrations.fastapi import setup_dishka
from fastapi import FastAPI
from whisperlivekit import TranscriptionEngine

from src.backend.entrypoint.ioc.registry import get_providers
from src.backend.entrypoint.config import create_config
from src.backend.entrypoint.setup import create_app, configure_app, configure_logging, create_container
from src.backend.presentation.root_router import root_router
from whisperlivekit.silero_vad_iterator import VADOnnxWrapper


def make_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        container = app.state.dishka_container
        await container.get(TranscriptionEngine) # Данная строка позволяет инициализировать заранее модели
        await container.get(VADOnnxWrapper)
        yield
        await app.state.dishka_container.close()

    config = create_config()

    app: FastAPI = create_app(lifespan=lifespan, config=config)
    configure_app(app=app, root_router=root_router, config=config)
    base_container: AsyncContainer = create_container(
        providers=(*get_providers(),),
        config=config
    )

    setup_dishka(container=base_container, app=app)
    configure_logging(config, config.log_level)

    return app


if __name__ == '__main__':
    uvicorn.run("run:make_app", log_level="info", factory=True)