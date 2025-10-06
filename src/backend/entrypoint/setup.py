import pathlib
import warnings
from collections.abc import Iterable
import logging
import os

from concurrent_log_handler import ConcurrentTimedRotatingFileHandler
from dishka import (
    Provider,
    AsyncContainer,
    make_async_container,
)
from fastapi import APIRouter, FastAPI
from starlette.middleware.gzip import GZipMiddleware
from starlette.staticfiles import StaticFiles

from src.backend.entrypoint.config import Config
from src.backend.presentation.error_handler import init_error_handlers
from src.backend.presentation.middlewares.x_backend import XBackendMiddleware


def create_app(lifespan, config: Config) -> FastAPI:
    docs_url = "/docs"
    if not config.debug:
        docs_url = "/n44444uHzE"
    return FastAPI(lifespan=lifespan, docs_url=docs_url)

def create_container(providers: Iterable[Provider], config: Config) -> AsyncContainer:
    return make_async_container(*providers, context={Config: config})

def configure_app(app: FastAPI, root_router: APIRouter, config: Config) -> None:
    app.include_router(root_router)
    app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=5)
    app.add_middleware(XBackendMiddleware, config.hostname)
    import whisperlivekit.web as webpkg
    web_dir = pathlib.Path(webpkg.__file__).parent
    app.mount("/web", StaticFiles(directory=str(web_dir)), name="web")
    init_error_handlers(app)


def configure_logging(config: Config, level=logging.DEBUG) -> None:
    warnings.filterwarnings("ignore", module="pyannote.audio.models.blocks.pooling")
    warnings.filterwarnings("ignore", module="pyannote.audio.utils.reproducibility")
    warnings.filterwarnings("ignore", module="sklearn")
    log_format = (
        "[%(asctime)s.%(msecs)03d] [%(process)d] [%(levelname)-5s]"
        "[%(module)-30s -> %(funcName)-30s:%(lineno)-3d] - %(message)s"
    )
    datefmt = "%Y-%m-%d %H:%M:%S"

    handlers = []

    stream_handler = logging.StreamHandler()
    handlers.append(stream_handler)

    if config.log_files_path:
        log_filename = os.path.join(config.log_files_path, "asr.log")
        rotating_file_handler = ConcurrentTimedRotatingFileHandler(
            log_filename, when="midnight", interval=1, backupCount=90, encoding="utf-8"
        )
        handlers.append(rotating_file_handler)

    logging.basicConfig(
        level=level,
        datefmt=datefmt,
        format=log_format,
        handlers=handlers,
    )
    logging.getLogger("python_multipart.multipart").setLevel(logging.INFO)
    logging.getLogger("pyannote").setLevel(logging.ERROR)
    logging.getLogger('numba').setLevel(logging.ERROR)

