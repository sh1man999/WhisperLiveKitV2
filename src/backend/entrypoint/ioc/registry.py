from typing import Iterable

from dishka import Provider
from dishka.integrations.fastapi import FastapiProvider

from src.backend.entrypoint.ioc.adapters import ConfigProvider, StorageProvider, TranscriptionProvider
from src.backend.entrypoint.ioc.interactors import InteractorProvider


def get_providers() -> Iterable[Provider]:
    return (
        FastapiProvider(),
        ConfigProvider(),
        StorageProvider(),
        TranscriptionProvider(),
        InteractorProvider(),
    )
