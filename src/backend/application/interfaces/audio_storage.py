from abc import abstractmethod
from typing import Protocol, BinaryIO


class AudioStorage(Protocol):

    @abstractmethod
    def upload(self, audio_file: BinaryIO, filename: str) -> str: ...