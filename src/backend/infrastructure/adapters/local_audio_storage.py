import os
from typing import BinaryIO

import logging

from src.backend.application.interfaces.audio_storage import AudioStorage
from src.backend.entrypoint.config import Config

logger = logging.getLogger(__name__)

class LocalAudioStorage(AudioStorage):
    def __init__(self, config: Config):
        self._base_path = config.audio_storage_path
        os.makedirs(self._base_path, exist_ok=True)

    def upload(self, audio_file: BinaryIO, filename: str) -> str:
        file_path = os.path.join(self._base_path, filename)
        with open(file_path, 'wb') as afp:
            afp.write(audio_file.read())
        logger.info(f'Audio file {file_path} has been uploaded')
        return str(file_path)
