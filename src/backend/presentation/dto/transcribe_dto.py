from pydantic import BaseModel, Field, field_validator

from src.backend.application.errors import AuthenticationError


class TranscribeDTO(BaseModel):
    diarization: bool = Field(False, validation_alias='diarize_speakers')
    token: str
    language: str | None = Field(default=None, validation_alias='lang')

    @field_validator('token')
    @classmethod
    def validate_token(cls, v: str):
        if not v:  # Проверка на пустую строку (e.g., token="")
            raise AuthenticationError("Токен не найден или пуст.")
        return v

