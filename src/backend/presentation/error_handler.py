from functools import partial
from typing import cast

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette import status as code

from src.backend.application.errors import AuthenticationError, UnsupportedAudioFormatError
from src.backend.domain.errors import Error, DoesNotExists, AlreadyExists, LargeFileError


class HTTPError(Error): ...


async def validate(_: Request, error: Exception, status: int) -> JSONResponse:
    error = cast(HTTPError, error)
    return JSONResponse(content={"message": error.message}, status_code=status)


def init_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(
        AuthenticationError,
        partial(validate, status=code.HTTP_401_UNAUTHORIZED),
    )
    app.add_exception_handler(
        DoesNotExists,
        partial(validate, status=code.HTTP_404_NOT_FOUND),
    )
    app.add_exception_handler(
        AlreadyExists,
        partial(validate, status=code.HTTP_409_CONFLICT),
    )
    app.add_exception_handler(
        LargeFileError, partial(validate, status=code.HTTP_413_REQUEST_ENTITY_TOO_LARGE)
    )
    app.add_exception_handler(
        UnsupportedAudioFormatError,
        partial(validate, status=code.HTTP_415_UNSUPPORTED_MEDIA_TYPE)
    )

