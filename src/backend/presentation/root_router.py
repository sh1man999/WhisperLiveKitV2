from fastapi import APIRouter
from starlette import status
from starlette.responses import HTMLResponse
from whisperlivekit import get_inline_ui_html

from src.backend.presentation.dto.health_check_dto import HealthCheckResponse
from src.backend.presentation.endpoints.transcribe import transcribe_router
from whisperlivekit.web.web_interface import get_inline_stream_ui_html

root_router = APIRouter()


@root_router.get(
    "/health",
    tags=["healthcheck"],
    summary="Perform a Health Check",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
    response_model=HealthCheckResponse,
)
def healthcheck() -> HealthCheckResponse:
    """
    ## Perform a Health Check
    Endpoint to perform a healthcheck on. This endpoint can primarily be used Docker
    to ensure a robust container orchestration and management is in place. Other
    services which rely on proper functioning of the API service will not deploy if this
    endpoint returns any other HTTP status code except 200 (OK).
    Returns:
        HealthCheckResponse: Returns a JSON response with the health status
    """
    return HealthCheckResponse(status="OK")


@root_router.get("/")
async def get():
    return HTMLResponse(get_inline_ui_html())

@root_router.get("/stream")
async def get_url_page():
    return HTMLResponse(get_inline_stream_ui_html())

root_sub_routers = (transcribe_router,)

for router in root_sub_routers:
    root_router.include_router(router)
