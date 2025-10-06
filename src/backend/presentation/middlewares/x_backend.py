from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

class XBackendMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp, hostname: str):
        super().__init__(app)
        self.hostname = hostname

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)
        response.headers["X-backend_node"] = self.hostname
        return response
