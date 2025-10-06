from pydantic import BaseModel


class HealthCheckResponse(BaseModel):
    """Response model to validate and return when performing a health check."""

    status: str = "OK"