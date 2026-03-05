"""
Pydantic schemas for API request/response and internal data structures.
"""
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request body for POST /ask."""
    question: str = Field(..., min_length=1, max_length=2000, description="User's legal question")


class SourceRef(BaseModel):
    """Reference to a source document and page."""
    document: str = Field(..., description="Source filename (e.g. tck.pdf)")
    page: int = Field(..., ge=0, description="Page number in the document")


class AskResponse(BaseModel):
    """Response body for POST /ask."""
    answer: str = Field(..., description="Model answer grounded in context")
    sources: list[SourceRef] = Field(default_factory=list, description="Cited documents and pages")
