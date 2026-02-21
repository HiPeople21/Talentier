"""Pydantic models for the recruiter candidate finder API."""

from pydantic import BaseModel
from typing import Optional


class SearchFilters(BaseModel):
    """Filters used to search for candidates."""
    skills: list[str] = []
    experience_level: Optional[str] = None  # junior, mid, senior, lead, principal
    location: Optional[str] = None
    description: Optional[str] = None  # free-text descriptors for AI matching
    page: int = 1


class Candidate(BaseModel):
    """A candidate parsed from LinkedIn search results."""
    id: str
    name: str
    headline: str
    location: str
    profile_url: str
    snippet: str
    matched_skills: list[str] = []
    avatar_initials: str = ""


class SearchResponse(BaseModel):
    """Response from the candidate search endpoint."""
    candidates: list[Candidate]
    total_results: int
    page: int
    has_more: bool
