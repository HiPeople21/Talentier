"""Pydantic models for the recruiter candidate finder API."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict


# ============================================================================
# Recruiter Candidate Finder Models
# ============================================================================

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


# ============================================================================
# GitHub Tools Models
# ============================================================================

class Repository(BaseModel):
    """Repository information model."""
    name: str
    full_name: str
    url: str = Field(alias="html_url")
    description: Optional[str] = None
    stars: int = Field(alias="stargazers_count")
    forks: int = Field(alias="forks_count")
    language: Optional[str] = None
    created_at: str
    updated_at: str
    
    class Config:
        populate_by_name = True


class RepositorySearchResponse(BaseModel):
    """Response model for repository search."""
    tag: str
    language: Optional[str] = None
    min_stars: int
    repositories: List[Dict]
    total_found: int
    success: bool
    error: Optional[str] = None


class ContributorInfo(BaseModel):
    """Contributor information model."""
    login: str
    contributions: int
    avatar_url: Optional[str] = None
    html_url: Optional[str] = None


class ContributorsResponse(BaseModel):
    """Response model for contributors."""
    owner: str
    repo: str
    contributors: List[ContributorInfo]
    total_contributors: int
    total_commits: int
    success: bool
    error: Optional[str] = None


class CommitInfo(BaseModel):
    """Single commit information."""
    sha: str
    message: str
    author: str
    date: str
    url: str
    diff: Optional[str] = None
    files_changed: int = 0
    additions: int = 0
    deletions: int = 0


class ContributorCommitsResponse(BaseModel):
    """Response model for contributor commits."""
    owner: str
    repo: str
    contributor: str
    commits: List[CommitInfo]
    total_commits: int
    success: bool
    error: Optional[str] = None
