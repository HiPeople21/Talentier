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
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None
    snippet: str
    matched_skills: list[str] = []
    avatar_initials: str = ""
    score: int = 0
    code_quality_score: Optional[int] = None
    summary: str = ""
    source: str = ""  # "linkedin" | "github"


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

class CodeQualityMetric(BaseModel):
    """Code quality grading for a commit."""
    commit_sha: str
    overall_score: float = Field(..., ge=0, le=100, description="Overall quality score 0-100")
    maintainability: float = Field(..., ge=0, le=100, description="Code structure and modularity grade")
    readability: float = Field(..., ge=0, le=100, description="Code clarity and documentation grade")
    test_coverage: float = Field(..., ge=0, le=100, description="Testing practices grade")
    complexity: float = Field(..., ge=0, le=100, description="Code simplicity grade (higher = simpler)")
    performance: float = Field(..., ge=0, le=100, description="Performance characteristics grade")
    security: float = Field(..., ge=0, le=100, description="Security practices grade")
    issues: List[str] = Field(default_factory=list, description="Critical issues identified")
    suggestions: List[str] = Field(default_factory=list, description="Grade justifications")


class CodeQualityResponse(BaseModel):
    """Response model for code quality analysis."""
    owner: str
    repo: str
    contributor: str
    metrics: List[CodeQualityMetric]
    total_analyzed: int
    success: bool
    error: Optional[str] = None
