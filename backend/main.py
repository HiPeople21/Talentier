"""FastAPI backend for the Recruiter Candidate Finder."""

from typing import Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

from models import SearchResponse
from scraper import SearchFilters, search_linkedin_profiles
from github_routes import router as github_router

app = FastAPI(
    title="Recruiter Candidate Finder",
    description="Find potential candidates on LinkedIn based on technical skills and filters",
    version="1.0.0",
)

# Include GitHub routes
app.include_router(github_router)

# Allow the React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/candidates", response_model=SearchResponse)
async def search_candidates(
    skills: Optional[str] = Query(None, description="Comma-separated list of skills"),
    experience_level: Optional[str] = Query(None, description="junior, mid, senior, lead, principal"),
    location: Optional[str] = Query(None, description="Location filter"),
    page: int = Query(1, ge=1, description="Page number"),
):
    """Search for candidates on LinkedIn based on filters."""
    skill_list = [s.strip() for s in skills.split(",") if s.strip()] if skills else []

    filters = SearchFilters(
        skills=skill_list,
        experience_level=experience_level,
        location=location,
        page=page,
    )

    candidates, has_more = await search_linkedin_profiles(filters)

    return SearchResponse(
        candidates=candidates,
        total_results=len(candidates),
        page=page,
        has_more=has_more,
    )


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "services": {
            "linkedin": "active",
            "github": "active"
        }
    }
