"""FastAPI backend for recruiter candidate search."""

from __future__ import annotations

import json
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from github import router as github_router
from models import SearchFilters
from scraper import stream_candidate_search

app = FastAPI(
    title="Recruiter Candidate Finder",
    description="Find candidates via LinkedIn profile search and GitHub contributor sourcing",
    version="1.0.0",
)

# Include GitHub routes
app.include_router(github_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/candidates/stream")
async def stream_candidates(
    skills: Optional[str] = Query(None, description="Comma-separated list of skills"),
    experience_level: Optional[str] = Query(None, description="junior, mid, senior, lead, principal"),
    location: Optional[str] = Query(None, description="Location filter"),
    description: Optional[str] = Query(None, description="Free-text description of ideal candidate"),
):
    """SSE endpoint streaming search progress and final results."""
    filters = SearchFilters(
        skills=[s.strip() for s in skills.split(",") if s.strip()] if skills else [],
        experience_level=experience_level,
        location=location,
        description=description,
    )

    async def event_generator():
        async for event in stream_candidate_search(filters):
            yield f"data: {json.dumps(event)}\n\n"
        yield "data: {\"type\": \"done\"}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "services": {"linkedin": "active", "github": "active"},
    }
