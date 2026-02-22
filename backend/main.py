"""FastAPI backend for the Recruiter Candidate Finder."""

import asyncio
import json
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from models import SearchResponse
from scraper import agent_status
from github import router as github_router

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
    description: Optional[str] = Query(None, description="Free-text descriptors for AI matching"),
    page: int = Query(1, ge=1, description="Page number"),
):
    """
    Search for candidates using the unified pipeline.

    Runs the full multi-source LangGraph pipeline (LinkedIn + GitHub)
    and returns results in the SearchResponse format.
    """
    from langgraph_pipeline import run_candidate_sourcing_pipeline
    from models import Candidate

    skill_list = [s.strip() for s in skills.split(",") if s.strip()] if skills else []
    tag_list = [s.lower().replace(" ", "-") for s in skill_list]

    result = await asyncio.to_thread(
        run_candidate_sourcing_pipeline,
        skills=skill_list,
        location=location or "",
        description=description or "",
        github_tags=tag_list,
        top_k=50,
    )

    import hashlib, re

    def _get_initials(name: str) -> str:
        parts = name.split()
        return "".join(p[0].upper() for p in parts[:2] if p)

    candidates = []
    for c in result["candidates"]:
        cid = hashlib.md5(c.name.encode()).hexdigest()[:12]
        best_url = ""
        for src in ("linkedin", "github"):
            url = c.profile_urls.get(src, "")
            if url:
                best_url = url
                break
        if not best_url:
            best_url = next(iter(c.profile_urls.values()), "")

        candidates.append(Candidate(
            id=cid,
            name=c.name,
            headline=c.headline or "",
            location=c.location or "",
            profile_url=best_url,
            snippet="",
            matched_skills=sorted(c.all_skills)[:6],
            avatar_initials=_get_initials(c.name),
        ))

    return SearchResponse(
        candidates=candidates,
        total_results=len(candidates),
        page=page,
        has_more=False,
    )


@app.post("/api/pipeline")
async def run_pipeline(
    skills: Optional[str] = Query(None, description="Comma-separated list of skills"),
    experience_level: Optional[str] = Query(None, description="junior, mid, senior, lead, principal"),
    location: Optional[str] = Query(None, description="Location filter"),
    description: Optional[str] = Query(None, description="Free-text descriptors for AI matching"),
    github_tags: Optional[str] = Query(None, description="Comma-separated GitHub topic tags"),
    top_k: int = Query(50, ge=1, le=200, description="Number of top candidates to return"),
):
    """
    Run the full multi-source candidate sourcing pipeline.

    This runs LinkedIn discovery (via the scraper agent) and GitHub contributor
    analysis in parallel, then merges, deduplicates, and ranks all candidates
    using the global CandidateSourcingState.
    """
    from langgraph_pipeline import run_candidate_sourcing_pipeline

    skill_list = [s.strip() for s in skills.split(",") if s.strip()] if skills else []
    tag_list = (
        [t.strip() for t in github_tags.split(",") if t.strip()]
        if github_tags
        else [s.lower().replace(" ", "-") for s in skill_list]
    )

    result = await asyncio.to_thread(
        run_candidate_sourcing_pipeline,
        skills=skill_list,
        location=location or "",
        description=description or "",
        github_tags=tag_list,
        top_k=top_k,
    )

    # Serialise the ranked candidates into a JSON-friendly format
    ranked = []
    for c in result["candidates"]:
        sources = {s.value: sc.profile_url for s, sc in c.source_candidates.items()}
        ranked.append({
            "canonical_id": c.canonical_id,
            "name": c.name,
            "email": c.email,
            "headline": c.headline,
            "location": c.location,
            "profile_urls": sources,
            "skills": sorted(c.all_skills),
            "final_score": round(c.final_score, 4) if c.final_score else None,
            "rank": c.rank,
            "num_sources": len(c.source_candidates),
        })

    return {
        "candidates": ranked,
        "total": len(ranked),
        "errors": [
            {"stage": str(e.get("stage", "")), "message": e.get("message", "")}
            for e in result.get("errors", [])
        ],
        "metrics": {
            "total_duration_ms": sum(
                m.duration_ms for m in result["metrics"] if m.duration_ms
            ),
            "stages": [
                {"stage": m.stage.value, "duration_ms": round(m.duration_ms or 0)}
                for m in result["metrics"]
            ],
        },
    }


@app.get("/api/agent-status")
async def get_agent_status():
    """SSE endpoint streaming the agent's thought process in real-time."""
    async def event_stream():
        last_index = 0
        while True:
            steps = agent_status.get("steps", [])
            while last_index < len(steps):
                step = steps[last_index]
                yield f"data: {json.dumps(step)}\n\n"
                last_index += 1

            if agent_status.get("done", False):
                yield f"data: {json.dumps({'type': 'done', 'message': 'Search complete'})}\n\n"
                break

            await asyncio.sleep(0.3)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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
