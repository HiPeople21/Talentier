"""Candidate search orchestration helpers."""

from __future__ import annotations

from typing import Any
from typing import Optional

from agent_events import mark_agent_done, push_agent_step, reset_agent_status
from github_candidate_search import search_github_candidates
from linkedin_search import search_linkedin_profiles
from models import Candidate, SearchResponse


def build_skill_terms(skills: Optional[str]) -> list[str]:
    """Build normalized skill terms."""
    if not skills:
        return []
    return [s.strip() for s in skills.split(",") if s.strip()]


def build_keywords(
    skills: Optional[str],
    description: Optional[str],
    experience_level: Optional[str],
) -> list[str]:
    """Build keyword list from query params."""
    keywords: list[str] = build_skill_terms(skills)
    if description and description.strip():
        keywords.append(description.strip())
    if experience_level and experience_level.strip():
        keywords.append(experience_level.strip())
    return keywords


def _dedupe_candidates(candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Deduplicate candidates across LinkedIn/GitHub and keep ordering."""
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()

    for candidate in candidates:
        linkedin_url = (candidate.get("linkedin_url") or "").strip().lower()
        github_url = (candidate.get("github_url") or "").strip().lower()
        name = (candidate.get("name") or "").strip().lower()

        if linkedin_url:
            key = f"li:{linkedin_url}"
        elif github_url:
            key = f"gh:{github_url}"
        else:
            key = f"nm:{name}"

        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
        if len(deduped) >= limit:
            break

    return deduped


def _merge_source_candidates(
    linkedin_candidates: list[dict[str, Any]],
    github_candidates: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """
    Merge source candidates while preserving GitHub representation.

    If GitHub candidates exist, reserve at least 30% (max 20) of the final list.
    """
    if limit <= 0:
        return []

    if not github_candidates:
        return _dedupe_candidates(linkedin_candidates, limit=limit)
    if not linkedin_candidates:
        return _dedupe_candidates(github_candidates, limit=limit)

    github_quota = min(max(limit // 3, 10), 20, len(github_candidates))
    linkedin_quota = max(0, limit - github_quota)

    primary_batch = linkedin_candidates[:linkedin_quota] + github_candidates[:github_quota]
    merged = _dedupe_candidates(primary_batch, limit=limit)
    if len(merged) >= limit:
        return merged

    overflow_pool = linkedin_candidates[linkedin_quota:] + github_candidates[github_quota:]
    merged = _dedupe_candidates(merged + overflow_pool, limit=limit)
    return merged


def search_candidates_with_status(
    skills: Optional[str],
    description: Optional[str],
    experience_level: Optional[str],
    location: Optional[str],
    page: int,
    limit: int = 50,
) -> SearchResponse:
    """Run LinkedIn + GitHub candidate sourcing and update agent status events."""
    skill_terms = build_skill_terms(skills)
    keywords = build_keywords(skills, description, experience_level)

    reset_agent_status()
    push_agent_step("start", "Starting candidate sourcing")
    push_agent_step(
        "thinking",
        "Building search query",
        f"Keywords: {', '.join(keywords) or 'none'}",
    )
    push_agent_step("searching", "Searching LinkedIn profiles")

    try:
        linkedin_profiles = search_linkedin_profiles(
            keywords=keywords,
            location=location or "",
            page=page,
            page_size=limit,
        )
    except Exception as exc:
        push_agent_step("error", "LinkedIn search failed", str(exc))
        mark_agent_done()
        raise

    push_agent_step("success", "LinkedIn search complete", f"Found {len(linkedin_profiles)} profiles")

    push_agent_step("searching", "Searching significant GitHub repositories")
    github_candidates, github_error = search_github_candidates(
        skill_terms=skill_terms,
        keywords=keywords,
        description=description or "",
        location=location or "",
        target_count=limit,
    )
    if github_error:
        push_agent_step("error", "GitHub sourcing issue", github_error)
    push_agent_step(
        "success",
        "GitHub sourcing complete",
        f"Found {len(github_candidates)} contributor leads",
    )

    merged_candidates = _merge_source_candidates(
        linkedin_candidates=linkedin_profiles,
        github_candidates=github_candidates,
        limit=limit,
    )

    push_agent_step(
        "success",
        "Candidate merge complete",
        f"Returning {len(merged_candidates)} candidates",
    )
    mark_agent_done()

    return SearchResponse(
        candidates=[Candidate(**p) for p in merged_candidates],
        total_results=len(merged_candidates),
        page=page,
        has_more=len(merged_candidates) == limit,
    )


def to_pipeline_response(search_response: SearchResponse) -> dict:
    """Map SearchResponse to frontend pipeline payload."""
    candidates = []
    for candidate in search_response.candidates:
        profile_urls = {}
        if candidate.linkedin_url:
            profile_urls["linkedin"] = candidate.linkedin_url
        if candidate.github_url:
            profile_urls["github"] = candidate.github_url
        if not profile_urls and candidate.profile_url:
            if "linkedin.com/" in candidate.profile_url:
                profile_urls["linkedin"] = candidate.profile_url
            elif "github.com/" in candidate.profile_url:
                profile_urls["github"] = candidate.profile_url
            else:
                profile_urls["profile"] = candidate.profile_url

        candidates.append(
            {
                "canonical_id": candidate.id,
                "name": candidate.name,
                "headline": candidate.headline,
                "location": candidate.location,
                "profile_urls": profile_urls,
                "skills": candidate.matched_skills,
                "snippet": candidate.snippet,
                "final_score": None,
                "rank": None,
                "num_sources": len(profile_urls),
            }
        )

    return {
        "candidates": candidates,
        "total": search_response.total_results,
        "errors": [],
        "metrics": {
            "total_duration_ms": 0,
            "stages": [
                {"stage": "linkedin_search", "duration_ms": 0},
                {"stage": "github_repo_search", "duration_ms": 0},
                {"stage": "github_contributor_sourcing", "duration_ms": 0},
                {"stage": "identity_merge", "duration_ms": 0},
            ],
        },
    }
