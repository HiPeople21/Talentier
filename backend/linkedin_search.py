"""LinkedIn profile search utilities."""

from __future__ import annotations

import hashlib
import logging
import re
from urllib.parse import urlparse, urlunparse

from ddgs import DDGS
from ddgs.exceptions import DDGSException, TimeoutException

logger = logging.getLogger(__name__)


def _clean_linkedin_url(url: str) -> str:
    """Normalize LinkedIn profile URL and strip query params."""
    if not url:
        return ""

    parsed = urlparse(url.strip())
    path = parsed.path.rstrip("/")
    cleaned = parsed._replace(path=path, params="", query="", fragment="")
    return urlunparse(cleaned)


def _extract_name_and_headline(title: str) -> tuple[str, str]:
    """Extract best-effort name/headline from DDGS title."""
    if not title:
        return "Unknown", ""

    normalized = re.sub(r"\s+", " ", title).strip()
    normalized = re.sub(r"\s*[\-|]\s*LinkedIn\s*$", "", normalized, flags=re.IGNORECASE)

    for sep in (" - ", " | ", " — ", " – "):
        if sep in normalized:
            left, right = normalized.split(sep, 1)
            return left.strip() or "Unknown", right.strip()

    return normalized, ""


def _avatar_initials(name: str) -> str:
    parts = [p for p in name.split() if p]
    return "".join(p[0].upper() for p in parts[:2])


def _run_ddgs_text_query(query: str, max_results: int) -> list[dict]:
    """Run DDGS text search with backend fallbacks."""
    raw_results: list[dict] = []
    backends = ["duckduckgo", "google", "yahoo", "brave", "auto"]
    last_error: Exception | None = None

    with DDGS() as ddgs:
        for backend in backends:
            try:
                raw_results = list(
                    ddgs.text(
                        query,
                        max_results=max_results,
                        backend=backend,
                    )
                )
                if raw_results:
                    return raw_results
            except (DDGSException, TimeoutException) as exc:
                last_error = exc
                continue
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue

    if last_error is not None:
        logger.warning(
            "LinkedIn search failed across DDGS backends for query '%s': %s",
            query,
            last_error,
        )
    return []


def find_linkedin_profile_for_identity(
    identity: str,
    keywords: list[str],
    location: str = "",
) -> dict | None:
    """Find the best LinkedIn profile match for a specific person identity."""
    identity = (identity or "").strip()
    if not identity:
        return None

    keyword_part = " ".join(k.strip() for k in keywords[:3] if k.strip())
    search_terms = " ".join(part for part in [f"\"{identity}\"", keyword_part, location.strip()] if part)
    query = f"site:linkedin.com/in {search_terms}".strip()
    results = _run_ddgs_text_query(query=query, max_results=8)

    for item in results:
        href = item.get("href", "")
        cleaned_url = _clean_linkedin_url(href)
        if "linkedin.com/in/" not in cleaned_url.lower():
            continue
        name, headline = _extract_name_and_headline(item.get("title", ""))
        return {
            "name": name,
            "headline": headline,
            "profile_url": cleaned_url,
            "linkedin_url": cleaned_url,
            "snippet": (item.get("body") or "").strip(),
            "avatar_initials": _avatar_initials(name),
        }

    return None


def search_linkedin_profiles(
    keywords: list[str],
    location: str = "",
    page: int = 1,
    page_size: int = 20,
) -> list[dict]:
    """
    Search LinkedIn profile pages (`linkedin.com/in/*`) via DDGS.

    Returns paginated, de-duplicated profile records.
    """
    page = max(page, 1)
    page_size = max(page_size, 1)
    # Over-fetch to offset non-profile hits and deduplication losses.
    fetch_count = max(200, page * page_size * 4)

    keyword_query = " ".join(k.strip() for k in keywords if k.strip())
    search_terms = " ".join(part for part in [keyword_query, location.strip()] if part)
    query = f"site:linkedin.com/in {search_terms}".strip()
    raw_results = _run_ddgs_text_query(query=query, max_results=fetch_count)
    if not raw_results:
        return []

    seen_urls: set[str] = set()
    profiles: list[dict] = []

    for item in raw_results:
        href = item.get("href", "")
        cleaned_url = _clean_linkedin_url(href)
        if "linkedin.com/in/" not in cleaned_url.lower():
            continue
        if cleaned_url.lower() in seen_urls:
            continue
        seen_urls.add(cleaned_url.lower())

        name, headline = _extract_name_and_headline(item.get("title", ""))
        snippet = (item.get("body") or "").strip()
        profile_id = hashlib.md5(cleaned_url.encode("utf-8")).hexdigest()[:12]

        profiles.append(
            {
                "id": profile_id,
                "name": name,
                "headline": headline,
                "location": location,
                "profile_url": cleaned_url,
                "linkedin_url": cleaned_url,
                "github_url": None,
                "snippet": snippet,
                "matched_skills": [k for k in keywords if k.strip()],
                "avatar_initials": _avatar_initials(name),
            }
        )

    start = (page - 1) * page_size
    end = start + page_size
    return profiles[start:end]
