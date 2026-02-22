"""
Candidate search pipeline:
  1. Ollama extracts keywords from free-text description
  2. LinkedIn: single DDG query (no concurrent requests = no rate limiting)
  3. GitHub: repos → contributors → commit analysis via Ollama → code quality score
  4. All candidates scored 1-100, sorted, and returned
"""

import asyncio
import hashlib
import json
import re
import time
from urllib.parse import urlparse

import requests
from ddgs import DDGS

from github.tools import get_github_tools
from models import Candidate, SearchFilters

# ── Config ────────────────────────────────────────────────────────────
CACHE_TTL = 300
DDG_TIMEOUT = 50          # DDG can take 30-45s; give it room
GITHUB_TIMEOUT = 180      # Includes Ollama commit analysis time
OLLAMA_TIMEOUT = 30
MAX_DDG_RESULTS = 50
MAX_GITHUB_REPOS = 2
MAX_GITHUB_CONTRIBS = 8   # Per repo — keeps analysis within timeout
MAX_COMMITS_PER_CONTRIB = 2
OLLAMA_MODEL = "llama3"
OLLAMA_URL = "http://localhost:11434/api/generate"

_cache: dict[str, tuple[float, list[Candidate]]] = {}


# ── Ollama helpers ────────────────────────────────────────────────────
def _ollama(prompt: str, timeout: int = OLLAMA_TIMEOUT) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def _extract_keywords(description: str) -> list[str]:
    print("[Ollama] Extracting keywords from description...")
    prompt = (
        "Extract the most important technical skills and job-related keywords from this "
        "job description. Return ONLY a comma-separated list of 3-8 terms, no explanation.\n\n"
        f"Description: {description}"
    )
    try:
        raw = _ollama(prompt)
        keywords = [
            re.sub(r"^[\-\*\"\'\`\s]+|[\-\*\"\'\`\s]+$", "", k).strip()
            for k in raw.split(",")
        ]
        keywords = [k for k in keywords if k and len(k) < 50]
        print(f"[Ollama] Keywords: {keywords}")
        return keywords
    except Exception as e:
        print(f"[Ollama] Keyword extraction failed: {e}")
        return []


def _analyze_commits(login: str, owner: str, repo: str, commits: list) -> tuple[int, str]:
    """Score a developer's commit quality via Ollama. Returns (score 0-100, summary)."""
    if not commits:
        return 50, f"Contributor to {owner}/{repo} — no commit data available."

    commits_text = ""
    for c in commits[:MAX_COMMITS_PER_CONTRIB]:
        diff_preview = (c.diff or "")[:600]
        commits_text += f"\nMessage: {c.message}\nDiff:\n{diff_preview}\n---"

    prompt = (
        f"Review GitHub commits for @{login} on {owner}/{repo}.\n\n"
        f"{commits_text}\n\n"
        "Score this developer from 0-100 considering: code quality, naming conventions, "
        "commit message clarity, code structure, best practices.\n"
        'Return ONLY valid JSON: {"score": 75, "summary": "One sentence about this developer."}'
    )
    try:
        raw = _ollama(prompt)
        m = re.search(r'\{[^{}]*"score"[^{}]*\}', raw, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            score = max(1, min(100, int(data.get("score", 50))))
            summary = data.get("summary", f"Contributor to {owner}/{repo}.")
            return score, summary
    except Exception as e:
        print(f"[Ollama] Commit analysis failed for @{login}: {e}")
    return 50, f"Active contributor to {owner}/{repo} with {len(commits)} analyzed commits."


# ── Helpers ───────────────────────────────────────────────────────────
def _get_initials(name: str) -> str:
    words = name.strip().split()
    if len(words) >= 2:
        return (words[0][0] + words[-1][0]).upper()
    elif words:
        return words[0][0].upper()
    return "?"


def _canonicalize_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    host = parsed.netloc.lower().replace("www.", "")
    path = re.sub(r"/+", "/", parsed.path).rstrip("/")
    return f"{host}{path}"


def _build_linkedin_query(filters: SearchFilters) -> str:
    skills = filters.skills[:2]
    location = (filters.location or "").strip()
    level = (filters.experience_level or "").strip().lower()
    exp_prefix = {"senior": "senior", "lead": "lead", "principal": "principal"}.get(level, "")
    terms = " ".join(skills) if skills else "software engineer"
    base = f"{exp_prefix} {terms}".strip() if exp_prefix else terms
    if location:
        return f"site:linkedin.com/in {base} {location}"
    return f"site:linkedin.com/in {base}"


_GITHUB_TAG_MAP = {
    "node.js": "nodejs", "node": "nodejs", "vue.js": "vuejs", "react.js": "reactjs",
    "next.js": "nextjs", "nuxt.js": "nuxtjs", "express.js": "expressjs",
    "three.js": "threejs", "c#": "csharp", "c++": "cpp", "f#": "fsharp",
    "machine learning": "machine-learning", "deep learning": "deep-learning",
    "natural language processing": "nlp", "computer vision": "computer-vision",
    "data science": "data-science", "ci/cd": "ci-cd", "graphql": "graphql",
}


def _skill_to_github_tag(skill: str) -> str:
    lowered = skill.lower().strip()
    return _GITHUB_TAG_MAP.get(lowered, re.sub(r"[^a-z0-9\-]", "-", lowered).strip("-"))


# ── LinkedIn search ───────────────────────────────────────────────────
def _search_linkedin(query: str) -> list[dict]:
    print(f"[LinkedIn] Query: {query}")
    try:
        with DDGS(timeout=DDG_TIMEOUT) as ddgs:
            raw = ddgs.text(query, max_results=MAX_DDG_RESULTS)
            results = list(raw) if raw else []
            print(f"[LinkedIn] Got {len(results)} results")
        return [
            {"title": r.get("title", ""), "href": r.get("href", ""), "snippet": r.get("body", "")}
            for r in results
        ]
    except Exception as e:
        print(f"[LinkedIn] Failed: {e}")
        return []


def _score_linkedin(matched_skills: list[str], total_skills: int, exp_level: str,
                    location: str, required_exp: str, required_loc: str) -> tuple[int, str]:
    """Score a LinkedIn candidate 1-100 using a deterministic formula."""
    skill_score = int((len(matched_skills) / max(total_skills, 1)) * 50)

    exp_score = 0
    if required_exp and required_exp.lower() in exp_level.lower():
        exp_score = 20

    loc_score = 0
    if required_loc and required_loc.lower() in location.lower():
        loc_score = 20

    quality_score = 10  # base

    score = max(1, min(100, skill_score + exp_score + loc_score + quality_score))

    reasons = []
    if matched_skills:
        reasons.append(f"matches {len(matched_skills)} required skill(s): {', '.join(matched_skills[:3])}")
    if exp_score:
        reasons.append(f"{required_exp} experience level")
    if loc_score:
        reasons.append(f"based in {required_loc}")
    summary = "LinkedIn profile — " + ("; ".join(reasons) if reasons else "partial skill match") + "."
    return score, summary


def _parse_linkedin_result(result: dict, search_skills: list[str], exp_level: str, location: str) -> Candidate | None:
    url = result.get("href", "")
    title = result.get("title", "")
    snippet = result.get("snippet", "")

    if "linkedin.com/in/" not in url.lower():
        return None

    title_cleaned = re.sub(r"\s*\|\s*LinkedIn\s*$", "", title, flags=re.IGNORECASE).strip()
    parts = [p.strip() for p in title_cleaned.split(" - ", 2)]
    name = parts[0] if parts else "Unknown"
    headline = " - ".join(parts[1:]) if len(parts) > 1 else "LinkedIn Profile"

    if not name or name.lower() in {"sign in", "log in", "linkedin"}:
        return None

    candidate_location = ""
    for pattern in [
        r"(?:Location|Area|Region|Based in)[:\s]+([^·\n.]+)",
        r"(?:located in|based in)\s+([^·\n.]+)",
    ]:
        m = re.search(pattern, snippet, re.IGNORECASE)
        if m:
            candidate_location = m.group(1).strip()[:50]
            break

    full_text = f"{title} {snippet}".lower()
    matched = [s for s in search_skills if s.lower() in full_text]

    score, summary = _score_linkedin(
        matched, len(search_skills), headline, candidate_location, exp_level, location
    )

    cid = hashlib.md5(url.encode()).hexdigest()[:12]
    return Candidate(
        id=cid,
        name=name,
        headline=headline,
        location=candidate_location,
        profile_url=url,
        linkedin_url=url,
        snippet=snippet[:300],
        matched_skills=matched[:6],
        avatar_initials=_get_initials(name),
        score=score,
        summary=summary,
        source="linkedin",
    )


# ── GitHub search + commit analysis ──────────────────────────────────
def _find_github_repos(gh, tags: list[str], max_results: int) -> list[dict]:
    """Try combined tag search first, then fall back to individual tags."""
    # Try combined OR search first
    if len(tags) > 1:
        resp = gh.search_repos_by_impact(
            tags=tags, min_stars=100, match_all_tags=False, max_results=max_results
        )
        print(f"[GitHub] Combined search ({tags}): success={resp.success}, found={len(resp.repositories)}, error={resp.error}")
        if resp.success and resp.repositories:
            return resp.repositories

    # Fall back: search each tag individually and merge by impact score
    seen: set[str] = set()
    merged: list[dict] = []
    for tag in tags:
        resp = gh.search_repos_by_impact(
            tags=[tag], min_stars=100, match_all_tags=False, max_results=max_results * 2
        )
        print(f"[GitHub] Tag '{tag}': success={resp.success}, found={len(resp.repositories)}, error={resp.error}")
        if resp.success:
            for repo in resp.repositories:
                fn = repo.get("full_name", "")
                if fn and fn not in seen:
                    seen.add(fn)
                    merged.append(repo)
        if merged:
            break  # got repos from first tag; that's enough

    merged.sort(key=lambda r: r.get("impact_score", 0), reverse=True)
    return merged[:max_results]


def _search_github_with_analysis(skills: list[str]) -> list[Candidate]:
    """Find top GitHub contributors and score them via commit analysis."""
    if not skills:
        return []

    tags = [_skill_to_github_tag(s) for s in skills[:3] if s.strip()]
    print(f"[GitHub] Searching repos for tags: {tags}")
    gh = get_github_tools()

    try:
        repos = _find_github_repos(gh, tags, MAX_GITHUB_REPOS)
        if not repos:
            print("[GitHub] No repos found after all attempts")
            return []
    except Exception as e:
        print(f"[GitHub] Repo search error: {e}")
        return []

    candidates: list[Candidate] = []
    seen_logins: set[str] = set()

    for repo in repos[:MAX_GITHUB_REPOS]:
        owner = (repo.get("owner") or {}).get("login", "")
        repo_name = repo.get("name", "")
        if not owner or not repo_name:
            continue

        print(f"[GitHub] Processing {owner}/{repo_name}...")
        try:
            contrib_resp = gh.get_contributors(owner, repo_name, max_results=MAX_GITHUB_CONTRIBS)
            if not contrib_resp.success:
                continue
        except Exception as e:
            print(f"[GitHub] Contributors failed for {owner}/{repo_name}: {e}")
            continue

        print(f"[GitHub] Analyzing {len(contrib_resp.contributors)} contributors for {owner}/{repo_name}...")
        for contributor in contrib_resp.contributors:
            login = contributor.login
            if login in seen_logins:
                continue
            seen_logins.add(login)

            # Fetch commits for code quality analysis
            commits = []
            try:
                commits_resp = gh.get_commits(owner, repo_name, login, max_results=MAX_COMMITS_PER_CONTRIB)
                if commits_resp.success:
                    commits = commits_resp.commits
            except Exception:
                pass

            code_score, summary = _analyze_commits(login, owner, repo_name, commits)

            # Final score: code quality (70%) + guaranteed skill relevance (30%)
            final_score = max(1, min(100, int(code_score * 0.7 + 30)))

            github_url = f"https://github.com/{login}"
            cid = hashlib.md5(github_url.encode()).hexdigest()[:12]
            candidates.append(Candidate(
                id=cid,
                name=login,
                headline=f"GitHub · Contributor to {owner}/{repo_name} — {contributor.contributions} commits",
                location="",
                profile_url=github_url,
                github_url=github_url,
                snippet=f"{contributor.contributions} commits to {owner}/{repo_name}.",
                matched_skills=skills[:6],
                avatar_initials=_get_initials(login),
                score=final_score,
                code_quality_score=code_score,
                summary=summary,
                source="github",
            ))
            print(f"[GitHub] @{login}: code={code_score}, final={final_score}")

    print(f"[GitHub] Total candidates: {len(candidates)}")
    return candidates


# ══════════════════════════════════════════════════════════════════════
# Public streaming API
# ══════════════════════════════════════════════════════════════════════

async def stream_candidate_search(filters: SearchFilters):
    """Async generator yielding SSE-style event dicts for the stream endpoint."""
    cache_key = (
        f"{sorted(filters.skills)}:{filters.experience_level}:"
        f"{filters.location}:{filters.description or ''}"
    )
    key = hashlib.md5(cache_key.encode()).hexdigest()

    if key in _cache:
        ts, cached = _cache[key]
        if time.time() - ts < CACHE_TTL:
            print("[Search] Cache hit")
            yield {
                "type": "results",
                "candidates": [c.model_dump() for c in cached],
                "total_results": len(cached),
                "has_more": False,
            }
            return

    # Step 1: Extract keywords from description via Ollama
    skills = list(filters.skills)
    if filters.description and filters.description.strip():
        yield {"type": "progress", "stage": "analyze", "message": "Analyzing your requirements with AI..."}
        try:
            extracted = await asyncio.wait_for(
                asyncio.to_thread(_extract_keywords, filters.description),
                timeout=float(OLLAMA_TIMEOUT),
            )
            seen_lower = {s.lower() for s in skills}
            for kw in extracted:
                if kw.lower() not in seen_lower:
                    skills.append(kw)
                    seen_lower.add(kw.lower())
            print(f"[Search] Final skills: {skills}")
        except Exception as e:
            print(f"[Ollama] Skipping extraction: {e}")

    merged_filters = filters.model_copy(update={"skills": skills})
    linkedin_query = _build_linkedin_query(merged_filters)
    print(f"[Search] LinkedIn query: {linkedin_query}")
    print(f"[Search] GitHub skills: {skills}")

    # Step 2: LinkedIn search (fast) + GitHub analysis (slow) in parallel
    yield {"type": "progress", "stage": "linkedin", "message": "Searching LinkedIn profiles..."}

    async def safe_linkedin() -> list[dict]:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_search_linkedin, linkedin_query),
                timeout=float(DDG_TIMEOUT),
            )
        except Exception as e:
            print(f"[LinkedIn] Timed out: {e}")
            return []

    async def safe_github() -> list[Candidate]:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_search_github_with_analysis, skills),
                timeout=float(GITHUB_TIMEOUT),
            )
        except Exception as e:
            print(f"[GitHub] Timed out: {e}")
            return []

    yield {
        "type": "progress",
        "stage": "search",
        "message": "Searching LinkedIn + analyzing GitHub contributor code quality...",
        "detail": "GitHub analysis includes commit review — this takes ~60s",
    }

    print("[Search] Launching LinkedIn + GitHub (with commit analysis) in parallel...")
    linkedin_raw, github_candidates = await asyncio.gather(safe_linkedin(), safe_github())
    print(f"[Search] Done — {len(linkedin_raw)} LinkedIn raw, {len(github_candidates)} GitHub scored")

    # Step 3: Parse LinkedIn, score, and merge
    yield {"type": "progress", "stage": "score", "message": "Ranking all candidates by score..."}

    linkedin_candidates: list[Candidate] = []
    seen: set[str] = set()
    for r in linkedin_raw:
        c = _parse_linkedin_result(
            r, skills,
            exp_level=filters.experience_level or "",
            location=filters.location or "",
        )
        if not c:
            continue
        dk = _canonicalize_url(c.profile_url)
        if dk and dk not in seen:
            seen.add(dk)
            linkedin_candidates.append(c)

    for c in github_candidates:
        dk = _canonicalize_url(c.profile_url)
        if dk and dk not in seen:
            seen.add(dk)

    all_candidates = linkedin_candidates + github_candidates
    all_candidates.sort(key=lambda c: c.score, reverse=True)

    li_count = len(linkedin_candidates)
    gh_count = len(github_candidates)
    print(f"[Search] Final: {li_count} LinkedIn + {gh_count} GitHub = {len(all_candidates)} total")
    if all_candidates:
        print(f"[Search] Score range: {all_candidates[-1].score}–{all_candidates[0].score}")

    if all_candidates:
        _cache[key] = (time.time(), all_candidates)

    yield {
        "type": "results",
        "candidates": [c.model_dump() for c in all_candidates],
        "total_results": len(all_candidates),
        "has_more": False,
    }


async def search_linkedin_profiles(
    filters: SearchFilters,
) -> tuple[list[Candidate], bool]:
    """Non-streaming fallback for legacy /api/candidates endpoint."""
    async for event in stream_candidate_search(filters):
        if event.get("type") == "results":
            raw_list = event.get("candidates", [])
            return [Candidate(**c) for c in raw_list], False
    return [], False
