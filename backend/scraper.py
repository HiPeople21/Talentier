"""
Candidate search pipeline:
  1. Ollama extracts keywords + generates prestige-targeted LinkedIn query from description
  2. LinkedIn: single DDG query targeting elite companies/universities (no concurrent = no rate limit)
  3. GitHub: repos → contributors → commit analysis via Ollama → code quality score
  4. LinkedIn candidates scored with prestige bonus (+20 top company, +10 top school)
  5. All candidates scored 1-100, sorted, and returned
"""

import asyncio
import hashlib
import json
import os
import re
import time
from urllib.parse import urlparse

import anthropic
import requests
from ddgs import DDGS

from github.tools import get_github_tools
from models import Candidate, SearchFilters

# ── Config ────────────────────────────────────────────────────────────
CACHE_TTL = 300
DDG_TIMEOUT = 20           # Per-request HTTP timeout inside DDGS session
GITHUB_TIMEOUT = 180      # Includes Claude commit analysis time
CLAUDE_TIMEOUT = 30        # Max seconds to wait for a Claude API response
MAX_DDG_RESULTS = 50
MAX_GITHUB_REPOS = 2
MAX_GITHUB_FETCH_CONTRIBS = 30  # How many contributors to pull from GitHub per repo
MAX_GITHUB_ANALYZE = 8          # How many of those to run through Claude commit review
MAX_COMMITS_PER_CONTRIB = 2
CLAUDE_MODEL = "claude-sonnet-4-6"

_cache: dict[str, tuple[float, list[Candidate]]] = {}
_search_status: dict[str, str] = {}   # thread-safe flag channel

# Single shared client — thread-safe, reuses connection pool
_claude_client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env


# ── Claude helpers ─────────────────────────────────────────────────────
def _claude(prompt: str) -> str:
    """Call Claude and return the text response."""
    msg = _claude_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip()


def _extract_keywords_and_query(
    description: str, exp_level: str, location: str
) -> tuple[list[str], str | None]:
    """
    Single Claude call: extracts keywords AND generates a prestige-targeted LinkedIn
    search query. Returns (keywords, linkedin_query_or_None).
    """
    print("[Claude] Generating keywords + prestige-targeted LinkedIn query...")
    exp_hint = f"Experience level: {exp_level}" if exp_level else ""
    loc_hint = f"Location: {location}" if location else ""
    context = "\n".join(filter(None, [exp_hint, loc_hint]))

    prompt = (
        "You are a technical recruiter assistant. Given a job description, do TWO things:\n\n"
        "1. Extract 3-8 key technical skills or domain keywords.\n"
        "2. Write ONE optimal DuckDuckGo search query to find ELITE candidates on LinkedIn. "
        "The query MUST start with 'site:linkedin.com/in' and should target prestigious "
        "employers or top universities using OR groups when relevant to the role. "
        "Good prestige employers include: Google, DeepMind, OpenAI, Anthropic, Meta, "
        "Microsoft, Apple, Amazon, NVIDIA, Tesla, Stripe, Databricks, Jane Street, "
        "Two Sigma, Citadel, Palantir, Hugging Face, Cohere, Mistral. "
        "Good prestige universities include: MIT, Stanford, CMU, Harvard, Caltech, "
        "Berkeley, Oxford, Cambridge, ETH Zurich, Imperial, Princeton, Cornell. "
        "Include 3-5 prestige options using OR. Also include the core job title/skill "
        "and experience level if provided. Keep the query under 120 chars.\n\n"
        f"Job description: {description}\n"
        f"{context}\n\n"
        "Return ONLY valid JSON with this exact structure (no extra text):\n"
        '{"keywords": ["skill1", "skill2"], "linkedin_query": "site:linkedin.com/in ..."}'
    )
    try:
        raw = _claude(prompt)
        # Find JSON block — Claude may wrap it in ```json ... ```
        m = re.search(r'\{[^{}]*"keywords"[^{}]*\}', raw, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            keywords = [
                re.sub(r"^[\-\*\"\'\`\s]+|[\-\*\"\'\`\s]+$", "", k).strip()
                for k in data.get("keywords", [])
            ]
            keywords = [k for k in keywords if k and len(k) < 50]
            query = data.get("linkedin_query", "").strip()
            if not query.startswith("site:linkedin.com/in"):
                query = None
            print(f"[Claude] Keywords: {keywords}")
            print(f"[Claude] LinkedIn query: {query}")
            return keywords, query
    except Exception as e:
        print(f"[Claude] Combined extraction failed: {e}")
    return [], None


def _analyze_commits(login: str, owner: str, repo: str, commits: list) -> tuple[int, str]:
    """Score a developer's commit quality via Claude. Returns (score 0-100, summary)."""
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
        raw = _claude(prompt)
        m = re.search(r'\{[^{}]*"score"[^{}]*\}', raw, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            score = max(1, min(100, int(data.get("score", 50))))
            summary = data.get("summary", f"Contributor to {owner}/{repo}.")
            return score, summary
    except Exception as e:
        print(f"[Claude] Commit analysis failed for @{login}: {e}")
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
    """Fallback LinkedIn query when Ollama is unavailable. Includes prestige targeting."""
    skills = filters.skills[:3]
    location = (filters.location or "").strip()
    level = (filters.experience_level or "").strip().lower()
    exp_prefix = {"senior": "senior", "lead": "lead", "principal": "principal", "junior": "junior"}.get(level, "")
    terms = " ".join(skills) if skills else "software engineer"
    base = f"{exp_prefix} {terms}".strip() if exp_prefix else terms
    prestige = "(Google OR OpenAI OR Meta OR Microsoft OR DeepMind)"
    if location:
        return f"site:linkedin.com/in {base} {prestige} {location}"
    return f"site:linkedin.com/in {base} {prestige}"


# ── Prestige scoring (tiered so scores spread out) ────────────────────
# Company tiers — different scores create differentiation between candidates
_PRESTIGE_COMPANIES_T1 = {  # 40 pts — most elite
    "google", "deepmind", "google deepmind", "openai", "anthropic", "apple", "nvidia",
    "google brain", "google research",
}
_PRESTIGE_COMPANIES_T2 = {  # 30 pts — top-tier
    "meta", "facebook", "microsoft", "amazon", "aws", "tesla", "spacex",
    "stripe", "databricks", "jane street", "two sigma", "citadel",
    "microsoft research", "fair", "msr", "xai", "deepseek",
}
_PRESTIGE_COMPANIES_T3 = {  # 20 pts — prestigious
    "netflix", "hugging face", "huggingface", "cohere", "mistral",
    "stability ai", "scale ai", "anyscale", "replicate", "weights & biases", "wandb",
    "allen institute", "ai2", "d.e. shaw", "renaissance", "palantir",
    "mckinsey", "goldman sachs", "jp morgan", "inflection", "adept", "together ai",
}

# School tiers
_PRESTIGE_SCHOOLS_T1 = {  # 20 pts
    "mit", "stanford", "carnegie mellon", "cmu", "harvard", "caltech",
    "berkeley", "uc berkeley", "eth zurich", "epfl",
}
_PRESTIGE_SCHOOLS_T2 = {  # 12 pts
    "cambridge", "oxford", "imperial college", "imperial", "princeton",
    "yale", "columbia", "georgia tech", "gatech", "cornell",
    "toronto", "waterloo", "university of toronto", "university of cambridge",
    "university of oxford", "tsinghua", "peking university",
}

# Research signal tiers
_RESEARCH_STRONG = {  # 20 pts
    "phd", "ph.d", "ph.d.", "research scientist", "research engineer",
    "professor", "postdoc", "postdoctoral", "doctoral",
    "machine learning researcher", "ai researcher", "deep learning researcher",
    "principal investigator",
}
_RESEARCH_MILD = {  # 10 pts
    "researcher", "research lead", "research director",
    "dissertation", "publications", "published", "arxiv",
    "paper", "papers", "co-author", "first author",
}


def _compute_prestige_breakdown(text: str) -> tuple[int, int, int]:
    """
    Returns (company_score, school_score, research_score) — all tiered.
    Company: T1=40, T2=30, T3=20. School: T1=20, T2=12. Research: strong=20, mild=10.
    All three stack — Google (40) + MIT (20) + PhD (20) = 80 prestige points.
    """
    lowered = text.lower()

    if any(c in lowered for c in _PRESTIGE_COMPANIES_T1):
        company = 40
    elif any(c in lowered for c in _PRESTIGE_COMPANIES_T2):
        company = 30
    elif any(c in lowered for c in _PRESTIGE_COMPANIES_T3):
        company = 20
    else:
        company = 0

    if any(s in lowered for s in _PRESTIGE_SCHOOLS_T1):
        school = 20
    elif any(s in lowered for s in _PRESTIGE_SCHOOLS_T2):
        school = 12
    else:
        school = 0

    if any(t in lowered for t in _RESEARCH_STRONG):
        research = 20
    elif any(t in lowered for t in _RESEARCH_MILD):
        research = 10
    else:
        research = 0

    return company, school, research


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



# ── LinkedIn → GitHub enrichment ──────────────────────────────────────
_GITHUB_NON_USER = {
    "orgs", "sponsors", "topics", "trending", "marketplace",
    "features", "explore", "apps", "settings", "login", "about",
    "collections", "events", "pulls", "issues", "notifications",
}

def _extract_github_username(text: str) -> str | None:
    """Extract a GitHub username mentioned anywhere in profile text."""
    m = re.search(
        r'github\.com/([a-zA-Z0-9][a-zA-Z0-9\-]{0,37}[a-zA-Z0-9]|[a-zA-Z0-9]{1,39})',
        text, re.IGNORECASE,
    )
    if not m:
        return None
    username = m.group(1)
    return None if username.lower() in _GITHUB_NON_USER else username


def _get_user_best_repo(username: str) -> tuple[str, str] | None:
    """Find the user's most-starred GitHub repository via the search API."""
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    try:
        resp = requests.get(
            "https://api.github.com/search/repositories",
            params={"q": f"user:{username}", "sort": "stars", "order": "desc", "per_page": 1},
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        items = resp.json().get("items", [])
        if items:
            repo = items[0]
            owner = (repo.get("owner") or {}).get("login", username)
            return owner, repo.get("name", "")
    except Exception as e:
        print(f"[LinkedIn→GitHub] Repo lookup failed for @{username}: {e}")
    return None


def _enhance_linkedin_with_github(candidate: Candidate) -> Candidate:
    """
    If the LinkedIn snippet mentions a GitHub URL, fetch their commits, run
    Ollama code-quality analysis, and blend the score:
        blended = linkedin_score × 0.6 + code_quality × 0.4
    Returns the (possibly enhanced) candidate.
    """
    full_text = f"{candidate.headline} {candidate.snippet}"
    username = _extract_github_username(full_text)
    if not username:
        return candidate

    github_url = f"https://github.com/{username}"
    print(f"[LinkedIn→GitHub] @{username} found on {candidate.name}'s profile")

    repo_info = _get_user_best_repo(username)
    if not repo_info:
        # Link the profile even without a code review
        return candidate.model_copy(update={"github_url": github_url})

    owner, repo_name = repo_info
    print(f"[LinkedIn→GitHub] Analysing @{username} via {owner}/{repo_name}...")

    gh = get_github_tools()
    commits: list = []
    try:
        cr = gh.get_commits(owner, repo_name, username, max_results=MAX_COMMITS_PER_CONTRIB)
        if cr.success:
            commits = cr.commits
    except Exception:
        pass

    code_score, code_summary = _analyze_commits(username, owner, repo_name, commits)

    # Blend: professional experience (60%) + code quality (40%)
    blended = max(1, min(100, int(candidate.score * 0.6 + code_score * 0.4)))

    enhanced_summary = (
        f"{candidate.summary} "
        f"GitHub code review ({owner}/{repo_name}): {code_summary}"
    )
    print(
        f"[LinkedIn→GitHub] {candidate.name}: "
        f"linkedin={candidate.score}, code={code_score}, blended={blended}"
    )

    return candidate.model_copy(update={
        "github_url": github_url,
        "code_quality_score": code_score,
        "score": blended,
        "summary": enhanced_summary[:600],
    })

# ── LinkedIn search ───────────────────────────────────────────────────
def _search_linkedin(query: str) -> list[dict]:
    _search_status.pop("linkedin_error", None)
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
        err_str = str(e)
        if "429" in err_str or "Too Many Requests" in err_str:
            print(f"[LinkedIn] Rate limited by search provider")
            _search_status["linkedin_error"] = "rate_limited"
        else:
            print(f"[LinkedIn] Failed: {e}")
        return []


def _score_linkedin(matched_skills: list[str], total_skills: int, exp_level: str,
                    location: str, required_exp: str, required_loc: str,
                    company_prestige: int = 0, school_prestige: int = 0,
                    research_score: int = 0) -> tuple[int, str]:
    """Score a LinkedIn candidate 1-100.

    Prestige (company + school + research) is the dominant signal. Tiered scoring
    creates natural spread: T1 company=40, T2=30, T3=20; T1 school=20, T2=12;
    strong research=20, mild=10.
      linkedin bonus : 20     (always — verified professional profile baseline)
      prestige total : 0–80   (T1 company 40 + T1 school 20 + strong research 20)
      skill match    : 0–12
      exp level      : 0–6
      location match : 0–3
    Total max ≈ 121 → capped at 100
    """
    skill_score = int((len(matched_skills) / max(total_skills, 1)) * 12)  # 0-12
    exp_score = 6 if (required_exp and required_exp.lower() in exp_level.lower()) else 0
    loc_score = 3 if (required_loc and required_loc.lower() in location.lower()) else 0
    linkedin_bonus = 20
    prestige_total = min(80, company_prestige + school_prestige + research_score)

    score = max(1, min(100, skill_score + exp_score + loc_score + linkedin_bonus + prestige_total))

    reasons = []
    if company_prestige:
        reasons.append("top-tier tech company")
    if school_prestige:
        reasons.append("elite university")
    if research_score:
        reasons.append("research/PhD background")
    if matched_skills:
        reasons.append(f"matches {len(matched_skills)} skill(s): {', '.join(matched_skills[:3])}")
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
    # Try multiple patterns roughly ordered by confidence
    _loc_patterns = [
        # Explicit label: "Location: San Francisco"
        r"(?:Location|Area|Region|Based in)[:\s]+([^·|\n]{3,50})",
        r"(?:located in|based in)\s+([^·|\n.]{3,50})",
        # LinkedIn dot-separator format: "· San Francisco, CA ·"
        r"·\s*([A-Z][a-zA-Z\s]{2,30}(?:,\s*[A-Z][a-zA-Z\s]{2,30})?)\s*·",
        # "City, Country" pattern at sentence boundary
        r"\b([A-Z][a-zA-Z ]{2,25},\s*(?:United States|United Kingdom|UK|Canada|Australia|"
        r"Germany|France|Netherlands|Sweden|Switzerland|India|Singapore|Japan|China|Israel|"
        r"Brazil|Norway|Denmark|Finland|Spain|Italy|South Korea))\b",
        # Well-known metro area phrases
        r"((?:San Francisco|New York|Greater London|Greater NYC|Bay Area|"
        r"Silicon Valley|Greater Boston|Greater Seattle|Greater Chicago|"
        r"Greater Los Angeles|Greater Toronto|Greater Vancouver|Greater Berlin|"
        r"Greater Munich)[a-zA-Z\s,]*(?:Area|Region|Metro)?)",
        # "City, ST" US two-letter state
        r"\b([A-Z][a-zA-Z ]{2,20},\s*[A-Z]{2})\b",
    ]
    for pattern in _loc_patterns:
        m = re.search(pattern, f"{title} {snippet}", re.IGNORECASE)
        if m:
            candidate_location = m.group(1).strip()[:60]
            break

    full_text = f"{title} {snippet}".lower()
    matched = [s for s in search_skills if s.lower() in full_text]

    company_p, school_p, research_p = _compute_prestige_breakdown(f"{title} {snippet}")

    score, summary = _score_linkedin(
        matched, len(search_skills), headline, candidate_location, exp_level, location,
        company_prestige=company_p, school_prestige=school_p, research_score=research_p,
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
MIN_GITHUB_STARS = 5000  # Target high-profile repos (OpenCV, YOLO, etc.)


def _find_github_repos(gh, tags: list[str], max_results: int) -> list[dict]:
    """Try combined tag search first, then fall back to individual tags."""
    # Try combined OR search first
    if len(tags) > 1:
        resp = gh.search_repos_by_impact(
            tags=tags, min_stars=MIN_GITHUB_STARS, match_all_tags=False, max_results=max_results
        )
        print(f"[GitHub] Combined search ({tags}): success={resp.success}, found={len(resp.repositories)}, error={resp.error}")
        if resp.success and resp.repositories:
            return resp.repositories

    # Fall back: search each tag individually and merge by impact score
    seen: set[str] = set()
    merged: list[dict] = []
    for tag in tags:
        resp = gh.search_repos_by_impact(
            tags=[tag], min_stars=MIN_GITHUB_STARS, match_all_tags=False, max_results=max_results * 2
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
            contrib_resp = gh.get_contributors(owner, repo_name, max_results=MAX_GITHUB_FETCH_CONTRIBS)
            if not contrib_resp.success:
                continue
        except Exception as e:
            print(f"[GitHub] Contributors failed for {owner}/{repo_name}: {e}")
            continue

        fetched = len(contrib_resp.contributors)
        analyze_limit = MAX_GITHUB_ANALYZE - len(candidates)  # budget remaining
        print(f"[GitHub] Fetched {fetched} contributors for {owner}/{repo_name}, analyzing top {analyze_limit}...")
        analyzed = 0
        for contributor in contrib_resp.contributors:
            if analyzed >= analyze_limit:
                break
            login = contributor.login
            if login in seen_logins:
                continue
            seen_logins.add(login)
            analyzed += 1

            # Fetch commits for code quality analysis
            commits = []
            try:
                commits_resp = gh.get_commits(owner, repo_name, login, max_results=MAX_COMMITS_PER_CONTRIB)
                if commits_resp.success:
                    commits = commits_resp.commits
            except Exception:
                pass

            code_score, summary = _analyze_commits(login, owner, repo_name, commits)

            # Capped at 50: prestige LinkedIn profiles (company + school + research)
            # should always outrank pure code contributors.
            final_score = max(1, min(50, int(code_score * 0.25 + 20)))

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

    # Step 1: Extract keywords + generate prestige-targeted LinkedIn query via Claude
    skills = list(filters.skills)
    ai_linkedin_query: str | None = None

    if filters.description and filters.description.strip():
        yield {"type": "progress", "stage": "analyze", "message": "Analyzing your requirements with AI..."}
        try:
            extracted, ai_linkedin_query = await asyncio.wait_for(
                asyncio.to_thread(
                    _extract_keywords_and_query,
                    filters.description,
                    filters.experience_level or "",
                    filters.location or "",
                ),
                timeout=float(CLAUDE_TIMEOUT),
            )
            seen_lower = {s.lower() for s in skills}
            for kw in extracted:
                if kw.lower() not in seen_lower:
                    skills.append(kw)
                    seen_lower.add(kw.lower())
            print(f"[Search] Final skills: {skills}")
        except Exception as e:
            print(f"[Claude] Skipping extraction: {e}")

    merged_filters = filters.model_copy(update={"skills": skills})
    # Use Claude-generated prestige query if available; fall back to deterministic builder
    linkedin_query = ai_linkedin_query or _build_linkedin_query(merged_filters)
    print(f"[Search] LinkedIn query: {linkedin_query}")
    print(f"[Search] GitHub skills: {skills}")

    # Step 2: LinkedIn search (fast) + GitHub analysis (slow) in parallel
    yield {"type": "progress", "stage": "linkedin", "message": "Searching LinkedIn profiles..."}

    async def safe_linkedin() -> list[dict]:
        # NOTE: Do NOT wrap in asyncio.wait_for — it cancels the wrapper but NOT the
        # underlying thread (asyncio.to_thread threads can't be interrupted). That causes
        # wait_for to fire and return [], while the thread keeps running and discards
        # its results. Instead let the thread run naturally; DDGS handles its own
        # per-request HTTP timeout via DDG_TIMEOUT passed to DDGS(timeout=...).
        try:
            return await asyncio.to_thread(_search_linkedin, linkedin_query)
        except Exception as e:
            print(f"[LinkedIn] Failed: {e}")
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

    # Inform frontend if LinkedIn was rate-limited
    if _search_status.get("linkedin_error") == "rate_limited":
        yield {
            "type": "progress",
            "stage": "warning",
            "message": "LinkedIn search rate-limited — showing GitHub results only.",
            "detail": "Try again in a minute or broaden your description.",
        }

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

    # Step 3.5: Enhance LinkedIn candidates whose snippet mentions a GitHub URL
    github_linked = [
        c for c in linkedin_candidates
        if _extract_github_username(f"{c.headline} {c.snippet}")
    ][:5]  # limit to top 5 to keep total time reasonable

    if github_linked:
        yield {
            "type": "progress",
            "stage": "enhance",
            "message": f"Found GitHub on {len(github_linked)} LinkedIn profile(s) — running code review...",
            "detail": "Blending professional experience with commit quality",
        }
        enhanced_map: dict[str, Candidate] = {}
        for lc in github_linked:
            try:
                enhanced = await asyncio.wait_for(
                    asyncio.to_thread(_enhance_linkedin_with_github, lc),
                    timeout=60.0,
                )
                enhanced_map[lc.id] = enhanced
            except Exception as e:
                print(f"[LinkedIn→GitHub] Timed out for {lc.name}: {e}")
        linkedin_candidates = [enhanced_map.get(c.id, c) for c in linkedin_candidates]

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
