"""
Candidate discovery utility module.

Provides standalone functions for multi-source candidate search
(LinkedIn, Google Scholar, GitHub) via DuckDuckGo + Brave Search,
and LLM-powered enrichment / ranking via Ollama.

Exported functions (called by the unified pipeline):
  plan_search_terms      – LLM generates search terms from skills/description
  discover_candidates    – multi-source discovery across LinkedIn, Scholar, GitHub
  enrich_candidates_llm  – LLM cleans and enriches raw profiles
  evaluate_candidates_llm – LLM scores candidates 1-10
  finalize_candidates    – deduplicates and caps results
"""

import datetime
import hashlib
import json
import random
import re
import subprocess
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import unquote, urlparse

import httpx
from bs4 import BeautifulSoup
from ddgs import DDGS
from github.tools import get_github_tools
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

# ── Config ────────────────────────────────────────────────────────────
OLLAMA_MODEL = "llama3.2"  # 2GB — ~2x faster than qwen2.5 while still capable
MAX_REFINE_LOOPS = 2
CACHE_TTL = 300

# ── Agent status (shared with SSE endpoint) ───────────────────────────
agent_status: dict = {"steps": [], "done": True}


def _emit(step_type: str, message: str, detail: str = ""):
    """Push a status update for the SSE stream."""
    agent_status["steps"].append({
        "type": step_type,
        "message": message,
        "detail": detail,
    })

llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0.7,
    num_predict=3072,
)

# ── Header Rotation Pool ─────────────────────────────────────────────
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
]

_ACCEPT_LANGUAGES = [
    "en-US,en;q=0.9",
    "en-GB,en;q=0.9",
    "en-US,en;q=0.9,es;q=0.8",
    "en-US,en;q=0.8",
    "en,en-US;q=0.9",
    "en-AU,en;q=0.9",
    "en-CA,en;q=0.9",
]

_REFERERS = [
    "https://www.google.com/",
    "https://duckduckgo.com/",
    "https://www.bing.com/",
    "",  # no referer (direct visit)
]


def _random_headers() -> dict:
    """Generate a unique set of browser-like headers for each request."""
    return {
        "User-Agent": random.choice(_USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": random.choice(_ACCEPT_LANGUAGES),
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": random.choice(_REFERERS),
        "DNT": random.choice(["1", "0"]),
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": random.choice(["none", "cross-site"]),
    }

# Also keep a static header dict for use in Scholar/GitHub HTTP calls
DDG_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

DDG_MAX_CONCURRENT = 4   # Max concurrent DDGS search calls

SOURCE_QUERY_TEMPLATES = [
    "linkedin.com/in {terms}",
]

# Discovery templates
SCHOLAR_DISCOVERY_TEMPLATE = "scholar.google.com {terms}"

MIN_SCHOLAR_CITATIONS = 100
MIN_GITHUB_CONTRIBUTIONS_LAST_YEAR = 100
MIN_GITHUB_PUBLIC_REPOS = 5
MIN_REPO_CONTRIBUTIONS = 10   # Min commits to repo to qualify as significant contributor

# GitHub repo impact thresholds — only crawl large, community-scale projects
MIN_REPO_STARS = 500
MIN_REPO_FORKS = 50
MIN_REPO_TOTAL_CONTRIBUTORS = 10  # Must be a multi-contributor community project

# Recency gate — candidates with no activity in this many years are skipped
RECENCY_CUTOFF_YEARS = 2

# Output filtering — only return the top N most qualified candidates
TOP_CANDIDATES_LIMIT = 20   # Max candidates returned to the recruiter
MIN_CANDIDATE_SCORE = 5     # Drop candidates scored below this threshold
ENRICH_BATCH_SIZE = 25      # Process at most this many candidates per LLM call

_scholar_quality_cache: dict[str, bool] = {}
_github_quality_cache: dict[str, bool] = {}


# ── Helpers ───────────────────────────────────────────────────────────
def _get_initials(name: str) -> str:
    words = name.strip().split()
    if len(words) >= 2:
        return (words[0][0] + words[-1][0]).upper()
    elif words:
        return words[0][0].upper()
    return "?"


def _extract_json(text: str) -> list[dict] | dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for pattern in [r"(\[.*\])", r"(\{.*\})"]:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
    return []


# ── Search Engines ────────────────────────────────────────────────────
def _search_ddg(query: str) -> list[dict]:
    """Search DuckDuckGo via the duckduckgo_search library (DDGS).

    Uses DDG's API endpoints rather than the HTML scraper, so it is far more
    resilient to rate-limiting and IP blocks.  Each call creates its own DDGS
    session so concurrent threads don't share state.

    Wrapped in a 15-second timeout to prevent indefinite hangs (curl_cffi
    impersonation issues can cause stalls).
    """
    def _do_search():
        with DDGS() as ddgs:
            raw = ddgs.text(query, max_results=10)
            return list(raw) if raw else []

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_do_search)
            results = future.result(timeout=15)
        return [
            {
                "title": r.get("title", ""),
                "href": r.get("href", ""),
                "snippet": r.get("body", ""),
            }
            for r in results
        ]
    except TimeoutError:
        print(f"[DDG] Search timed out for '{query[:60]}'")
        return []
    except Exception as e:
        print(f"[DDG] Search failed for '{query[:60]}': {e}")
        return []


def _search_brave(query: str) -> list[dict]:
    """Search Brave Search — zero bot protection, fast, reliable fallback."""
    try:
        headers = _random_headers()
        query_encoded = urllib.parse.quote(query)
        url = f"https://search.brave.com/search?q={query_encoded}&source=web"

        cmd = [
            "curl", "-s", "-m", "10",
            "-A", headers["User-Agent"],
            "-H", f"Accept-Language: {headers['Accept-Language']}",
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        html = result.stdout

        if not html:
            return []

        soup = BeautifulSoup(html, "html.parser")

        results = []
        seen = set()
        for a in soup.find_all("a", href=re.compile(r"linkedin\.com/in/")):
            href = a.get("href", "")
            if href in seen:
                continue
            seen.add(href)

            title = a.get_text(strip=True) or ""
            snippet = ""
            parent = a.find_parent(attrs={"class": True})
            if parent:
                desc = parent.find(
                    ["p", "span", "div"],
                    class_=lambda c: c and ("description" in str(c).lower() or "snippet" in str(c).lower()),
                )
                if desc:
                    snippet = desc.get_text(strip=True)

            if not title:
                match = re.search(r"linkedin\.com/in/([^/\?]+)", href)
                title = match.group(1).replace("-", " ").title() if match else href

            results.append({"title": title, "href": href, "snippet": snippet})

        return results
    except Exception as e:
        print(f"[Brave] Search failed: {e}")
        return []


def _search_with_fallback(query: str) -> list[dict]:
    """Try DDGS first, fall back to Brave Search if it fails."""
    results = _search_ddg(query)
    if results:
        return results

    # Fallback to Brave Search
    brave_results = _search_brave(query)
    if brave_results:
        return brave_results

    return []


# ── URL / Source Utilities ────────────────────────────────────────────
def _canonicalize_url(url: str) -> str:
    """Normalize profile URLs to improve deduplication across queries."""
    if not url:
        return ""
    parsed = urlparse(url)
    host = parsed.netloc.lower().replace("www.", "")
    # Normalize LinkedIn regional domains to a single domain
    # e.g. in.linkedin.com, uk.linkedin.com, ca.linkedin.com → linkedin.com
    host = re.sub(r"^[a-z]{2}\.linkedin\.com$", "linkedin.com", host)
    path = re.sub(r"/+", "/", parsed.path).rstrip("/")
    # Strip query params and fragments (tracking junk)
    canonical = f"{host}{path}"
    return canonical


def _normalize_name(name: str) -> str:
    """Normalize a person's name for deduplication.

    Strips middle initials, suffixes (Jr, Sr, PhD, etc.), and
    reduces to lowercase alpha-only for fuzzy matching.
    """
    if not name:
        return ""
    # Remove common suffixes
    name = re.sub(r"\b(jr|sr|ii|iii|iv|phd|md|mba|ms|bs|pe)\b", "", name, flags=re.IGNORECASE)
    # Remove single-letter middle initials (e.g. "John A. Doe" → "John Doe")
    name = re.sub(r"\b[A-Za-z]\.?\s+", " ", name)
    # Keep only lowercase alpha
    name = re.sub(r"[^a-z]", "", name.lower())
    return name


def _detect_source(url: str) -> str | None:
    lowered = url.lower()
    if "linkedin.com/in/" in lowered:
        return "LinkedIn"
    if "scholar.google.com/citations" in lowered:
        return "Google Scholar"
    if "github.com/" in lowered:
        parsed = urlparse(url)
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) == 1:
            blocked = {
                "features", "topics", "collections", "trending", "marketplace",
                "sponsors", "about", "pricing", "login", "join", "orgs",
                "organizations", "enterprise", "site", "events", "settings",
                "apps", "search", "explore", "pulls", "issues", "new",
            }
            if parts[0].lower() not in blocked:
                return "GitHub"
    return None


def _extract_first_int(text: str, pattern: str) -> int:
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return 0
    try:
        return int(match.group(1).replace(",", ""))
    except Exception:
        return 0


def _is_recent_enough(date_str: str | None) -> bool:
    """Return True if the ISO-8601 date is within RECENCY_CUTOFF_YEARS of today."""
    if not date_str:
        return False
    try:
        dt = datetime.datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
            days=365 * RECENCY_CUTOFF_YEARS
        )
        return dt >= cutoff
    except Exception:
        return False


def _skills_to_github_tags(skills: list[str]) -> list[str]:
    """Convert skill names to lowercase, hyphenated GitHub topic tags."""
    tags = []
    for skill in skills:
        tag = skill.lower().strip().replace(" ", "-")
        if tag:
            tags.append(tag)
    return tags[:5]


def _is_substantial_scholar_contributor(url: str, title: str, snippet: str) -> bool:
    """Return True if the Scholar profile has enough citations AND recent publications."""
    key = _canonicalize_url(url)
    if key in _scholar_quality_cache:
        return _scholar_quality_cache[key]

    parsed = urlparse(url)
    if "user=" not in parsed.query:
        _scholar_quality_cache[key] = False
        return False

    combined = f"{title} {snippet}"
    citations = _extract_first_int(combined, r"cited by\s*([0-9,]+)")

    current_year = datetime.datetime.now().year
    recent_years = {str(y) for y in range(current_year - 2, current_year + 1)}
    has_recent_snippet = any(y in combined for y in recent_years)

    if citations == 0 or not has_recent_snippet:
        try:
            resp = httpx.get(url, headers=DDG_HEADERS, follow_redirects=True, timeout=10.0)
            if resp.status_code < 400:
                page = resp.text
                if citations == 0:
                    citations = _extract_first_int(page, r"Cited by\s*([0-9,]+)")
                if not has_recent_snippet:
                    has_recent_snippet = any(y in page for y in recent_years)
        except Exception:
            pass

    ok = citations >= MIN_SCHOLAR_CITATIONS and has_recent_snippet
    _scholar_quality_cache[key] = ok
    return ok


def _is_substantial_github_contributor(url: str) -> bool:
    key = _canonicalize_url(url)
    if key in _github_quality_cache:
        return _github_quality_cache[key]

    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) != 1:
        _github_quality_cache[key] = False
        return False
    username = parts[0]

    contributions = 0
    public_repos = 0
    is_user = False

    try:
        profile_resp = httpx.get(url, headers=DDG_HEADERS, follow_redirects=True, timeout=12.0)
        if profile_resp.status_code < 400:
            contributions = _extract_first_int(
                profile_resp.text,
                r'([0-9,]+)\s+contributions?\s+in\s+the\s+last\s+year',
            )
    except Exception:
        contributions = 0

    try:
        api_resp = httpx.get(
            f"https://api.github.com/users/{username}",
            headers={"Accept": "application/vnd.github+json", "User-Agent": DDG_HEADERS["User-Agent"]},
            timeout=12.0,
        )
        if api_resp.status_code < 400:
            data = api_resp.json()
            public_repos = int(data.get("public_repos", 0) or 0)
            is_user = (data.get("type", "") == "User")
    except Exception:
        public_repos = 0
        is_user = False

    ok = is_user and (
        contributions >= MIN_GITHUB_CONTRIBUTIONS_LAST_YEAR
        or public_repos >= MIN_GITHUB_PUBLIC_REPOS
    )
    _github_quality_cache[key] = ok
    return ok


def _extract_name_from_result(url: str, title: str) -> str:
    title = re.sub(r"\s*[\-|:]\s*(LinkedIn|Google Scholar|GitHub).*$", "", title).strip()
    if title and title.lower() not in {
        "linkedin", "google scholar", "github"
    }:
        return title.split(" - ")[0].strip()

    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    if not parts:
        return "Unknown"

    handle = parts[-1]
    handle = re.sub(r"[_\-]+", " ", handle).strip()
    return handle.title() if handle else "Unknown"


def _parse_candidate_result(result: dict, search_skills: list[str]) -> dict | None:
    """Parse a DDG search result into a cross-source candidate dict."""
    url = result.get("href", "")
    title = result.get("title", "")
    snippet = result.get("snippet", "")
    source = _detect_source(url)

    if not source:
        return None

    # Parse title where possible
    title_cleaned = re.sub(r"\s*[\|–-]\s*(LinkedIn|Google Scholar|GitHub).*$", "", title).strip()
    parts = [p.strip() for p in title_cleaned.split(" - ", 2)]
    name = _extract_name_from_result(url, title_cleaned)
    headline = " - ".join(parts[1:]) if len(parts) > 1 else ""

    if not name or len(name) < 2 or name.lower() in ["linkedin", "sign in", "log in", ""]:
        return None

    location = ""
    for pattern in [
        r"(?:Location|Area|Based in)[:\s]+([^·\n.]+)",
        r"(?:located in|based in)\s+([^·\n.]+)",
    ]:
        match = re.search(pattern, snippet, re.IGNORECASE)
        if match:
            location = match.group(1).strip()[:50]
            break

    full_text = f"{title} {snippet}".lower()
    matched = [s for s in search_skills if s.lower() in full_text]

    if source == "Google Scholar" and not _is_substantial_scholar_contributor(url, title, snippet):
        return None
    if source == "GitHub" and not _is_substantial_github_contributor(url):
        return None

    return {
        "name": name,
        "headline": f"[{source}] {headline or 'Profile'}",
        "location": location,
        "profile_url": url,
        "snippet": snippet[:300],
        "matched_skills": matched,
    }


def _extract_authors_from_scholar_result(result: dict) -> list[str]:
    """Extract author names from a Google Scholar search result snippet."""
    snippet = result.get("snippet", "")
    authors: list[str] = []
    patterns = [
        r"^([A-Z][a-zA-Z\-']+ [A-Z][a-zA-Z\-']+"
        r"(?:,\s*[A-Z][a-zA-Z\-']+ [A-Z][a-zA-Z\-']+)*)\s*[-–]",
        r"([A-Z][a-zA-Z\-']+ [A-Z][a-zA-Z\-']+"
        r"(?:,\s*[A-Z][a-zA-Z\-']+ [A-Z][a-zA-Z\-']+)*)\s*[-–]\s*\d{4}",
    ]
    for pattern in patterns:
        match = re.search(pattern, snippet)
        if match:
            for part in match.group(1).split(","):
                name = part.strip()
                if len(name.split()) >= 2:
                    authors.append(name)
            break
    return authors[:3]


_github_user_cache: dict[str, dict] = {}


def _fetch_github_user(username: str) -> dict:
    """Fetch and cache GitHub user API data for a given username."""
    if username in _github_user_cache:
        return _github_user_cache[username]
    try:
        resp = httpx.get(
            f"https://api.github.com/users/{username}",
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": DDG_HEADERS["User-Agent"],
            },
            timeout=8.0,
        )
        data = resp.json() if resp.status_code < 400 else {}
    except Exception:
        data = {}
    _github_user_cache[username] = data
    return data


def _process_repo_contributors(owner: str, repo_name: str, repo_html_url: str) -> list[dict]:
    """Fetch and enrich contributors for a repo using GitHubTools."""
    github_tools = get_github_tools()
    contribs_response = github_tools.get_contributors(owner, repo_name, max_results=30)
    if not contribs_response.success:
        print(f"[GitHub] Contributors fetch failed for {owner}/{repo_name}: {contribs_response.error}")
        return []

    all_contribs = contribs_response.contributors
    if len(all_contribs) < MIN_REPO_TOTAL_CONTRIBUTORS:
        print(f"[GitHub] Skip {owner}/{repo_name}: only {len(all_contribs)} contributors")
        return []

    qualified = [c for c in all_contribs if c.contributions >= MIN_REPO_CONTRIBUTIONS]

    def _enrich(contrib) -> dict | None:
        ud = _fetch_github_user(contrib.login)
        if not ud or ud.get("type", "") != "User":
            return None
        if not _is_recent_enough(ud.get("updated_at")):
            print(f"[GitHub] Skip {contrib.login}: stale account")
            return None
        blog = (ud.get("blog") or "").strip()
        website = blog if blog.startswith("http") else (f"https://{blog}" if blog else "")
        return {
            "login": contrib.login,
            "contributions": contrib.contributions,
            "name": ud.get("name") or contrib.login,
            "location": ud.get("location") or "",
            "website": website,
        }

    enriched: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(8, len(qualified))) as executor:
        futures = {executor.submit(_enrich, c): c for c in qualified}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result:
                    enriched.append(result)
            except Exception:
                pass

    return enriched


def _contributors_to_candidates(
    contributors: list[dict],
    repo_url: str,
    skills: list[str],
) -> list[dict]:
    """Convert GitHub contributor records into candidate dicts.

    Only ONE entry per person — prefers personal website over GitHub profile.
    """
    candidates: list[dict] = []
    repo_name = "/".join(repo_url.split("github.com/")[-1].split("/")[:2])

    for contrib in contributors:
        username = contrib.get("login", "")
        name = contrib.get("name") or username
        github_url = f"https://github.com/{username}"
        website = contrib.get("website", "")
        location = contrib.get("location", "")
        n_contribs = contrib.get("contributions", 0)
        snippet = f"Contributor to {repo_name} with {n_contribs} commits."

        # Prefer personal website; fall back to GitHub profile — only ONE entry
        if website:
            candidates.append({
                "name": name,
                "headline": f"[Personal Website] Contributor to {repo_name} · GitHub: {username}",
                "location": location,
                "profile_url": website,
                "snippet": snippet,
                "matched_skills": skills,
            })
        else:
            candidates.append({
                "name": name,
                "headline": f"[GitHub] Contributor to {repo_name}",
                "location": location,
                "profile_url": github_url,
                "snippet": snippet,
                "matched_skills": skills,
            })

    return candidates


# ══════════════════════════════════════════════════════════════════════
# LangGraph Agent Nodes
# ══════════════════════════════════════════════════════════════════════

def _build_linkedin_queries(terms: str, filters: dict) -> list[str]:
    """Generate a diverse set of LinkedIn DDG queries for broad candidate coverage."""
    skills = filters.get("skills", [])
    location = filters.get("location", "").strip()

    bases: list[str] = []
    if terms:
        bases.append(terms)
    if len(skills) >= 2:
        pair = f"{skills[0]} {skills[1]}"
        if pair.lower() != terms.lower():
            bases.append(pair)
    if skills:
        solo = skills[0]
        if solo.lower() not in {b.lower() for b in bases}:
            bases.append(solo)
    if len(skills) >= 3:
        alt = f"{skills[0]} {skills[2]}"
        if alt.lower() not in {b.lower() for b in bases}:
            bases.append(alt)
    if len(skills) >= 3:
        alt2 = f"{skills[1]} {skills[2]}"
        if alt2.lower() not in {b.lower() for b in bases}:
            bases.append(alt2)

    loc_variants: list[str] = []
    if location:
        loc_variants.append(location)
        city = location.split(",")[0].strip()
        if city and city.lower() != location.lower():
            loc_variants.append(city)

    role_suffixes = ["engineer", "developer", "researcher", "scientist"]
    seniority_prefixes = ["senior", "staff"]

    queries: list[str] = []

    for base in bases[:4]:
        if loc_variants:
            for loc in loc_variants[:2]:
                queries.append(f"site:linkedin.com/in {base} {loc}")
                queries.append(f"site:linkedin.com/in {base} {role_suffixes[0]} {loc}")
                queries.append(f"site:linkedin.com/in {base} {role_suffixes[1]} {loc}")
                queries.append(f"site:linkedin.com/in {seniority_prefixes[0]} {base} {loc}")
        else:
            queries.append(f"site:linkedin.com/in {base}")
            queries.append(f"site:linkedin.com/in {base} {role_suffixes[0]}")
            queries.append(f"site:linkedin.com/in {base} {role_suffixes[1]}")
            queries.append(f"site:linkedin.com/in {seniority_prefixes[0]} {base}")

    if bases:
        plain_loc = f" {loc_variants[0]}" if loc_variants else ""
        queries.append(f"linkedin.com/in {bases[0]}{plain_loc}")
        if len(bases) > 1:
            queries.append(f"linkedin.com/in {bases[1]}{plain_loc}")
        queries.append(f"site:linkedin.com/in {bases[0]} remote")

    seen: set[str] = set()
    deduped: list[str] = []
    for q in queries:
        k = q.lower().strip()
        if k not in seen:
            seen.add(k)
            deduped.append(q)

    return deduped[:16]


# ══════════════════════════════════════════════════════════════════════
# Standalone utility functions (called by the pipeline)
# ══════════════════════════════════════════════════════════════════════

def plan_search_terms(
    skills: list[str],
    description: str = "",
    refinement_feedback: str = "",
) -> str:
    """Use LLM to generate compact search terms from skills/description."""
    skills_str = ", ".join(skills)

    refinement_note = ""
    if refinement_feedback:
        refinement_note = f"\n\nPrevious search had issues: {refinement_feedback}\nBroaden or adjust accordingly."

    description_note = ""
    if description:
        description_note = f"\n6. Consider these extra descriptors: {description}"

    messages = [
        SystemMessage(content=(
            "You are a recruitment search specialist. Produce compact skill/domain "
            "search terms to find candidates across LinkedIn, Google Scholar, and GitHub.\n\n"
            "Rules:\n"
            "1. Focus on the 1-3 most important skills or technologies\n"
            "2. Do NOT include experience level (junior/senior/etc.) — we want all levels\n"
            "3. Do NOT include location — location is handled separately\n"
            "4. Return ONLY the search terms string — nothing else\n"
            "5. Keep it concise: 2-6 words"
            f"{description_note}\n\n"
            "Example: 'python machine learning' or 'react typescript frontend'"
        )),
        HumanMessage(content=(
            f"Skills: {skills_str or 'any'}\n"
            f"Description: {description or 'none'}"
            f"{refinement_note}"
        )),
    ]

    _emit("thinking", "Planning search query...", "Analyzing filters to build optimal search")
    print("[Scraper] Planning search query...")
    response = llm.invoke(messages)
    query = response.content.strip().strip('"').strip("'").strip("`").splitlines()[0].strip()

    _emit("success", f"Search terms ready", query)
    print(f"[Scraper] Planned terms: {query}")
    return query


def discover_candidates(
    search_terms: str,
    filters: dict,
) -> list[dict]:
    """
    Two-phase candidate discovery across LinkedIn, Scholar, and GitHub.

    Returns a list of parsed candidate dicts (not yet enriched).
    """
    skills = filters.get("skills", [])
    terms = search_terms.strip() or "software engineer"

    linkedin_queries = _build_linkedin_queries(terms, filters)
    scholar_query = SCHOLAR_DISCOVERY_TEMPLATE.format(terms=terms)

    phase1_map: dict[str, str] = {}
    for i, q in enumerate(linkedin_queries):
        phase1_map[f"linkedin_{i}"] = q
    phase1_map["scholar_papers"] = scholar_query

    # ── Phase 1: LinkedIn + Scholar DDG searches in parallel ─────────
    _emit("searching", "Searching multiple sources...", f"Running {len(phase1_map)} parallel queries")
    phase1_results: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=min(DDG_MAX_CONCURRENT, len(phase1_map))) as executor:
        future_map = {executor.submit(_search_with_fallback, q): key for key, q in phase1_map.items()}
        for future in as_completed(future_map):
            key = future_map[future]
            try:
                phase1_results[key] = future.result()
            except Exception as e:
                print(f"[DDG] Phase 1 query failed ({key}): {e}")
                phase1_results[key] = []

    all_linkedin_raw: list[dict] = []
    for key, results in phase1_results.items():
        if key.startswith("linkedin_"):
            all_linkedin_raw.extend(results)

    scholar_paper_raw = phase1_results.get("scholar_papers", [])

    # ── Extract author names from Scholar discovery results ───────────
    author_names: list[str] = []
    for result in scholar_paper_raw:
        author_names.extend(_extract_authors_from_scholar_result(result))
    seen_names: set[str] = set()
    unique_authors: list[str] = []
    for a in author_names:
        if a not in seen_names:
            seen_names.add(a)
            unique_authors.append(a)
    unique_authors = unique_authors[:10]

    # ── Phase 2a: GitHub repos via GitHub API (impact-ranked) ────────
    github_candidates: list[dict] = []
    tags = _skills_to_github_tags(skills)
    if tags:
        try:
            gh = get_github_tools()
            repo_response = gh.search_repos_by_impact(
                tags=tags,
                min_stars=MIN_REPO_STARS,
                match_all_tags=False,
                max_results=10,
            )
            if repo_response.success:
                print(f"[GitHub] Found {repo_response.total_found} impact-ranked repos for tags {tags}")
                for repo_dict in repo_response.repositories[:5]:
                    owner = (repo_dict.get("owner") or {}).get("login", "")
                    repo_name = repo_dict.get("name", "")
                    repo_html_url = repo_dict.get("html_url", f"https://github.com/{owner}/{repo_name}")
                    if not owner or not repo_name:
                        continue
                    stars = repo_dict.get("stargazers_count", 0)
                    forks = repo_dict.get("forks_count", 0)
                    if forks < MIN_REPO_FORKS:
                        print(f"[GitHub] Skip {owner}/{repo_name}: {forks} forks < {MIN_REPO_FORKS}")
                        continue
                    print(f"[GitHub] Processing {owner}/{repo_name} ({stars}⭐, {forks} forks)")
                    contribs = _process_repo_contributors(owner, repo_name, repo_html_url)
                    github_candidates.extend(_contributors_to_candidates(contribs, repo_html_url, skills))
            else:
                print(f"[GitHub] Repo search failed: {repo_response.error}")
        except Exception as e:
            print(f"[GitHub] Repo discovery error: {e}")

    # ── Phase 2b: Scholar author names → citation profile lookups ─────
    scholar_candidates: list[dict] = []
    if unique_authors:
        with ThreadPoolExecutor(max_workers=min(DDG_MAX_CONCURRENT, len(unique_authors))) as executor:
            future_map = {
                executor.submit(_search_with_fallback, f"scholar.google.com/citations {author}"): author
                for author in unique_authors
            }
            for future in as_completed(future_map):
                author = future_map[future]
                try:
                    results = future.result()
                    for result in results:
                        url = result.get("href", "")
                        if "scholar.google.com/citations" in url and "user=" in url:
                            parsed = _parse_candidate_result(result, skills)
                            if parsed:
                                scholar_candidates.append(parsed)
                except Exception as e:
                    print(f"[Scholar] Author lookup failed ({author}): {e}")

    # ── Parse all LinkedIn results (deduplicated across queries) ───────
    linkedin_candidates: list[dict] = []
    seen_li: set[str] = set()
    for r in all_linkedin_raw:
        parsed = _parse_candidate_result(r, skills)
        if not parsed:
            continue
        key = _canonicalize_url(parsed.get("profile_url", ""))
        if key and key not in seen_li:
            seen_li.add(key)
            linkedin_candidates.append(parsed)

    # ── Deduplicate and merge all sources ─────────────────────────────
    candidates: list[dict] = []
    seen_urls: set[str] = set(seen_li)
    candidates.extend(linkedin_candidates)
    for c in scholar_candidates + github_candidates:
        key = _canonicalize_url(c.get("profile_url", ""))
        if key and key not in seen_urls:
            seen_urls.add(key)
            candidates.append(c)

    _emit(
        "success",
        f"Found {len(candidates)} candidate profiles",
        f"LinkedIn: {len(linkedin_candidates)}, Scholar: {len(scholar_candidates)}, GitHub: {len(github_candidates)}",
    )
    print(
        f"[Scraper] {len(candidates)} unique profiles "
        f"(LinkedIn: {len(linkedin_candidates)}, "
        f"Scholar: {len(scholar_candidates)}, "
        f"GitHub/Personal: {len(github_candidates)})"
    )

    return candidates


def enrich_candidates_llm(
    candidates: list[dict],
    filters: dict,
) -> list[dict]:
    """LLM cleans and enriches candidate data, in batches."""
    if not candidates:
        return []

    system_prompt = (
        "You are a recruitment data analyst. Clean and enrich these REAL candidate "
        "search results. For each candidate, return a JSON object with:\n"
        '- "name": Cleaned name\n'
        '- "headline": Cleaned role/title\n'
        '- "location": Location if available, else "Unknown"\n'
        '- "profile_url": Keep the original profile URL exactly as-is\n'
        '- "snippet": Cleaned snippet\n'
        '- "matched_skills": Skills from the search that appear in their profile\n'
        '- "experience_level": Infer from title (junior/mid/senior/lead/principal)\n\n'
        "Return ONLY a JSON array. Do NOT invent information."
    )

    _emit("thinking", "Enriching candidate profiles...", "Cleaning data and extracting experience levels")
    print("[Scraper] Enriching candidate data...")

    all_enriched: list[dict] = []
    for batch_start in range(0, len(candidates), ENRICH_BATCH_SIZE):
        batch = candidates[batch_start : batch_start + ENRICH_BATCH_SIZE]
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=(
                f"Filters: skills={filters.get('skills')}, "
                f"level={filters.get('experience_level')}, "
                f"location={filters.get('location')}, "
                f"description={filters.get('description', 'none')}\n\n"
                f"Candidates:\n{json.dumps(batch, indent=2)}"
            )),
        ]
        response = llm.invoke(messages)
        enriched = _extract_json(response.content)
        if isinstance(enriched, dict):
            enriched = [enriched]

        if isinstance(enriched, list):
            for i, e in enumerate(enriched):
                orig_idx = batch_start + i
                if orig_idx < len(candidates) and not e.get("profile_url"):
                    e["profile_url"] = candidates[orig_idx].get("profile_url", "")
            all_enriched.extend(enriched)
        else:
            all_enriched.extend(batch)

    # Deduplicate by URL and by normalized name
    deduped: list[dict] = []
    seen_urls: set[str] = set()
    seen_names: set[str] = set()
    for c in all_enriched:
        url_key = _canonicalize_url(c.get("profile_url", ""))
        name_key = _normalize_name(c.get("name", ""))
        if url_key and url_key in seen_urls:
            continue
        if name_key and len(name_key) > 3 and name_key in seen_names:
            continue
        if url_key:
            seen_urls.add(url_key)
        if name_key:
            seen_names.add(name_key)
        deduped.append(c)

    _emit("success", f"Enriched {len(deduped)} candidate profiles", "Data cleaned and structured")
    print(f"[Scraper] {len(deduped)} candidates enriched (deduped from {len(all_enriched)})")
    return deduped


def evaluate_candidates_llm(
    candidates: list[dict],
    filters: dict,
) -> list[dict]:
    """LLM scores and ranks candidates 1-10 by relevance, in batches."""
    if not candidates:
        return []

    system_prompt = (
        "You are a senior tech recruiter evaluating REAL candidates from LinkedIn, "
        "Google Scholar, and GitHub open-source contributor profiles.\n"
        "Score each candidate 1-10 based on: skill match, experience level match, "
        "location match, profile quality, and how well they match the user's description.\n"
        "Be strict — reserve 8-10 for outstanding matches, 6-7 for good matches, "
        "1-4 for poor matches. Only score 5 if you are genuinely unsure.\n\n"
        "Return ONLY a JSON array with:\n"
        '- "index": The candidate\'s _batch_idx value\n'
        '- "score": 1-10\n'
        '- "reason": One sentence explaining the score\n'
        "Return ONLY the JSON array."
    )

    _emit("thinking", "Evaluating and ranking candidates...", "Scoring each candidate on skill match, experience, and relevance")
    print("[Scraper] Evaluating and ranking candidates...")

    eval_map: dict[int, dict] = {}
    for batch_start in range(0, len(candidates), ENRICH_BATCH_SIZE):
        batch = candidates[batch_start : batch_start + ENRICH_BATCH_SIZE]
        indexed_batch = [{"_batch_idx": i, **c} for i, c in enumerate(batch)]
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=(
                f"Required: skills={filters.get('skills')}, "
                f"level={filters.get('experience_level', 'any')}, "
                f"location={filters.get('location', 'any')}, "
                f"description={filters.get('description', 'none')}\n\n"
                f"Candidates:\n{json.dumps(indexed_batch, indent=2)}"
            )),
        ]
        response = llm.invoke(messages)
        evaluations = _extract_json(response.content)
        if not isinstance(evaluations, list):
            evaluations = []
        for ev in evaluations:
            if isinstance(ev, dict) and "index" in ev:
                global_idx = batch_start + ev["index"]
                eval_map[global_idx] = ev

    evaluated = []
    for i, c in enumerate(candidates):
        ev = eval_map.get(i, {"score": 5, "reason": "Not evaluated"})
        c["_score"] = ev.get("score", 5)
        c["_reason"] = ev.get("reason", "")
        evaluated.append(c)

    evaluated.sort(key=lambda x: x.get("_score", 0), reverse=True)

    _emit("success", f"Ranked {len(evaluated)} candidates by relevance", "Candidates sorted by match score")
    print(f"[Scraper] {len(evaluated)} candidates scored and ranked")
    return evaluated


def finalize_candidates(evaluated: list[dict]) -> list[dict]:
    """Deduplicate and cap evaluated candidates to the top qualified ones."""
    top = [c for c in evaluated if c.get("_score", 0) >= MIN_CANDIDATE_SCORE]

    final: list[dict] = []
    seen_urls: set[str] = set()
    seen_names: set[str] = set()
    for c in top:
        url_key = _canonicalize_url(c.get("profile_url", ""))
        name_key = _normalize_name(c.get("name", ""))
        if url_key and url_key in seen_urls:
            continue
        if name_key and len(name_key) > 3 and name_key in seen_names:
            continue
        if url_key:
            seen_urls.add(url_key)
        if name_key:
            seen_names.add(name_key)
        final.append(c)

    final = final[:TOP_CANDIDATES_LIMIT]
    print(
        f"[Scraper] Finalized: {len(evaluated)} evaluated → "
        f"{len(final)} unique top candidates (score ≥ {MIN_CANDIDATE_SCORE}, cap {TOP_CANDIDATES_LIMIT})"
    )
    return final
