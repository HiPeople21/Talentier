import asyncio
import hashlib
import json
import random
import re
import time
import urllib.parse
import urllib.request
from typing import Literal
from urllib.parse import unquote

import httpx
from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from models import Candidate, SearchFilters

# ── Config ────────────────────────────────────────────────────────────
OLLAMA_MODEL = "llama3.2"  # 2GB — ~2x faster than qwen2.5 while still capable
MAX_REFINE_LOOPS = 1
CACHE_TTL = 300

_cache: dict[str, tuple[float, list[Candidate]]] = {}

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

DDG_URL = "https://html.duckduckgo.com/html/"


# ── Agent State ───────────────────────────────────────────────────────
class AgentState(TypedDict):
    filters: dict
    search_query: str
    raw_results: list[dict]
    parsed_candidates: list[dict]
    enriched_candidates: list[dict]
    evaluated_candidates: list[dict]
    refinement_feedback: str
    refinement_count: int
    final_candidates: list[dict]


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


def _extract_ddg_url(href: str) -> str:
    if "uddg=" in href:
        match = re.search(r"uddg=([^&]+)", href)
        if match:
            return unquote(match.group(1))
    return href

def _search_brave(query: str) -> list[dict]:
    """Search Brave Search — zero bot protection, fast, reliable."""
    import subprocess
    print(f"[Agent] 🔍 [Engine 0] Brave Search: {query}")
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
            print("[Engine 0] Empty response")
            return []
        
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        
        results = []
        # Brave uses <a> tags with LinkedIn hrefs throughout the page
        seen = set()
        for a in soup.find_all("a", href=re.compile(r"linkedin\.com/in/")):
            href = a.get("href", "")
            if href in seen:
                continue
            seen.add(href)
            
            # Find the nearest text content for title/snippet
            title = a.get_text(strip=True) or ""
            
            # Walk up to find a snippet/description near this link
            snippet = ""
            parent = a.find_parent(attrs={"class": True})
            if parent:
                desc = parent.find(["p", "span", "div"], class_=lambda c: c and ("description" in str(c).lower() or "snippet" in str(c).lower()))
                if desc:
                    snippet = desc.get_text(strip=True)
            
            if not title:
                # Use URL as title fallback
                match = re.search(r"linkedin\.com/in/([^/\?]+)", href)
                title = match.group(1).replace("-", " ").title() if match else href
            
            results.append({"title": title, "href": href, "snippet": snippet})
        
        print(f"[Agent] ✅ [Engine 0] Found {len(results)} LinkedIn results")
        return results
    except Exception as e:
        print(f"[Engine 0 Error] {e}")
        return []

def _search_ddg_html(query: str) -> list[dict]:
    """Search DuckDuckGo HTML endpoint directly via native urllib POST."""
    print(f"[Agent] 🔍 [Engine 1] DuckDuckGo HTML: {query}")
    try:
        url = "https://html.duckduckgo.com/html/"
        data = urllib.parse.urlencode({"q": query}).encode("utf-8")
        
        headers = _random_headers()
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        
        req = urllib.request.Request(
            url,
            data=data,
            headers=headers,
        )
        
        html = urllib.request.urlopen(req, timeout=10).read()
        
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        
        results = []
        for el in soup.select(".result"):
            link_el = el.select_one("a.result__url")
            snip_el = el.select_one("a.result__snippet")
            
            if not link_el: continue
            
            title = link_el.get_text(strip=True)
            raw_href = link_el.get("href", "")
            href = _extract_ddg_url(raw_href)
            snippet = snip_el.get_text(strip=True) if snip_el else ""
            
            if "linkedin.com/in" in href:
                results.append({"title": title, "href": href, "snippet": snippet})
                
        print(f"[Agent] ✅ [Engine 1] Found {len(results)} LinkedIn results")
        return results
    except Exception as e:
        print(f"[Engine 1 Error] {e}")
        return []


def _search_ddgs_package(query: str) -> list[dict]:
    """Search DuckDuckGo using the duckduckgo_search Python package."""
    print(f"[Agent] 🔍 [Engine 2] duckduckgo_search package: {query}")
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            raw = list(ddgs.text(query, max_results=15))
        
        results = []
        for r in raw:
            href = r.get("href", "")
            if "linkedin.com/in" in href:
                results.append({
                    "title": r.get("title", ""),
                    "href": href,
                    "snippet": r.get("body", "")
                })
        
        print(f"[Agent] ✅ [Engine 2] Found {len(results)} LinkedIn results")
        return results
    except Exception as e:
        print(f"[Engine 2 Error] {e}")
        return []


def _search_google_pkg(query: str) -> list[dict]:
    """Search Google using the googlesearch-python package."""
    print(f"[Agent] 🔍 [Engine 3] googlesearch-python: {query}")
    try:
        from googlesearch import search as gsearch
        raw = gsearch(query, num_results=15, advanced=True, sleep_interval=1)
        
        results = []
        for r in raw:
            if "linkedin.com/in" in r.url:
                results.append({
                    "title": r.title,
                    "href": r.url,
                    "snippet": r.description
                })
        
        print(f"[Agent] ✅ [Engine 3] Found {len(results)} LinkedIn results")
        return results
    except Exception as e:
        print(f"[Engine 3 Error] {e}")
        return []


def _search_multi(query: str) -> list[dict]:
    """Try multiple search engines in order until one returns results."""
    engines = [
        _search_brave,        # Engine 0: Brave Search (no bot protection)
        _search_ddg_html,     # Engine 1: DuckDuckGo HTML
        _search_ddgs_package, # Engine 2: duckduckgo_search package
        _search_google_pkg,   # Engine 3: googlesearch-python
    ]
    
    for engine in engines:
        results = engine(query)
        if results:
            # Deduplicate
            unique = []
            seen = set()
            for r in results:
                if r["href"] not in seen:
                    seen.add(r["href"])
                    unique.append(r)
            return unique
    
    print("[Agent] ⚠️ All search engines failed to return results")
    return []


def _parse_linkedin_result(result: dict, search_skills: list[str]) -> dict | None:
    url = result.get("href", "")
    title = result.get("title", "")
    snippet = result.get("snippet", "")

    if "linkedin.com/in/" not in url:
        return None

    title_cleaned = re.sub(r"\s*[\|–-]\s*LinkedIn.*$", "", title).strip()
    parts = [p.strip() for p in title_cleaned.split(" - ", 2)]
    name = parts[0] if parts else "Unknown"
    headline = " - ".join(parts[1:]) if len(parts) > 1 else ""

    if not name or len(name) < 2 or name.lower() in ["linkedin", "sign in", "log in"]:
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

    return {
        "name": name,
        "headline": headline,
        "location": location,
        "profile_url": url,
        "snippet": snippet[:300],
        "matched_skills": matched,
    }


# ══════════════════════════════════════════════════════════════════════
# LangGraph Agent Nodes (all 3 LLM calls preserved for full agency)
# ══════════════════════════════════════════════════════════════════════

# ── Node 1: Plan Search (LLM) ────────────────────────────────────────
def plan_search(state: AgentState) -> dict:
    """LLM agent plans the optimal search query for DuckDuckGo."""
    filters = state["filters"]
    skills = ", ".join(filters.get("skills", []))
    level = filters.get("experience_level", "")
    location = filters.get("location", "")
    description = filters.get("description", "")
    refinement = state.get("refinement_feedback", "")

    refinement_note = ""
    if refinement:
        refinement_note = f"\nPrevious search had issues: {refinement}\nAdjust the query to be broader."

    description_note = ""
    if description:
        description_note = f"\n7. Consider these extra descriptors: {description}"

    messages = [
        SystemMessage(content=(
            "You are a recruitment search agent. Create a DuckDuckGo search query "
            "to find LinkedIn profiles matching the criteria.\n\n"
            "Rules:\n"
            "1. MUST start exactly with: site:linkedin.com/in\n"
            "2. Include 1-2 most important skills\n"
            "3. Include experience level keywords if specified\n"
            "4. Include location if specified\n"
            "5. Return ONLY the query string\n"
            "6. Keep concise: 5-10 words total"
            f"{description_note}\n\n"
            "Example: site:linkedin.com/in Python senior engineer San Francisco"
        )),
        HumanMessage(content=(
            f"Skills: {skills or 'any'}\nLevel: {level or 'any'}\n"
            f"Location: {location or 'any'}\n"
            f"Description: {description or 'none'}{refinement_note}"
        )),
    ]

    _emit("thinking", "Planning search query...", "Analyzing filters to build optimal LinkedIn search")
    print("[Agent] 🧠 Planning search query...")
    response = llm.invoke(messages)
    query = response.content.strip().strip('"\'`')
    if "\n" in query:
        query = query.split("\n")[0].strip()

    _emit("success", f"Search query ready", query)
    print(f"[Agent] ✅ Query: {query}")
    return {"search_query": query}


# ── Node 2: Search LinkedIn (DDG HTTP — no LLM) ──────────────────────
def search_linkedin(state: AgentState) -> dict:
    """Fetch REAL LinkedIn profiles from DuckDuckGo."""
    query = state["search_query"]
    skills = state["filters"].get("skills", [])

    _emit("searching", "Searching for LinkedIn profiles...", f"Query: {query}")
    print(f"[Agent] 🔍 Searching (multi-engine)...")
    raw_results = _search_multi(query)
    print(f"[Agent] ✅ {len(raw_results)} results found")

    candidates = []
    seen = set()
    for r in raw_results:
        parsed = _parse_linkedin_result(r, skills)
        if parsed and parsed["profile_url"] not in seen:
            seen.add(parsed["profile_url"])
            candidates.append(parsed)

    _emit("success", f"Found {len(candidates)} LinkedIn profiles", f"{len(raw_results)} total results, {len(candidates)} profile pages")
    print(f"[Agent] ✅ {len(candidates)} LinkedIn profiles parsed")
    return {"raw_results": raw_results, "parsed_candidates": candidates}


# ── Node 3: Enrich (LLM) ─────────────────────────────────────────────
def enrich_candidates(state: AgentState) -> dict:
    """LLM agent cleans and enriches the real candidate data."""
    candidates = state["parsed_candidates"]
    filters = state["filters"]

    if not candidates:
        return {"enriched_candidates": []}

    messages = [
        SystemMessage(content=(
            "You are a recruitment data analyst. Clean these REAL LinkedIn results.\n"
            "For each candidate return a JSON object with:\n"
            '- "name": Cleaned name\n'
            '- "headline": Cleaned title/role\n'
            '- "location": Location or "Unknown"\n'
            '- "profile_url": KEEP the original LinkedIn URL exactly\n'
            '- "snippet": Cleaned summary\n'
            '- "matched_skills": Skills from the search matching their profile\n'
            '- "experience_level": Infer from title (junior/mid/senior/lead/principal)\n\n'
            "Return ONLY a JSON array. Do NOT invent information."
        )),
        HumanMessage(content=(
            f"Filters: skills={filters.get('skills')}, "
            f"level={filters.get('experience_level')}, "
            f"location={filters.get('location')}, "
            f"description={filters.get('description', 'none')}\n\n"
            f"Candidates:\n{json.dumps(candidates, indent=2)}"
        )),
    ]

    _emit("thinking", "Enriching candidate profiles...", "Cleaning data and extracting experience levels")
    print("[Agent] 🧠 Enriching candidate data...")
    response = llm.invoke(messages)
    enriched = _extract_json(response.content)
    if isinstance(enriched, dict):
        enriched = [enriched]

    if isinstance(enriched, list):
        for i, e in enumerate(enriched):
            if i < len(candidates) and not e.get("profile_url"):
                e["profile_url"] = candidates[i].get("profile_url", "")

    count = len(enriched) if isinstance(enriched, list) else 0
    _emit("success", f"Enriched {count} candidate profiles", "Data cleaned and structured")
    print(f"[Agent] ✅ {count} candidates enriched")
    return {"enriched_candidates": enriched if isinstance(enriched, list) else candidates}


# ── Node 4: Evaluate (LLM) ───────────────────────────────────────────
def evaluate_candidates(state: AgentState) -> dict:
    """LLM agent scores and ranks candidates by relevance."""
    candidates = state["enriched_candidates"]
    filters = state["filters"]

    if not candidates:
        return {
            "evaluated_candidates": [],
            "refinement_feedback": "No profiles found. Broaden search.",
        }

    messages = [
        SystemMessage(content=(
            "You are a senior tech recruiter. Score each candidate 1-10 based on:\n"
            "skill match, experience level match, location match, profile quality,"
            " and how well they match the user's description.\n\n"
            "Return ONLY a JSON array with:\n"
            '- "index": Position (0-based)\n'
            '- "score": 1-10\n'
            '- "reason": One sentence\n'
        )),
        HumanMessage(content=(
            f"Required: skills={filters.get('skills')}, "
            f"level={filters.get('experience_level', 'any')}, "
            f"location={filters.get('location', 'any')}, "
            f"description={filters.get('description', 'none')}\n\n"
            f"Candidates:\n{json.dumps(candidates, indent=2)}"
        )),
    ]

    _emit("thinking", "Evaluating and ranking candidates...", "Scoring each candidate on skill match, experience, and relevance")
    print("[Agent] 🧠 Evaluating and ranking candidates...")
    response = llm.invoke(messages)
    evaluations = _extract_json(response.content)
    if not isinstance(evaluations, list):
        evaluations = []

    eval_map = {}
    for ev in evaluations:
        if isinstance(ev, dict) and "index" in ev:
            eval_map[ev["index"]] = ev

    evaluated = []
    for i, c in enumerate(candidates):
        ev = eval_map.get(i, {"score": 5, "reason": "Not evaluated"})
        c["_score"] = ev.get("score", 5)
        c["_reason"] = ev.get("reason", "")
        evaluated.append(c)

    evaluated.sort(key=lambda x: x.get("_score", 0), reverse=True)

    feedback = ""
    if len(evaluated) < 3:
        feedback = "Too few results. Broaden the query."

    _emit("success", f"Ranked {len(evaluated)} candidates by relevance", "Candidates sorted by match score")
    print(f"[Agent] ✅ {len(evaluated)} candidates scored and ranked")
    return {"evaluated_candidates": evaluated, "refinement_feedback": feedback}


# ── Node 5: Decide ───────────────────────────────────────────────────
def decide(state: AgentState) -> Literal["refine", "finish"]:
    feedback = state.get("refinement_feedback", "")
    count = state.get("refinement_count", 0)
    if feedback and count < MAX_REFINE_LOOPS:
        _emit("refining", "Refining search...", "Results were insufficient, trying a broader query")
        print("[Agent] 🔄 Refining — results were insufficient")
        return "refine"
    _emit("success", "Results look good — finalizing", "")
    print("[Agent] ✅ Results look good — finalizing")
    return "finish"


def prepare_refinement(state: AgentState) -> dict:
    return {"refinement_count": state.get("refinement_count", 0) + 1}


def finalize(state: AgentState) -> dict:
    return {"final_candidates": state.get("evaluated_candidates", [])}


# ══════════════════════════════════════════════════════════════════════
# Build the LangGraph — fully agentic with 3 LLM-powered nodes
# ══════════════════════════════════════════════════════════════════════
def _build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("plan_search", plan_search)         # LLM plans query
    graph.add_node("search_linkedin", search_linkedin) # DDG HTTP fetch
    graph.add_node("enrich", enrich_candidates)        # LLM enriches data
    graph.add_node("evaluate", evaluate_candidates)    # LLM scores & ranks
    graph.add_node("refine", prepare_refinement)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("plan_search")
    graph.add_edge("plan_search", "search_linkedin")
    graph.add_edge("search_linkedin", "enrich")
    graph.add_edge("enrich", "evaluate")

    graph.add_conditional_edges(
        "evaluate",
        decide,
        {"refine": "refine", "finish": "finalize"},
    )

    graph.add_edge("refine", "plan_search")
    graph.add_edge("finalize", END)

    return graph


agent_graph = _build_graph().compile()


# ══════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════
def _to_candidate(data: dict, search_skills: list[str]) -> Candidate:
    name = data.get("name", "Unknown")
    cid = hashlib.md5(f"{name}-{data.get('headline', '')}".encode()).hexdigest()[:12]

    profile_url = data.get("profile_url", "")
    if not profile_url or "linkedin.com" not in profile_url:
        slug = re.sub(r"[^a-z0-9-]", "", name.lower().replace(" ", "-"))
        profile_url = f"https://www.linkedin.com/in/{slug}"

    cand_skills = data.get("matched_skills", [])
    if not cand_skills:
        text = f"{data.get('headline', '')} {data.get('snippet', '')}".lower()
        cand_skills = [s for s in search_skills if s.lower() in text]

    return Candidate(
        id=cid,
        name=name,
        headline=data.get("headline", ""),
        location=data.get("location", ""),
        profile_url=profile_url,
        snippet=data.get("snippet", ""),
        matched_skills=cand_skills[:6],
        avatar_initials=_get_initials(name),
    )


async def search_linkedin_profiles(
    filters: SearchFilters,
) -> tuple[list[Candidate], bool]:
    """Run the fully agentic LangGraph pipeline."""
    cache_key = f"{sorted(filters.skills)}:{filters.experience_level}:{filters.location}:{filters.description}:{filters.page}"
    key = hashlib.md5(cache_key.encode()).hexdigest()

    if key in _cache:
        ts, cached = _cache[key]
        if time.time() - ts < CACHE_TTL:
            return cached, False

    initial_state: AgentState = {
        "filters": {
            "skills": filters.skills,
            "experience_level": filters.experience_level or "",
            "location": filters.location or "",
            "description": filters.description or "",
        },
        "search_query": "",
        "raw_results": [],
        "parsed_candidates": [],
        "enriched_candidates": [],
        "evaluated_candidates": [],
        "refinement_feedback": "",
        "refinement_count": 0,
        "final_candidates": [],
    }

    try:
        # Reset agent status for SSE streaming
        agent_status["steps"] = []
        agent_status["done"] = False
        _emit("start", "Starting AI agent search...", "Initializing LangGraph pipeline")

        print("\n[Agent] 🚀 Starting agentic candidate search...")
        # Run synchronous LangGraph execution in a separate thread to prevent blocking
        # the FastAPI event loop so the SSE endpoint can stream agent thoughts in real-time.
        result = await asyncio.to_thread(agent_graph.invoke, initial_state)
        raw_list = result.get("final_candidates", [])

        candidates = []
        for data in raw_list:
            try:
                candidates.append(_to_candidate(data, filters.skills))
            except Exception:
                continue

        print(f"[Agent] 🎯 Done — {len(candidates)} real candidates returned\n")

        if candidates:
            _cache[key] = (time.time(), candidates)

        agent_status["done"] = True
        return candidates, False

    except Exception as e:
        _emit("error", f"Agent error: {str(e)}", "")
        agent_status["done"] = True
        print(f"[Agent Error] {e}")
        import traceback
        traceback.print_exc()
        return [], False
