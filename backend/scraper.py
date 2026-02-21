"""
LangGraph-powered agentic LinkedIn candidate discovery using REAL data.

Uses DuckDuckGo HTML search (no API key, no rate limits) to find actual
LinkedIn profiles, then uses the Ollama LLM agent to enrich and rank them.

Agent graph:
  plan_search → search_linkedin → enrich → evaluate → decide → [refine] or [finalize]
"""

import hashlib
import json
import re
import time
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
OLLAMA_MODEL = "qwen2.5"
MAX_REFINE_LOOPS = 1
CACHE_TTL = 300

_cache: dict[str, tuple[float, list[Candidate]]] = {}

llm = ChatOllama(
    model=OLLAMA_MODEL,
    temperature=0.7,
    num_predict=4096,
)

DDG_URL = "https://html.duckduckgo.com/html/"
DDG_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


# ── Agent State ───────────────────────────────────────────────────────
class AgentState(TypedDict):
    filters: dict
    search_query: str
    raw_results: list[dict]          # Raw DuckDuckGo search results
    parsed_candidates: list[dict]    # Parsed from search results
    enriched_candidates: list[dict]  # LLM-enriched candidates
    evaluated_candidates: list[dict] # Scored and ranked
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
    """Extract the actual URL from DuckDuckGo's redirect wrapper."""
    # DDG wraps URLs like //duckduckgo.com/l/?uddg=<encoded_url>&rut=...
    if "uddg=" in href:
        match = re.search(r"uddg=([^&]+)", href)
        if match:
            return unquote(match.group(1))
    return href


def _search_ddg(query: str) -> list[dict]:
    """Search DuckDuckGo HTML endpoint and return parsed results."""
    try:
        resp = httpx.post(
            DDG_URL,
            data={"q": query},
            headers=DDG_HEADERS,
            follow_redirects=True,
            timeout=15.0,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"[DDG] Request failed: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []

    for link in soup.select(".result__a"):
        title = link.get_text(strip=True)
        raw_href = link.get("href", "")
        url = _extract_ddg_url(raw_href)

        # Get the snippet
        snippet_el = link.find_parent("div", class_="result")
        snippet = ""
        if snippet_el:
            snippet_tag = snippet_el.select_one(".result__snippet")
            if snippet_tag:
                snippet = snippet_tag.get_text(strip=True)

        results.append({
            "title": title,
            "href": url,
            "snippet": snippet,
        })

    return results


def _parse_linkedin_result(result: dict, search_skills: list[str]) -> dict | None:
    """Parse a DDG search result into a candidate dict."""
    url = result.get("href", "")
    title = result.get("title", "")
    snippet = result.get("snippet", "")

    if "linkedin.com/in/" not in url:
        return None

    # Parse: "Jane Doe - Senior Engineer - Google | LinkedIn"
    title_cleaned = re.sub(r"\s*\|\s*LinkedIn.*$", "", title).strip()
    title_cleaned = re.sub(r"\s*-\s*LinkedIn.*$", "", title_cleaned).strip()
    parts = [p.strip() for p in title_cleaned.split(" - ", 2)]
    name = parts[0] if parts else "Unknown"
    headline = " - ".join(parts[1:]) if len(parts) > 1 else ""

    if not name or name.lower() in ["linkedin", "sign in", "log in", ""]:
        return None

    # Extract location from snippet
    location = ""
    loc_patterns = [
        r"(?:Location|Area|Region|Based in)[:\s]+([^·\n.]+)",
        r"(?:located in|based in)\s+([^·\n.]+)",
    ]
    for pattern in loc_patterns:
        match = re.search(pattern, snippet, re.IGNORECASE)
        if match:
            location = match.group(1).strip()[:50]
            break

    # Match skills
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
# LangGraph Agent Nodes
# ══════════════════════════════════════════════════════════════════════

def plan_search(state: AgentState) -> dict:
    """LLM plans the optimal search query."""
    filters = state["filters"]
    skills = ", ".join(filters.get("skills", []))
    level = filters.get("experience_level", "")
    location = filters.get("location", "")
    refinement = state.get("refinement_feedback", "")

    refinement_note = ""
    if refinement:
        refinement_note = f"\n\nPrevious search had issues: {refinement}\nAdjust the query accordingly."

    messages = [
        SystemMessage(content=(
            "You are a recruitment search specialist. Create a DuckDuckGo search query "
            "to find LinkedIn profiles (/in/ pages) matching the given criteria.\n\n"
            "Rules:\n"
            "1. Include 'linkedin.com/in' in the query (NOT site: — DDG doesn't support it well)\n"
            "2. Include the most important 1-2 skills\n"
            "3. Include experience level keywords if specified\n"
            "4. Include location if specified\n"
            "5. Return ONLY the search query string — nothing else\n"
            "6. Keep it concise — 5-8 words max after 'linkedin.com/in'\n\n"
            "Example: linkedin.com/in Python senior software engineer San Francisco"
        )),
        HumanMessage(content=(
            f"Skills: {skills or 'any'}\n"
            f"Level: {level or 'any'}\n"
            f"Location: {location or 'any'}"
            f"{refinement_note}"
        )),
    ]

    response = llm.invoke(messages)
    query = response.content.strip().strip('"').strip("'").strip("`")
    if "\n" in query:
        query = query.split("\n")[0].strip()

    print(f"[Agent] Planned query: {query}")
    return {"search_query": query}


def search_linkedin(state: AgentState) -> dict:
    """Fetch REAL LinkedIn profiles from DuckDuckGo."""
    query = state["search_query"]
    filters = state["filters"]
    skills = filters.get("skills", [])

    print(f"[Agent] Searching DuckDuckGo: {query}")
    raw_results = _search_ddg(query)
    print(f"[Agent] Got {len(raw_results)} raw results")

    # Parse LinkedIn profiles
    candidates = []
    seen_urls = set()
    for r in raw_results:
        parsed = _parse_linkedin_result(r, skills)
        if parsed and parsed["profile_url"] not in seen_urls:
            seen_urls.add(parsed["profile_url"])
            candidates.append(parsed)

    print(f"[Agent] Parsed {len(candidates)} LinkedIn profiles")

    return {
        "raw_results": raw_results,
        "parsed_candidates": candidates,
    }


def enrich_candidates(state: AgentState) -> dict:
    """LLM cleans and enriches the real candidate data."""
    candidates = state["parsed_candidates"]
    filters = state["filters"]

    if not candidates:
        return {"enriched_candidates": []}

    messages = [
        SystemMessage(content=(
            "You are a recruitment data analyst. Clean and enrich these REAL LinkedIn "
            "search results. For each candidate, return a JSON object with:\n"
            '- "name": Cleaned name\n'
            '- "headline": Cleaned role/title\n'
            '- "location": Location if available, else "Unknown"\n'
            '- "profile_url": Keep the original LinkedIn URL exactly as-is\n'
            '- "snippet": Cleaned snippet\n'
            '- "matched_skills": Skills from the search that appear in their profile\n'
            '- "experience_level": Infer from title (junior/mid/senior/lead/principal)\n\n'
            "Return ONLY a JSON array. Do NOT invent information."
        )),
        HumanMessage(content=(
            f"Filters: skills={filters.get('skills')}, "
            f"level={filters.get('experience_level')}, "
            f"location={filters.get('location')}\n\n"
            f"Candidates:\n{json.dumps(candidates, indent=2)}"
        )),
    ]

    response = llm.invoke(messages)
    enriched = _extract_json(response.content)
    if isinstance(enriched, dict):
        enriched = [enriched]

    # Preserve original URLs in case LLM strips them
    if isinstance(enriched, list):
        for i, e in enumerate(enriched):
            if i < len(candidates) and not e.get("profile_url"):
                e["profile_url"] = candidates[i].get("profile_url", "")

    return {"enriched_candidates": enriched if isinstance(enriched, list) else candidates}


def evaluate_candidates(state: AgentState) -> dict:
    """LLM scores and ranks candidates by relevance."""
    candidates = state["enriched_candidates"]
    filters = state["filters"]

    if not candidates:
        return {
            "evaluated_candidates": [],
            "refinement_feedback": "No LinkedIn profiles found. Try broader skills.",
        }

    messages = [
        SystemMessage(content=(
            "You are a senior tech recruiter evaluating REAL LinkedIn candidates.\n"
            "Score each candidate 1-10 based on: skill match, experience level match, "
            "location match, and profile quality.\n\n"
            "Return ONLY a JSON array with:\n"
            '- "index": Position (0-based)\n'
            '- "score": 1-10\n'
            '- "reason": One sentence\n'
            "Return ONLY the JSON array."
        )),
        HumanMessage(content=(
            f"Required: skills={filters.get('skills')}, "
            f"level={filters.get('experience_level', 'any')}, "
            f"location={filters.get('location', 'any')}\n\n"
            f"Candidates:\n{json.dumps(candidates, indent=2)}"
        )),
    ]

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
        feedback = "Too few results. Broaden the query — fewer skills, drop location."

    return {
        "evaluated_candidates": evaluated,
        "refinement_feedback": feedback,
    }


def decide(state: AgentState) -> Literal["refine", "finish"]:
    feedback = state.get("refinement_feedback", "")
    count = state.get("refinement_count", 0)
    if feedback and count < MAX_REFINE_LOOPS:
        return "refine"
    return "finish"


def prepare_refinement(state: AgentState) -> dict:
    return {"refinement_count": state.get("refinement_count", 0) + 1}


def finalize(state: AgentState) -> dict:
    return {"final_candidates": state.get("evaluated_candidates", [])}


# ══════════════════════════════════════════════════════════════════════
# Build the LangGraph
# ══════════════════════════════════════════════════════════════════════
def _build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("plan_search", plan_search)
    graph.add_node("search_linkedin", search_linkedin)
    graph.add_node("enrich", enrich_candidates)
    graph.add_node("evaluate", evaluate_candidates)
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
    """Run the LangGraph agent to discover REAL LinkedIn candidates."""
    cache_key = f"{sorted(filters.skills)}:{filters.experience_level}:{filters.location}:{filters.page}"
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
        print("[Agent] Starting LangGraph candidate search...")
        result = agent_graph.invoke(initial_state)
        raw_list = result.get("final_candidates", [])

        candidates = []
        for data in raw_list:
            try:
                candidates.append(_to_candidate(data, filters.skills))
            except Exception:
                continue

        print(f"[Agent] Returning {len(candidates)} real candidates")

        if candidates:
            _cache[key] = (time.time(), candidates)

        return candidates, False

    except Exception as e:
        print(f"[LangGraph Agent Error] {e}")
        import traceback
        traceback.print_exc()
        return [], False
