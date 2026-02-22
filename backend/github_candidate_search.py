"""GitHub-driven candidate sourcing helpers."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict

from github.tools import get_github_tools
from linkedin_search import find_linkedin_profile_for_identity


def _avatar_initials(name: str) -> str:
    parts = [p for p in name.split() if p]
    return "".join(p[0].upper() for p in parts[:2]) or "GH"


def _slugify_tag(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9\-\s]", "", value).strip().lower()
    cleaned = re.sub(r"\s+", "-", cleaned)
    return cleaned.strip("-")


def build_github_tags(skill_terms: list[str], description: str = "") -> list[str]:
    """Build GitHub topic tags from skill terms."""
    tags: list[str] = []
    for skill in skill_terms:
        tag = _slugify_tag(skill)
        if tag:
            tags.append(tag)

    if not tags and description.strip():
        words = re.findall(r"[A-Za-z][A-Za-z0-9\-\+]{2,}", description.lower())
        tags.extend(_slugify_tag(w) for w in words[:5] if _slugify_tag(w))

    deduped = []
    seen = set()
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            deduped.append(tag)
    return deduped[:8]


def search_github_candidates(
    skill_terms: list[str],
    keywords: list[str],
    description: str,
    location: str,
    target_count: int = 50,
    min_repo_stars: int = 100,
    min_repo_clones_14d: int = 30,
) -> tuple[list[dict], str | None]:
    """
    Source candidates from GitHub:
    1) Search high-impact repos by tags
    2) Keep significant repos (stars/clones threshold)
    3) Aggregate top contributors
    4) Try finding contributor LinkedIn profiles
    """
    tags = build_github_tags(skill_terms=skill_terms, description=description)
    if not tags:
        return [], "No GitHub tags could be derived from skills/description."

    github_tools = get_github_tools()
    repo_response = github_tools.search_repos_by_impact(
        tags=tags,
        min_stars=max(100, min_repo_stars // 2),
        max_results=30,
        match_all_tags=False,
        include_clones=True,
    )
    if not repo_response.success:
        # Fallback: per-tag repo search, then aggregate.
        repos = []
        fallback_errors: list[str] = []
        for tag in tags[:4]:
            fallback = github_tools.search_repos(
                tag=tag,
                min_stars=50,
                max_results=15,
            )
            if fallback.success and fallback.repositories:
                repos.extend(fallback.repositories)
            elif fallback.error:
                fallback_errors.append(f"{tag}: {fallback.error}")
        if not repos:
            reason = repo_response.error or "; ".join(fallback_errors) or "No repositories found."
            return [], f"GitHub repo search failed. {reason}"
    else:
        repos = repo_response.repositories

    significant_repos = []
    for repo in repos:
        metrics = repo.get("impact_metrics", {})
        stars = int(metrics.get("stars", repo.get("stargazers_count", 0)) or 0)
        clones = int(metrics.get("clones_14d", 0) or 0) + int(
            metrics.get("unique_clones_14d", 0) or 0
        )
        if stars >= min_repo_stars or clones >= min_repo_clones_14d:
            significant_repos.append(repo)

    if not significant_repos:
        significant_repos = repos[:12]

    contributor_scores: dict[str, float] = defaultdict(float)
    contributor_profiles: dict[str, dict] = {}

    for repo in significant_repos[:10]:
        owner = (repo.get("owner") or {}).get("login", "")
        repo_name = repo.get("name", "")
        repo_full_name = repo.get("full_name", f"{owner}/{repo_name}")
        impact_score = float(repo.get("impact_score", 0.0) or 0.0)
        if not owner or not repo_name:
            continue

        contributors_response = github_tools.get_contributors(
            owner=owner,
            repo=repo_name,
            max_results=15,
        )
        if not contributors_response.success:
            continue

        for contributor in contributors_response.contributors:
            login = contributor.login
            weighted = float(contributor.contributions) * (1.0 + (impact_score / 100.0))
            contributor_scores[login] += weighted

            if login not in contributor_profiles:
                contributor_profiles[login] = {
                    "login": login,
                    "github_url": contributor.html_url or f"https://github.com/{login}",
                    "repos": set(),
                }
            contributor_profiles[login]["repos"].add(repo_full_name)

    ranked = sorted(
        contributor_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )
    if not ranked:
        return [], "GitHub repos found, but no contributors were returned."

    github_candidates: list[dict] = []
    linkedin_lookup_limit = min(max(target_count, 15), 25)
    contributor_limit = min(max(target_count * 2, 40), 120)

    for idx, (login, score) in enumerate(ranked[:contributor_limit]):
        profile = contributor_profiles.get(login, {})
        github_url = profile.get("github_url", f"https://github.com/{login}")
        repo_names = sorted(profile.get("repos", set()))

        linkedin_match = None
        if idx < linkedin_lookup_limit:
            guesses = [login, login.replace("-", " "), login.replace("_", " ")]
            for guess in guesses:
                linkedin_match = find_linkedin_profile_for_identity(
                    identity=guess,
                    keywords=keywords,
                    location=location,
                )
                if linkedin_match:
                    break

        candidate_name = (linkedin_match or {}).get("name") or login
        headline = (linkedin_match or {}).get("headline") or "Open-source contributor"
        linkedin_url = (linkedin_match or {}).get("profile_url")

        candidate_id = hashlib.md5(f"github:{login}".encode("utf-8")).hexdigest()[:12]
        top_repos = ", ".join(repo_names[:3])
        snippet = (
            f"GitHub impact score {round(score, 1)} from {len(repo_names)} significant repos."
            f" Top repos: {top_repos}" if top_repos else f"GitHub impact score {round(score, 1)}."
        )

        github_candidates.append(
            {
                "id": candidate_id,
                "name": candidate_name,
                "headline": headline,
                "location": location,
                "profile_url": linkedin_url or github_url,
                "linkedin_url": linkedin_url,
                "github_url": github_url,
                "snippet": snippet,
                "matched_skills": skill_terms[:8],
                "avatar_initials": (linkedin_match or {}).get("avatar_initials")
                or _avatar_initials(candidate_name),
            }
        )

    return github_candidates[:target_count], None
