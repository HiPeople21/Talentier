"""
Complete LangGraph Implementation for Candidate Sourcing Pipeline

This file shows how to wire up all the nodes using the state structure from state.py
"""

from langgraph.graph import StateGraph, END, START
from typing import Literal
import hashlib
import json
import math
import re
from datetime import datetime, timezone

from state import (
    CandidateSourcingState,
    SourceType,
    SourceResult,
    SourceCandidate,
    ScoreType,
    ScoreMetadata,
    PipelineStage,
    StageMetrics,
    RankingStrategy,
    DefaultRankingStrategy,
    create_initial_state,
    get_all_source_results,
)
from github.tools import get_github_tools
from github.code_quality_agent import get_code_quality_agent


# ============================================================================
# NODE IMPLEMENTATIONS
# ============================================================================

def linkedin_query_node(state: CandidateSourcingState) -> dict:
    """
    Discover candidates across LinkedIn, Scholar, and GitHub (via DDG/Brave).

    Calls the scraper utility functions directly:
      plan_search_terms → discover_candidates → enrich_candidates_llm → finalize_candidates
    then converts results into SourceCandidate objects for the global state.
    """
    start_time = datetime.now(timezone.utc)

    query_config = state["query_config"]
    skills = query_config.get("skills", [])
    location = query_config.get("location", "")
    description = query_config.get("description", "")
    experience_level = query_config.get("experience_level", "")

    query_dict = {
        "skills": skills,
        "location": location,
        "description": description,
        "experience_level": experience_level,
    }
    cache_key = hashlib.sha256(
        json.dumps(query_dict, sort_keys=True).encode()
    ).hexdigest()

    try:
        from scraper import (
            plan_search_terms,
            discover_candidates,
            enrich_candidates_llm,
            finalize_candidates,
            _emit,
            agent_status,
        )

        # Reset agent status for SSE streaming
        agent_status["steps"] = []
        agent_status["done"] = False
        _emit("start", "Starting AI agent search...", "Initializing candidate discovery")

        # 1. Plan search terms via LLM
        search_terms = plan_search_terms(skills, description)

        # 2. Discover candidates across LinkedIn, Scholar, GitHub
        filters_dict = {
            "skills": skills,
            "location": location,
            "description": description,
            "experience_level": experience_level,
        }
        raw_candidates = discover_candidates(search_terms, filters_dict)

        # 3. Enrich via LLM
        enriched = enrich_candidates_llm(raw_candidates, filters_dict)

        # 4. Finalize (dedup + quality gate + cap)
        final_list = finalize_candidates(enriched)

        agent_status["done"] = True

        # Convert scraper dicts → SourceCandidate objects
        candidates: list[SourceCandidate] = []
        for raw in final_list:
            name = raw.get("name", "Unknown")
            profile_url = raw.get("profile_url", "")
            cid = hashlib.md5(
                f"{name}-{raw.get('headline', '')}".encode()
            ).hexdigest()[:12]

            # Detect source from URL
            source_label = "linkedin"
            if "scholar.google" in profile_url:
                source_label = "scholar"
            elif "github.com" in profile_url:
                source_label = "github_discovery"

            candidate = SourceCandidate(
                source=SourceType.LINKEDIN,
                source_id=cid,
                name=name,
                headline=raw.get("headline"),
                location=raw.get("location"),
                profile_url=profile_url,
                bio=raw.get("snippet"),
                skills=raw.get("matched_skills", []),
                raw_data={
                    **raw,
                    "_discovery_source": source_label,
                },
                source_specific={
                    "experience_level": raw.get("experience_level", ""),
                    "discovery_source": source_label,
                },
                query_fingerprint=cache_key,
            )
            candidates.append(candidate)

        result = SourceResult(
            source=SourceType.LINKEDIN,
            candidates=candidates,
            query_params=query_dict,
            query_fingerprint=cache_key,
            status="completed",
            total_fetched=len(candidates),
            started_at=start_time,
            completed_at=datetime.now(timezone.utc),
            fetch_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
        )

        metrics = StageMetrics(
            stage=PipelineStage.SOURCE_QUERY,
            started_at=start_time,
            completed_at=datetime.now(timezone.utc),
            duration_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
            items_out=len(candidates),
            llm_calls=2,
            custom={
                "source": "linkedin",
                "raw_discovered": len(raw_candidates),
                "enriched": len(enriched),
                "final_returned": len(final_list),
            },
        )

        return {
            "linkedin_result": result,
            "stage_metrics": [metrics],
            "cache_keys": {"linkedin_query": cache_key},
        }

    except Exception as e:
        error_dict = {
            "stage": PipelineStage.SOURCE_QUERY,
            "source": SourceType.LINKEDIN,
            "error_type": type(e).__name__,
            "message": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "recoverable": False,
        }

        result = SourceResult(
            source=SourceType.LINKEDIN,
            query_fingerprint=cache_key,
            status="failed",
            error=str(e),
        )

        return {"linkedin_result": result, "errors": [error_dict]}


def github_query_node(state: CandidateSourcingState) -> dict:
    """Query GitHub repositories by tag and extract top contributors."""
    start_time = datetime.now(timezone.utc)

    query_config = state["query_config"]
    github_tags = query_config.get("github_tags", [])
    min_stars = query_config.get("min_stars", 100)
    language = query_config.get("language")
    max_repos = query_config.get("max_repos", 30)
    max_contributors = query_config.get("max_contributors_per_repo", 10)

    query_dict = {"tags": github_tags, "min_stars": min_stars, "language": language}
    cache_key = hashlib.sha256(json.dumps(query_dict, sort_keys=True).encode()).hexdigest()

    try:
        github_tools = get_github_tools()

        all_candidates: list[SourceCandidate] = []
        seen_users: set[str] = set()
        api_calls = 0

        for tag in github_tags:
            # Use the real RepositorySearcher
            repo_response = github_tools.search_repos(
                tag=tag,
                language=language,
                min_stars=min_stars,
                max_results=max_repos,
            )
            api_calls += 1

            if not repo_response.success:
                continue

            for repo_dict in repo_response.repositories:
                owner_login = (repo_dict.get("owner") or {}).get("login", "")
                repo_name = repo_dict.get("name", "")
                if not owner_login or not repo_name:
                    continue

                # Use the real CommitAnalyzer via GitHubTools
                contrib_response = github_tools.get_contributors(
                    owner=owner_login,
                    repo=repo_name,
                    max_results=max_contributors,
                )
                api_calls += 1

                if not contrib_response.success:
                    continue

                for contrib in contrib_response.contributors:
                    if contrib.login in seen_users:
                        # If we already have this user, just append the repo
                        for existing in all_candidates:
                            if existing.source_id == contrib.login:
                                existing.source_specific.setdefault("repos", []).append(
                                    f"{owner_login}/{repo_name}"
                                )
                                existing.source_specific["total_contributions"] = (
                                    existing.source_specific.get("total_contributions", 0)
                                    + contrib.contributions
                                )
                                if tag not in existing.tags:
                                    existing.tags.append(tag)
                                break
                        continue

                    seen_users.add(contrib.login)

                    candidate = SourceCandidate(
                        source=SourceType.GITHUB,
                        source_id=contrib.login,
                        name=contrib.login,
                        profile_url=contrib.html_url or f"https://github.com/{contrib.login}",
                        raw_data={
                            "login": contrib.login,
                            "contributions": contrib.contributions,
                            "avatar_url": contrib.avatar_url,
                            "html_url": contrib.html_url,
                        },
                        source_specific={
                            "total_contributions": contrib.contributions,
                            "repos": [f"{owner_login}/{repo_name}"],
                            "primary_language": repo_dict.get("language"),
                        },
                        tags=[tag],
                        query_fingerprint=cache_key,
                    )
                    all_candidates.append(candidate)

        result = SourceResult(
            source=SourceType.GITHUB,
            candidates=all_candidates,
            query_params=query_dict,
            query_fingerprint=cache_key,
            status="completed",
            total_fetched=len(all_candidates),
            started_at=start_time,
            completed_at=datetime.now(timezone.utc),
            fetch_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
        )

        metrics = StageMetrics(
            stage=PipelineStage.SOURCE_QUERY,
            started_at=start_time,
            completed_at=datetime.now(timezone.utc),
            duration_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
            items_out=len(all_candidates),
            api_calls=api_calls,
            custom={"source": "github", "repos_searched": len(seen_users)},
        )

        return {
            "github_result": result,
            "stage_metrics": [metrics],
            "cache_keys": {"github_query": cache_key},
        }

    except Exception as e:
        error_dict = {
            "stage": PipelineStage.SOURCE_QUERY,
            "source": SourceType.GITHUB,
            "error_type": type(e).__name__,
            "message": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "recoverable": False,
        }

        result = SourceResult(
            source=SourceType.GITHUB,
            query_fingerprint=cache_key,
            status="failed",
            error=str(e),
        )

        return {"github_result": result, "errors": [error_dict]}


def linkedin_scoring_node(state: CandidateSourcingState) -> dict:
    """
    Score LinkedIn candidates using the scraper's LLM evaluation.

    Calls evaluate_candidates_llm on the enriched candidate data stored
    in each SourceCandidate's raw_data, then converts scores into proper
    ScoreMetadata objects.
    """
    start_time = datetime.now(timezone.utc)

    result = state["linkedin_result"]
    query_config = state["query_config"]

    if result.status != "completed":
        return {}  # Skip if query failed

    from scraper import evaluate_candidates_llm

    # Extract the raw enriched dicts that the query node stored
    raw_dicts = [dict(c.raw_data) for c in result.candidates]

    filters_dict = {
        "skills": query_config.get("skills", []),
        "location": query_config.get("location", ""),
        "description": query_config.get("description", ""),
        "experience_level": query_config.get("experience_level", ""),
    }

    # Run LLM evaluation (results come back sorted, but we match by profile_url)
    evaluated = evaluate_candidates_llm(raw_dicts, filters_dict)

    # Build a lookup from profile_url → (score, reason)
    score_lookup: dict[str, tuple[float, str]] = {}
    for ev in evaluated:
        url = ev.get("profile_url", "")
        if url:
            score_lookup[url] = (ev.get("_score", 5), ev.get("_reason", ""))

    for candidate in result.candidates:
        scraper_score, scraper_reason = score_lookup.get(
            candidate.profile_url, (5, "Not evaluated")
        )
        discovery_source = candidate.raw_data.get("_discovery_source", "linkedin")

        # Convert 1-10 scraper score → 0-100 raw score
        raw_score = float(scraper_score) * 10.0

        # Skills match bonus
        matched_skills = set(candidate.skills) & set(query_config.get("skills", []))
        total_skills = max(len(query_config.get("skills", [])), 1)
        skills_match_pct = (len(matched_skills) / total_skills) * 100

        component_scores = {
            "llm_relevance": raw_score,
            "skills_match": round(skills_match_pct, 2),
            "discovery_source": discovery_source,
        }

        score_metadata = ScoreMetadata(
            score_type=ScoreType.RAW,
            raw_value=round(raw_score, 2),
            model_name="scraper_evaluate_llm",
            model_version="1.0.0",
            source=SourceType.LINKEDIN,
            component_scores=component_scores,
            reasoning=scraper_reason or f"Scraper LLM score: {scraper_score}/10",
            confidence=min(scraper_score / 10.0, 1.0),
        )

        candidate.scores.append(score_metadata)

    # Normalize scores across the batch
    ranker = DefaultRankingStrategy()
    result.candidates = ranker.normalize_scores(result.candidates, SourceType.LINKEDIN)

    result.total_scored = len(result.candidates)
    result.score_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

    metrics = StageMetrics(
        stage=PipelineStage.SOURCE_SCORING,
        started_at=start_time,
        completed_at=datetime.now(timezone.utc),
        duration_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
        items_in=len(result.candidates),
        items_out=len(result.candidates),
        custom={"source": "linkedin", "scoring_method": "scraper_agent_llm"},
    )

    return {"linkedin_result": result, "stage_metrics": [metrics]}


def github_scoring_node(state: CandidateSourcingState) -> dict:
    """Score GitHub candidates using contribution metrics and AI code quality analysis."""
    start_time = datetime.now(timezone.utc)

    result = state["github_result"]
    query_config = state["query_config"]

    if result.status != "completed":
        return {}

    github_tools = get_github_tools()
    code_quality_agent = get_code_quality_agent()
    max_commits_to_grade = query_config.get("max_commits_to_grade", 3)

    llm_calls = 0
    api_calls = 0

    for candidate in result.candidates:
        contributions = candidate.source_specific.get("total_contributions", 0)
        repos = candidate.source_specific.get("repos", [])
        num_repos = len(repos)

        # --- Component 1: Contribution activity score ---
        activity_score = min(math.log1p(contributions) / math.log1p(1000) * 100, 100)

        # --- Component 2: Repo breadth score ---
        breadth_score = min(num_repos / 5 * 100, 100)

        # --- Component 3: Code quality via CodeQualityAgent (sample commits) ---
        quality_score = 50.0  # default if we can't fetch commits
        quality_details: dict = {}
        graded_commits = 0

        if repos and max_commits_to_grade > 0:
            # Pick the first repo and sample commits
            sample_repo = repos[0]
            parts = sample_repo.split("/", 1)
            if len(parts) == 2:
                owner, repo_name = parts
                try:
                    commits_response = github_tools.get_commits(
                        owner=owner,
                        repo=repo_name,
                        contributor=candidate.source_id,
                        max_results=max_commits_to_grade,
                    )
                    api_calls += 1

                    if commits_response.success and commits_response.commits:
                        total_quality = 0.0
                        for commit in commits_response.commits:
                            if not commit.diff:
                                continue
                            metric = code_quality_agent.analyze_commit(
                                commit_sha=commit.sha,
                                diff=commit.diff,
                                message=commit.message,
                            )
                            llm_calls += 1
                            graded_commits += 1
                            total_quality += metric.overall_score
                            quality_details[commit.sha[:8]] = {
                                "overall": metric.overall_score,
                                "maintainability": metric.maintainability,
                                "readability": metric.readability,
                                "complexity": metric.complexity,
                                "performance": metric.performance,
                                "security": metric.security,
                            }

                        if graded_commits > 0:
                            quality_score = total_quality / graded_commits
                except Exception:
                    pass  # graceful degradation — keep default quality_score

        # --- Combine component scores ---
        raw_score = (
            0.35 * activity_score
            + 0.15 * breadth_score
            + 0.50 * quality_score
        )

        component_scores = {
            "activity_score": round(activity_score, 2),
            "breadth_score": round(breadth_score, 2),
            "code_quality_score": round(quality_score, 2),
            "total_contributions": contributions,
            "repos_count": num_repos,
            "commits_graded": graded_commits,
        }

        reasoning_parts = [
            f"{contributions} contributions across {num_repos} repos",
            f"activity={activity_score:.0f}",
            f"breadth={breadth_score:.0f}",
            f"code_quality={quality_score:.0f} ({graded_commits} commits graded)",
        ]

        score_metadata = ScoreMetadata(
            score_type=ScoreType.RAW,
            raw_value=round(raw_score, 2),
            model_name="github_composite+code_quality_agent",
            model_version="1.0.0",
            source=SourceType.GITHUB,
            component_scores=component_scores,
            reasoning="; ".join(reasoning_parts),
        )

        candidate.scores.append(score_metadata)

        # Attach code quality details for downstream auditability
        if quality_details:
            candidate.source_specific["code_quality_details"] = quality_details

    # Normalize
    ranker = DefaultRankingStrategy()
    result.candidates = ranker.normalize_scores(result.candidates, SourceType.GITHUB)

    result.total_scored = len(result.candidates)
    result.score_time_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

    metrics = StageMetrics(
        stage=PipelineStage.SOURCE_SCORING,
        started_at=start_time,
        completed_at=datetime.now(timezone.utc),
        duration_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
        items_in=len(result.candidates),
        items_out=len(result.candidates),
        llm_calls=llm_calls,
        api_calls=api_calls,
        custom={"source": "github", "total_commits_graded": sum(
            c.source_specific.get("code_quality_details", {}).__len__()
            for c in result.candidates
        )},
    )

    return {"github_result": result, "stage_metrics": [metrics]}


def identity_resolution_node(state: CandidateSourcingState) -> dict:
    """Match candidates across sources."""
    start_time = datetime.now(timezone.utc)
    
    from state import ResolvedIdentity, IdentitySignal
    import uuid
    import re
    from collections import defaultdict
    
    # Get all completed sources
    all_source_results = get_all_source_results(state)
    all_candidates = []
    
    for result in all_source_results:
        for candidate in result.candidates:
            all_candidates.append((result.source, candidate))
    
    # Build signals
    candidate_signals = {}
    for source, candidate in all_candidates:
        signals = []
        
        if candidate.email:
            signals.append(IdentitySignal(
                signal_type="email",
                value=candidate.email.lower(),
                confidence=1.0,
                source=source
            ))
        
        if candidate.name:
            signals.append(IdentitySignal(
                signal_type="name",
                value=candidate.name.lower(),
                confidence=0.7,
                source=source
            ))
        
        # Extract username from URL
        if candidate.profile_url:
            if source == SourceType.LINKEDIN:
                match = re.search(r'/in/([^/]+)', candidate.profile_url)
                if match:
                    signals.append(IdentitySignal(
                        signal_type="username",
                        value=match.group(1).lower(),
                        confidence=0.8,
                        source=source
                    ))
            elif source == SourceType.GITHUB:
                match = re.search(r'github\.com/([^/]+)', candidate.profile_url)
                if match:
                    signals.append(IdentitySignal(
                        signal_type="username",
                        value=match.group(1).lower(),
                        confidence=0.8,
                        source=source
                    ))
        
        candidate_signals[(source, candidate.source_id)] = signals
    
    # Simple clustering: email-based and singleton
    email_clusters = defaultdict(list)
    for (source, source_id), signals in candidate_signals.items():
        email_signal = next((s for s in signals if s.signal_type == "email"), None)
        if email_signal:
            email_clusters[email_signal.value].append((source, source_id))
    
    resolved_identities = []
    identity_clusters = {}
    processed = set()
    
    # Process email clusters
    for email, cluster in email_clusters.items():
        canonical_id = str(uuid.uuid4())
        
        sources = set()
        source_ids = {}
        all_signals = []
        
        for source, source_id in cluster:
            sources.add(source)
            source_ids[source] = source_id
            all_signals.extend(candidate_signals[(source, source_id)])
            processed.add((source, source_id))
        
        # Get name
        primary_name = None
        for source, source_id in cluster:
            cand = next(c for s, c in all_candidates if s == source and c.source_id == source_id)
            if cand.name:
                primary_name = cand.name
                break
        
        resolved = ResolvedIdentity(
            canonical_id=canonical_id,
            signals=all_signals,
            sources=sources,
            source_ids=source_ids,
            primary_name=primary_name,
            primary_email=email,
            resolution_confidence=0.95,
            resolution_method="exact_email"
        )
        
        resolved_identities.append(resolved)
        
        for source, source_id in cluster:
            identity_clusters[f"{source.value}:{source_id}"] = canonical_id
    
    # Process singletons
    for source, candidate in all_candidates:
        if (source, candidate.source_id) in processed:
            continue
        
        canonical_id = str(uuid.uuid4())
        
        resolved = ResolvedIdentity(
            canonical_id=canonical_id,
            signals=candidate_signals.get((source, candidate.source_id), []),
            sources={source},
            source_ids={source: candidate.source_id},
            primary_name=candidate.name,
            primary_email=candidate.email,
            resolution_confidence=1.0,
            resolution_method="singleton"
        )
        
        resolved_identities.append(resolved)
        identity_clusters[f"{source.value}:{candidate.source_id}"] = canonical_id
    
    metrics = StageMetrics(
        stage=PipelineStage.IDENTITY_RESOLUTION,
        started_at=start_time,
        completed_at=datetime.now(timezone.utc),
        duration_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
        items_in=len(all_candidates),
        items_out=len(resolved_identities),
        custom={
            "multi_source_matches": sum(1 for r in resolved_identities if len(r.sources) > 1)
        }
    )
    
    return {
        "resolved_identities": resolved_identities,
        "identity_clusters": identity_clusters,
        "stage_metrics": [metrics]
    }


def aggregation_node(state: CandidateSourcingState) -> dict:
    """Merge candidates across sources."""
    start_time = datetime.now(timezone.utc)
    
    from state import AggregatedCandidate
    
    resolved_identities = state["resolved_identities"]
    
    # Build lookup
    candidate_lookup = {}
    for source_type in [SourceType.LINKEDIN, SourceType.GITHUB]:
        result_key = f"{source_type.value}_result"
        if result_key in state:
            result = state[result_key]
            if result.status == "completed":
                for candidate in result.candidates:
                    candidate_lookup[(source_type, candidate.source_id)] = candidate
    
    aggregated_candidates = []
    
    for identity in resolved_identities:
        source_candidates = {}
        
        for source, source_id in identity.source_ids.items():
            candidate = candidate_lookup.get((source, source_id))
            if candidate:
                source_candidates[source] = candidate
        
        if not source_candidates:
            continue
        
        # Merge profile
        name = identity.primary_name
        email = identity.primary_email
        profile_urls = {}
        location = None
        headline = None
        
        for source, candidate in source_candidates.items():
            profile_urls[source] = candidate.profile_url
            if not location and candidate.location:
                location = candidate.location
            if not headline and candidate.headline:
                headline = candidate.headline
        
        # Merge skills
        all_skills = set()
        skill_sources = {}
        for source, candidate in source_candidates.items():
            for skill in candidate.skills:
                all_skills.add(skill)
                if skill not in skill_sources:
                    skill_sources[skill] = set()
                skill_sources[skill].add(source)
        
        # Collect scores
        all_scores = []
        for candidate in source_candidates.values():
            all_scores.extend(candidate.scores)
        
        aggregated = AggregatedCandidate(
            canonical_id=identity.canonical_id,
            identity=identity,
            source_candidates=source_candidates,
            name=name or "Unknown",
            email=email,
            profile_urls=profile_urls,
            location=location,
            headline=headline,
            all_skills=all_skills,
            skill_sources=skill_sources,
            all_scores=all_scores,
        )
        
        aggregated_candidates.append(aggregated)
    
    metrics = StageMetrics(
        stage=PipelineStage.AGGREGATION,
        started_at=start_time,
        completed_at=datetime.now(timezone.utc),
        duration_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
        items_in=len(resolved_identities),
        items_out=len(aggregated_candidates),
    )
    
    return {
        "aggregated_candidates": aggregated_candidates,
        "stage_metrics": [metrics]
    }


def ranking_node(state: CandidateSourcingState) -> dict:
    """Compute final scores and rank."""
    start_time = datetime.now(timezone.utc)
    
    aggregated_candidates = state["aggregated_candidates"]
    ranking_strategy = state["ranking_strategy"]
    top_k = state["top_k"]
    
    ranker = DefaultRankingStrategy()
    
    # Compute final scores
    for candidate in aggregated_candidates:
        final_score = ranker.compute_weighted_score(candidate, ranking_strategy)
        
        final_score_metadata = ScoreMetadata(
            score_type=ScoreType.WEIGHTED,
            raw_value=final_score,
            weighted_value=final_score,
            model_name="DefaultRankingStrategy",
            model_version="1.0.0",
            source=SourceType.LINKEDIN,
            reasoning=f"Weighted score: {final_score:.3f}"
        )
        
        candidate.final_score = final_score
        candidate.final_score_metadata = final_score_metadata
    
    # Rank
    ranked_candidates = ranker.rank_candidates(aggregated_candidates, ranking_strategy)
    final_ranked_candidates = ranked_candidates[:top_k]
    
    metrics = StageMetrics(
        stage=PipelineStage.FINAL_RANKING,
        started_at=start_time,
        completed_at=datetime.now(timezone.utc),
        duration_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
        items_in=len(aggregated_candidates),
        items_out=len(final_ranked_candidates),
    )
    
    return {
        "final_ranked_candidates": final_ranked_candidates,
        "current_stage": PipelineStage.COMPLETE,
        "stage_metrics": [metrics]
    }


# ============================================================================
# CONDITIONAL EDGES
# ============================================================================

def should_continue_to_identity_resolution(state: CandidateSourcingState) -> Literal["identity_resolution", "end"]:
    """
    Check if both sources are complete before proceeding to identity resolution.
    
    This is the key to parallel execution: we wait for both LinkedIn and GitHub
    to finish before merging.
    """
    linkedin_status = state["linkedin_result"].status
    github_status = state["github_result"].status
    
    # If both completed or failed, proceed
    if linkedin_status in ["completed", "failed"] and github_status in ["completed", "failed"]:
        # Check if at least one succeeded
        if linkedin_status == "completed" or github_status == "completed":
            return "identity_resolution"
        else:
            # Both failed
            return "end"
    
    # Still waiting (shouldn't happen in graph structure, but safe)
    return "end"


# ============================================================================
# BUILD GRAPH
# ============================================================================

def create_candidate_sourcing_graph():
    """
    Build the LangGraph for candidate sourcing.
    
    Graph structure:
    
                    START
                      │
                      ├─────────────────────┐
                      ▼                     ▼
              linkedin_query          github_query
                      │                     │
                      ▼                     ▼
              linkedin_scoring        github_scoring
                      │                     │
                      └─────────┬───────────┘
                                ▼
                      identity_resolution
                                │
                                ▼
                          aggregation
                                │
                                ▼
                            ranking
                                │
                                ▼
                              END
    """
    
    # Create graph
    graph = StateGraph(CandidateSourcingState)
    
    # Add nodes
    # graph.add_node("linkedin_query", linkedin_query_node)
    # graph.add_node("linkedin_scoring", linkedin_scoring_node)
    graph.add_node("github_query", github_query_node)
    graph.add_node("github_scoring", github_scoring_node)
    graph.add_node("identity_resolution", identity_resolution_node)
    graph.add_node("aggregation", aggregation_node)
    graph.add_node("ranking", ranking_node)
    
    # LinkedIn disabled for now
    # graph.add_edge(START, "linkedin_query")
    graph.add_edge(START, "github_query")
    
    # Sequential within each source
    # graph.add_edge("linkedin_query", "linkedin_scoring")
    graph.add_edge("github_query", "github_scoring")
    
    # Merge point: both sources → identity_resolution
    # graph.add_edge("linkedin_scoring", "identity_resolution")
    graph.add_edge("github_scoring", "identity_resolution")
    
    # Sequential aggregation → ranking → END
    graph.add_edge("identity_resolution", "aggregation")
    graph.add_edge("aggregation", "ranking")
    graph.add_edge("ranking", END)
    
    return graph.compile()


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def run_candidate_sourcing_pipeline(
    skills: list[str],
    location: str,
    description: str,
    github_tags: list[str],
    top_k: int = 50
):
    """
    Run the complete candidate sourcing pipeline.
    
    Example:
        results = run_candidate_sourcing_pipeline(
            skills=["Python", "React", "AWS"],
            location="San Francisco",
            description="Senior full-stack engineer",
            github_tags=["python", "react"],
            top_k=50
        )
    """
    
    # Create initial state
    query_config = {
        "skills": skills,
        "location": location,
        "description": description,
        "github_tags": github_tags,
        "min_stars": 100
    }
    
    ranking_strategy = RankingStrategy(
        strategy_name="default",
        version="1.0.0",
        source_weights={
            SourceType.LINKEDIN: 0.4,
            SourceType.GITHUB: 0.6
        }
    )
    
    initial_state = create_initial_state(
        query_config=query_config,
        ranking_strategy=ranking_strategy,
        top_k=top_k
    )
    
    # Build and run graph
    graph = create_candidate_sourcing_graph()
    result = graph.invoke(initial_state)
    
    # Extract results
    final_candidates = result["final_ranked_candidates"]
    stage_metrics = result["stage_metrics"]
    errors = result.get("errors", [])
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Pipeline Execution Summary")
    print(f"{'='*60}")
    print(f"Execution ID: {result['execution_id']}")
    print(f"Status: {result['current_stage']}")
    print(f"Total Candidates: {len(final_candidates)}")
    print(f"\nSource Results:")
    print(f"  LinkedIn: {result['linkedin_result'].status} ({result['linkedin_result'].total_fetched} fetched, {result['linkedin_result'].total_scored} scored)")
    print(f"  GitHub: {result['github_result'].status} ({result['github_result'].total_fetched} fetched, {result['github_result'].total_scored} scored)")
    print(f"\nIdentity Resolution:")
    print(f"  Total Identities: {len(result['resolved_identities'])}")
    print(f"  Multi-source Matches: {sum(1 for r in result['resolved_identities'] if len(r.sources) > 1)}")
    print(f"\nTop 10 Candidates:")
    for i, candidate in enumerate(final_candidates[:10], 1):
        sources = ", ".join(s.value for s in candidate.source_candidates.keys())
        print(f"  {i}. {candidate.name} - Score: {candidate.final_score:.3f} - Sources: [{sources}]")
    
    if errors:
        print(f"\nErrors:")
        for error in errors:
            print(f"  - {error['stage']}: {error['message']}")
    
    print(f"\nPerformance Metrics:")
    total_duration = sum(m.duration_ms for m in stage_metrics if m.duration_ms)
    print(f"  Total Duration: {total_duration:.0f}ms")
    for metric in stage_metrics:
        if metric.duration_ms:
            print(f"  {metric.stage}: {metric.duration_ms:.0f}ms")
    
    print(f"{'='*60}\n")
    
    return {
        "candidates": final_candidates,
        "metrics": stage_metrics,
        "errors": errors,
        "state": result
    }


if __name__ == "__main__":
    # Example usage
    results = run_candidate_sourcing_pipeline(
        skills=["Python", "Machine Learning", "NLP"],
        location="Remote",
        description="AI/ML Engineer with strong Python and NLP experience",
        github_tags=["machine-learning", "nlp", "pytorch"],
        top_k=30
    )
