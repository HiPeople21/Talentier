"""
Complete LangGraph Implementation for Candidate Sourcing Pipeline

This file shows how to wire up all the nodes using the state structure from state.py
"""

from langgraph.graph import StateGraph, END, START
from typing import Literal
import hashlib
import json
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


# ============================================================================
# NODE IMPLEMENTATIONS
# ============================================================================

def linkedin_query_node(state: CandidateSourcingState) -> dict:
    """Query LinkedIn and populate linkedin_result."""
    start_time = datetime.now(timezone.utc)
    
    query_config = state["query_config"]
    skills = query_config.get("skills", [])
    location = query_config.get("location")
    description = query_config.get("description")
    
    # Generate cache key
    query_dict = {"skills": skills, "location": location, "description": description}
    cache_key = hashlib.sha256(json.dumps(query_dict, sort_keys=True).encode()).hexdigest()
    
    try:
        # Import your existing scraper
        from scraper import scrape_linkedin
        
        # Query LinkedIn
        raw_candidates = scrape_linkedin(skills, location, description)
        
        # Convert to SourceCandidate objects
        candidates = []
        for raw in raw_candidates:
            candidate = SourceCandidate(
                source=SourceType.LINKEDIN,
                source_id=raw["id"],
                name=raw["name"],
                headline=raw.get("headline"),
                location=raw.get("location"),
                profile_url=raw["profile_url"],
                snippet=raw.get("snippet"),
                skills=raw.get("matched_skills", []),
                raw_data=raw,
                query_fingerprint=cache_key
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
            fetch_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        )
        
        metrics = StageMetrics(
            stage=PipelineStage.SOURCE_QUERY,
            started_at=start_time,
            completed_at=datetime.now(timezone.utc),
            duration_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
            items_out=len(candidates),
            custom={"source": "linkedin"}
        )
        
        return {
            "linkedin_result": result,
            "stage_metrics": [metrics],
            "cache_keys": {"linkedin_query": cache_key}
        }
        
    except Exception as e:
        error_dict = {
            "stage": PipelineStage.SOURCE_QUERY,
            "source": SourceType.LINKEDIN,
            "error_type": type(e).__name__,
            "message": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "recoverable": False
        }
        
        result = SourceResult(
            source=SourceType.LINKEDIN,
            query_fingerprint=cache_key,
            status="failed",
            error=str(e)
        )
        
        return {
            "linkedin_result": result,
            "errors": [error_dict]
        }


def github_query_node(state: CandidateSourcingState) -> dict:
    """Query GitHub and populate github_result."""
    start_time = datetime.now(timezone.utc)
    
    query_config = state["query_config"]
    github_tags = query_config.get("github_tags", [])
    min_stars = query_config.get("min_stars", 100)
    
    query_dict = {"tags": github_tags, "min_stars": min_stars}
    cache_key = hashlib.sha256(json.dumps(query_dict, sort_keys=True).encode()).hexdigest()
    
    try:
        # Import your GitHub tools
        from github.tools import search_repositories, get_top_contributors
        
        # Query GitHub repositories
        all_candidates = []
        seen_users = set()
        
        for tag in github_tags:
            repos = search_repositories(tag, min_stars=min_stars)
            
            for repo in repos:
                contributors = get_top_contributors(
                    repo["owner"]["login"],
                    repo["name"],
                    max_contributors=10
                )
                
                for contrib in contributors:
                    if contrib["login"] in seen_users:
                        continue
                    
                    seen_users.add(contrib["login"])
                    
                    candidate = SourceCandidate(
                        source=SourceType.GITHUB,
                        source_id=contrib["login"],
                        name=contrib.get("name", contrib["login"]),
                        profile_url=contrib["html_url"],
                        raw_data=contrib,
                        source_specific={
                            "contributions": contrib["contributions"],
                            "repos": [repo["full_name"]],
                            "primary_language": repo.get("language")
                        },
                        tags=[tag],
                        query_fingerprint=cache_key
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
            fetch_time_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        )
        
        metrics = StageMetrics(
            stage=PipelineStage.SOURCE_QUERY,
            started_at=start_time,
            completed_at=datetime.now(timezone.utc),
            duration_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
            items_out=len(all_candidates),
            custom={"source": "github"}
        )
        
        return {
            "github_result": result,
            "stage_metrics": [metrics],
            "cache_keys": {"github_query": cache_key}
        }
        
    except Exception as e:
        error_dict = {
            "stage": PipelineStage.SOURCE_QUERY,
            "source": SourceType.GITHUB,
            "error_type": type(e).__name__,
            "message": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "recoverable": False
        }
        
        result = SourceResult(
            source=SourceType.GITHUB,
            query_fingerprint=cache_key,
            status="failed",
            error=str(e)
        )
        
        return {
            "github_result": result,
            "errors": [error_dict]
        }


def linkedin_scoring_node(state: CandidateSourcingState) -> dict:
    """Score LinkedIn candidates using LLM."""
    start_time = datetime.now(timezone.utc)
    
    result = state["linkedin_result"]
    query_config = state["query_config"]
    
    if result.status != "completed":
        return {}  # Skip if query failed
    
    # TODO: Implement LLM scoring
    # For now, use placeholder scores
    for i, candidate in enumerate(result.candidates):
        # Placeholder: score based on skills match
        matched_skills = set(candidate.skills) & set(query_config.get("skills", []))
        raw_score = (len(matched_skills) / max(len(query_config.get("skills", [])), 1)) * 100
        
        score_metadata = ScoreMetadata(
            score_type=ScoreType.RAW,
            raw_value=raw_score,
            model_name="placeholder",
            model_version="1.0.0",
            source=SourceType.LINKEDIN,
            component_scores={"skills_match": raw_score},
            reasoning=f"Matched {len(matched_skills)} skills"
        )
        
        candidate.scores.append(score_metadata)
    
    # Normalize scores
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
        custom={"source": "linkedin"}
    )
    
    return {
        "linkedin_result": result,
        "stage_metrics": [metrics]
    }


def github_scoring_node(state: CandidateSourcingState) -> dict:
    """Score GitHub candidates using LLM or code quality metrics."""
    start_time = datetime.now(timezone.utc)
    
    result = state["github_result"]
    
    if result.status != "completed":
        return {}
    
    # TODO: Implement code quality scoring
    # For now, use contribution count as proxy
    for candidate in result.candidates:
        contributions = candidate.source_specific.get("contributions", 0)
        raw_score = min(contributions / 10, 100)  # Cap at 100
        
        score_metadata = ScoreMetadata(
            score_type=ScoreType.RAW,
            raw_value=raw_score,
            model_name="placeholder",
            model_version="1.0.0",
            source=SourceType.GITHUB,
            component_scores={
                "contributions": contributions,
                "activity_score": raw_score
            },
            reasoning=f"{contributions} contributions"
        )
        
        candidate.scores.append(score_metadata)
    
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
        custom={"source": "github"}
    )
    
    return {
        "github_result": result,
        "stage_metrics": [metrics]
    }


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
    graph.add_node("linkedin_query", linkedin_query_node)
    graph.add_node("linkedin_scoring", linkedin_scoring_node)
    graph.add_node("github_query", github_query_node)
    graph.add_node("github_scoring", github_scoring_node)
    graph.add_node("identity_resolution", identity_resolution_node)
    graph.add_node("aggregation", aggregation_node)
    graph.add_node("ranking", ranking_node)
    
    # Parallel execution: START → linkedin_query AND github_query
    graph.add_edge(START, "linkedin_query")
    graph.add_edge(START, "github_query")
    
    # Sequential within each source
    graph.add_edge("linkedin_query", "linkedin_scoring")
    graph.add_edge("github_query", "github_scoring")
    
    # Merge point: both sources → identity_resolution
    graph.add_edge("linkedin_scoring", "identity_resolution")
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
