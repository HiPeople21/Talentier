"""
Production-grade LangGraph State Architecture for Multi-Source Candidate Sourcing Pipeline

Design Principles:
- Non-destructive updates (preserve all raw data)
- Source isolation (parallel execution friendly)
- Extensible to new sources (StackOverflow, Kaggle, etc.)
- Identity resolution across sources
- Auditability and debugging support
- ML-ready metadata structure
- Incremental recomputation support
"""

from typing import Annotated, TypedDict, Literal, Optional, Any
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from enum import Enum
import operator


# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================

class SourceType(str, Enum):
    """All candidate sources in the system."""
    LINKEDIN = "linkedin"
    GITHUB = "github"


class ScoreType(str, Enum):
    """Types of scores maintained throughout the pipeline."""
    RAW = "raw"  # Original score from source-specific LLM
    NORMALIZED = "normalized"  # 0-1 normalized across source
    WEIGHTED = "weighted"  # Final weighted score
    ENSEMBLE = "ensemble"  # ML-based ensemble score


class PipelineStage(str, Enum):
    """Pipeline execution stages for telemetry."""
    INPUT = "input"
    SOURCE_QUERY = "source_query"
    SOURCE_SCORING = "source_scoring"
    IDENTITY_RESOLUTION = "identity_resolution"
    AGGREGATION = "aggregation"
    FINAL_RANKING = "final_ranking"
    COMPLETE = "complete"


# ============================================================================
# SCORING MODELS
# ============================================================================

class ScoreMetadata(BaseModel):
    """Metadata about how a score was computed."""
    model_config = ConfigDict(frozen=True)  # Immutable for auditability
    
    score_type: ScoreType
    raw_value: float = Field(..., description="Original score before any transformations")
    normalized_value: Optional[float] = Field(None, description="0-1 normalized score")
    weighted_value: Optional[float] = Field(None, description="Score after applying weights")
    
    # Auditability
    model_name: str = Field(..., description="LLM model used (e.g., 'gpt-4o', 'claude-3-5-sonnet')")
    model_version: str = Field(..., description="Model version or timestamp")
    prompt_hash: Optional[str] = Field(None, description="Hash of prompt for reproducibility")
    
    # Context
    scored_at: datetime = Field(default_factory=datetime.utcnow)
    source: SourceType = Field(..., description="Which source this score came from")
    
    # Components (for debugging)
    component_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Breakdown of score components (e.g., skills_match, experience_match)"
    )
    
    # Statistical context
    percentile: Optional[float] = Field(None, description="Percentile within source batch")
    z_score: Optional[float] = Field(None, description="Z-score for normalization")
    
    # Reasoning
    reasoning: Optional[str] = Field(None, description="LLM explanation for debugging")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Model confidence")


class RankingStrategy(BaseModel):
    """Configuration for how to rank candidates."""
    strategy_name: str = Field(..., description="Name of ranking strategy")
    version: str = Field(..., description="Strategy version")
    
    # Weights by source
    source_weights: dict[SourceType, float] = Field(
        default_factory=lambda: {
            SourceType.LINKEDIN: 0.4,
            SourceType.GITHUB: 0.6,
        },
        description="Weight for each source (sum should = 1.0)"
    )
    
    # Feature weights (for ML-based ranking)
    feature_weights: dict[str, float] = Field(
        default_factory=dict,
        description="Weights for individual features"
    )
    
    # Normalization strategy
    normalization_method: Literal["min-max", "z-score", "rank", "softmax"] = "min-max"
    
    # Identity resolution
    identity_boost: float = Field(
        default=0.15,
        description="Bonus for candidates found in multiple sources"
    )
    
    # Decay factors
    recency_decay: Optional[float] = Field(None, description="Decay factor for older data")
    
    class Config:
        use_enum_values = True


# ============================================================================
# IDENTITY RESOLUTION MODELS
# ============================================================================

class IdentitySignal(BaseModel):
    """A signal that helps match candidates across sources."""
    signal_type: Literal["email", "name", "username", "url", "profile_id"]
    value: str
    confidence: float = Field(..., ge=0, le=1, description="Confidence in this signal")
    source: SourceType


class ResolvedIdentity(BaseModel):
    """A unified identity across multiple sources."""
    canonical_id: str = Field(..., description="UUID for this person")
    
    # All signals that contributed to this identity
    signals: list[IdentitySignal] = Field(default_factory=list)
    
    # Which sources found this person
    sources: set[SourceType] = Field(default_factory=set)
    
    # Source-specific IDs
    source_ids: dict[SourceType, str] = Field(
        default_factory=dict,
        description="Original ID from each source"
    )
    
    # Merged profile data
    primary_name: Optional[str] = None
    primary_email: Optional[str] = None
    primary_url: Optional[str] = None
    
    # Metadata
    resolved_at: datetime = Field(default_factory=datetime.utcnow)
    resolution_confidence: float = Field(..., ge=0, le=1)
    resolution_method: str = Field(..., description="e.g., 'exact_email', 'fuzzy_name'")
    
    class Config:
        use_enum_values = True


# ============================================================================
# SOURCE-SPECIFIC CANDIDATE MODELS
# ============================================================================

class SourceCandidate(BaseModel):
    """A candidate from a specific source with all metadata."""
    # Identity
    source: SourceType
    source_id: str = Field(..., description="ID from the source system")
    
    # Raw data (preserved for audit)
    raw_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Complete raw response from source"
    )
    
    # Normalized profile fields
    name: Optional[str] = None
    email: Optional[str] = None
    profile_url: Optional[str] = None
    location: Optional[str] = None
    headline: Optional[str] = None
    bio: Optional[str] = None
    
    # Skills/tags
    skills: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    
    # Scoring
    scores: list[ScoreMetadata] = Field(
        default_factory=list,
        description="All scores for this candidate (raw, normalized, weighted)"
    )
    
    # Source-specific data
    source_specific: dict[str, Any] = Field(
        default_factory=dict,
        description="Source-specific fields (e.g., github_repos, linkedin_connections)"
    )
    
    # Metadata
    fetched_at: datetime = Field(default_factory=datetime.utcnow)
    query_fingerprint: Optional[str] = Field(
        None,
        description="Hash of query params for cache invalidation"
    )
    
    class Config:
        use_enum_values = True


class SourceResult(BaseModel):
    """Results from a single source (e.g., LinkedIn or GitHub)."""
    source: SourceType
    candidates: list[SourceCandidate] = Field(default_factory=list)
    
    # Query context
    query_params: dict[str, Any] = Field(default_factory=dict)
    query_fingerprint: str = Field(..., description="Hash for caching")
    
    # Statistics
    total_fetched: int = 0
    total_scored: int = 0
    fetch_time_ms: Optional[float] = None
    score_time_ms: Optional[float] = None
    
    # Status
    status: Literal["pending", "in_progress", "completed", "failed"] = "pending"
    error: Optional[str] = None
    
    # Metadata
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Caching
    cache_hit: bool = False
    cached_at: Optional[datetime] = None
    
    class Config:
        use_enum_values = True


# ============================================================================
# AGGREGATED CANDIDATE MODEL
# ============================================================================

class AggregatedCandidate(BaseModel):
    """A candidate with data merged across all sources."""
    # Unified identity
    canonical_id: str
    identity: ResolvedIdentity
    
    # Source contributions
    source_candidates: dict[SourceType, SourceCandidate] = Field(
        default_factory=dict,
        description="Original candidate objects from each source"
    )
    
    # Merged profile (best data from all sources)
    name: str
    email: Optional[str] = None
    profile_urls: dict[SourceType, str] = Field(default_factory=dict)
    location: Optional[str] = None
    headline: Optional[str] = None
    bio: Optional[str] = None
    
    # Merged skills (union across sources)
    all_skills: set[str] = Field(default_factory=set)
    skill_sources: dict[str, set[SourceType]] = Field(
        default_factory=dict,
        description="Which sources mentioned each skill"
    )
    
    # All scores from all sources
    all_scores: list[ScoreMetadata] = Field(default_factory=list)
    
    # Final aggregated score
    final_score: Optional[float] = Field(
        None,
        description="Final weighted/ensemble score"
    )
    final_score_metadata: Optional[ScoreMetadata] = None
    
    # Ranking
    rank: Optional[int] = Field(None, description="Final rank in result list")
    
    # Multi-source bonus
    multi_source_boost: float = Field(
        default=0.0,
        description="Boost applied for being in multiple sources"
    )
    
    class Config:
        use_enum_values = True


# ============================================================================
# TELEMETRY AND DEBUGGING
# ============================================================================

class StageMetrics(BaseModel):
    """Performance metrics for a pipeline stage."""
    stage: PipelineStage
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    
    # Counts
    items_in: int = 0
    items_out: int = 0
    items_filtered: int = 0
    
    # Resource usage
    llm_calls: int = 0
    llm_tokens_in: int = 0
    llm_tokens_out: int = 0
    api_calls: int = 0
    cache_hits: int = 0
    
    # Errors
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    
    # Custom metrics
    custom: dict[str, Any] = Field(default_factory=dict)


class PipelineVersion(BaseModel):
    """Versioning information for reproducibility."""
    pipeline_version: str = Field(..., description="Semantic version of pipeline")
    state_schema_version: str = Field(..., description="Version of this state schema")
    
    # Code versions
    code_commit_hash: Optional[str] = None
    deployed_at: Optional[datetime] = None
    
    # Model versions
    model_versions: dict[str, str] = Field(
        default_factory=dict,
        description="Versions of all models used"
    )


# ============================================================================
# MAIN LANGGRAPH STATE
# ============================================================================

class CandidateSourcingState(TypedDict):
    """
    Main LangGraph state for the candidate sourcing pipeline.
    
    This state is designed for:
    - Parallel execution (LinkedIn and GitHub run independently)
    - Non-destructive updates (all data is preserved)
    - Extensibility (easy to add new sources)
    - Auditability (full provenance tracking)
    - ML-readiness (structured scores and features)
    
    State Flow:
    1. INPUT: query_config is set
    2. SOURCE_QUERY: linkedin_result and github_result are populated (parallel)
    3. SOURCE_SCORING: scores added to candidates in each result
    4. IDENTITY_RESOLUTION: resolved_identities is populated
    5. AGGREGATION: aggregated_candidates is populated
    6. FINAL_RANKING: final_ranked_candidates is populated
    """
    
    # ========================================================================
    # SECTION 1: INPUT CONFIGURATION
    # Purpose: Store all user inputs and configuration. Never modified after init.
    # ========================================================================
    
    query_config: dict[str, Any]
    """
    Original query parameters from user:
    - skills: list[str]
    - location: str
    - description: str
    - experience_level: str
    - github_tags: list[str]
    - etc.
    """
    
    ranking_strategy: RankingStrategy
    """How to weight and combine results from different sources."""
    
    pipeline_version: PipelineVersion
    """Version tracking for reproducibility."""
    
    # ========================================================================
    # SECTION 2: SOURCE-SPECIFIC RESULTS
    # Purpose: Isolated storage for each source. Enables parallel execution.
    # Updated by: Source-specific nodes (linkedin_query_node, github_query_node, etc.)
    # ========================================================================
    
    linkedin_result: SourceResult
    """Complete LinkedIn search and scoring results."""
    
    github_result: SourceResult
    """Complete GitHub search and scoring results."""
    
    # ========================================================================
    # SECTION 3: IDENTITY RESOLUTION
    # Purpose: Map same person across different sources.
    # Updated by: identity_resolution_node
    # Why separate: Identity resolution is a distinct concern from scoring
    # ========================================================================
    
    resolved_identities: Annotated[list[ResolvedIdentity], operator.add]
    """
    List of unified identities across sources.
    Uses operator.add to support incremental identity resolution.
    """
    
    identity_clusters: dict[str, list[str]]
    """
    Mapping from source_id (e.g., 'linkedin:12345') to canonical_id.
    Used for fast lookups during aggregation.
    """
    
    # ========================================================================
    # SECTION 4: AGGREGATION
    # Purpose: Merge candidates across sources, combining all data.
    # Updated by: aggregation_node
    # ========================================================================
    
    aggregated_candidates: Annotated[list[AggregatedCandidate], operator.add]
    """
    Candidates with data merged across all sources.
    Each represents a unique person with all their source profiles.
    Uses operator.add to support incremental aggregation.
    """
    
    # ========================================================================
    # SECTION 5: FINAL RANKING
    # Purpose: Store final ranked output.
    # Updated by: ranking_node
    # ========================================================================
    
    final_ranked_candidates: list[AggregatedCandidate]
    """
    Final sorted list of candidates.
    This is the ultimate output of the pipeline.
    """
    
    top_k: int
    """Number of top candidates to return (e.g., 50)."""
    
    # ========================================================================
    # SECTION 6: EXECUTION METADATA
    # Purpose: Telemetry, debugging, and observability.
    # Updated by: All nodes (append-only)
    # ========================================================================
    
    stage_metrics: Annotated[list[StageMetrics], operator.add]
    """
    Performance metrics for each pipeline stage.
    Append-only for audit trail.
    """
    
    current_stage: PipelineStage
    """Current stage of pipeline execution."""
    
    execution_id: str
    """Unique ID for this pipeline run."""
    
    started_at: datetime
    """Pipeline start time."""
    
    # ========================================================================
    # SECTION 7: ERROR HANDLING
    # Purpose: Track errors without halting the pipeline.
    # ========================================================================
    
    errors: Annotated[list[dict[str, Any]], operator.add]
    """
    All errors encountered during execution.
    Format: {stage, error_type, message, timestamp, recoverable}
    """
    
    warnings: Annotated[list[str], operator.add]
    """Non-fatal warnings."""
    
    # ========================================================================
    # SECTION 8: CACHING AND INCREMENTAL RECOMPUTATION
    # Purpose: Support partial re-runs and result caching.
    # ========================================================================
    
    cache_keys: dict[str, str]
    """
    Cache keys for each expensive operation.
    Format: {operation_name: cache_key_hash}
    """
    
    recompute_flags: dict[str, bool]
    """
    Flags to force recomputation of specific stages.
    Format: {stage_name: should_recompute}
    """


# ============================================================================
# HELPER FUNCTIONS FOR STATE UPDATES
# ============================================================================

def create_initial_state(
    query_config: dict[str, Any],
    ranking_strategy: Optional[RankingStrategy] = None,
    pipeline_version: Optional[PipelineVersion] = None,
    top_k: int = 50,
    execution_id: Optional[str] = None,
) -> CandidateSourcingState:
    """
    Create initial state for a new pipeline run.
    
    This is the entry point for the pipeline.
    """
    import uuid
    
    if ranking_strategy is None:
        ranking_strategy = RankingStrategy(
            strategy_name="default",
            version="1.0.0"
        )
    
    if pipeline_version is None:
        pipeline_version = PipelineVersion(
            pipeline_version="1.0.0",
            state_schema_version="1.0.0"
        )
    
    if execution_id is None:
        execution_id = str(uuid.uuid4())
    
    return CandidateSourcingState(
        # Input
        query_config=query_config,
        ranking_strategy=ranking_strategy,
        pipeline_version=pipeline_version,
        
        # Source results (initialized as empty)
        linkedin_result=SourceResult(
            source=SourceType.LINKEDIN,
            query_fingerprint="",
            status="pending"
        ),
        github_result=SourceResult(
            source=SourceType.GITHUB,
            query_fingerprint="",
            status="pending"
        ),
        
        # Identity resolution
        resolved_identities=[],
        identity_clusters={},
        
        # Aggregation
        aggregated_candidates=[],
        
        # Ranking
        final_ranked_candidates=[],
        top_k=top_k,
        
        # Execution metadata
        stage_metrics=[],
        current_stage=PipelineStage.INPUT,
        execution_id=execution_id,
        started_at=datetime.utcnow(),
        
        # Error handling
        errors=[],
        warnings=[],
        
        # Caching
        cache_keys={},
        recompute_flags={},
    )


def update_source_result(
    state: CandidateSourcingState,
    source: SourceType,
    result: SourceResult
) -> dict[str, SourceResult]:
    """
    Update a specific source result in state.
    
    Returns a dict with the appropriate key for state update.
    """
    if source == SourceType.LINKEDIN:
        return {"linkedin_result": result}
    elif source == SourceType.GITHUB:
        return {"github_result": result}
    # elif source == SourceType.STACKOVERFLOW:
    #     return {"stackoverflow_result": result}
    else:
        raise ValueError(f"Unknown source type: {source}")


def get_all_source_results(state: CandidateSourcingState) -> list[SourceResult]:
    """Get all source results that are completed."""
    results = []
    
    if state["linkedin_result"].status == "completed":
        results.append(state["linkedin_result"])
    
    if state["github_result"].status == "completed":
        results.append(state["github_result"])
    
    return results


# ============================================================================
# PLUGGABLE RANKING STRATEGY INTERFACE
# ============================================================================

class RankingStrategyInterface:
    """
    Abstract interface for ranking strategies.
    
    Implement this to create custom ranking logic that can be swapped in
    without changing the pipeline structure.
    """
    
    def normalize_scores(
        self,
        candidates: list[SourceCandidate],
        source: SourceType
    ) -> list[SourceCandidate]:
        """
        Normalize scores within a source to 0-1 range.
        
        Should update each candidate's scores list with normalized scores.
        """
        raise NotImplementedError
    
    def compute_weighted_score(
        self,
        aggregated_candidate: AggregatedCandidate,
        strategy: RankingStrategy
    ) -> float:
        """
        Compute final weighted score for an aggregated candidate.
        
        Can incorporate:
        - Weighted average of source scores
        - Multi-source boost
        - ML-based features
        """
        raise NotImplementedError
    
    def rank_candidates(
        self,
        candidates: list[AggregatedCandidate],
        strategy: RankingStrategy
    ) -> list[AggregatedCandidate]:
        """
        Rank candidates and assign rank field.
        
        Returns sorted list with rank assigned.
        """
        raise NotImplementedError


# ============================================================================
# DEFAULT RANKING IMPLEMENTATION
# ============================================================================

class DefaultRankingStrategy(RankingStrategyInterface):
    """
    Default ranking strategy using weighted average of normalized scores.
    
    This can be swapped out for ML-based strategies later.
    """
    
    def normalize_scores(
        self,
        candidates: list[SourceCandidate],
        source: SourceType
    ) -> list[SourceCandidate]:
        """Normalize using min-max scaling."""
        if not candidates:
            return candidates
        
        # Extract raw scores
        raw_scores = []
        for candidate in candidates:
            raw_score = next(
                (s.raw_value for s in candidate.scores if s.score_type == ScoreType.RAW),
                None
            )
            if raw_score is not None:
                raw_scores.append(raw_score)
        
        if not raw_scores:
            return candidates
        
        # Min-max normalization
        min_score = min(raw_scores)
        max_score = max(raw_scores)
        score_range = max_score - min_score if max_score > min_score else 1.0
        
        # Update candidates
        for candidate in candidates:
            raw_score_obj = next(
                (s for s in candidate.scores if s.score_type == ScoreType.RAW),
                None
            )
            if raw_score_obj:
                normalized = (raw_score_obj.raw_value - min_score) / score_range
                
                # Create new normalized score metadata
                normalized_score = ScoreMetadata(
                    score_type=ScoreType.NORMALIZED,
                    raw_value=raw_score_obj.raw_value,
                    normalized_value=normalized,
                    model_name=raw_score_obj.model_name,
                    model_version=raw_score_obj.model_version,
                    source=source,
                    component_scores=raw_score_obj.component_scores,
                    reasoning=f"Normalized using min-max: ({raw_score_obj.raw_value} - {min_score}) / {score_range}"
                )
                
                candidate.scores.append(normalized_score)
        
        return candidates
    
    def compute_weighted_score(
        self,
        aggregated_candidate: AggregatedCandidate,
        strategy: RankingStrategy
    ) -> float:
        """Compute weighted average of normalized scores across sources."""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for source, source_candidate in aggregated_candidate.source_candidates.items():
            # Get normalized score
            normalized_score = next(
                (s.normalized_value for s in source_candidate.scores 
                 if s.score_type == ScoreType.NORMALIZED and s.normalized_value is not None),
                None
            )
            
            if normalized_score is not None:
                weight = strategy.source_weights.get(source, 0.0)
                weighted_sum += normalized_score * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        base_score = weighted_sum / total_weight
        
        # Apply multi-source boost
        num_sources = len(aggregated_candidate.source_candidates)
        if num_sources > 1:
            boost = strategy.identity_boost * (num_sources - 1)
            base_score = min(1.0, base_score + boost)
        
        return base_score
    
    def rank_candidates(
        self,
        candidates: list[AggregatedCandidate],
        strategy: RankingStrategy
    ) -> list[AggregatedCandidate]:
        """Sort by final_score and assign ranks."""
        # Sort by score (descending)
        sorted_candidates = sorted(
            candidates,
            key=lambda c: c.final_score if c.final_score is not None else 0.0,
            reverse=True
        )
        
        # Assign ranks
        for rank, candidate in enumerate(sorted_candidates, start=1):
            candidate.rank = rank
        
        return sorted_candidates
