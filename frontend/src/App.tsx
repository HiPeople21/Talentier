import { useState, useEffect, useRef, useCallback } from "react";
import FilterSidebar from "./components/FilterSidebar";
import CandidateGrid from "./components/CandidateGrid";
import CandidateModal from "./components/CandidateModal";
import { searchCandidates, subscribeAgentStatus } from "./api";
import type { AgentStep } from "./api";
import type { Candidate, SearchFilters } from "./types";

export default function App() {
  const [candidates, setCandidates] = useState<Candidate[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);
  const [selectedCandidate, setSelectedCandidate] = useState<Candidate | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeFilters, setActiveFilters] = useState<SearchFilters | null>(null);
  const [page, setPage] = useState(1);
  const [hasMore] = useState(true); // Always allow fetching more
  const [agentSteps, setAgentSteps] = useState<AgentStep[]>([]);
  const [isLoadingMore, setIsLoadingMore] = useState(false);

  const sentinelRef = useRef<HTMLDivElement>(null);
  const unsubRef = useRef<(() => void) | null>(null);

  async function handleSearch(filters: SearchFilters) {
    setIsLoading(true);
    setError(null);
    setActiveFilters(filters);
    setPage(1);
    setCandidates([]);
    setAgentSteps([]);

    // Subscribe to agent status SSE
    if (unsubRef.current) unsubRef.current();
    unsubRef.current = subscribeAgentStatus(
      (step) => setAgentSteps((prev) => [...prev, step]),
      () => { } // done handled by search completing
    );

    try {
      const data = await searchCandidates(filters, 1);
      setCandidates(data.candidates);
      setHasSearched(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed");
      setCandidates([]);
    } finally {
      setIsLoading(false);
      if (unsubRef.current) {
        unsubRef.current();
        unsubRef.current = null;
      }
    }
  }

  const handleLoadMore = useCallback(async () => {
    if (!activeFilters || isLoading || isLoadingMore) return;
    const nextPage = page + 1;
    setIsLoadingMore(true);
    setAgentSteps([]);

    // Subscribe to agent status for load more
    if (unsubRef.current) unsubRef.current();
    unsubRef.current = subscribeAgentStatus(
      (step) => setAgentSteps((prev) => [...prev, step]),
      () => { }
    );

    try {
      const data = await searchCandidates(activeFilters, nextPage);
      if (data.candidates.length > 0) {
        setCandidates((prev) => [...prev, ...data.candidates]);
        setPage(nextPage);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load more");
    } finally {
      setIsLoadingMore(false);
      if (unsubRef.current) {
        unsubRef.current();
        unsubRef.current = null;
      }
    }
  }, [activeFilters, isLoading, isLoadingMore, page]);

  // Infinite scroll via IntersectionObserver
  useEffect(() => {
    if (!hasSearched || !hasMore || isLoading || isLoadingMore) return;

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries[0].isIntersecting && candidates.length > 0) {
          handleLoadMore();
        }
      },
      { threshold: 0.1 }
    );

    const sentinel = sentinelRef.current;
    if (sentinel) observer.observe(sentinel);

    return () => {
      if (sentinel) observer.unobserve(sentinel);
    };
  }, [hasSearched, hasMore, isLoading, isLoadingMore, candidates.length, handleLoadMore]);

  // Get the step icon
  function stepIcon(type: string) {
    switch (type) {
      case "start": return "🚀";
      case "thinking": return "🧠";
      case "searching": return "🔍";
      case "success": return "✅";
      case "refining": return "🔄";
      case "error": return "❌";
      default: return "⏳";
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <div className="header-content">
          <div className="logo">
            <div className="logo-icon">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
                <circle cx="9" cy="7" r="4" />
                <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
                <path d="M16 3.13a4 4 0 0 1 0 7.75" />
              </svg>
            </div>
            <div>
              <h1>TalentScope</h1>
              <span className="tagline">LinkedIn Candidate Discovery</span>
            </div>
          </div>
          {hasSearched && (
            <div className="result-count">
              <span className="count-number">{candidates.length}</span>
              <span className="count-label">candidates found</span>
            </div>
          )}
        </div>
      </header>

      <main className="app-main">
        <FilterSidebar onSearch={handleSearch} isLoading={isLoading} />

        <section className="results-section">
          {error && (
            <div className="error-banner">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" />
                <line x1="15" y1="9" x2="9" y2="15" />
                <line x1="9" y1="9" x2="15" y2="15" />
              </svg>
              {error}
            </div>
          )}

          {activeFilters && hasSearched && (
            <div className="active-filters-bar">
              {activeFilters.skills.map((skill) => (
                <span key={skill} className="active-filter-pill">
                  {skill}
                </span>
              ))}
              {activeFilters.experience_level && (
                <span className="active-filter-pill">
                  {activeFilters.experience_level}
                </span>
              )}
              {activeFilters.location && (
                <span className="active-filter-pill">
                  📍 {activeFilters.location}
                </span>
              )}
            </div>
          )}

          {/* Agent Thought Process */}
          {(isLoading || isLoadingMore) && agentSteps.length > 0 && (
            <div className="agent-thoughts">
              <div className="agent-thoughts-header">
                <span className="agent-thoughts-icon">🤖</span>
                AI Agent Thinking...
              </div>
              <div className="agent-steps">
                {agentSteps.map((step, i) => (
                  <div key={i} className={`agent-step agent-step-${step.type}`}>
                    <span className="step-icon">{stepIcon(step.type)}</span>
                    <div className="step-content">
                      <span className="step-message">{step.message}</span>
                      {step.detail && (
                        <span className="step-detail">{step.detail}</span>
                      )}
                    </div>
                  </div>
                ))}
                <div className="agent-step agent-step-active">
                  <span className="spinner" />
                  <span className="step-message">Processing...</span>
                </div>
              </div>
            </div>
          )}

          <CandidateGrid
            candidates={candidates}
            isLoading={isLoading && !hasSearched}
            hasSearched={hasSearched}
            onCandidateClick={setSelectedCandidate}
          />

          {/* Infinite scroll sentinel */}
          {hasSearched && candidates.length > 0 && (
            <div ref={sentinelRef} className="scroll-sentinel">
              {isLoadingMore && (
                <div className="loading-more">
                  <span className="spinner" /> AI agent is finding more candidates...
                </div>
              )}
            </div>
          )}
        </section>
      </main>

      {selectedCandidate && (
        <CandidateModal
          candidate={selectedCandidate}
          onClose={() => setSelectedCandidate(null)}
        />
      )}
    </div>
  );
}
