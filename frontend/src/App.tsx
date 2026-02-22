import { useState, useRef } from "react";
import FilterSidebar from "./components/FilterSidebar";
import CandidateGrid from "./components/CandidateGrid";
import CandidateModal from "./components/CandidateModal";
import { streamCandidates } from "./api";
import type { Candidate, SearchFilters } from "./types";

const PAGE_SIZE = 10;

interface ProgressStep {
    stage: string;
    message: string;
    detail?: string;
}

export default function App() {
    const [allCandidates, setAllCandidates] = useState<Candidate[]>([]);
    const [visibleCount, setVisibleCount] = useState(PAGE_SIZE);
    const [isLoading, setIsLoading] = useState(false);
    const [hasSearched, setHasSearched] = useState(false);
    const [selectedCandidate, setSelectedCandidate] = useState<Candidate | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [activeFilters, setActiveFilters] = useState<SearchFilters | null>(null);
    const [progressSteps, setProgressSteps] = useState<ProgressStep[]>([]);
    const [totalResults, setTotalResults] = useState(0);

    const unsubRef = useRef<(() => void) | null>(null);

    function handleSearch(filters: SearchFilters) {
        // Cancel any in-flight search
        if (unsubRef.current) {
            unsubRef.current();
            unsubRef.current = null;
        }

        setIsLoading(true);
        setError(null);
        setActiveFilters(filters);
        setAllCandidates([]);
        setVisibleCount(PAGE_SIZE);
        setProgressSteps([]);
        setTotalResults(0);
        setHasSearched(false);

        unsubRef.current = streamCandidates(
            filters,
            // onProgress
            (stage, message, detail) => {
                setProgressSteps((prev) => [...prev, { stage, message, detail }]);
            },
            // onResults
            (candidates, total) => {
                setAllCandidates(candidates);
                setTotalResults(total);
                setHasSearched(true);
                setIsLoading(false);
            },
            // onError
            (message) => {
                setError(message);
                setIsLoading(false);
            },
            // onDone
            () => {
                setIsLoading(false);
            },
        );
    }

    const visibleCandidates = allCandidates.slice(0, visibleCount);
    const hasMore = visibleCount < allCandidates.length;

    function loadMore() {
        setVisibleCount((n) => n + PAGE_SIZE);
    }

    function stageIcon(stage: string) {
        switch (stage) {
            case "analyze": return "🧠";
            case "linkedin": return "💼";
            case "search": return "🔍";
            case "score": return "📊";
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
                            <span className="tagline">AI-Powered Candidate Discovery</span>
                        </div>
                    </div>
                    {hasSearched && (
                        <div className="result-count">
                            <span className="count-number">{totalResults}</span>
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

                    {/* Search progress steps */}
                    {isLoading && progressSteps.length > 0 && (
                        <div className="agent-thoughts">
                            <div className="agent-thoughts-header">
                                <span className="agent-thoughts-icon">🤖</span>
                                Searching for candidates...
                            </div>
                            <div className="agent-steps">
                                {progressSteps.map((step, i) => (
                                    <div key={i} className="agent-step agent-step-success">
                                        <span className="step-icon">{stageIcon(step.stage)}</span>
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
                                    <span className="step-message">Working...</span>
                                </div>
                            </div>
                        </div>
                    )}

                    <CandidateGrid
                        candidates={visibleCandidates}
                        isLoading={isLoading && !hasSearched}
                        hasSearched={hasSearched}
                        onCandidateClick={setSelectedCandidate}
                    />

                    {hasSearched && hasMore && (
                        <div className="load-more-container">
                            <button className="load-more-btn" onClick={loadMore}>
                                Load more ({allCandidates.length - visibleCount} remaining)
                            </button>
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
