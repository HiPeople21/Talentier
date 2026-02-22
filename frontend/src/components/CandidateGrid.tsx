import type { Candidate } from "../types";

interface Props {
    candidates: Candidate[];
    isLoading: boolean;
    hasSearched: boolean;
    onCandidateClick: (candidate: Candidate) => void;
}

function scoreColor(score: number): string {
    if (score >= 80) return "score-high";
    if (score >= 55) return "score-mid";
    return "score-low";
}

export default function CandidateGrid({
    candidates,
    isLoading,
    hasSearched,
    onCandidateClick,
}: Props) {
    if (isLoading && !hasSearched) {
        return (
            <div className="candidate-list">
                {Array.from({ length: 4 }).map((_, i) => (
                    <div key={i} className="candidate-row skeleton">
                        <div className="row-rank skeleton-line narrow" />
                        <div className="skeleton-avatar" />
                        <div className="skeleton-text" style={{ flex: 1 }}>
                            <div className="skeleton-line wide" />
                            <div className="skeleton-line medium" />
                        </div>
                        <div className="skeleton-chips">
                            <div className="skeleton-chip" />
                            <div className="skeleton-chip" />
                        </div>
                    </div>
                ))}
            </div>
        );
    }

    if (!hasSearched) {
        return (
            <div className="empty-state">
                <div className="empty-icon">
                    <svg width="80" height="80" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round">
                        <circle cx="11" cy="11" r="8" />
                        <line x1="21" y1="21" x2="16.65" y2="16.65" />
                    </svg>
                </div>
                <h2>Find Your Next Great Hire</h2>
                <p>Describe the candidate you're looking for, then hit Search to discover top profiles from LinkedIn and GitHub.</p>
            </div>
        );
    }

    if (candidates.length === 0 && !isLoading) {
        return (
            <div className="empty-state">
                <div className="empty-icon">
                    <svg width="80" height="80" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
                        <circle cx="9" cy="7" r="4" />
                        <line x1="23" y1="11" x2="17" y2="11" />
                    </svg>
                </div>
                <h2>No Candidates Found</h2>
                <p>
                    No matching candidates found. Try adjusting your filters or describing the candidate differently.
                </p>
            </div>
        );
    }

    return (
        <div className="candidate-list">
            {candidates.map((candidate, index) => {
                const hue = candidate.name
                    .split("")
                    .reduce((acc, c) => acc + c.charCodeAt(0), 0) % 360;

                const isGitHub = candidate.source === "github";
                const profileUrl = candidate.github_url || candidate.linkedin_url || candidate.profile_url;

                return (
                    <div
                        key={candidate.id}
                        className="candidate-row"
                        onClick={() => onCandidateClick(candidate)}
                    >
                        <div className="row-rank">{index + 1}</div>
                        <div
                            className="avatar avatar-sm"
                            style={{
                                background: `linear-gradient(135deg, hsl(${hue}, 70%, 50%), hsl(${(hue + 60) % 360}, 70%, 40%))`,
                            }}
                        >
                            {candidate.avatar_initials}
                        </div>
                        <div className="row-info">
                            <div className="row-name-row">
                                <span className="candidate-name">{candidate.name}</span>
                                {candidate.source && (
                                    <span className={`source-badge source-${candidate.source}`}>
                                        {isGitHub ? (
                                            <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor">
                                                <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z" />
                                            </svg>
                                        ) : (
                                            <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor">
                                                <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
                                            </svg>
                                        )}
                                        {isGitHub ? "GitHub" : "LinkedIn"}
                                    </span>
                                )}
                            </div>
                            <span className="candidate-headline">{candidate.headline}</span>
                            {candidate.summary && (
                                <span className="row-summary">{candidate.summary}</span>
                            )}
                        </div>
                        <div className="row-location">
                            {candidate.location || "—"}
                        </div>
                        <div className="row-score-skills">
                            {candidate.score > 0 && (
                                <span className={`score-badge ${scoreColor(candidate.score)}`}>
                                    {candidate.score}
                                </span>
                            )}
                            {candidate.matched_skills.slice(0, 2).map((skill) => (
                                <span key={skill} className="skill-tag">
                                    {skill}
                                </span>
                            ))}
                        </div>
                        <div className="row-action">
                            <a
                                className={isGitHub ? "github-link" : "linkedin-link"}
                                href={profileUrl}
                                target="_blank"
                                rel="noopener noreferrer"
                                onClick={(e) => e.stopPropagation()}
                                title={isGitHub ? "View GitHub Profile" : "View LinkedIn Profile"}
                            >
                                {isGitHub ? (
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                                        <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z" />
                                    </svg>
                                ) : (
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                                        <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
                                    </svg>
                                )}
                            </a>
                        </div>
                    </div>
                );
            })}
        </div>
    );
}
