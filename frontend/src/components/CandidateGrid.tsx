import type { Candidate } from "../types";

interface Props {
    candidates: Candidate[];
    isLoading: boolean;
    hasSearched: boolean;
    onCandidateClick: (candidate: Candidate) => void;
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
                <p>Select technical skills and filters, then search to discover candidates on LinkedIn.</p>
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
                    The AI agent couldn't find matching candidates.
                    Try adjusting your filters or broadening your search criteria.
                </p>
            </div>
        );
    }

    return (
        <div className="candidate-list">
            {/* Table header */}
            <div className="candidate-row row-header">
                <div className="row-rank">#</div>
                <div className="row-avatar" />
                <div className="row-info">Candidate</div>
                <div className="row-location">Location</div>
                <div className="row-skills">Skills</div>
                <div className="row-action" />
            </div>

            {candidates.map((candidate, index) => {
                const hue = candidate.name
                    .split("")
                    .reduce((acc, c) => acc + c.charCodeAt(0), 0) % 360;

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
                            <span className="candidate-name">{candidate.name}</span>
                            <span className="candidate-headline">{candidate.headline}</span>
                        </div>
                        <div className="row-location">
                            {candidate.location || "—"}
                        </div>
                        <div className="row-skills">
                            {candidate.matched_skills.slice(0, 3).map((skill) => (
                                <span key={skill} className="skill-tag">
                                    {skill}
                                </span>
                            ))}
                        </div>
                        <div className="row-action">
                            <a
                                className="linkedin-link"
                                href={candidate.profile_url}
                                target="_blank"
                                rel="noopener noreferrer"
                                onClick={(e) => e.stopPropagation()}
                            >
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                                    <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
                                </svg>
                            </a>
                        </div>
                    </div>
                );
            })}
        </div>
    );
}
