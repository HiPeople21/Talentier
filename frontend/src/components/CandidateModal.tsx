import type { Candidate } from "../types";
import { useEffect } from "react";

interface Props {
    candidate: Candidate;
    onClose: () => void;
}

function scoreColor(score: number): string {
    if (score >= 80) return "score-high";
    if (score >= 55) return "score-mid";
    return "score-low";
}

function scoreLabel(score: number): string {
    if (score >= 80) return "Excellent match";
    if (score >= 55) return "Good match";
    return "Partial match";
}

export default function CandidateModal({ candidate, onClose }: Props) {
    const hue = candidate.name
        .split("")
        .reduce((acc, c) => acc + c.charCodeAt(0), 0) % 360;

    const isGitHub = candidate.source === "github";

    useEffect(() => {
        function handleKey(e: KeyboardEvent) {
            if (e.key === "Escape") onClose();
        }
        document.addEventListener("keydown", handleKey);
        return () => document.removeEventListener("keydown", handleKey);
    }, [onClose]);

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-content" onClick={(e) => e.stopPropagation()}>
                <button className="modal-close" onClick={onClose}>×</button>

                <div className="modal-header">
                    <div
                        className="modal-avatar"
                        style={{
                            background: `linear-gradient(135deg, hsl(${hue}, 70%, 50%), hsl(${(hue + 60) % 360}, 70%, 40%))`,
                        }}
                    >
                        {candidate.avatar_initials}
                    </div>
                    <div className="modal-header-info">
                        <div className="modal-name-row">
                            <h2 className="modal-name">{candidate.name}</h2>
                            <div className="modal-badges">
                                {candidate.source && (
                                    <span className={`source-badge source-${candidate.source}`}>
                                        {isGitHub ? (
                                            <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">
                                                <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z" />
                                            </svg>
                                        ) : (
                                            <svg width="13" height="13" viewBox="0 0 24 24" fill="currentColor">
                                                <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
                                            </svg>
                                        )}
                                        {isGitHub ? "GitHub" : "LinkedIn"}
                                    </span>
                                )}
                            </div>
                        </div>
                        <p className="modal-headline">{candidate.headline}</p>
                        {candidate.location && (
                            <p className="modal-location">
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z" />
                                    <circle cx="12" cy="10" r="3" />
                                </svg>
                                {candidate.location}
                            </p>
                        )}
                    </div>
                </div>

                {/* Score section */}
                {candidate.score > 0 && (
                    <div className="modal-score-section">
                        <div className={`modal-score-badge ${scoreColor(candidate.score)}`}>
                            <span className="modal-score-number">{candidate.score}</span>
                            <span className="modal-score-label">{scoreLabel(candidate.score)}</span>
                        </div>
                        {isGitHub && candidate.code_quality_score != null && (
                            <div className="modal-code-quality">
                                <span className="code-quality-label">Code Quality</span>
                                <div className="code-quality-bar-wrap">
                                    <div
                                        className={`code-quality-bar ${scoreColor(candidate.code_quality_score)}`}
                                        style={{ width: `${candidate.code_quality_score}%` }}
                                    />
                                </div>
                                <span className="code-quality-value">{candidate.code_quality_score}/100</span>
                            </div>
                        )}
                    </div>
                )}

                {/* AI Summary */}
                {candidate.summary && (
                    <div className="modal-section">
                        <h3>Why this candidate?</h3>
                        <p className="modal-summary">{candidate.summary}</p>
                    </div>
                )}

                {candidate.matched_skills.length > 0 && (
                    <div className="modal-section">
                        <h3>Matched Skills</h3>
                        <div className="modal-skills">
                            {candidate.matched_skills.map((skill) => (
                                <span key={skill} className="skill-tag large">
                                    {skill}
                                </span>
                            ))}
                        </div>
                    </div>
                )}

                {candidate.snippet && (
                    <div className="modal-section">
                        <h3>Profile Snippet</h3>
                        <p className="modal-snippet">{candidate.snippet}</p>
                    </div>
                )}

                <div className="modal-actions">
                    {candidate.linkedin_url && (
                        <a
                            className="btn-primary btn-linkedin"
                            href={candidate.linkedin_url}
                            target="_blank"
                            rel="noopener noreferrer"
                        >
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                                <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
                            </svg>
                            LinkedIn Profile
                        </a>
                    )}
                    {candidate.github_url && (
                        <a
                            className="btn-primary btn-github"
                            href={candidate.github_url}
                            target="_blank"
                            rel="noopener noreferrer"
                        >
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                                <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z" />
                            </svg>
                            GitHub Profile
                        </a>
                    )}
                    {!candidate.linkedin_url && !candidate.github_url && (
                        <a
                            className="btn-primary"
                            href={candidate.profile_url}
                            target="_blank"
                            rel="noopener noreferrer"
                        >
                            View Profile
                        </a>
                    )}
                    <button className="btn-secondary" onClick={onClose}>
                        Close
                    </button>
                </div>
            </div>
        </div>
    );
}
