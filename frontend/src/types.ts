/** TypeScript interfaces for the Recruiter Candidate Finder. */

export interface Candidate {
    id: string;
    name: string;
    headline: string;
    location: string;
    profile_url: string;
    linkedin_url?: string;
    github_url?: string;
    snippet: string;
    matched_skills: string[];
    avatar_initials: string;
    score: number;
    code_quality_score?: number;
    summary: string;
    source: string; // "linkedin" | "github"
}

export interface SearchFilters {
    skills: string[];
    experience_level: string;
    location: string;
    description: string;
}
