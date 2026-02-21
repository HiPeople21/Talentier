/** TypeScript interfaces for the Recruiter Candidate Finder. */

export interface Candidate {
    id: string;
    name: string;
    headline: string;
    location: string;
    profile_url: string;
    snippet: string;
    matched_skills: string[];
    avatar_initials: string;
}

export interface SearchFilters {
    skills: string[];
    experience_level: string;
    location: string;
    description: string;
}

export interface SearchResponse {
    candidates: Candidate[];
    total_results: number;
    page: number;
    has_more: boolean;
}
