/** API client for the FastAPI backend. */

import type { Candidate, SearchFilters } from "./types";

const API_BASE = "http://localhost:8000";

export interface ProgressEvent {
    type: "progress";
    stage: string;
    message: string;
    detail?: string;
}

export interface ResultsEvent {
    type: "results";
    candidates: Candidate[];
    total_results: number;
    has_more: boolean;
}

/**
 * Stream candidate search results via SSE.
 * Returns an unsubscribe function.
 */
export function streamCandidates(
    filters: SearchFilters,
    onProgress: (stage: string, message: string, detail?: string) => void,
    onResults: (candidates: Candidate[], total: number) => void,
    onError: (message: string) => void,
    onDone: () => void,
): () => void {
    const params = new URLSearchParams();

    if (filters.skills.length > 0) {
        params.set("skills", filters.skills.join(","));
    }
    if (filters.experience_level) {
        params.set("experience_level", filters.experience_level);
    }
    if (filters.location) {
        params.set("location", filters.location);
    }
    if (filters.description) {
        params.set("description", filters.description);
    }

    const url = `${API_BASE}/api/candidates/stream?${params.toString()}`;
    const evtSource = new EventSource(url);

    evtSource.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);

            if (data.type === "progress") {
                onProgress(data.stage ?? "", data.message ?? "", data.detail);
            } else if (data.type === "results") {
                onResults(data.candidates ?? [], data.total_results ?? 0);
            } else if (data.type === "done") {
                evtSource.close();
                onDone();
            }
        } catch {
            // ignore parse errors
        }
    };

    evtSource.onerror = () => {
        evtSource.close();
        onError("Connection lost. Please try again.");
        onDone();
    };

    return () => evtSource.close();
}

export async function healthCheck(): Promise<boolean> {
    try {
        const resp = await fetch(`${API_BASE}/api/health`);
        return resp.ok;
    } catch {
        return false;
    }
}
