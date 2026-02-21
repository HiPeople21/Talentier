"""
GitHub Tools Integration
Unified module for searching GitHub repositories and analyzing commits.
Ready for FastAPI integration.
"""

from collections import defaultdict
from dotenv import load_dotenv
from typing import List, Dict, Optional
import os
import requests
import time

from models import (
    Repository,
    RepositorySearchResponse,
    ContributorInfo,
    ContributorsResponse,
    CommitInfo,
    ContributorCommitsResponse,
)

# Load .env file
load_dotenv()

BASE_URL = "https://api.github.com"

# ============================================================================
# GitHub API Client
# ============================================================================

class GitHubClient:
    """Base client for GitHub API operations."""
    
    def __init__(self, token: Optional[str] = None):
        """Initialize GitHub API client."""
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.session = requests.Session()
        if self.token:
            self.session.headers.update({"Authorization": f"token {self.token}"})
        self.session.headers.update({"Accept": "application/vnd.github.v3+json"})
    
    def _get(self, url: str, params: Optional[Dict] = None, retries: int = 3) -> Optional[Dict]:
        """Make GET request with retry logic."""
        for attempt in range(retries):
            try:
                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
        return None


# ============================================================================
# Repository Searcher
# ============================================================================

class RepositorySearcher(GitHubClient):
    """Search GitHub repositories by tags and filters."""
    
    def search_by_tag(
        self,
        tag: str,
        language: Optional[str] = None,
        min_stars: int = 0,
        sort: str = "stars",
        order: str = "desc",
        max_results: int = 30
    ) -> RepositorySearchResponse:
        """
        Search repositories by tag/topic.
        
        Args:
            tag: Topic/tag to search for
            language: Programming language filter
            min_stars: Minimum star count
            sort: Sort by 'stars', 'forks', 'updated', or 'score'
            order: 'asc' or 'desc'
            max_results: Maximum results to return
            
        Returns:
            RepositorySearchResponse with results
        """
        try:
            repos = []
            query = f"topic:{tag}"
            
            if language:
                query += f" language:{language}"
            
            if min_stars > 0:
                query += f" stars:>={min_stars}"
            
            page = 1
            per_page = min(100, max_results)
            
            while len(repos) < max_results:
                params = {
                    "q": query,
                    "sort": sort,
                    "order": order,
                    "per_page": per_page,
                    "page": page
                }
                
                data = self._get(f"{BASE_URL}/search/repositories", params=params)
                
                if not data or "items" not in data or not data["items"]:
                    break
                
                repos.extend(data["items"])
                page += 1
                time.sleep(0.3)
            
            return RepositorySearchResponse(
                tag=tag,
                language=language,
                min_stars=min_stars,
                repositories=repos[:max_results],
                total_found=len(repos[:max_results]),
                success=True
            )
        
        except Exception as e:
            return RepositorySearchResponse(
                tag=tag,
                language=language,
                min_stars=min_stars,
                repositories=[],
                total_found=0,
                success=False,
                error=str(e)
            )
    
    def search_by_multiple_tags(
        self,
        tags: List[str],
        language: Optional[str] = None,
        min_stars: int = 0,
        max_results: int = 30
    ) -> RepositorySearchResponse:
        """
        Search for repositories that have all specified tags.
        
        Args:
            tags: List of tags to search (repos must have ALL tags)
            language: Programming language filter
            min_stars: Minimum star count
            max_results: Maximum results to return
            
        Returns:
            RepositorySearchResponse with repos containing all tags
        """
        try:
            repos = []
            # Build query with all tags - repos must have ALL of them
            query = " ".join([f"topic:{tag}" for tag in tags])
            
            if language:
                query += f" language:{language}"
            
            if min_stars > 0:
                query += f" stars:>={min_stars}"
            
            page = 1
            per_page = min(100, max_results)
            
            while len(repos) < max_results:
                params = {
                    "q": query,
                    "sort": "stars",
                    "order": "desc",
                    "per_page": per_page,
                    "page": page
                }
                
                data = self._get(f"{BASE_URL}/search/repositories", params=params)
                
                if not data or "items" not in data or not data["items"]:
                    break
                
                repos.extend(data["items"])
                page += 1
                time.sleep(0.3)
            
            return RepositorySearchResponse(
                tag=", ".join(tags),
                language=language,
                min_stars=min_stars,
                repositories=repos[:max_results],
                total_found=len(repos[:max_results]),
                success=True
            )
        
        except Exception as e:
            return RepositorySearchResponse(
                tag=", ".join(tags),
                language=language,
                min_stars=min_stars,
                repositories=[],
                total_found=0,
                success=False,
                error=str(e)
            )


# ============================================================================
# Commit Analyzer
# ============================================================================

class CommitAnalyzer(GitHubClient):
    """Analyze commits and contributors in GitHub repositories."""
    
    def _get_commit_diff(self, owner: str, repo: str, sha: str) -> Optional[Dict]:
        """
        Get detailed commit information including diff.
        
        Args:
            owner: Repository owner
            repo: Repository name
            sha: Commit SHA
            
        Returns:
            Commit details with patch (diff)
        """
        try:
            url = f"{BASE_URL}/repos/{owner}/{repo}/commits/{sha}"
            return self._get(url)
        except Exception as e:
            return None
    
    def get_contributors(
        self,
        owner: str,
        repo: str,
        max_results: int = 50
    ) -> ContributorsResponse:
        """
        Get top contributors for a repository.
        
        Args:
            owner: Repository owner
            repo: Repository name
            max_results: Maximum contributors to fetch
            
        Returns:
            ContributorsResponse with contributor list
        """
        try:
            contributors = []
            page = 1
            per_page = min(100, max_results)
            
            while len(contributors) < max_results:
                params = {"per_page": per_page, "page": page}
                url = f"{BASE_URL}/repos/{owner}/{repo}/contributors"
                
                data = self._get(url, params=params)
                
                if not data:
                    break
                
                contributors.extend(data)
                page += 1
                time.sleep(0.2)
            
            total_commits = sum(c.get("contributions", 0) for c in contributors[:max_results])
            
            return ContributorsResponse(
                owner=owner,
                repo=repo,
                contributors=[ContributorInfo(**c) for c in contributors[:max_results]],
                total_contributors=len(contributors[:max_results]),
                total_commits=total_commits,
                success=True
            )
        
        except Exception as e:
            return ContributorsResponse(
                owner=owner,
                repo=repo,
                contributors=[],
                total_contributors=0,
                total_commits=0,
                success=False,
                error=str(e)
            )
    
    def get_contributor_commits(
        self,
        owner: str,
        repo: str,
        contributor: str,
        max_results: int = 50
    ) -> ContributorCommitsResponse:
        """
        Get commits by a specific contributor.
        
        Args:
            owner: Repository owner
            repo: Repository name
            contributor: Contributor login
            max_results: Maximum commits to fetch
            
        Returns:
            ContributorCommitsResponse with commits
        """
        try:
            commits = []
            page = 1
            per_page = min(100, max_results)
            
            while len(commits) < max_results:
                params = {
                    "author": contributor,
                    "per_page": per_page,
                    "page": page
                }
                url = f"{BASE_URL}/repos/{owner}/{repo}/commits"
                
                data = self._get(url, params=params)
                
                if not data:
                    break
                
                commits.extend(data)
                page += 1
                time.sleep(0.2)
            
            commit_list = []
            for commit in commits[:max_results]:
                # Fetch full commit details including diff
                commit_sha = commit.get("sha", "")
                full_commit = self._get_commit_diff(owner, repo, commit_sha)
                
                # Build diff from all file patches
                diff_str = ""
                files_changed = 0
                additions = 0
                deletions = 0
                
                if full_commit and "files" in full_commit:
                    files = full_commit.get("files", [])
                    files_changed = len(files)
                    
                    for file in files:
                        # Collect patch from each file
                        if "patch" in file:
                            diff_str += file.get("patch", "") + "\n"
                        additions += file.get("additions", 0)
                        deletions += file.get("deletions", 0)
                
                commit_info = CommitInfo(
                    sha=commit.get("sha", ""),
                    message=commit.get("commit", {}).get("message", "").split('\n')[0],
                    author=commit.get("commit", {}).get("author", {}).get("name", ""),
                    date=commit.get("commit", {}).get("author", {}).get("date", ""),
                    url=commit.get("html_url", ""),
                    diff=diff_str.strip(),
                    files_changed=files_changed,
                    additions=additions,
                    deletions=deletions
                )
                commit_list.append(commit_info)
                time.sleep(0.1)  # Rate limit for diff fetches
            
            return ContributorCommitsResponse(
                owner=owner,
                repo=repo,
                contributor=contributor,
                commits=commit_list,
                total_commits=len(commit_list),
                success=True
            )
        
        except Exception as e:
            return ContributorCommitsResponse(
                owner=owner,
                repo=repo,
                contributor=contributor,
                commits=[],
                total_commits=0,
                success=False,
                error=str(e)
            )
    
    def get_all_contributors_commits(
        self,
        owner: str,
        repo: str,
        max_contributors: int = 10,
        commits_per_contributor: int = 20
    ) -> Dict[str, ContributorCommitsResponse]:
        """
        Get commits for top contributors.
        
        Args:
            owner: Repository owner
            repo: Repository name
            max_contributors: Number of top contributors
            commits_per_contributor: Commits per contributor
            
        Returns:
            Dictionary mapping contributor login to their commits
        """
        contributors_resp = self.get_contributors(owner, repo, max_contributors)
        
        if not contributors_resp.success:
            return {}
        
        results = {}
        for contributor in contributors_resp.contributors:
            commits_resp = self.get_contributor_commits(
                owner, repo, contributor.login, max_results=commits_per_contributor
            )
            results[contributor.login] = commits_resp
            time.sleep(0.3)
        
        return results


# ============================================================================
# Unified GitHub Tools API
# ============================================================================

class GitHubTools:
    """Unified interface for GitHub repository and commit operations."""
    
    def __init__(self, token: Optional[str] = None):
        """Initialize GitHub tools."""
        self.searcher = RepositorySearcher(token=token)
        self.analyzer = CommitAnalyzer(token=token)
    
    def search_repos(
        self,
        tag: str,
        language: Optional[str] = None,
        min_stars: int = 0,
        sort: str = "stars",
        order: str = "desc",
        max_results: int = 30
    ) -> RepositorySearchResponse:
        """Search repositories by tag."""
        return self.searcher.search_by_tag(
            tag=tag,
            language=language,
            min_stars=min_stars,
            sort=sort,
            order=order,
            max_results=max_results
        )
    
    def search_multiple_repos(
        self,
        tags: List[str],
        language: Optional[str] = None,
        min_stars: int = 0,
        max_results: int = 30
    ) -> RepositorySearchResponse:
        """Search for repos with all specified tags."""
        return self.searcher.search_by_multiple_tags(
            tags=tags,
            language=language,
            min_stars=min_stars,
            max_results=max_results
        )
    
    def get_contributors(
        self,
        owner: str,
        repo: str,
        max_results: int = 50
    ) -> ContributorsResponse:
        """Get repository contributors."""
        return self.analyzer.get_contributors(owner, repo, max_results)
    
    def get_commits(
        self,
        owner: str,
        repo: str,
        contributor: str,
        max_results: int = 50
    ) -> ContributorCommitsResponse:
        """Get commits by a contributor."""
        return self.analyzer.get_contributor_commits(owner, repo, contributor, max_results)
    
    def get_all_contributor_commits(
        self,
        owner: str,
        repo: str,
        max_contributors: int = 10
    ) -> Dict[str, ContributorCommitsResponse]:
        """Get commits for all top contributors."""
        return self.analyzer.get_all_contributors_commits(owner, repo, max_contributors)


# Singleton instance
_github_tools = None

def get_github_tools() -> GitHubTools:
    """Get or create the GitHub tools instance."""
    global _github_tools
    if _github_tools is None:
        _github_tools = GitHubTools()
    return _github_tools
