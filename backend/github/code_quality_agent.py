"""Code Quality Grading Agent using LangGraph."""

from typing import TypedDict, Optional, List, Annotated
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaLLM
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from models import CodeQualityMetric

LLM_MAX_RETRIES = 3
LLM_RETRY_BACKOFF = 2  # seconds (doubled each attempt)
LLM_TIMEOUT = 120  # seconds per invocation


# ============================================================================
# Models
# ============================================================================

class AnalysisState(TypedDict):
    """State for the code quality analysis graph."""
    commit_sha: str
    diff: str
    message: str
    analysis: Optional[str]
    extracted_metrics: Optional[CodeQualityMetric]
    error: Optional[str]


# ============================================================================
# LangGraph Agent
# ============================================================================

class CodeQualityAgent:
    """AI agent for grading code quality using LLM with objective scoring criteria."""
    
    def __init__(self, model: str = "mistral"):
        """Initialize the code quality grading agent."""
        self.llm = OllamaLLM(model=model, temperature=0.1)
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AnalysisState)
        
        # Add nodes
        workflow.add_node("analyze_diff", self._analyze_diff)
        workflow.add_node("extract_metrics", self._extract_metrics)
        workflow.add_node("validate_metrics", self._validate_metrics)
        
        # Add edges
        workflow.add_edge("analyze_diff", "extract_metrics")
        workflow.add_edge("extract_metrics", "validate_metrics")
        workflow.add_edge("validate_metrics", END)
        
        # Set entry point
        workflow.set_entry_point("analyze_diff")
        
        return workflow.compile()
    
    def _analyze_diff(self, state: AnalysisState) -> AnalysisState:
        """Grade the diff using LLM."""
        print(f"[CODE QUALITY] Grading commit {state['commit_sha'][:8]}...")
        print(f"[CODE QUALITY] Commit message: {state['message'][:50]}...")
        print(f"[CODE QUALITY] Diff length: {len(state['diff'])} characters")
        try:
            prompt = f"""You are a code quality grading system. Grade the following code diff objectively based on software engineering best practices.

GRADING CRITERIA (Score each 0-100):

1. MAINTAINABILITY (0-100): Evaluate code structure, modularity, and how easy it would be to modify
   - 90-100: Excellent structure, highly modular, well-organized
   - 70-89: Good structure with minor improvements needed
   - 50-69: Acceptable but could be better organized
   - 30-49: Poor structure, difficult to maintain
   - 0-29: Very poor, significant refactoring needed

2. READABILITY (0-100): Assess code clarity, naming, and documentation
   - 90-100: Crystal clear, self-documenting, excellent naming
   - 70-89: Clear and well-named with good comments
   - 50-69: Understandable but could be clearer
   - 30-49: Confusing, poor naming conventions
   - 0-29: Very hard to understand

3. TEST COVERAGE (0-100): Evaluate testing practices
   - 90-100: Comprehensive tests, edge cases covered
   - 70-89: Good test coverage for main functionality
   - 50-69: Basic tests present
   - 30-49: Minimal or incomplete tests
   - 0-29: No tests or very poor coverage

4. COMPLEXITY (0-100): Assess whether code is unnecessarily complex (higher score = simpler)
   - 90-100: Simple, elegant solution
   - 70-89: Appropriately complex for the problem
   - 50-69: More complex than needed
   - 30-49: Overly complex, hard to follow
   - 0-29: Extremely convoluted

5. PERFORMANCE (0-100): Identify performance issues
   - 90-100: Highly optimized, no concerns
   - 70-89: Good performance characteristics
   - 50-69: Acceptable but room for optimization
   - 30-49: Performance issues present
   - 0-29: Serious performance problems

6. SECURITY (0-100): Check for security vulnerabilities
   - 90-100: Secure, follows best practices
   - 70-89: Generally secure with minor concerns
   - 50-69: Some security issues to address
   - 30-49: Notable security vulnerabilities
   - 0-29: Critical security flaws

Commit message: {state['message']}

Code diff:
{state['diff'][:2000]}...

Provide ONLY a JSON response with numeric grades and critical issues:
{{
    "overall_score": <average of all scores, 0-100>,
    "maintainability": <0-100>,
    "readability": <0-100>,
    "test_coverage": <0-100>,
    "complexity": <0-100>,
    "performance": <0-100>,
    "security": <0-100>,
    "issues": ["issue 1", "issue 2"],
    "suggestions": ["grade justification 1", "grade justification 2"]
}}

Be objective and consistent. Base scores on concrete observations from the code."""
            
            print(f"[CODE QUALITY] Sending prompt to LLM...")
            last_err = None
            for attempt in range(1, LLM_MAX_RETRIES + 1):
                try:
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(self.llm.invoke, prompt)
                        analysis = future.result(timeout=LLM_TIMEOUT)
                    break
                except FuturesTimeoutError:
                    last_err = TimeoutError(f"LLM call timed out after {LLM_TIMEOUT}s")
                    print(f"[CODE QUALITY] LLM attempt {attempt}/{LLM_MAX_RETRIES} timed out after {LLM_TIMEOUT}s.")
                except Exception as exc:
                    last_err = exc
                    print(f"[CODE QUALITY] LLM attempt {attempt}/{LLM_MAX_RETRIES} failed: {exc}.")
                if attempt < LLM_MAX_RETRIES:
                    wait = LLM_RETRY_BACKOFF * (2 ** (attempt - 1))
                    print(f"[CODE QUALITY] Retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise last_err
            print(f"[CODE QUALITY] LLM grading completed, response length: {len(analysis)} characters")
            state["analysis"] = analysis
            
        except Exception as e:
            print(f"[CODE QUALITY] ERROR in grading: {str(e)}")
            state["error"] = str(e)
        
        return state
    
    def _extract_metrics(self, state: AnalysisState) -> AnalysisState:
        """Extract grades from the LLM response."""
        print(f"[CODE QUALITY] Extracting grades from LLM response...")
        try:
            if state.get("error"):
                print(f"[CODE QUALITY] Skipping extraction due to previous error")
                return state
            
            analysis_text = state["analysis"]
            
            # Extract JSON from the response
            print(f"[CODE QUALITY] Searching for JSON in LLM response...")
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if not json_match:
                print(f"[CODE QUALITY] ERROR: Could not extract JSON from grading response")
                state["error"] = "Could not extract JSON from grading response"
                return state
            
            json_str = json_match.group()
            print(f"[CODE QUALITY] Parsing JSON response...")
            metrics_dict = json.loads(json_str)
            print(f"[CODE QUALITY] JSON parsed successfully")
            
            # Extract and normalize issues (handle both string arrays and object arrays)
            issues_raw = metrics_dict.get("issues", [])
            issues = []
            for issue in issues_raw:
                if isinstance(issue, str):
                    issues.append(issue)
                elif isinstance(issue, dict):
                    # Extract description or any string value
                    issues.append(issue.get("description", str(issue)))
                else:
                    issues.append(str(issue))
            
            # Extract and normalize suggestions (handle both string arrays and object arrays)
            suggestions_raw = metrics_dict.get("suggestions", [])
            suggestions = []
            for suggestion in suggestions_raw:
                if isinstance(suggestion, str):
                    suggestions.append(suggestion)
                elif isinstance(suggestion, dict):
                    # Extract description or any string value
                    suggestions.append(suggestion.get("description", str(suggestion)))
                else:
                    suggestions.append(str(suggestion))
            
            # Create CodeQualityMetric
            print(f"[CODE QUALITY] Creating grade report object...")
            metrics = CodeQualityMetric(
                commit_sha=state["commit_sha"],
                overall_score=metrics_dict.get("overall_score", 0),
                maintainability=metrics_dict.get("maintainability", 0),
                readability=metrics_dict.get("readability", 0),
                test_coverage=metrics_dict.get("test_coverage", 0),
                complexity=metrics_dict.get("complexity", 0),
                performance=metrics_dict.get("performance", 0),
                security=metrics_dict.get("security", 0),
                issues=issues,
                suggestions=suggestions
            )
            print(f"[CODE QUALITY] Grades extracted successfully - Overall score: {metrics.overall_score}")
            
            state["extracted_metrics"] = metrics
            
        except json.JSONDecodeError as e:
            print(f"[CODE QUALITY] ERROR: JSON parsing failed - {str(e)}")
            state["error"] = f"JSON parsing error: {str(e)}"
        except Exception as e:
            print(f"[CODE QUALITY] ERROR: Extraction failed - {str(e)}")
            state["error"] = f"Extraction error: {str(e)}"
        
        return state
    
    def _validate_metrics(self, state: AnalysisState) -> AnalysisState:
        """Validate and normalize grades."""
        print(f"[CODE QUALITY] Validating grades...")
        try:
            if state.get("error") or not state.get("extracted_metrics"):
                print(f"[CODE QUALITY] Skipping validation due to error or missing grades")
                return state
            
            metrics = state["extracted_metrics"]
            
            # Ensure all metrics are within valid range
            for field in ["overall_score", "maintainability", "readability", 
                         "test_coverage", "complexity", "performance", "security"]:
                value = getattr(metrics, field)
                if value > 100:
                    setattr(metrics, field, 100)
                elif value < 0:
                    setattr(metrics, field, 0)
            
            state["extracted_metrics"] = metrics
            print(f"[CODE QUALITY] Grades validated successfully")
            
        except Exception as e:
            print(f"[CODE QUALITY] ERROR: Validation failed - {str(e)}")
            state["error"] = f"Validation error: {str(e)}"
        
        return state
    
    def analyze_commit(
        self,
        commit_sha: str,
        diff: str,
        message: str
    ) -> CodeQualityMetric:
        """Grade a single commit and return quality scores."""
        print(f"\n[CODE QUALITY] ===== Starting grading for commit {commit_sha[:8]} =====")
        
        # Handle empty diff
        if not diff or diff.strip() == "":
            print(f"[CODE QUALITY] WARNING: Empty diff for commit {commit_sha[:8]}")
            return CodeQualityMetric(
                commit_sha=commit_sha,
                overall_score=50,
                maintainability=50,
                readability=50,
                test_coverage=0,
                complexity=50,
                performance=50,
                security=50,
                issues=["No code changes found in diff"],
                suggestions=["Cannot grade empty diff"]
            )
        
        initial_state: AnalysisState = {
            "commit_sha": commit_sha,
            "diff": diff,
            "message": message,
            "analysis": None,
            "extracted_metrics": None,
            "error": None
        }
        
        print(f"[CODE QUALITY] Invoking LangGraph workflow...")
        result = self.graph.invoke(initial_state)
        print(f"[CODE QUALITY] LangGraph workflow completed")
        
        if result.get("error"):
            print(f"[CODE QUALITY] Analysis failed with error: {result.get('error')}")
            return CodeQualityMetric(
                commit_sha=commit_sha,
                overall_score=0,
                maintainability=0,
                readability=0,
                test_coverage=0,
                complexity=0,
                performance=0,
                security=0,
                issues=[f"Grading error: {result['error']}"],
                suggestions=["Unable to grade this commit"]
            )
        
        print(f"[CODE QUALITY] ===== Grading complete for commit {commit_sha[:8]} =====\n")
        return result["extracted_metrics"] or CodeQualityMetric(
            commit_sha=commit_sha,
            overall_score=0,
            issues=["Failed to extract grades"],
            suggestions=["Unable to grade this commit"]
        )
    
    async def analyze_commits_batch(
        self,
        commits: List[dict]
    ) -> List[CodeQualityMetric]:
        """Grade multiple commits."""
        print(f"\n[CODE QUALITY] ========================================")
        print(f"[CODE QUALITY] Starting batch grading of {len(commits)} commits")
        print(f"[CODE QUALITY] ========================================\n")
        results = []
        for idx, commit in enumerate(commits, 1):
            print(f"[CODE QUALITY] Processing commit {idx}/{len(commits)}")
            metric = self.analyze_commit(
                commit_sha=commit.get("sha", "unknown"),
                diff=commit.get("diff", ""),
                message=commit.get("message", "")
            )
            results.append(metric)
        
        print(f"\n[CODE QUALITY] ========================================")
        print(f"[CODE QUALITY] Batch grading complete: {len(results)} commits graded")
        print(f"[CODE QUALITY] ========================================\n")
        return results


# Singleton instance
_code_quality_agent = None

def get_code_quality_agent() -> CodeQualityAgent:
    """Get or create the code quality agent instance."""
    global _code_quality_agent
    if _code_quality_agent is None:
        _code_quality_agent = CodeQualityAgent()
    return _code_quality_agent
