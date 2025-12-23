"""Mock implementations of core interfaces for testing and development.

These mocks can be used for:
- Unit testing without external dependencies
- Local development without live API access
- Demo scenarios and integration testing
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.models.reddit_data import RedditPost
from src.models.issue import IssueAnalysis, ICMIssue, AnalyzedPost
from src.interfaces.reddit_client import IRedditClient
from src.interfaces.llm_analyzer import ILLMAnalyzer
from src.interfaces.icm_manager import IICMManager
from src.interfaces.post_tracker import IPostTracker


class MockRedditClient(IRedditClient):
    """Mock Reddit client for testing.
    
    Attributes:
        posts: List of posts to return from queries.
        calls: List of method calls for verification.
    """
    
    def __init__(self, posts: Optional[List[RedditPost]] = None):
        """Initialize with optional preset posts.
        
        Args:
            posts: List of RedditPost objects to return from queries.
        """
        self.posts = posts or []
        self.calls: List[tuple] = []
    
    def get_recent_posts(self, subreddit: str, limit: int = 100, time_filter: str = "hour") -> List[RedditPost]:
        self.calls.append(("get_recent_posts", subreddit, limit, time_filter))
        return [p for p in self.posts if p.subreddit == subreddit][:limit]
    
    def get_post_by_id(self, post_id: str) -> Optional[RedditPost]:
        self.calls.append(("get_post_by_id", post_id))
        for post in self.posts:
            if post.id == post_id:
                return post
        return None
    
    def get_hot_posts(self, subreddit: str, limit: int = 100) -> List[RedditPost]:
        self.calls.append(("get_hot_posts", subreddit, limit))
        return [p for p in self.posts if p.subreddit == subreddit][:limit]
    
    def get_rising_posts(self, subreddit: str, limit: int = 100) -> List[RedditPost]:
        self.calls.append(("get_rising_posts", subreddit, limit))
        return [p for p in self.posts if p.subreddit == subreddit][:limit]


class MockLLMAnalyzer(ILLMAnalyzer):
    """Mock LLM analyzer for testing.
    
    Attributes:
        default_analysis: Default analysis to return if no specific one is set.
        post_analyses: Mapping of post_id to specific analysis results.
        duplicate_results: Mapping of summary to duplicate check results.
        calls: List of method calls for verification.
    """
    
    def __init__(self, default_analysis: Optional[IssueAnalysis] = None):
        """Initialize with optional default analysis.
        
        Args:
            default_analysis: Default IssueAnalysis to return.
        """
        self.default_analysis = default_analysis or IssueAnalysis.no_issue()
        self.post_analyses: dict = {}
        self.duplicate_results: dict = {}
        self.calls: List[tuple] = []
    
    def set_analysis_for_post(self, post_id: str, analysis: IssueAnalysis) -> None:
        """Set a specific analysis result for a post.
        
        Args:
            post_id: The post ID to set the analysis for.
            analysis: The analysis to return for this post.
        """
        self.post_analyses[post_id] = analysis
    
    def set_duplicate_result(self, summary: str, is_duplicate: bool) -> None:
        """Set a duplicate check result.
        
        Args:
            summary: The issue summary to match.
            is_duplicate: Whether to report as duplicate.
        """
        self.duplicate_results[summary] = is_duplicate
    
    def analyze_post(self, post: RedditPost) -> IssueAnalysis:
        self.calls.append(("analyze_post", post.id))
        return self.post_analyses.get(post.id, self.default_analysis)
    
    def analyze_posts_batch(self, posts: List[RedditPost]) -> List[IssueAnalysis]:
        self.calls.append(("analyze_posts_batch", [p.id for p in posts]))
        return [self.analyze_post(post) for post in posts]
    
    def check_duplicate(self, analysis: IssueAnalysis, existing_issues: List[ICMIssue]) -> bool:
        self.calls.append(("check_duplicate", analysis.summary))
        return self.duplicate_results.get(analysis.summary, False)
    
    def get_model_name(self) -> str:
        return "mock-model"


class MockICMManager(IICMManager):
    """Mock ICM manager for testing.
    
    Attributes:
        issues: List of ICM issues.
        calls: List of method calls for verification.
    """
    
    def __init__(self, existing_issues: Optional[List[ICMIssue]] = None):
        """Initialize with optional existing issues.
        
        Args:
            existing_issues: List of pre-existing ICM issues.
        """
        self.issues = existing_issues or []
        self.calls: List[tuple] = []
        self._next_id = 1
    
    def get_current_issues(self) -> List[ICMIssue]:
        self.calls.append(("get_current_issues",))
        return list(self.issues)
    
    def create_new_icm(
        self,
        title: str,
        description: str,
        severity: str,
        source_url: str,
        category: str,
        tags: List[str]
    ) -> ICMIssue:
        self.calls.append(("create_new_icm", title, severity, category))
        icm = ICMIssue(
            id=f"ICM-{self._next_id:04d}",
            title=title,
            description=description,
            severity=severity,
            source_url=source_url,
            category=category,
            tags=tags,
            created_at=datetime.now(timezone.utc),
            status="open",
        )
        self._next_id += 1
        self.issues.append(icm)
        return icm
    
    def get_issue_by_id(self, issue_id: str) -> Optional[ICMIssue]:
        self.calls.append(("get_issue_by_id", issue_id))
        for issue in self.issues:
            if issue.id == issue_id:
                return issue
        return None
    
    def update_issue_status(self, issue_id: str, status: str) -> bool:
        self.calls.append(("update_issue_status", issue_id, status))
        for issue in self.issues:
            if issue.id == issue_id:
                issue.status = status
                return True
        return False
    
    def add_comment_to_issue(self, issue_id: str, comment: str) -> bool:
        self.calls.append(("add_comment_to_issue", issue_id, comment))
        return any(issue.id == issue_id for issue in self.issues)


class InMemoryPostTracker(IPostTracker):
    """In-memory post tracker for testing.
    
    Stores analyzed posts in memory without persistence.
    Useful for testing and short-lived processes.
    
    Attributes:
        analyzed_posts: Dictionary of post_id to AnalyzedPost.
        calls: List of method calls for verification.
    """
    
    def __init__(self):
        """Initialize an empty tracker."""
        self.analyzed_posts: dict = {}
        self.calls: List[tuple] = []
    
    def is_analyzed(self, post_id: str) -> bool:
        self.calls.append(("is_analyzed", post_id))
        return post_id in self.analyzed_posts

    def are_analyzed(self, post_ids: List[str]) -> Dict[str, bool]:
        """Check if multiple posts have already been analyzed (batch operation)."""
        self.calls.append(("are_analyzed", post_ids))
        return {post_id: post_id in self.analyzed_posts for post_id in post_ids}
    
    def mark_analyzed(
        self,
        post_id: str,
        subreddit: str,
        analysis: IssueAnalysis,
        icm_created: bool,
        icm_id: Optional[str] = None,
        post_title: Optional[str] = None,
        post_url: Optional[str] = None
    ) -> None:
        self.calls.append(("mark_analyzed", post_id, icm_created))
        self.analyzed_posts[post_id] = AnalyzedPost(
            post_id=post_id,
            subreddit=subreddit,
            analyzed_at=datetime.now(timezone.utc),
            analysis_result=analysis,
            icm_created=icm_created,
            icm_id=icm_id,
            post_title=post_title,
            post_url=post_url,
        )
    
    def get_analyzed_post(self, post_id: str) -> Optional[AnalyzedPost]:
        self.calls.append(("get_analyzed_post", post_id))
        return self.analyzed_posts.get(post_id)
    
    def get_analyzed_posts(
        self,
        since: Optional[datetime] = None,
        subreddit: Optional[str] = None,
        limit: int = 1000
    ) -> List[AnalyzedPost]:
        self.calls.append(("get_analyzed_posts", since, subreddit))
        results = list(self.analyzed_posts.values())
        if since:
            results = [p for p in results if p.analyzed_at >= since]
        if subreddit:
            results = [p for p in results if p.subreddit == subreddit]
        return results[:limit]
    
    def get_posts_with_issues(
        self,
        since: Optional[datetime] = None,
        subreddit: Optional[str] = None,
        limit: int = 1000
    ) -> List[AnalyzedPost]:
        self.calls.append(("get_posts_with_issues", since, subreddit))
        results = [p for p in self.analyzed_posts.values() if p.analysis_result.is_issue]
        if since:
            results = [p for p in results if p.analyzed_at >= since]
        if subreddit:
            results = [p for p in results if p.subreddit == subreddit]
        return results[:limit]
    
    def get_statistics(
        self,
        since: Optional[datetime] = None,
        subreddit: Optional[str] = None
    ) -> dict:
        self.calls.append(("get_statistics", since, subreddit))
        posts = list(self.analyzed_posts.values())
        if since:
            posts = [p for p in posts if p.analyzed_at >= since]
        if subreddit:
            posts = [p for p in posts if p.subreddit == subreddit]
        
        issues = [p for p in posts if p.analysis_result.is_issue]
        
        return {
            "total_analyzed": len(posts),
            "issues_detected": len(issues),
            "icms_created": sum(1 for p in posts if p.icm_created),
            "by_category": {},
            "by_severity": {},
            "average_confidence": 0.0,
        }
    
    def cleanup_old_records(self, older_than: datetime) -> int:
        self.calls.append(("cleanup_old_records", older_than))
        to_delete = [
            post_id for post_id, post in self.analyzed_posts.items()
            if post.analyzed_at < older_than
        ]
        for post_id in to_delete:
            del self.analyzed_posts[post_id]
        return len(to_delete)
