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
        """Get recent posts from a subreddit.

        Args:
            subreddit: Name of the subreddit to query.
            limit: Maximum number of posts to return.
            time_filter: Time filter for posts (e.g., "hour", "day").

        Returns:
            List of RedditPost objects matching the subreddit.
        """
        self.calls.append(("get_recent_posts", subreddit, limit, time_filter))
        return [p for p in self.posts if p.subreddit == subreddit][:limit]
    
    def get_post_by_id(self, post_id: str) -> Optional[RedditPost]:
        """Get a specific post by its ID.

        Args:
            post_id: The unique identifier of the post.

        Returns:
            The RedditPost if found, None otherwise.
        """
        self.calls.append(("get_post_by_id", post_id))
        for post in self.posts:
            if post.id == post_id:
                return post
        return None
    
    def get_hot_posts(self, subreddit: str, limit: int = 100) -> List[RedditPost]:
        """Get hot posts from a subreddit.

        Args:
            subreddit: Name of the subreddit to query.
            limit: Maximum number of posts to return.

        Returns:
            List of RedditPost objects matching the subreddit.
        """
        self.calls.append(("get_hot_posts", subreddit, limit))
        return [p for p in self.posts if p.subreddit == subreddit][:limit]
    
    def get_rising_posts(self, subreddit: str, limit: int = 100) -> List[RedditPost]:
        """Get rising posts from a subreddit.

        Args:
            subreddit: Name of the subreddit to query.
            limit: Maximum number of posts to return.

        Returns:
            List of RedditPost objects matching the subreddit.
        """
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
        """Analyze a single post for potential issues.

        Args:
            post: The RedditPost to analyze.

        Returns:
            IssueAnalysis result for the post.
        """
        self.calls.append(("analyze_post", post.id))
        return self.post_analyses.get(post.id, self.default_analysis)
    
    def analyze_posts_batch(self, posts: List[RedditPost]) -> List[IssueAnalysis]:
        """Analyze multiple posts in a batch operation.

        Args:
            posts: List of RedditPost objects to analyze.

        Returns:
            List of IssueAnalysis results corresponding to each post.
        """
        self.calls.append(("analyze_posts_batch", [p.id for p in posts]))
        return [self.analyze_post(post) for post in posts]
    
    def check_duplicate(self, analysis: IssueAnalysis, existing_issues: List[ICMIssue]) -> bool:
        """Check if an analysis represents a duplicate of existing issues.

        Args:
            analysis: The IssueAnalysis to check.
            existing_issues: List of existing ICMIssue objects to compare against.

        Returns:
            True if the analysis is a duplicate, False otherwise.
        """
        self.calls.append(("check_duplicate", analysis.summary))
        return self.duplicate_results.get(analysis.summary, False)
    
    def get_model_name(self) -> str:
        """Get the name of the LLM model.

        Returns:
            The model name string.
        """
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
        """Get all current ICM issues.

        Returns:
            List of all ICMIssue objects.
        """
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
        """Create a new ICM issue.

        Args:
            title: Title of the issue.
            description: Detailed description of the issue.
            severity: Severity level of the issue.
            source_url: URL of the source that triggered the issue.
            category: Category classification for the issue.
            tags: List of tags to associate with the issue.

        Returns:
            The newly created ICMIssue object.
        """
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
        """Get an issue by its ID.

        Args:
            issue_id: The unique identifier of the issue.

        Returns:
            The ICMIssue if found, None otherwise.
        """
        self.calls.append(("get_issue_by_id", issue_id))
        for issue in self.issues:
            if issue.id == issue_id:
                return issue
        return None
    
    def update_issue_status(self, issue_id: str, status: str) -> bool:
        """Update the status of an issue.

        Args:
            issue_id: The unique identifier of the issue.
            status: The new status to set.

        Returns:
            True if the issue was found and updated, False otherwise.
        """
        self.calls.append(("update_issue_status", issue_id, status))
        for issue in self.issues:
            if issue.id == issue_id:
                issue.status = status
                return True
        return False
    
    def add_comment_to_issue(self, issue_id: str, comment: str) -> bool:
        """Add a comment to an issue.

        Args:
            issue_id: The unique identifier of the issue.
            comment: The comment text to add.

        Returns:
            True if the issue was found, False otherwise.
        """
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
        """Check if a post has been analyzed.

        Args:
            post_id: The unique identifier of the post.

        Returns:
            True if the post has been analyzed, False otherwise.
        """
        self.calls.append(("is_analyzed", post_id))
        return post_id in self.analyzed_posts

    def are_analyzed(self, post_ids: List[str]) -> Dict[str, bool]:
        """Check if multiple posts have already been analyzed (batch operation).

        Args:
            post_ids: List of post IDs to check.

        Returns:
            Dictionary mapping each post_id to its analyzed status.
        """
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
        """Mark a post as analyzed and store the result.

        Args:
            post_id: The unique identifier of the post.
            subreddit: Name of the subreddit the post belongs to.
            analysis: The IssueAnalysis result for the post.
            icm_created: Whether an ICM was created for this post.
            icm_id: The ID of the created ICM, if any.
            post_title: Title of the post.
            post_url: URL of the post.
        """
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
        """Get the analysis result for a specific post.

        Args:
            post_id: The unique identifier of the post.

        Returns:
            The AnalyzedPost if found, None otherwise.
        """
        self.calls.append(("get_analyzed_post", post_id))
        return self.analyzed_posts.get(post_id)
    
    def get_analyzed_posts(
        self,
        since: Optional[datetime] = None,
        subreddit: Optional[str] = None,
        limit: int = 1000
    ) -> List[AnalyzedPost]:
        """Get analyzed posts with optional filtering.

        Args:
            since: Only return posts analyzed after this datetime.
            subreddit: Filter to posts from this subreddit.
            limit: Maximum number of posts to return.

        Returns:
            List of AnalyzedPost objects matching the filters.
        """
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
        """Get analyzed posts that were identified as issues.

        Args:
            since: Only return posts analyzed after this datetime.
            subreddit: Filter to posts from this subreddit.
            limit: Maximum number of posts to return.

        Returns:
            List of AnalyzedPost objects that are issues.
        """
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
        """Get statistics about analyzed posts.

        Args:
            since: Only include posts analyzed after this datetime.
            subreddit: Filter to posts from this subreddit.

        Returns:
            Dictionary containing statistics including total_analyzed,
            issues_detected, icms_created, by_category, by_severity,
            and average_confidence.
        """
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
        """Remove records older than the specified datetime.

        Args:
            older_than: Remove records analyzed before this datetime.

        Returns:
            Number of records deleted.
        """
        self.calls.append(("cleanup_old_records", older_than))
        to_delete = [
            post_id for post_id, post in self.analyzed_posts.items()
            if post.analyzed_at < older_than
        ]
        for post_id in to_delete:
            del self.analyzed_posts[post_id]
        return len(to_delete)
