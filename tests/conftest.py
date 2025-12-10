"""Pytest fixtures and shared test utilities."""

import pytest
import tempfile
import os
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from src.models.reddit_data import RedditPost, RedditComment
from src.models.issue import IssueAnalysis, ICMIssue, AnalyzedPost
from src.interfaces.reddit_client import IRedditClient
from src.interfaces.llm_analyzer import ILLMAnalyzer
from src.interfaces.icm_manager import IICMManager
from src.interfaces.post_tracker import IPostTracker
from src.tracking.sqlite_tracker import SQLitePostTracker
from src.logging.llm_logger import LLMLogger

# ============================================================
# Sample Data Fixtures
# ============================================================

@pytest.fixture
def sample_comment() -> RedditComment:
    """Create a sample Reddit comment."""
    return RedditComment(
        id="comment_123",
        post_id="post_456",
        body="I'm having the same issue! My Xbox keeps disconnecting from Xbox Live.",
        author="user123",
        created_utc=datetime(2024, 1, 15, 10, 30, 0),
        score=25,
        parent_id=None,
    )

@pytest.fixture
def sample_comments() -> List[RedditComment]:
    """Create a list of sample comments."""
    return [
        RedditComment(
            id="c1",
            post_id="post_456",
            body="Same here! Getting error code 0x87e50033",
            author="gamer1",
            created_utc=datetime(2024, 1, 15, 11, 0, 0),
            score=15,
            parent_id=None,
        ),
        RedditComment(
            id="c2",
            post_id="post_456",
            body="This started happening after the last update",
            author="gamer2",
            created_utc=datetime(2024, 1, 15, 11, 30, 0),
            score=10,
            parent_id="c1",
        ),
        RedditComment(
            id="c3",
            post_id="post_456",
            body="Try hard resetting your console, that fixed it for me",
            author="helpful_user",
            created_utc=datetime(2024, 1, 15, 12, 0, 0),
            score=20,
            parent_id=None,
        ),
    ]

@pytest.fixture
def sample_post(sample_comments) -> RedditPost:
    """Create a sample Reddit post."""
    return RedditPost(
        id="post_456",
        subreddit="xbox",
        title="Xbox keeps disconnecting from Xbox Live after update",
        body="Ever since the latest update, my Xbox Series X keeps losing connection to Xbox Live. "
             "I've tried restarting my router and console but nothing works. Anyone else having this issue?",
        author="frustrated_gamer",
        created_utc=datetime(2024, 1, 15, 10, 0, 0),
        url="https://reddit.com/r/xbox/comments/post_456",
        score=150,
        comments=sample_comments,
        num_comments=45,
        flair="Tech Support",
    )

@pytest.fixture
def sample_post_no_issue() -> RedditPost:
    """Create a sample Reddit post that doesn't describe an issue."""
    return RedditPost(
        id="post_789",
        subreddit="xbox",
        title="Just got my first Xbox! What games should I play?",
        body="Finally bought an Xbox Series X and looking for game recommendations. "
             "I'm into RPGs and shooters. What are your favorites?",
        author="new_gamer",
        created_utc=datetime(2024, 1, 15, 14, 0, 0),
        url="https://reddit.com/r/xbox/comments/post_789",
        score=50,
        comments=[],
        num_comments=25,
        flair="Discussion",
    )

@pytest.fixture
def sample_issue_analysis() -> IssueAnalysis:
    """Create a sample issue analysis."""
    return IssueAnalysis(
        is_issue=True,
        confidence=0.92,
        summary="Users experiencing Xbox Live connectivity issues after recent system update",
        category="connectivity",
        severity="high",
        affected_users_estimate=50,
        keywords=["Xbox Live", "disconnect", "update", "0x87e50033"],
        raw_response='{"is_issue": true, "confidence": 0.92, ...}',
    )

@pytest.fixture
def sample_icm_issue() -> ICMIssue:
    """Create a sample ICM issue."""
    return ICMIssue(
        id="ICM-2024-001",
        title="[Reddit] Xbox Live connectivity issues after update",
        description="Users are reporting connectivity issues with Xbox Live...",
        severity="high",
        source_url="https://reddit.com/r/xbox/comments/post_456",
        category="connectivity",
        tags=["reddit", "r/xbox", "connectivity", "Xbox Live"],
        created_at=datetime(2024, 1, 15, 12, 0, 0),
        status="open",
        source_post_id="post_456",
    )

# ============================================================
# Mock Implementations
# ============================================================

class MockRedditClient(IRedditClient):
    """Mock Reddit client for testing."""
    
    def __init__(self, posts: Optional[List[RedditPost]] = None):
        self.posts = posts or []
        self.calls = []
    
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
    """Mock LLM analyzer for testing."""
    
    def __init__(self, default_analysis: Optional[IssueAnalysis] = None):
        self.default_analysis = default_analysis or IssueAnalysis.no_issue()
        self.post_analyses = {}  # Map post_id -> IssueAnalysis
        self.duplicate_results = {}  # Map summary -> bool
        self.calls = []
    
    def set_analysis_for_post(self, post_id: str, analysis: IssueAnalysis):
        """Set a specific analysis result for a post."""
        self.post_analyses[post_id] = analysis
    
    def set_duplicate_result(self, summary: str, is_duplicate: bool):
        """Set a duplicate check result."""
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
    """Mock ICM manager for testing."""
    
    def __init__(self, existing_issues: Optional[List[ICMIssue]] = None):
        self.issues = existing_issues or []
        self.calls = []
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
            created_at=datetime.utcnow(),
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
    """In-memory post tracker for testing."""
    
    def __init__(self):
        self.analyzed_posts = {}
        self.calls = []
    
    def is_analyzed(self, post_id: str) -> bool:
        self.calls.append(("is_analyzed", post_id))
        return post_id in self.analyzed_posts
    
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
            analyzed_at=datetime.utcnow(),
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

# ============================================================
# Fixture Instances
# ============================================================

@pytest.fixture
def mock_reddit_client(sample_post, sample_post_no_issue) -> MockRedditClient:
    """Create a mock Reddit client with sample posts."""
    return MockRedditClient(posts=[sample_post, sample_post_no_issue])

@pytest.fixture
def mock_llm_analyzer(sample_issue_analysis) -> MockLLMAnalyzer:
    """Create a mock LLM analyzer."""
    analyzer = MockLLMAnalyzer(default_analysis=IssueAnalysis.no_issue())
    analyzer.set_analysis_for_post("post_456", sample_issue_analysis)
    return analyzer

@pytest.fixture
def mock_icm_manager() -> MockICMManager:
    """Create a mock ICM manager."""
    return MockICMManager()

@pytest.fixture
def in_memory_tracker() -> InMemoryPostTracker:
    """Create an in-memory post tracker."""
    return InMemoryPostTracker()

@pytest.fixture
def temp_db_path() -> str:
    """Create a temporary database path for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    # Cleanup
    if os.path.exists(path):
        os.unlink(path)

@pytest.fixture
def sqlite_tracker(temp_db_path) -> SQLitePostTracker:
    """Create a SQLite tracker with a temporary database."""
    return SQLitePostTracker(db_path=temp_db_path)

@pytest.fixture
def temp_log_dir() -> str:
    """Create a temporary directory for logs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def llm_logger(temp_log_dir) -> LLMLogger:
    """Create an LLM logger with temporary storage."""
    return LLMLogger(log_dir=temp_log_dir, log_to_console=False)
