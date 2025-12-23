"""Pytest fixtures and shared test utilities."""

import pytest
import tempfile
import os
from datetime import datetime, timezone
from typing import List, Optional
from pathlib import Path

from src.models.reddit_data import RedditPost, RedditComment
from src.models.issue import IssueAnalysis, ICMIssue, AnalyzedPost
from src.interfaces.reddit_client import IRedditClient
from src.interfaces.llm_analyzer import ILLMAnalyzer
from src.interfaces.icm_manager import IICMManager
from src.interfaces.post_tracker import IPostTracker
from src.tracking.sqlite_tracker import SQLitePostTracker
from src.llm_logging.llm_logger import LLMLogger
from src.testing.mocks import (
    MockRedditClient,
    MockLLMAnalyzer,
    MockICMManager,
    InMemoryPostTracker,
)

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
