"""Tests for the post tracker implementations."""

import pytest
from datetime import datetime, timedelta

from src.models.issue import IssueAnalysis
from src.tracking.sqlite_tracker import SQLitePostTracker

class TestSQLitePostTracker:
    """Tests for SQLitePostTracker."""

    def test_is_analyzed_false_initially(self, sqlite_tracker):
        """Test that new posts are not marked as analyzed.

        Args:
            sqlite_tracker: SQLitePostTracker fixture for testing.
        """
        assert sqlite_tracker.is_analyzed("new_post") is False

    def test_mark_analyzed(self, sqlite_tracker, sample_issue_analysis):
        """Test marking a post as analyzed.

        Args:
            sqlite_tracker: SQLitePostTracker fixture for testing.
            sample_issue_analysis: Sample IssueAnalysis fixture with an issue.
        """
        sqlite_tracker.mark_analyzed(
            post_id="p1",
            subreddit="xbox",
            analysis=sample_issue_analysis,
            icm_created=True,
            icm_id="ICM-001",
            post_title="Test Post",
            post_url="https://reddit.com/r/xbox/p1",
        )
        
        assert sqlite_tracker.is_analyzed("p1") is True

    def test_get_analyzed_post(self, sqlite_tracker, sample_issue_analysis):
        """Test retrieving an analyzed post.

        Args:
            sqlite_tracker: SQLitePostTracker fixture for testing.
            sample_issue_analysis: Sample IssueAnalysis fixture with an issue.
        """
        sqlite_tracker.mark_analyzed(
            post_id="p1",
            subreddit="xbox",
            analysis=sample_issue_analysis,
            icm_created=True,
            icm_id="ICM-001",
            post_title="Test Post",
            post_url="https://reddit.com/r/xbox/p1",
        )
        
        result = sqlite_tracker.get_analyzed_post("p1")
        
        assert result is not None
        assert result.post_id == "p1"
        assert result.icm_created is True
        assert result.icm_id == "ICM-001"
        assert result.analysis_result.is_issue is True
        assert result.analysis_result.category == "connectivity"

    def test_get_analyzed_post_not_found(self, sqlite_tracker):
        """Test retrieving a non-existent post.

        Args:
            sqlite_tracker: SQLitePostTracker fixture for testing.
        """
        result = sqlite_tracker.get_analyzed_post("nonexistent")
        assert result is None

    def test_get_analyzed_posts(self, sqlite_tracker, sample_issue_analysis):
        """Test retrieving multiple analyzed posts.

        Args:
            sqlite_tracker: SQLitePostTracker fixture for testing.
            sample_issue_analysis: Sample IssueAnalysis fixture with an issue.
        """
        no_issue = IssueAnalysis.no_issue()
        
        sqlite_tracker.mark_analyzed(
            post_id="p1",
            subreddit="xbox",
            analysis=sample_issue_analysis,
            icm_created=True,
            icm_id="ICM-001",
        )
        sqlite_tracker.mark_analyzed(
            post_id="p2",
            subreddit="xbox",
            analysis=no_issue,
            icm_created=False,
        )
        
        results = sqlite_tracker.get_analyzed_posts()
        
        assert len(results) == 2

    def test_get_analyzed_posts_filter_by_subreddit(self, sqlite_tracker, sample_issue_analysis):
        """Test filtering analyzed posts by subreddit.

        Args:
            sqlite_tracker: SQLitePostTracker fixture for testing.
            sample_issue_analysis: Sample IssueAnalysis fixture with an issue.
        """
        sqlite_tracker.mark_analyzed(
            post_id="p1",
            subreddit="xbox",
            analysis=sample_issue_analysis,
            icm_created=True,
        )
        sqlite_tracker.mark_analyzed(
            post_id="p2",
            subreddit="xboxone",
            analysis=sample_issue_analysis,
            icm_created=True,
        )
        
        results = sqlite_tracker.get_analyzed_posts(subreddit="xbox")
        
        assert len(results) == 1
        assert results[0].subreddit == "xbox"

    def test_get_posts_with_issues(self, sqlite_tracker, sample_issue_analysis):
        """Test getting only posts with issues.

        Args:
            sqlite_tracker: SQLitePostTracker fixture for testing.
            sample_issue_analysis: Sample IssueAnalysis fixture with an issue.
        """
        no_issue = IssueAnalysis.no_issue()
        
        sqlite_tracker.mark_analyzed(
            post_id="p1",
            subreddit="xbox",
            analysis=sample_issue_analysis,
            icm_created=True,
        )
        sqlite_tracker.mark_analyzed(
            post_id="p2",
            subreddit="xbox",
            analysis=no_issue,
            icm_created=False,
        )
        
        results = sqlite_tracker.get_posts_with_issues()
        
        assert len(results) == 1
        assert results[0].post_id == "p1"
        assert results[0].analysis_result.is_issue is True

    def test_get_statistics(self, sqlite_tracker, sample_issue_analysis):
        """Test getting statistics.

        Args:
            sqlite_tracker: SQLitePostTracker fixture for testing.
            sample_issue_analysis: Sample IssueAnalysis fixture with an issue.
        """
        no_issue = IssueAnalysis.no_issue()
        
        sqlite_tracker.mark_analyzed(
            post_id="p1",
            subreddit="xbox",
            analysis=sample_issue_analysis,
            icm_created=True,
        )
        sqlite_tracker.mark_analyzed(
            post_id="p2",
            subreddit="xbox",
            analysis=no_issue,
            icm_created=False,
        )
        sqlite_tracker.mark_analyzed(
            post_id="p3",
            subreddit="xbox",
            analysis=sample_issue_analysis,
            icm_created=False,  # Duplicate, no ICM created
        )
        
        stats = sqlite_tracker.get_statistics()
        
        assert stats["total_analyzed"] == 3
        assert stats["issues_detected"] == 2
        assert stats["icms_created"] == 1

    def test_cleanup_old_records(self, sqlite_tracker, sample_issue_analysis):
        """Test cleaning up old records.

        Args:
            sqlite_tracker: SQLitePostTracker fixture for testing.
            sample_issue_analysis: Sample IssueAnalysis fixture with an issue.
        """
        sqlite_tracker.mark_analyzed(
            post_id="p1",
            subreddit="xbox",
            analysis=sample_issue_analysis,
            icm_created=True,
        )
        
        # Clean up records older than 1 hour from now (should delete nothing)
        cutoff = datetime.utcnow() - timedelta(hours=1)
        deleted = sqlite_tracker.cleanup_old_records(cutoff)
        
        assert deleted == 0
        assert sqlite_tracker.is_analyzed("p1") is True
        
        # Clean up records older than now (should delete the record)
        cutoff = datetime.utcnow() + timedelta(seconds=1)
        deleted = sqlite_tracker.cleanup_old_records(cutoff)
        
        assert deleted == 1
        assert sqlite_tracker.is_analyzed("p1") is False

    def test_mark_analyzed_updates_existing(self, sqlite_tracker, sample_issue_analysis):
        """Test that marking an already analyzed post updates the record.

        Args:
            sqlite_tracker: SQLitePostTracker fixture for testing.
            sample_issue_analysis: Sample IssueAnalysis fixture with an issue.
        """
        no_issue = IssueAnalysis.no_issue()
        
        # First analysis
        sqlite_tracker.mark_analyzed(
            post_id="p1",
            subreddit="xbox",
            analysis=no_issue,
            icm_created=False,
        )
        
        # Re-analyze with issue
        sqlite_tracker.mark_analyzed(
            post_id="p1",
            subreddit="xbox",
            analysis=sample_issue_analysis,
            icm_created=True,
            icm_id="ICM-001",
        )
        
        result = sqlite_tracker.get_analyzed_post("p1")
        
        # Should have updated values
        assert result.analysis_result.is_issue is True
        assert result.icm_created is True
        assert result.icm_id == "ICM-001"

    def test_keywords_serialization(self, sqlite_tracker):
        """Test that keywords are properly serialized and deserialized.

        Args:
            sqlite_tracker: SQLitePostTracker fixture for testing.
        """
        analysis = IssueAnalysis(
            is_issue=True,
            confidence=0.9,
            summary="Test issue",
            category="connectivity",
            severity="high",
            keywords=["keyword1", "keyword2", "keyword3"],
        )
        
        sqlite_tracker.mark_analyzed(
            post_id="p1",
            subreddit="xbox",
            analysis=analysis,
            icm_created=False,
        )
        
        result = sqlite_tracker.get_analyzed_post("p1")
        
        assert result.analysis_result.keywords == ["keyword1", "keyword2", "keyword3"]

class TestInMemoryPostTracker:
    """Tests for InMemoryPostTracker (from conftest)."""

    def test_is_analyzed(self, in_memory_tracker, sample_issue_analysis):
        """Test is_analyzed functionality.

        Args:
            in_memory_tracker: InMemoryPostTracker fixture for testing.
            sample_issue_analysis: Sample IssueAnalysis fixture with an issue.
        """
        assert in_memory_tracker.is_analyzed("p1") is False
        
        in_memory_tracker.mark_analyzed(
            post_id="p1",
            subreddit="xbox",
            analysis=sample_issue_analysis,
            icm_created=False,
        )
        
        assert in_memory_tracker.is_analyzed("p1") is True

    def test_get_analyzed_posts(self, in_memory_tracker, sample_issue_analysis):
        """Test getting analyzed posts.

        Args:
            in_memory_tracker: InMemoryPostTracker fixture for testing.
            sample_issue_analysis: Sample IssueAnalysis fixture with an issue.
        """
        in_memory_tracker.mark_analyzed(
            post_id="p1",
            subreddit="xbox",
            analysis=sample_issue_analysis,
            icm_created=True,
        )
        
        results = in_memory_tracker.get_analyzed_posts()
        
        assert len(results) == 1
        assert results[0].post_id == "p1"
