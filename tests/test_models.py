"""Tests for data models."""

import pytest
from datetime import datetime

from src.models.reddit_data import RedditPost, RedditComment
from src.models.issue import IssueAnalysis, ICMIssue, AnalyzedPost, Severity, IssueCategory

class TestRedditComment:
    """Tests for RedditComment model."""

    def test_create_comment(self, sample_comment):
        """Test creating a comment."""
        assert sample_comment.id == "comment_123"
        assert sample_comment.post_id == "post_456"
        assert sample_comment.author == "user123"
        assert sample_comment.score == 25

    def test_comment_to_dict(self, sample_comment):
        """Test converting comment to dictionary."""
        data = sample_comment.to_dict()
        
        assert data["id"] == "comment_123"
        assert data["post_id"] == "post_456"
        assert data["body"] == sample_comment.body
        assert data["parent_id"] is None

    def test_comment_from_dict(self, sample_comment):
        """Test creating comment from dictionary."""
        data = sample_comment.to_dict()
        restored = RedditComment.from_dict(data)
        
        assert restored.id == sample_comment.id
        assert restored.body == sample_comment.body
        assert restored.author == sample_comment.author

    def test_comment_with_parent(self):
        """Test comment with parent ID."""
        comment = RedditComment(
            id="c2",
            post_id="p1",
            body="Reply",
            author="user",
            created_utc=datetime.utcnow(),
            score=5,
            parent_id="c1",
        )
        
        assert comment.parent_id == "c1"
        data = comment.to_dict()
        assert data["parent_id"] == "c1"


class TestRedditPost:
    """Tests for RedditPost model."""

    def test_create_post(self, sample_post):
        """Test creating a post."""
        assert sample_post.id == "post_456"
        assert sample_post.subreddit == "xbox"
        assert sample_post.score == 150
        assert len(sample_post.comments) == 3

    def test_post_to_dict(self, sample_post):
        """Test converting post to dictionary."""
        data = sample_post.to_dict()
        
        assert data["id"] == "post_456"
        assert data["subreddit"] == "xbox"
        assert len(data["comments"]) == 3
        assert data["flair"] == "Tech Support"

    def test_post_from_dict(self, sample_post):
        """Test creating post from dictionary."""
        data = sample_post.to_dict()
        restored = RedditPost.from_dict(data)
        
        assert restored.id == sample_post.id
        assert restored.title == sample_post.title
        assert len(restored.comments) == len(sample_post.comments)

    def test_get_full_text(self, sample_post):
        """Test getting full text content."""
        full_text = sample_post.get_full_text()
        
        assert sample_post.title in full_text
        assert sample_post.body in full_text

    def test_get_full_text_no_body(self):
        """Test getting full text when no body."""
        post = RedditPost(
            id="p1",
            subreddit="xbox",
            title="Just a title",
            body="",
            author="user",
            created_utc=datetime.utcnow(),
            url="https://reddit.com/r/xbox/p1",
            score=10,
        )
        
        full_text = post.get_full_text()
        assert full_text == "Just a title"

    def test_get_comment_thread_text(self, sample_post):
        """Test getting comment thread text."""
        thread_text = sample_post.get_comment_thread_text()
        
        assert "gamer1" in thread_text
        assert "0x87e50033" in thread_text

    def test_get_comment_thread_text_limit(self, sample_post):
        """Test limiting comment thread."""
        thread_text = sample_post.get_comment_thread_text(max_comments=1)
        
        # Should only have first comment
        assert "gamer1" in thread_text
        assert "gamer2" not in thread_text


class TestIssueAnalysis:
    """Tests for IssueAnalysis model."""

    def test_create_analysis(self, sample_issue_analysis):
        """Test creating an analysis."""
        assert sample_issue_analysis.is_issue is True
        assert sample_issue_analysis.confidence == 0.92
        assert sample_issue_analysis.category == "connectivity"
        assert sample_issue_analysis.severity == "high"

    def test_analysis_to_dict(self, sample_issue_analysis):
        """Test converting analysis to dictionary."""
        data = sample_issue_analysis.to_dict()
        
        assert data["is_issue"] is True
        assert data["confidence"] == 0.92
        assert "Xbox Live" in data["keywords"]

    def test_analysis_from_dict(self, sample_issue_analysis):
        """Test creating analysis from dictionary."""
        data = sample_issue_analysis.to_dict()
        restored = IssueAnalysis.from_dict(data)
        
        assert restored.is_issue == sample_issue_analysis.is_issue
        assert restored.confidence == sample_issue_analysis.confidence
        assert restored.keywords == sample_issue_analysis.keywords

    def test_no_issue_factory(self):
        """Test creating no-issue analysis."""
        analysis = IssueAnalysis.no_issue()
        
        assert analysis.is_issue is False
        assert analysis.confidence == 1.0
        assert analysis.affected_users_estimate == 0

    def test_confidence_validation(self):
        """Test confidence must be between 0 and 1."""
        with pytest.raises(ValueError):
            IssueAnalysis(
                is_issue=True,
                confidence=1.5,  # Invalid
                summary="Test",
                category="other",
                severity="low",
            )

    def test_confidence_validation_negative(self):
        """Test confidence cannot be negative."""
        with pytest.raises(ValueError):
            IssueAnalysis(
                is_issue=True,
                confidence=-0.1,  # Invalid
                summary="Test",
                category="other",
                severity="low",
            )


class TestICMIssue:
    """Tests for ICMIssue model."""

    def test_create_icm(self, sample_icm_issue):
        """Test creating an ICM issue."""
        assert sample_icm_issue.id == "ICM-2024-001"
        assert sample_icm_issue.severity == "high"
        assert sample_icm_issue.status == "open"

    def test_icm_to_dict(self, sample_icm_issue):
        """Test converting ICM to dictionary."""
        data = sample_icm_issue.to_dict()
        
        assert data["id"] == "ICM-2024-001"
        assert "connectivity" in data["tags"]

    def test_icm_from_dict(self, sample_icm_issue):
        """Test creating ICM from dictionary."""
        data = sample_icm_issue.to_dict()
        restored = ICMIssue.from_dict(data)
        
        assert restored.id == sample_icm_issue.id
        assert restored.title == sample_icm_issue.title
        assert restored.tags == sample_icm_issue.tags


class TestAnalyzedPost:
    """Tests for AnalyzedPost model."""

    def test_create_analyzed_post(self, sample_issue_analysis):
        """Test creating an analyzed post record."""
        analyzed = AnalyzedPost(
            post_id="p1",
            subreddit="xbox",
            analyzed_at=datetime.utcnow(),
            analysis_result=sample_issue_analysis,
            icm_created=True,
            icm_id="ICM-001",
        )
        
        assert analyzed.post_id == "p1"
        assert analyzed.icm_created is True
        assert analyzed.analysis_result.is_issue is True

    def test_analyzed_post_to_dict(self, sample_issue_analysis):
        """Test converting analyzed post to dictionary."""
        analyzed = AnalyzedPost(
            post_id="p1",
            subreddit="xbox",
            analyzed_at=datetime(2024, 1, 15, 10, 0, 0),
            analysis_result=sample_issue_analysis,
            icm_created=True,
            icm_id="ICM-001",
        )
        
        data = analyzed.to_dict()
        assert data["post_id"] == "p1"
        assert data["icm_id"] == "ICM-001"
        assert data["analysis_result"]["is_issue"] is True

    def test_analyzed_post_from_dict(self, sample_issue_analysis):
        """Test creating analyzed post from dictionary."""
        analyzed = AnalyzedPost(
            post_id="p1",
            subreddit="xbox",
            analyzed_at=datetime(2024, 1, 15, 10, 0, 0),
            analysis_result=sample_issue_analysis,
            icm_created=True,
            icm_id="ICM-001",
        )
        
        data = analyzed.to_dict()
        restored = AnalyzedPost.from_dict(data)
        
        assert restored.post_id == analyzed.post_id
        assert restored.icm_id == analyzed.icm_id


class TestEnums:
    """Tests for enum types."""

    def test_severity_values(self):
        """Test severity enum values."""
        assert Severity.LOW.value == "low"
        assert Severity.CRITICAL.value == "critical"

    def test_issue_category_values(self):
        """Test issue category enum values."""
        assert IssueCategory.CONNECTIVITY.value == "connectivity"
        assert IssueCategory.GAME_PASS.value == "game_pass"
