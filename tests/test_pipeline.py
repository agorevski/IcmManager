"""Tests for the issue detection pipeline."""

import pytest
from datetime import datetime

from src.pipeline.issue_detector import IssueDetectorPipeline, PipelineConfig, PipelineResult
from src.models.issue import IssueAnalysis
from tests.conftest import MockRedditClient, MockLLMAnalyzer, MockICMManager, InMemoryPostTracker

class TestPipelineConfig:
    """Tests for PipelineConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PipelineConfig()
        
        assert config.subreddits == ["xbox"]
        assert config.posts_per_subreddit == 100
        assert config.min_confidence == 0.7
        assert config.min_severity == "medium"

    def test_custom_config(self):
        """Test custom configuration."""
        config = PipelineConfig(
            subreddits=["xbox", "xboxone"],
            min_confidence=0.9,
            min_severity="high",
        )
        
        assert len(config.subreddits) == 2
        assert config.min_confidence == 0.9

class TestIssueDetectorPipeline:
    """Tests for IssueDetectorPipeline."""

    def test_run_with_no_posts(self, mock_icm_manager, in_memory_tracker):
        """Test pipeline with no posts."""
        empty_reddit = MockRedditClient(posts=[])
        empty_analyzer = MockLLMAnalyzer()
        
        pipeline = IssueDetectorPipeline(
            reddit_client=empty_reddit,
            llm_analyzer=empty_analyzer,
            icm_manager=mock_icm_manager,
            post_tracker=in_memory_tracker,
        )
        
        result = pipeline.run()
        
        assert result.posts_fetched == 0
        assert result.posts_analyzed == 0
        assert result.icms_created == 0

    def test_run_detects_issue_and_creates_icm(
        self,
        mock_reddit_client,
        mock_llm_analyzer,
        mock_icm_manager,
        in_memory_tracker,
        sample_issue_analysis,
    ):
        """Test that pipeline detects issues and creates ICMs."""
        pipeline = IssueDetectorPipeline(
            reddit_client=mock_reddit_client,
            llm_analyzer=mock_llm_analyzer,
            icm_manager=mock_icm_manager,
            post_tracker=in_memory_tracker,
        )
        
        result = pipeline.run()
        
        # Should have fetched posts
        assert result.posts_fetched > 0
        
        # Should have analyzed posts
        assert result.posts_analyzed == 2  # sample_post and sample_post_no_issue
        
        # Should have detected one issue (from sample_post)
        assert result.issues_detected == 1
        
        # Should have created one ICM
        assert result.icms_created == 1
        assert len(result.created_icms) == 1

    def test_run_skips_already_analyzed_posts(
        self,
        mock_reddit_client,
        mock_llm_analyzer,
        mock_icm_manager,
        in_memory_tracker,
        sample_issue_analysis,
    ):
        """Test that already analyzed posts are skipped."""
        # Pre-mark one post as analyzed
        in_memory_tracker.mark_analyzed(
            post_id="post_456",
            subreddit="xbox",
            analysis=sample_issue_analysis,
            icm_created=True,
        )
        
        pipeline = IssueDetectorPipeline(
            reddit_client=mock_reddit_client,
            llm_analyzer=mock_llm_analyzer,
            icm_manager=mock_icm_manager,
            post_tracker=in_memory_tracker,
        )
        
        result = pipeline.run()
        
        # Should only analyze the post that wasn't already analyzed
        assert result.posts_analyzed == 1
        
        # No new issues since the issue post was already analyzed
        assert result.issues_detected == 0

    def test_run_skips_duplicates(
        self,
        mock_reddit_client,
        mock_llm_analyzer,
        mock_icm_manager,
        in_memory_tracker,
        sample_issue_analysis,
    ):
        """Test that duplicate issues don't create new ICMs."""
        # Configure analyzer to report duplicate
        mock_llm_analyzer.set_duplicate_result(sample_issue_analysis.summary, True)
        
        pipeline = IssueDetectorPipeline(
            reddit_client=mock_reddit_client,
            llm_analyzer=mock_llm_analyzer,
            icm_manager=mock_icm_manager,
            post_tracker=in_memory_tracker,
        )
        
        result = pipeline.run()
        
        # Issue detected but skipped as duplicate
        assert result.issues_detected == 1
        assert result.duplicates_skipped == 1
        assert result.icms_created == 0

    def test_run_filters_by_confidence(
        self,
        mock_reddit_client,
        mock_icm_manager,
        in_memory_tracker,
    ):
        """Test that low confidence issues are filtered out."""
        low_confidence_analysis = IssueAnalysis(
            is_issue=True,
            confidence=0.5,  # Below default threshold of 0.7
            summary="Low confidence issue",
            category="other",
            severity="high",
        )
        
        analyzer = MockLLMAnalyzer()
        analyzer.set_analysis_for_post("post_456", low_confidence_analysis)
        
        pipeline = IssueDetectorPipeline(
            reddit_client=mock_reddit_client,
            llm_analyzer=analyzer,
            icm_manager=mock_icm_manager,
            post_tracker=in_memory_tracker,
        )
        
        result = pipeline.run()
        
        # Issue detected but not above confidence threshold
        assert result.issues_detected == 0
        assert result.icms_created == 0

    def test_run_filters_by_severity(
        self,
        mock_reddit_client,
        mock_icm_manager,
        in_memory_tracker,
    ):
        """Test that low severity issues are filtered out."""
        low_severity_analysis = IssueAnalysis(
            is_issue=True,
            confidence=0.9,
            summary="Low severity issue",
            category="other",
            severity="low",  # Below default threshold of "medium"
        )
        
        analyzer = MockLLMAnalyzer()
        analyzer.set_analysis_for_post("post_456", low_severity_analysis)
        
        config = PipelineConfig(min_severity="medium")
        
        pipeline = IssueDetectorPipeline(
            reddit_client=mock_reddit_client,
            llm_analyzer=analyzer,
            icm_manager=mock_icm_manager,
            post_tracker=in_memory_tracker,
            config=config,
        )
        
        result = pipeline.run()
        
        # Issue detected but not above severity threshold
        assert result.issues_detected == 0
        assert result.icms_created == 0

    def test_run_for_posts(
        self,
        sample_post,
        mock_llm_analyzer,
        mock_icm_manager,
        in_memory_tracker,
    ):
        """Test running pipeline for specific posts."""
        pipeline = IssueDetectorPipeline(
            reddit_client=MockRedditClient(),  # Empty, not used
            llm_analyzer=mock_llm_analyzer,
            icm_manager=mock_icm_manager,
            post_tracker=in_memory_tracker,
        )
        
        result = pipeline.run_for_posts([sample_post])
        
        assert result.posts_fetched == 1
        assert result.posts_analyzed == 1
        assert result.issues_detected == 1
        assert result.icms_created == 1

    def test_icm_created_with_correct_data(
        self,
        sample_post,
        mock_llm_analyzer,
        mock_icm_manager,
        in_memory_tracker,
        sample_issue_analysis,
    ):
        """Test that created ICM has correct data."""
        pipeline = IssueDetectorPipeline(
            reddit_client=MockRedditClient(),
            llm_analyzer=mock_llm_analyzer,
            icm_manager=mock_icm_manager,
            post_tracker=in_memory_tracker,
        )
        
        result = pipeline.run_for_posts([sample_post])
        
        assert len(result.created_icms) == 1
        icm = result.created_icms[0]
        
        # Check ICM data
        assert "[Reddit]" in icm.title
        assert icm.severity == "high"
        assert icm.category == "connectivity"
        assert sample_post.url in icm.source_url
        assert "reddit" in icm.tags
        assert f"r/{sample_post.subreddit}" in icm.tags

    def test_post_marked_as_analyzed_after_processing(
        self,
        sample_post,
        mock_llm_analyzer,
        mock_icm_manager,
        in_memory_tracker,
    ):
        """Test that posts are marked as analyzed after processing."""
        pipeline = IssueDetectorPipeline(
            reddit_client=MockRedditClient(),
            llm_analyzer=mock_llm_analyzer,
            icm_manager=mock_icm_manager,
            post_tracker=in_memory_tracker,
        )
        
        pipeline.run_for_posts([sample_post])
        
        # Post should be marked as analyzed
        assert in_memory_tracker.is_analyzed(sample_post.id)
        
        # Should have recorded the analysis
        analyzed = in_memory_tracker.get_analyzed_post(sample_post.id)
        assert analyzed is not None
        assert analyzed.icm_created is True

    def test_error_handling(
        self,
        sample_post,
        mock_icm_manager,
        in_memory_tracker,
    ):
        """Test that errors are captured in results."""
        # Create analyzer that raises an error
        class ErrorAnalyzer(MockLLMAnalyzer):
            def analyze_post(self, post):
                raise Exception("LLM API Error")
        
        pipeline = IssueDetectorPipeline(
            reddit_client=MockRedditClient(),
            llm_analyzer=ErrorAnalyzer(),
            icm_manager=mock_icm_manager,
            post_tracker=in_memory_tracker,
        )
        
        result = pipeline.run_for_posts([sample_post])
        
        # Should have captured the error
        assert len(result.errors) > 0
        assert "LLM API Error" in result.errors[0]

    def test_pipeline_result_duration(
        self,
        mock_reddit_client,
        mock_llm_analyzer,
        mock_icm_manager,
        in_memory_tracker,
    ):
        """Test that pipeline result includes duration."""
        pipeline = IssueDetectorPipeline(
            reddit_client=mock_reddit_client,
            llm_analyzer=mock_llm_analyzer,
            icm_manager=mock_icm_manager,
            post_tracker=in_memory_tracker,
        )
        
        result = pipeline.run()
        
        # Should have non-zero duration
        assert result.run_duration_seconds >= 0

class TestPipelineThresholds:
    """Tests for pipeline threshold logic."""

    def test_meets_thresholds_confidence(self):
        """Test confidence threshold checking."""
        config = PipelineConfig(min_confidence=0.8)
        pipeline = IssueDetectorPipeline(
            reddit_client=MockRedditClient(),
            llm_analyzer=MockLLMAnalyzer(),
            icm_manager=MockICMManager(),
            post_tracker=InMemoryPostTracker(),
            config=config,
        )
        
        high_conf = IssueAnalysis(
            is_issue=True, confidence=0.9, summary="Test",
            category="other", severity="high"
        )
        low_conf = IssueAnalysis(
            is_issue=True, confidence=0.7, summary="Test",
            category="other", severity="high"
        )
        
        assert pipeline._meets_thresholds(high_conf) is True
        assert pipeline._meets_thresholds(low_conf) is False

    def test_meets_thresholds_severity(self):
        """Test severity threshold checking."""
        config = PipelineConfig(min_severity="high", min_confidence=0.0)
        pipeline = IssueDetectorPipeline(
            reddit_client=MockRedditClient(),
            llm_analyzer=MockLLMAnalyzer(),
            icm_manager=MockICMManager(),
            post_tracker=InMemoryPostTracker(),
            config=config,
        )
        
        high_sev = IssueAnalysis(
            is_issue=True, confidence=0.9, summary="Test",
            category="other", severity="high"
        )
        critical_sev = IssueAnalysis(
            is_issue=True, confidence=0.9, summary="Test",
            category="other", severity="critical"
        )
        medium_sev = IssueAnalysis(
            is_issue=True, confidence=0.9, summary="Test",
            category="other", severity="medium"
        )
        
        assert pipeline._meets_thresholds(high_sev) is True
        assert pipeline._meets_thresholds(critical_sev) is True
        assert pipeline._meets_thresholds(medium_sev) is False
