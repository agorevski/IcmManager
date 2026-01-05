"""Tests for the issue detection pipeline."""

import pytest
from datetime import datetime

from src.pipeline.issue_detector import IssueDetectorPipeline, PipelineConfig, PipelineResult
from src.models.issue import IssueAnalysis
from src.testing.mocks import MockRedditClient, MockLLMAnalyzer, MockICMManager, InMemoryPostTracker

class TestPipelineConfig:
    """Tests for PipelineConfig."""

    def test_default_config(self):
        """Test default configuration values.

        Verifies that PipelineConfig initializes with expected defaults
        including subreddits, posts_per_subreddit, min_confidence, and min_severity.
        """
        config = PipelineConfig()
        
        assert config.subreddits == ["xbox"]
        assert config.posts_per_subreddit == 100
        assert config.min_confidence == 0.7
        assert config.min_severity == "medium"

    def test_custom_config(self):
        """Test custom configuration.

        Verifies that PipelineConfig accepts and stores custom values
        for subreddits, min_confidence, and min_severity.
        """
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
        """Test pipeline with no posts.

        Verifies that the pipeline handles empty post lists gracefully,
        returning zero counts for fetched, analyzed, and created items.

        Args:
            mock_icm_manager: Fixture providing a mock ICM manager.
            in_memory_tracker: Fixture providing an in-memory post tracker.
        """
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
        """Test that pipeline detects issues and creates ICMs.

        Verifies the complete pipeline flow from fetching posts to analyzing
        them and creating ICMs for detected issues.

        Args:
            mock_reddit_client: Fixture providing a mock Reddit client with sample posts.
            mock_llm_analyzer: Fixture providing a mock LLM analyzer.
            mock_icm_manager: Fixture providing a mock ICM manager.
            in_memory_tracker: Fixture providing an in-memory post tracker.
            sample_issue_analysis: Fixture providing a sample issue analysis.
        """
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
        """Test that already analyzed posts are skipped.

        Verifies that posts previously marked as analyzed are not
        re-processed by the pipeline, preventing duplicate work.

        Args:
            mock_reddit_client: Fixture providing a mock Reddit client with sample posts.
            mock_llm_analyzer: Fixture providing a mock LLM analyzer.
            mock_icm_manager: Fixture providing a mock ICM manager.
            in_memory_tracker: Fixture providing an in-memory post tracker.
            sample_issue_analysis: Fixture providing a sample issue analysis.
        """
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
        """Test that duplicate issues don't create new ICMs.

        Verifies that when an issue is identified as a duplicate of an
        existing issue, no new ICM is created for it.

        Args:
            mock_reddit_client: Fixture providing a mock Reddit client with sample posts.
            mock_llm_analyzer: Fixture providing a mock LLM analyzer.
            mock_icm_manager: Fixture providing a mock ICM manager.
            in_memory_tracker: Fixture providing an in-memory post tracker.
            sample_issue_analysis: Fixture providing a sample issue analysis.
        """
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
        """Test that low confidence issues are filtered out.

        Verifies that issues with confidence scores below the configured
        minimum threshold are not counted as detected issues.

        Args:
            mock_reddit_client: Fixture providing a mock Reddit client with sample posts.
            mock_icm_manager: Fixture providing a mock ICM manager.
            in_memory_tracker: Fixture providing an in-memory post tracker.
        """
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
        """Test that low severity issues are filtered out.

        Verifies that issues with severity levels below the configured
        minimum threshold are not counted as detected issues.

        Args:
            mock_reddit_client: Fixture providing a mock Reddit client with sample posts.
            mock_icm_manager: Fixture providing a mock ICM manager.
            in_memory_tracker: Fixture providing an in-memory post tracker.
        """
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
        """Test running pipeline for specific posts.

        Verifies that run_for_posts processes a provided list of posts
        rather than fetching from Reddit.

        Args:
            sample_post: Fixture providing a sample Reddit post.
            mock_llm_analyzer: Fixture providing a mock LLM analyzer.
            mock_icm_manager: Fixture providing a mock ICM manager.
            in_memory_tracker: Fixture providing an in-memory post tracker.
        """
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
        """Test that created ICM has correct data.

        Verifies that ICMs are created with the correct title, severity,
        category, source URL, and tags from the analyzed post.

        Args:
            sample_post: Fixture providing a sample Reddit post.
            mock_llm_analyzer: Fixture providing a mock LLM analyzer.
            mock_icm_manager: Fixture providing a mock ICM manager.
            in_memory_tracker: Fixture providing an in-memory post tracker.
            sample_issue_analysis: Fixture providing a sample issue analysis.
        """
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
        """Test that posts are marked as analyzed after processing.

        Verifies that processed posts are tracked as analyzed and their
        analysis results are recorded for future reference.

        Args:
            sample_post: Fixture providing a sample Reddit post.
            mock_llm_analyzer: Fixture providing a mock LLM analyzer.
            mock_icm_manager: Fixture providing a mock ICM manager.
            in_memory_tracker: Fixture providing an in-memory post tracker.
        """
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
        """Test that errors are captured in results.

        Verifies that when the LLM analyzer raises an exception, the error
        is captured in the pipeline results rather than crashing.

        Args:
            sample_post: Fixture providing a sample Reddit post.
            mock_icm_manager: Fixture providing a mock ICM manager.
            in_memory_tracker: Fixture providing an in-memory post tracker.
        """
        # Create analyzer that raises an API error
        class ErrorAnalyzer(MockLLMAnalyzer):
            def analyze_post(self, post):
                raise ValueError("LLM API Error")
        
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
        """Test that pipeline result includes duration.

        Verifies that the pipeline run result includes a non-negative
        run_duration_seconds value.

        Args:
            mock_reddit_client: Fixture providing a mock Reddit client with sample posts.
            mock_llm_analyzer: Fixture providing a mock LLM analyzer.
            mock_icm_manager: Fixture providing a mock ICM manager.
            in_memory_tracker: Fixture providing an in-memory post tracker.
        """
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
        """Test confidence threshold checking.

        Verifies that _meets_thresholds correctly filters analyses based
        on the configured min_confidence value.
        """
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
        """Test severity threshold checking.

        Verifies that _meets_thresholds correctly filters analyses based
        on the configured min_severity value, accepting high and critical
        while rejecting medium and lower.
        """
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
