"""Main pipeline orchestrator for issue detection."""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from src.interfaces.reddit_client import IRedditClient
from src.interfaces.llm_analyzer import ILLMAnalyzer
from src.interfaces.icm_manager import IICMManager
from src.interfaces.post_tracker import IPostTracker
from src.models.reddit_data import RedditPost
from src.models.issue import ICMIssue, IssueAnalysis

logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for the issue detection pipeline.
    
    Attributes:
        subreddits: List of subreddits to monitor.
        posts_per_subreddit: Maximum posts to fetch per subreddit.
        time_filter: Time filter for fetching posts.
        min_confidence: Minimum confidence threshold for creating ICMs.
        min_severity: Minimum severity level for creating ICMs.
        include_rising: Whether to include rising posts.
        include_hot: Whether to include hot posts.
        skip_low_engagement: Skip posts with low engagement.
        min_score: Minimum post score to analyze.
        min_comments: Minimum number of comments to analyze.
    """
    subreddits: List[str] = field(default_factory=lambda: ["xbox"])
    posts_per_subreddit: int = 100
    time_filter: str = "hour"
    min_confidence: float = 0.7
    min_severity: str = "medium"
    include_rising: bool = True
    include_hot: bool = True
    skip_low_engagement: bool = False
    min_score: int = 0
    min_comments: int = 0

@dataclass 
class PipelineResult:
    """Result of a pipeline run.
    
    Attributes:
        posts_fetched: Total posts fetched from Reddit.
        posts_analyzed: Posts that were analyzed (not previously seen).
        issues_detected: Number of issues detected.
        icms_created: Number of new ICMs created.
        duplicates_skipped: Issues that were duplicates of existing ICMs.
        created_icms: List of ICM issues that were created.
        errors: List of any errors encountered.
        run_duration_seconds: Total runtime in seconds.
    """
    posts_fetched: int = 0
    posts_analyzed: int = 0
    issues_detected: int = 0
    icms_created: int = 0
    duplicates_skipped: int = 0
    created_icms: List[ICMIssue] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    run_duration_seconds: float = 0.0

class IssueDetectorPipeline:
    """Main pipeline for detecting Xbox issues from Reddit posts.
    
    Orchestrates the flow of:
    1. Fetching posts from Reddit
    2. Filtering out previously analyzed posts
    3. Analyzing posts with LLM
    4. Checking for duplicate issues
    5. Creating new ICMs for unique issues
    6. Recording analyzed posts
    
    Attributes:
        reddit_client: Interface for fetching Reddit data.
        llm_analyzer: Interface for LLM-based analysis.
        icm_manager: Interface for ICM operations.
        post_tracker: Interface for tracking analyzed posts.
        config: Pipeline configuration.
    """

    # Severity ordering for filtering
    SEVERITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}

    def __init__(
        self,
        reddit_client: IRedditClient,
        llm_analyzer: ILLMAnalyzer,
        icm_manager: IICMManager,
        post_tracker: IPostTracker,
        config: Optional[PipelineConfig] = None
    ):
        """Initialize the pipeline.
        
        Args:
            reddit_client: Reddit data source.
            llm_analyzer: LLM for issue analysis.
            icm_manager: ICM system interface.
            post_tracker: Post tracking storage.
            config: Optional pipeline configuration.
        """
        self.reddit_client = reddit_client
        self.llm_analyzer = llm_analyzer
        self.icm_manager = icm_manager
        self.post_tracker = post_tracker
        self.config = config or PipelineConfig()

    def run(self) -> PipelineResult:
        """Execute the full pipeline.
        
        Returns:
            PipelineResult with statistics about the run.
        """
        start_time = datetime.utcnow()
        result = PipelineResult()
        
        try:
            # Step 1: Fetch posts from all configured subreddits
            all_posts = self._fetch_posts()
            result.posts_fetched = len(all_posts)
            logger.info(f"Fetched {len(all_posts)} posts from {len(self.config.subreddits)} subreddits")
            
            # Step 2: Filter out already-analyzed posts
            new_posts = self._filter_new_posts(all_posts)
            logger.info(f"Found {len(new_posts)} new posts to analyze")
            
            if not new_posts:
                logger.info("No new posts to analyze")
                result.run_duration_seconds = (datetime.utcnow() - start_time).total_seconds()
                return result
            
            # Step 3: Get existing ICM issues for duplicate detection
            existing_issues = self.icm_manager.get_current_issues()
            logger.info(f"Retrieved {len(existing_issues)} existing ICM issues")
            
            # Step 4: Analyze each post and process issues
            for post in new_posts:
                try:
                    self._process_post(post, existing_issues, result)
                    result.posts_analyzed += 1
                except Exception as e:
                    error_msg = f"Error processing post {post.id}: {e}"
                    logger.error(error_msg)
                    result.errors.append(error_msg)
            
        except Exception as e:
            error_msg = f"Pipeline error: {e}"
            logger.error(error_msg)
            result.errors.append(error_msg)
        
        result.run_duration_seconds = (datetime.utcnow() - start_time).total_seconds()
        logger.info(
            f"Pipeline completed: {result.posts_analyzed} analyzed, "
            f"{result.issues_detected} issues, {result.icms_created} ICMs created "
            f"in {result.run_duration_seconds:.2f}s"
        )
        
        return result

    def run_for_posts(self, posts: List[RedditPost]) -> PipelineResult:
        """Run the pipeline for a specific list of posts.
        
        Useful for testing or processing specific posts.
        
        Args:
            posts: List of Reddit posts to process.
            
        Returns:
            PipelineResult with statistics about the run.
        """
        start_time = datetime.utcnow()
        result = PipelineResult()
        result.posts_fetched = len(posts)
        
        try:
            # Filter out already-analyzed posts
            new_posts = self._filter_new_posts(posts)
            
            if not new_posts:
                result.run_duration_seconds = (datetime.utcnow() - start_time).total_seconds()
                return result
            
            # Get existing issues
            existing_issues = self.icm_manager.get_current_issues()
            
            # Process each post
            for post in new_posts:
                try:
                    self._process_post(post, existing_issues, result)
                    result.posts_analyzed += 1
                except Exception as e:
                    result.errors.append(f"Error processing post {post.id}: {e}")
                    
        except Exception as e:
            result.errors.append(f"Pipeline error: {e}")
        
        result.run_duration_seconds = (datetime.utcnow() - start_time).total_seconds()
        return result

    def _fetch_posts(self) -> List[RedditPost]:
        """Fetch posts from all configured subreddits."""
        all_posts = []
        seen_ids = set()
        
        for subreddit in self.config.subreddits:
            try:
                # Fetch recent posts
                posts = self.reddit_client.get_recent_posts(
                    subreddit=subreddit,
                    limit=self.config.posts_per_subreddit,
                    time_filter=self.config.time_filter,
                )
                
                for post in posts:
                    if post.id not in seen_ids:
                        all_posts.append(post)
                        seen_ids.add(post.id)
                
                # Optionally fetch rising posts
                if self.config.include_rising:
                    rising = self.reddit_client.get_rising_posts(
                        subreddit=subreddit,
                        limit=self.config.posts_per_subreddit // 2,
                    )
                    for post in rising:
                        if post.id not in seen_ids:
                            all_posts.append(post)
                            seen_ids.add(post.id)
                
                # Optionally fetch hot posts
                if self.config.include_hot:
                    hot = self.reddit_client.get_hot_posts(
                        subreddit=subreddit,
                        limit=self.config.posts_per_subreddit // 2,
                    )
                    for post in hot:
                        if post.id not in seen_ids:
                            all_posts.append(post)
                            seen_ids.add(post.id)
                            
            except Exception as e:
                logger.error(f"Error fetching from r/{subreddit}: {e}")
        
        # Apply engagement filters if configured
        if self.config.skip_low_engagement:
            all_posts = [
                p for p in all_posts
                if p.score >= self.config.min_score 
                and p.num_comments >= self.config.min_comments
            ]
        
        return all_posts

    def _filter_new_posts(self, posts: List[RedditPost]) -> List[RedditPost]:
        """Filter out posts that have already been analyzed."""
        new_posts = []
        for post in posts:
            if not self.post_tracker.is_analyzed(post.id):
                new_posts.append(post)
        return new_posts

    def _process_post(
        self,
        post: RedditPost,
        existing_issues: List[ICMIssue],
        result: PipelineResult
    ) -> None:
        """Process a single post through the pipeline."""
        # Analyze the post
        analysis = self.llm_analyzer.analyze_post(post)
        
        icm_created = False
        icm_id = None
        
        if analysis.is_issue and self._meets_thresholds(analysis):
            result.issues_detected += 1
            
            # Check for duplicates
            is_duplicate = self.llm_analyzer.check_duplicate(analysis, existing_issues)
            
            if is_duplicate:
                result.duplicates_skipped += 1
                logger.info(f"Post {post.id}: Duplicate issue detected, skipping ICM creation")
            else:
                # Create new ICM
                icm = self._create_icm(post, analysis)
                icm_created = True
                icm_id = icm.id
                result.icms_created += 1
                result.created_icms.append(icm)
                
                # Add to existing issues for future duplicate checks in this run
                existing_issues.append(icm)
                
                logger.info(f"Post {post.id}: Created ICM {icm.id}")
        else:
            logger.debug(f"Post {post.id}: No issue detected or below thresholds")
        
        # Record the analysis
        self.post_tracker.mark_analyzed(
            post_id=post.id,
            subreddit=post.subreddit,
            analysis=analysis,
            icm_created=icm_created,
            icm_id=icm_id,
            post_title=post.title,
            post_url=post.url,
        )

    def _meets_thresholds(self, analysis: IssueAnalysis) -> bool:
        """Check if an analysis meets the configured thresholds."""
        # Check confidence threshold
        if analysis.confidence < self.config.min_confidence:
            return False
        
        # Check severity threshold
        analysis_severity = self.SEVERITY_ORDER.get(analysis.severity, 0)
        min_severity = self.SEVERITY_ORDER.get(self.config.min_severity, 0)
        
        if analysis_severity < min_severity:
            return False
        
        return True

    def _create_icm(self, post: RedditPost, analysis: IssueAnalysis) -> ICMIssue:
        """Create a new ICM from a post and its analysis."""
        # Build description
        description = self._build_icm_description(post, analysis)
        
        # Build tags
        tags = ["reddit", f"r/{post.subreddit}", analysis.category]
        tags.extend(analysis.keywords[:5])  # Add up to 5 keywords as tags
        
        return self.icm_manager.create_new_icm(
            title=f"[Reddit] {analysis.summary[:100]}",
            description=description,
            severity=analysis.severity,
            source_url=post.url,
            category=analysis.category,
            tags=tags,
        )

    def _build_icm_description(self, post: RedditPost, analysis: IssueAnalysis) -> str:
        """Build a detailed ICM description."""
        lines = [
            f"## Issue Summary",
            f"{analysis.summary}",
            "",
            f"## Source",
            f"- **Subreddit:** r/{post.subreddit}",
            f"- **Post Title:** {post.title}",
            f"- **Post URL:** {post.url}",
            f"- **Post Score:** {post.score}",
            f"- **Comments:** {post.num_comments}",
            "",
            f"## Analysis",
            f"- **Category:** {analysis.category}",
            f"- **Severity:** {analysis.severity}",
            f"- **Confidence:** {analysis.confidence:.1%}",
            f"- **Estimated Affected Users:** {analysis.affected_users_estimate}",
            "",
            f"## Keywords",
            f"{', '.join(analysis.keywords) if analysis.keywords else 'None extracted'}",
            "",
            f"## Post Content",
            f"### Title",
            f"{post.title}",
        ]
        
        if post.body:
            lines.extend([
                "",
                f"### Body",
                f"{post.body[:1000]}",
            ])
        
        if post.comments:
            lines.extend([
                "",
                f"### Sample Comments",
                post.get_comment_thread_text(max_comments=10)[:1500],
            ])
        
        return "\n".join(lines)
