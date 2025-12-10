"""Abstract interface for tracking analyzed Reddit posts."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional

from src.models.issue import IssueAnalysis, AnalyzedPost

class IPostTracker(ABC):
    """Abstract interface for tracking which posts have been analyzed.
    
    This interface defines the contract for post tracking storage.
    Implementations could include:
    - SQLite-based tracker (default)
    - PostgreSQL tracker
    - CosmosDB tracker
    - In-memory tracker (for testing)
    """

    @abstractmethod
    def is_analyzed(self, post_id: str) -> bool:
        """Check if a post has already been analyzed.
        
        Args:
            post_id: The Reddit post ID.
            
        Returns:
            True if the post has been analyzed, False otherwise.
        """
        pass

    @abstractmethod
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
        """Record that a post has been analyzed.
        
        Args:
            post_id: The Reddit post ID.
            subreddit: Subreddit the post was from.
            analysis: The analysis result.
            icm_created: Whether an ICM was created.
            icm_id: ID of the created ICM, if any.
            post_title: Title of the post.
            post_url: URL to the post.
        """
        pass

    @abstractmethod
    def get_analyzed_post(self, post_id: str) -> Optional[AnalyzedPost]:
        """Get the analysis record for a specific post.
        
        Args:
            post_id: The Reddit post ID.
            
        Returns:
            AnalyzedPost if found, None otherwise.
        """
        pass

    @abstractmethod
    def get_analyzed_posts(
        self,
        since: Optional[datetime] = None,
        subreddit: Optional[str] = None,
        limit: int = 1000
    ) -> List[AnalyzedPost]:
        """Retrieve history of analyzed posts.
        
        Args:
            since: Only return posts analyzed after this time.
            subreddit: Filter by subreddit name.
            limit: Maximum number of records to return.
            
        Returns:
            List of AnalyzedPost records.
        """
        pass

    @abstractmethod
    def get_posts_with_issues(
        self,
        since: Optional[datetime] = None,
        subreddit: Optional[str] = None,
        limit: int = 1000
    ) -> List[AnalyzedPost]:
        """Get posts where issues were detected.
        
        Args:
            since: Only return posts analyzed after this time.
            subreddit: Filter by subreddit name.
            limit: Maximum number of records to return.
            
        Returns:
            List of AnalyzedPost records where is_issue=True.
        """
        pass

    @abstractmethod
    def get_statistics(
        self,
        since: Optional[datetime] = None,
        subreddit: Optional[str] = None
    ) -> dict:
        """Get statistics about analyzed posts.
        
        Args:
            since: Only include posts analyzed after this time.
            subreddit: Filter by subreddit name.
            
        Returns:
            Dictionary with statistics like:
            - total_analyzed: Total posts analyzed
            - issues_detected: Posts with issues
            - icms_created: Number of ICMs created
            - by_category: Breakdown by issue category
            - by_severity: Breakdown by severity
        """
        pass

    @abstractmethod
    def cleanup_old_records(self, older_than: datetime) -> int:
        """Remove old analysis records to manage storage.
        
        Args:
            older_than: Delete records older than this timestamp.
            
        Returns:
            Number of records deleted.
        """
        pass
