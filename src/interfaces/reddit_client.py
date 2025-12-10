"""Abstract interface for Reddit data source."""

from abc import ABC, abstractmethod
from typing import List, Optional

from src.models.reddit_data import RedditPost


class IRedditClient(ABC):
    """Abstract interface for fetching Reddit posts and comments.
    
    This interface defines the contract for Reddit data sources.
    Implementations could include:
    - PRAW-based client for live Reddit API access
    - Mock client for testing
    - File-based client for replaying saved data
    """

    @abstractmethod
    def get_recent_posts(
        self,
        subreddit: str,
        limit: int = 100,
        time_filter: str = "hour"
    ) -> List[RedditPost]:
        """Fetch recent posts from a subreddit with their comment chains.
        
        Args:
            subreddit: Name of the subreddit (without r/ prefix).
            limit: Maximum number of posts to fetch.
            time_filter: Time filter for posts ('hour', 'day', 'week', 'month', 'year', 'all').
            
        Returns:
            List of RedditPost objects with their associated comments.
        """
        pass

    @abstractmethod
    def get_post_by_id(self, post_id: str) -> Optional[RedditPost]:
        """Fetch a specific post by its ID.
        
        Args:
            post_id: The Reddit post ID.
            
        Returns:
            RedditPost object if found, None otherwise.
        """
        pass

    @abstractmethod
    def get_hot_posts(
        self,
        subreddit: str,
        limit: int = 100
    ) -> List[RedditPost]:
        """Fetch hot/trending posts from a subreddit.
        
        Args:
            subreddit: Name of the subreddit (without r/ prefix).
            limit: Maximum number of posts to fetch.
            
        Returns:
            List of RedditPost objects with their associated comments.
        """
        pass

    @abstractmethod
    def get_rising_posts(
        self,
        subreddit: str,
        limit: int = 100
    ) -> List[RedditPost]:
        """Fetch rising posts from a subreddit.
        
        Args:
            subreddit: Name of the subreddit (without r/ prefix).
            limit: Maximum number of posts to fetch.
            
        Returns:
            List of RedditPost objects with their associated comments.
        """
        pass
