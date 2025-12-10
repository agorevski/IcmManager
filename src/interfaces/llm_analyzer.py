"""Abstract interface for LLM-based issue analysis."""

from abc import ABC, abstractmethod
from typing import List

from src.models.reddit_data import RedditPost
from src.models.issue import IssueAnalysis, ICMIssue

class ILLMAnalyzer(ABC):
    """Abstract interface for LLM-based analysis of Reddit posts.
    
    This interface defines the contract for LLM analyzers.
    Implementations could include:
    - Azure OpenAI analyzer
    - OpenAI analyzer
    - Local model analyzer (Ollama)
    - Mock analyzer for testing
    """

    @abstractmethod
    def analyze_post(self, post: RedditPost) -> IssueAnalysis:
        """Analyze a Reddit post and its comments to detect issues.
        
        The analyzer should examine:
        - Post title and body
        - Comment thread for user reports
        - Engagement metrics (score, number of comments)
        
        Args:
            post: The Reddit post to analyze.
            
        Returns:
            IssueAnalysis containing the detection result.
        """
        pass

    @abstractmethod
    def analyze_posts_batch(self, posts: List[RedditPost]) -> List[IssueAnalysis]:
        """Analyze multiple posts in batch for efficiency.
        
        Args:
            posts: List of Reddit posts to analyze.
            
        Returns:
            List of IssueAnalysis results, one per post in the same order.
        """
        pass

    @abstractmethod
    def check_duplicate(
        self,
        analysis: IssueAnalysis,
        existing_issues: List[ICMIssue]
    ) -> bool:
        """Check if an analysis represents a duplicate of an existing issue.
        
        Uses semantic similarity to determine if the detected issue
        matches any existing ICM issues.
        
        Args:
            analysis: The new issue analysis to check.
            existing_issues: List of existing ICM issues.
            
        Returns:
            True if the analysis is a duplicate, False otherwise.
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name/identifier of the LLM model being used.
        
        Returns:
            Model name string (e.g., 'gpt-5.1', 'gpt-4-turbo').
        """
        pass
