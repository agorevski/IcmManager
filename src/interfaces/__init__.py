"""Abstract interfaces for the ICM Manager components."""

from src.interfaces.reddit_client import IRedditClient
from src.interfaces.llm_analyzer import ILLMAnalyzer
from src.interfaces.icm_manager import IICMManager
from src.interfaces.post_tracker import IPostTracker

__all__ = [
    "IRedditClient",
    "ILLMAnalyzer",
    "IICMManager",
    "IPostTracker",
]
