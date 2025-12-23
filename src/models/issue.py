"""Data models for issue analysis and ICM management."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional
from enum import Enum

class Severity(str, Enum):
    """Severity levels for detected issues."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class IssueCategory(str, Enum):
    """Categories for Xbox-related issues."""
    CONNECTIVITY = "connectivity"
    PERFORMANCE = "performance"
    GAME_CRASH = "game_crash"
    ACCOUNT = "account"
    PURCHASE = "purchase"
    UPDATE = "update"
    HARDWARE = "hardware"
    GAME_PASS = "game_pass"
    CLOUD_GAMING = "cloud_gaming"
    SOCIAL = "social"
    OTHER = "other"

@dataclass
class IssueAnalysis:
    """Result of LLM analysis on a Reddit post.
    
    Attributes:
        is_issue: Whether the post indicates a user experiencing an issue.
        confidence: Confidence score from 0.0 to 1.0.
        summary: Brief summary of the issue.
        category: Category of the issue.
        severity: Severity level of the issue.
        affected_users_estimate: Estimated number of users affected based on post engagement.
        keywords: Key terms extracted from the post related to the issue.
        raw_response: The raw LLM response for debugging.
    """
    is_issue: bool
    confidence: float
    summary: str
    category: str
    severity: str
    affected_users_estimate: int = 1
    keywords: List[str] = field(default_factory=list)
    raw_response: Optional[str] = None

    def __post_init__(self):
        """Validate the analysis data."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

    def to_dict(self) -> dict:
        """Convert the analysis to a dictionary representation."""
        return {
            "is_issue": self.is_issue,
            "confidence": self.confidence,
            "summary": self.summary,
            "category": self.category,
            "severity": self.severity,
            "affected_users_estimate": self.affected_users_estimate,
            "keywords": self.keywords,
            "raw_response": self.raw_response,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "IssueAnalysis":
        """Create an IssueAnalysis from a dictionary."""
        return cls(
            is_issue=data["is_issue"],
            confidence=data["confidence"],
            summary=data["summary"],
            category=data["category"],
            severity=data["severity"],
            affected_users_estimate=data.get("affected_users_estimate", 1),
            keywords=data.get("keywords", []),
            raw_response=data.get("raw_response"),
        )

    @classmethod
    def no_issue(cls) -> "IssueAnalysis":
        """Create an analysis indicating no issue was detected."""
        return cls(
            is_issue=False,
            confidence=1.0,
            summary="No issue detected",
            category=IssueCategory.OTHER.value,
            severity=Severity.LOW.value,
            affected_users_estimate=0,
        )

    @classmethod
    def analysis_error(cls, error: Exception) -> "IssueAnalysis":
        """Create an analysis indicating an error occurred during analysis.
        
        This provides a distinct error state that callers can identify,
        rather than silently returning a no_issue result.
        
        Args:
            error: The exception that occurred during analysis.
            
        Returns:
            An IssueAnalysis with is_issue=False, confidence=0.0,
            and an error summary that identifies this as a failure.
        """
        return cls(
            is_issue=False,
            confidence=0.0,
            summary=f"Analysis failed: {type(error).__name__}: {error}",
            category=IssueCategory.OTHER.value,
            severity=Severity.LOW.value,
            affected_users_estimate=0,
            raw_response=None,
        )

@dataclass
class ICMIssue:
    """Represents an ICM (Incident/Case Management) issue.
    
    Attributes:
        id: Unique identifier for the ICM.
        title: Title/summary of the issue.
        description: Detailed description of the issue.
        severity: Severity level.
        source_url: URL to the Reddit post that triggered this ICM.
        category: Category of the issue.
        tags: List of tags for categorization.
        created_at: When the ICM was created.
        status: Current status of the ICM.
        source_post_id: Reddit post ID that triggered this ICM.
    """
    id: str
    title: str
    description: str
    severity: str
    source_url: str
    category: str
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "open"
    source_post_id: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert the ICM issue to a dictionary representation."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity,
            "source_url": self.source_url,
            "category": self.category,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "status": self.status,
            "source_post_id": self.source_post_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ICMIssue":
        """Create an ICMIssue from a dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)
            
        return cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            severity=data["severity"],
            source_url=data["source_url"],
            category=data["category"],
            tags=data.get("tags", []),
            created_at=created_at,
            status=data.get("status", "open"),
            source_post_id=data.get("source_post_id"),
        )

@dataclass
class AnalyzedPost:
    """Record of a post that has been analyzed.
    
    Attributes:
        post_id: Reddit post ID.
        subreddit: Subreddit the post was from.
        analyzed_at: When the analysis was performed.
        analysis_result: The result of the LLM analysis.
        icm_created: Whether an ICM was created for this post.
        icm_id: ID of the created ICM, if any.
        post_title: Title of the post for reference.
        post_url: URL to the post.
    """
    post_id: str
    subreddit: str
    analyzed_at: datetime
    analysis_result: IssueAnalysis
    icm_created: bool
    icm_id: Optional[str] = None
    post_title: Optional[str] = None
    post_url: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert the analyzed post to a dictionary representation."""
        return {
            "post_id": self.post_id,
            "subreddit": self.subreddit,
            "analyzed_at": self.analyzed_at.isoformat(),
            "analysis_result": self.analysis_result.to_dict(),
            "icm_created": self.icm_created,
            "icm_id": self.icm_id,
            "post_title": self.post_title,
            "post_url": self.post_url,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AnalyzedPost":
        """Create an AnalyzedPost from a dictionary."""
        analyzed_at = data.get("analyzed_at")
        if isinstance(analyzed_at, str):
            analyzed_at = datetime.fromisoformat(analyzed_at)
        elif analyzed_at is None:
            analyzed_at = datetime.now(timezone.utc)
            
        return cls(
            post_id=data["post_id"],
            subreddit=data["subreddit"],
            analyzed_at=analyzed_at,
            analysis_result=IssueAnalysis.from_dict(data["analysis_result"]),
            icm_created=data["icm_created"],
            icm_id=data.get("icm_id"),
            post_title=data.get("post_title"),
            post_url=data.get("post_url"),
        )
