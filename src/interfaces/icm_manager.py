"""Abstract interface for ICM (Incident/Case Management) operations."""

from abc import ABC, abstractmethod
from typing import List, Optional

from src.models.issue import ICMIssue

class IICMManager(ABC):
    """Abstract interface for ICM operations.
    
    This interface defines the contract for ICM management.
    The actual implementation will be provided by the user
    and will connect to Microsoft's ICM system.
    
    Note: This is a stub interface - the actual implementation
    is assumed to exist and will be injected at runtime.
    """

    @abstractmethod
    def get_current_issues(self) -> List[ICMIssue]:
        """Get list of outstanding issues to avoid duplicates.
        
        Returns:
            List of currently open/active ICM issues.
        """
        pass

    @abstractmethod
    def create_new_icm(
        self,
        title: str,
        description: str,
        severity: str,
        source_url: str,
        category: str,
        tags: List[str]
    ) -> ICMIssue:
        """Create a new ICM for investigation.
        
        Args:
            title: Title/summary of the issue.
            description: Detailed description of the issue.
            severity: Severity level ('low', 'medium', 'high', 'critical').
            source_url: URL to the Reddit post that triggered this ICM.
            category: Category of the issue.
            tags: List of tags for categorization.
            
        Returns:
            The created ICMIssue with its assigned ID.
        """
        pass

    @abstractmethod
    def get_issue_by_id(self, issue_id: str) -> Optional[ICMIssue]:
        """Get a specific ICM issue by its ID.
        
        Args:
            issue_id: The ICM issue ID.
            
        Returns:
            ICMIssue if found, None otherwise.
        """
        pass

    @abstractmethod
    def update_issue_status(self, issue_id: str, status: str) -> bool:
        """Update the status of an ICM issue.
        
        Args:
            issue_id: The ICM issue ID.
            status: New status value.
            
        Returns:
            True if update succeeded, False otherwise.
        """
        pass

    @abstractmethod
    def add_comment_to_issue(self, issue_id: str, comment: str) -> bool:
        """Add a comment to an existing ICM issue.
        
        Args:
            issue_id: The ICM issue ID.
            comment: Comment text to add.
            
        Returns:
            True if comment was added, False otherwise.
        """
        pass
