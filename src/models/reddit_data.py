"""Data models for Reddit posts and comments."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class RedditComment:
    """Represents a single Reddit comment.
    
    Attributes:
        id: Unique identifier for the comment.
        post_id: ID of the parent post this comment belongs to.
        body: The text content of the comment.
        author: Username of the comment author.
        created_utc: UTC timestamp when the comment was created.
        score: The upvote/downvote score of the comment.
        parent_id: ID of the parent comment (for nested replies) or None if top-level.
    """
    id: str
    post_id: str
    body: str
    author: str
    created_utc: datetime
    score: int
    parent_id: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert the comment to a dictionary representation.

        Returns:
            dict: A dictionary containing all comment fields with the
                created_utc converted to ISO format string.
        """
        return {
            "id": self.id,
            "post_id": self.post_id,
            "body": self.body,
            "author": self.author,
            "created_utc": self.created_utc.isoformat(),
            "score": self.score,
            "parent_id": self.parent_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RedditComment":
        """Create a RedditComment from a dictionary.

        Args:
            data: A dictionary containing comment fields. Must include keys:
                id, post_id, body, author, created_utc, score. Optional:
                parent_id.

        Returns:
            RedditComment: A new RedditComment instance populated from the
                dictionary data.
        """
        return cls(
            id=data["id"],
            post_id=data["post_id"],
            body=data["body"],
            author=data["author"],
            created_utc=datetime.fromisoformat(data["created_utc"]),
            score=data["score"],
            parent_id=data.get("parent_id"),
        )


@dataclass
class RedditPost:
    """Represents a Reddit post with its associated comments.
    
    Attributes:
        id: Unique identifier for the post.
        subreddit: Name of the subreddit (without r/ prefix).
        title: The title of the post.
        body: The text content of the post (selftext).
        author: Username of the post author.
        created_utc: UTC timestamp when the post was created.
        url: Full URL to the Reddit post.
        score: The upvote/downvote score of the post.
        comments: List of comments on the post.
        num_comments: Total number of comments on the post.
        flair: Optional flair text for the post.
    """
    id: str
    subreddit: str
    title: str
    body: str
    author: str
    created_utc: datetime
    url: str
    score: int
    comments: List[RedditComment] = field(default_factory=list)
    num_comments: int = 0
    flair: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert the post to a dictionary representation.

        Returns:
            dict: A dictionary containing all post fields with the
                created_utc converted to ISO format string and comments
                converted to dictionaries.
        """
        return {
            "id": self.id,
            "subreddit": self.subreddit,
            "title": self.title,
            "body": self.body,
            "author": self.author,
            "created_utc": self.created_utc.isoformat(),
            "url": self.url,
            "score": self.score,
            "comments": [c.to_dict() for c in self.comments],
            "num_comments": self.num_comments,
            "flair": self.flair,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RedditPost":
        """Create a RedditPost from a dictionary.

        Args:
            data: A dictionary containing post fields. Must include keys:
                id, subreddit, title, body, author, created_utc, url, score.
                Optional: comments, num_comments, flair.

        Returns:
            RedditPost: A new RedditPost instance populated from the
                dictionary data, including any nested comments.
        """
        comments = [RedditComment.from_dict(c) for c in data.get("comments", [])]
        return cls(
            id=data["id"],
            subreddit=data["subreddit"],
            title=data["title"],
            body=data["body"],
            author=data["author"],
            created_utc=datetime.fromisoformat(data["created_utc"]),
            url=data["url"],
            score=data["score"],
            comments=comments,
            num_comments=data.get("num_comments", len(comments)),
            flair=data.get("flair"),
        )

    def get_full_text(self) -> str:
        """Get the full text content of the post including title and body.

        Returns:
            str: The title and body separated by a blank line if body exists,
                otherwise just the title.
        """
        if self.body:
            return f"{self.title}\n\n{self.body}"
        return self.title

    def get_comment_thread_text(self, max_comments: int = 50) -> str:
        """Get a text representation of the comment thread.

        Args:
            max_comments: Maximum number of comments to include. Defaults to 50.

        Returns:
            str: A formatted string with each comment showing author, score,
                and body text, separated by blank lines. Returns empty string
                if no comments exist.
        """
        if not self.comments:
            return ""
        
        lines = []
        for comment in self.comments[:max_comments]:
            lines.append(f"[{comment.author}] (score: {comment.score}): {comment.body}")
        
        return "\n\n".join(lines)
