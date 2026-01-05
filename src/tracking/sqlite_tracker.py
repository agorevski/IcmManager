"""SQLite-based implementation of the post tracker interface."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from src.interfaces.post_tracker import IPostTracker
from src.models.issue import IssueAnalysis, AnalyzedPost

class SQLitePostTracker(IPostTracker):
    """SQLite-based implementation for tracking analyzed posts.
    
    Stores analysis records in a local SQLite database file.
    This implementation is suitable for single-process usage
    and can be swapped for a more robust database later.
    
    Attributes:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: str = "data/post_tracker.db"):
        """Initialize the SQLite post tracker.
        
        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory.

        Returns:
            sqlite3.Connection: A connection to the SQLite database with
                Row factory enabled for dict-like row access.
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_database(self) -> None:
        """Initialize the database schema.

        Creates the analyzed_posts table and indexes if they don't exist.
        This method is called automatically during initialization.
        """
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analyzed_posts (
                    post_id TEXT PRIMARY KEY,
                    subreddit TEXT NOT NULL,
                    analyzed_at TEXT NOT NULL,
                    is_issue INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    summary TEXT NOT NULL,
                    category TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    affected_users_estimate INTEGER DEFAULT 1,
                    keywords TEXT,
                    raw_response TEXT,
                    icm_created INTEGER NOT NULL DEFAULT 0,
                    icm_id TEXT,
                    post_title TEXT,
                    post_url TEXT
                )
            """)
            
            # Create indexes for common queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_analyzed_at 
                ON analyzed_posts(analyzed_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_subreddit 
                ON analyzed_posts(subreddit)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_is_issue 
                ON analyzed_posts(is_issue)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_category 
                ON analyzed_posts(category)
            """)
            
            conn.commit()

    def is_analyzed(self, post_id: str) -> bool:
        """Check if a post has already been analyzed.

        Args:
            post_id: The unique identifier of the post to check.

        Returns:
            bool: True if the post has been analyzed, False otherwise.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT 1 FROM analyzed_posts WHERE post_id = ?",
                (post_id,)
            )
            return cursor.fetchone() is not None

    def are_analyzed(self, post_ids: List[str]) -> Dict[str, bool]:
        """Check if multiple posts have already been analyzed (batch operation).

        Args:
            post_ids: List of post identifiers to check.

        Returns:
            Dict[str, bool]: A dictionary mapping each post_id to a boolean
                indicating whether it has been analyzed.
        """
        if not post_ids:
            return {}
        
        with self._get_connection() as conn:
            # Use parameterized IN clause with placeholders
            placeholders = ",".join("?" * len(post_ids))
            cursor = conn.execute(
                f"SELECT post_id FROM analyzed_posts WHERE post_id IN ({placeholders})",
                post_ids
            )
            analyzed_ids = {row[0] for row in cursor.fetchall()}
        
        return {post_id: post_id in analyzed_ids for post_id in post_ids}

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
            post_id: The unique identifier of the post.
            subreddit: The subreddit where the post was found.
            analysis: The analysis result containing issue details.
            icm_created: Whether an ICM ticket was created for this post.
            icm_id: The ICM ticket identifier if one was created.
            post_title: The title of the post.
            post_url: The URL of the post.
        """
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO analyzed_posts (
                    post_id, subreddit, analyzed_at, is_issue, confidence,
                    summary, category, severity, affected_users_estimate,
                    keywords, raw_response, icm_created, icm_id, 
                    post_title, post_url
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                post_id,
                subreddit,
                datetime.now(timezone.utc).isoformat(),
                1 if analysis.is_issue else 0,
                analysis.confidence,
                analysis.summary,
                analysis.category,
                analysis.severity,
                analysis.affected_users_estimate,
                json.dumps(analysis.keywords),
                analysis.raw_response,
                1 if icm_created else 0,
                icm_id,
                post_title,
                post_url,
            ))
            conn.commit()

    def get_analyzed_post(self, post_id: str) -> Optional[AnalyzedPost]:
        """Get the analysis record for a specific post.

        Args:
            post_id: The unique identifier of the post to retrieve.

        Returns:
            Optional[AnalyzedPost]: The analyzed post record if found,
                None otherwise.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM analyzed_posts WHERE post_id = ?",
                (post_id,)
            )
            row = cursor.fetchone()
            
            if row is None:
                return None
            
            return self._row_to_analyzed_post(row)

    def get_analyzed_posts(
        self,
        since: Optional[datetime] = None,
        subreddit: Optional[str] = None,
        limit: int = 1000
    ) -> List[AnalyzedPost]:
        """Retrieve history of analyzed posts.

        Args:
            since: Only return posts analyzed after this datetime.
            subreddit: Filter results to a specific subreddit.
            limit: Maximum number of results to return.

        Returns:
            List[AnalyzedPost]: List of analyzed post records, ordered by
                analyzed_at descending.
        """
        query = "SELECT * FROM analyzed_posts WHERE 1=1"
        params = []
        
        if since:
            query += " AND analyzed_at >= ?"
            params.append(since.isoformat())
        
        if subreddit:
            query += " AND subreddit = ?"
            params.append(subreddit)
        
        query += " ORDER BY analyzed_at DESC LIMIT ?"
        params.append(limit)
        
        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [self._row_to_analyzed_post(row) for row in cursor.fetchall()]

    def get_posts_with_issues(
        self,
        since: Optional[datetime] = None,
        subreddit: Optional[str] = None,
        limit: int = 1000
    ) -> List[AnalyzedPost]:
        """Get posts where issues were detected.

        Args:
            since: Only return posts analyzed after this datetime.
            subreddit: Filter results to a specific subreddit.
            limit: Maximum number of results to return.

        Returns:
            List[AnalyzedPost]: List of analyzed post records where is_issue
                is True, ordered by analyzed_at descending.
        """
        query = "SELECT * FROM analyzed_posts WHERE is_issue = 1"
        params = []
        
        if since:
            query += " AND analyzed_at >= ?"
            params.append(since.isoformat())
        
        if subreddit:
            query += " AND subreddit = ?"
            params.append(subreddit)
        
        query += " ORDER BY analyzed_at DESC LIMIT ?"
        params.append(limit)
        
        with self._get_connection() as conn:
            cursor = conn.execute(query, params)
            return [self._row_to_analyzed_post(row) for row in cursor.fetchall()]

    def _build_where_clause(
        self,
        since: Optional[datetime] = None,
        subreddit: Optional[str] = None,
        is_issue: Optional[bool] = None,
        icm_created: Optional[bool] = None
    ) -> tuple:
        """Build WHERE clause and parameters for queries.

        Args:
            since: Filter for records analyzed after this datetime.
            subreddit: Filter for a specific subreddit.
            is_issue: Filter for posts that are/aren't issues.
            icm_created: Filter for posts with/without ICM tickets.

        Returns:
            tuple: A tuple of (where_clause_string, params_list) where
                where_clause_string is the SQL WHERE clause and params_list
                contains the corresponding parameter values.
        """
        conditions = []
        params = []
        
        if since:
            conditions.append("analyzed_at >= ?")
            params.append(since.isoformat())
        
        if subreddit:
            conditions.append("subreddit = ?")
            params.append(subreddit)
        
        if is_issue is not None:
            conditions.append("is_issue = ?")
            params.append(1 if is_issue else 0)
        
        if icm_created is not None:
            conditions.append("icm_created = ?")
            params.append(1 if icm_created else 0)
        
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)
        else:
            where_clause = ""
        
        return where_clause, params

    def get_statistics(
        self,
        since: Optional[datetime] = None,
        subreddit: Optional[str] = None
    ) -> dict:
        """Get statistics about analyzed posts.

        Args:
            since: Only include posts analyzed after this datetime.
            subreddit: Filter statistics to a specific subreddit.

        Returns:
            dict: A dictionary containing:
                - total_analyzed: Total number of analyzed posts.
                - issues_detected: Number of posts with detected issues.
                - icms_created: Number of ICM tickets created.
                - by_category: Issue counts grouped by category.
                - by_severity: Issue counts grouped by severity.
                - average_confidence: Average confidence score for issues.
        """
        base_where, base_params = self._build_where_clause(since=since, subreddit=subreddit)
        issue_where, issue_params = self._build_where_clause(since=since, subreddit=subreddit, is_issue=True)
        icm_where, icm_params = self._build_where_clause(since=since, subreddit=subreddit, icm_created=True)
        
        with self._get_connection() as conn:
            # Total analyzed
            query = f"SELECT COUNT(*) FROM analyzed_posts {base_where}"
            cursor = conn.execute(query, base_params)
            total_analyzed = cursor.fetchone()[0]
            
            # Issues detected
            query = f"SELECT COUNT(*) FROM analyzed_posts {issue_where}"
            cursor = conn.execute(query, issue_params)
            issues_detected = cursor.fetchone()[0]
            
            # ICMs created
            query = f"SELECT COUNT(*) FROM analyzed_posts {icm_where}"
            cursor = conn.execute(query, icm_params)
            icms_created = cursor.fetchone()[0]
            
            # By category
            query = f"""SELECT category, COUNT(*) as count 
                FROM analyzed_posts {issue_where}
                GROUP BY category"""
            cursor = conn.execute(query, issue_params)
            by_category = {row["category"]: row["count"] for row in cursor.fetchall()}
            
            # By severity
            query = f"""SELECT severity, COUNT(*) as count 
                FROM analyzed_posts {issue_where}
                GROUP BY severity"""
            cursor = conn.execute(query, issue_params)
            by_severity = {row["severity"]: row["count"] for row in cursor.fetchall()}
            
            # Average confidence for detected issues
            query = f"SELECT AVG(confidence) FROM analyzed_posts {issue_where}"
            cursor = conn.execute(query, issue_params)
            avg_confidence = cursor.fetchone()[0] or 0.0
            
            return {
                "total_analyzed": total_analyzed,
                "issues_detected": issues_detected,
                "icms_created": icms_created,
                "by_category": by_category,
                "by_severity": by_severity,
                "average_confidence": round(avg_confidence, 3),
            }

    def cleanup_old_records(self, older_than: datetime) -> int:
        """Remove old analysis records to manage storage.

        Args:
            older_than: Delete records analyzed before this datetime.

        Returns:
            int: The number of records deleted.
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM analyzed_posts WHERE analyzed_at < ?",
                (older_than.isoformat(),)
            )
            deleted = cursor.rowcount
            conn.commit()
            return deleted

    def _row_to_analyzed_post(self, row: sqlite3.Row) -> AnalyzedPost:
        """Convert a database row to an AnalyzedPost object.

        Args:
            row: A sqlite3.Row object from a query result.

        Returns:
            AnalyzedPost: The converted AnalyzedPost object with nested
                IssueAnalysis.
        """
        keywords = json.loads(row["keywords"]) if row["keywords"] else []
        
        analysis = IssueAnalysis(
            is_issue=bool(row["is_issue"]),
            confidence=row["confidence"],
            summary=row["summary"],
            category=row["category"],
            severity=row["severity"],
            affected_users_estimate=row["affected_users_estimate"],
            keywords=keywords,
            raw_response=row["raw_response"],
        )
        
        return AnalyzedPost(
            post_id=row["post_id"],
            subreddit=row["subreddit"],
            analyzed_at=datetime.fromisoformat(row["analyzed_at"]),
            analysis_result=analysis,
            icm_created=bool(row["icm_created"]),
            icm_id=row["icm_id"],
            post_title=row["post_title"],
            post_url=row["post_url"],
        )
