"""SQLite-based implementation of the post tracker interface."""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

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
        """Get a database connection with row factory."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_database(self) -> None:
        """Initialize the database schema."""
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
        """Check if a post has already been analyzed."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT 1 FROM analyzed_posts WHERE post_id = ?",
                (post_id,)
            )
            return cursor.fetchone() is not None

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
        """Record that a post has been analyzed."""
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
        """Get the analysis record for a specific post."""
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
        """Retrieve history of analyzed posts."""
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
        """Get posts where issues were detected."""
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

    def get_statistics(
        self,
        since: Optional[datetime] = None,
        subreddit: Optional[str] = None
    ) -> dict:
        """Get statistics about analyzed posts."""
        base_where = "WHERE 1=1"
        params = []
        
        if since:
            base_where += " AND analyzed_at >= ?"
            params.append(since.isoformat())
        
        if subreddit:
            base_where += " AND subreddit = ?"
            params.append(subreddit)
        
        with self._get_connection() as conn:
            # Total analyzed
            cursor = conn.execute(
                f"SELECT COUNT(*) FROM analyzed_posts {base_where}",
                params
            )
            total_analyzed = cursor.fetchone()[0]
            
            # Issues detected
            cursor = conn.execute(
                f"SELECT COUNT(*) FROM analyzed_posts {base_where} AND is_issue = 1",
                params
            )
            issues_detected = cursor.fetchone()[0]
            
            # ICMs created
            cursor = conn.execute(
                f"SELECT COUNT(*) FROM analyzed_posts {base_where} AND icm_created = 1",
                params
            )
            icms_created = cursor.fetchone()[0]
            
            # By category
            cursor = conn.execute(
                f"""SELECT category, COUNT(*) as count 
                    FROM analyzed_posts {base_where} AND is_issue = 1
                    GROUP BY category""",
                params
            )
            by_category = {row["category"]: row["count"] for row in cursor.fetchall()}
            
            # By severity
            cursor = conn.execute(
                f"""SELECT severity, COUNT(*) as count 
                    FROM analyzed_posts {base_where} AND is_issue = 1
                    GROUP BY severity""",
                params
            )
            by_severity = {row["severity"]: row["count"] for row in cursor.fetchall()}
            
            # Average confidence for detected issues
            cursor = conn.execute(
                f"""SELECT AVG(confidence) FROM analyzed_posts 
                    {base_where} AND is_issue = 1""",
                params
            )
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
        """Remove old analysis records to manage storage."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM analyzed_posts WHERE analyzed_at < ?",
                (older_than.isoformat(),)
            )
            deleted = cursor.rowcount
            conn.commit()
            return deleted

    def _row_to_analyzed_post(self, row: sqlite3.Row) -> AnalyzedPost:
        """Convert a database row to an AnalyzedPost object."""
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
