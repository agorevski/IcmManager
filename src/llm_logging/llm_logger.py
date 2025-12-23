"""LLM input/output logging functionality."""

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

class LLMLogger:
    """Logger for LLM requests and responses.
    
    Logs all LLM interactions to JSON files for debugging,
    auditing, and analysis purposes.
    
    Attributes:
        log_dir: Directory where log files are stored.
        logger: Standard Python logger for console output.
    """

    def __init__(
        self,
        log_dir: str = "logs",
        log_to_console: bool = True,
        console_level: int = logging.INFO
    ):
        """Initialize the LLM logger.
        
        Args:
            log_dir: Directory for storing log files.
            log_to_console: Whether to also log to console.
            console_level: Logging level for console output.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up standard Python logger for console output
        self.logger = logging.getLogger("llm_logger")
        self.logger.setLevel(logging.DEBUG)
        
        if log_to_console and not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(console_level)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Track current log file (one per day)
        self._current_date: Optional[str] = None
        self._current_file_handle = None

    def _get_log_file_path(self) -> Path:
        """Get the path to the current log file (one file per day)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return self.log_dir / f"llm_log_{today}.jsonl"

    def _ensure_log_file(self) -> None:
        """Ensure the log file exists and is open."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._current_date != today:
            if self._current_file_handle:
                self._current_file_handle.close()
            self._current_date = today
            self._current_file_handle = None

    def _write_log_entry(self, entry: Dict[str, Any]) -> None:
        """Write a log entry to the current log file."""
        self._ensure_log_file()
        log_path = self._get_log_file_path()
        
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def generate_request_id(self) -> str:
        """Generate a unique request ID for tracking.
        
        Returns:
            Unique request ID string.
        """
        return str(uuid.uuid4())

    def log_request(
        self,
        request_id: str,
        model: str,
        prompt: str,
        context: Optional[Dict[str, Any]] = None,
        post_id: Optional[str] = None,
        subreddit: Optional[str] = None
    ) -> None:
        """Log the input to the LLM.
        
        Args:
            request_id: Unique identifier for this request.
            model: Name of the LLM model being used.
            prompt: The prompt sent to the LLM.
            context: Additional context data.
            post_id: Reddit post ID being analyzed.
            subreddit: Subreddit name.
        """
        entry = {
            "type": "request",
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": model,
            "prompt": prompt,
            "prompt_length": len(prompt),
            "context": context or {},
            "post_id": post_id,
            "subreddit": subreddit,
        }
        
        self._write_log_entry(entry)
        self.logger.info(
            f"LLM Request [{request_id[:8]}]: model={model}, "
            f"prompt_length={len(prompt)}, post_id={post_id}"
        )

    def log_response(
        self,
        request_id: str,
        response: str,
        tokens_used: Optional[int] = None,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        latency_ms: Optional[float] = None,
        model: Optional[str] = None
    ) -> None:
        """Log the output from the LLM.
        
        Args:
            request_id: Unique identifier matching the request.
            response: The response from the LLM.
            tokens_used: Total tokens used (if available).
            prompt_tokens: Tokens used for the prompt.
            completion_tokens: Tokens used for the completion.
            latency_ms: Time taken for the request in milliseconds.
            model: Model name (if different from request).
        """
        entry = {
            "type": "response",
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "response": response,
            "response_length": len(response),
            "tokens_used": tokens_used,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": latency_ms,
            "model": model,
        }
        
        self._write_log_entry(entry)
        self.logger.info(
            f"LLM Response [{request_id[:8]}]: "
            f"response_length={len(response)}, "
            f"tokens={tokens_used}, latency_ms={latency_ms}"
        )

    def log_error(
        self,
        request_id: str,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log any errors during LLM calls.
        
        Args:
            request_id: Unique identifier matching the request.
            error: The exception that occurred.
            context: Additional context about the error.
        """
        entry = {
            "type": "error",
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {},
        }
        
        self._write_log_entry(entry)
        self.logger.error(
            f"LLM Error [{request_id[:8]}]: {type(error).__name__}: {error}"
        )

    def log_analysis_result(
        self,
        request_id: str,
        post_id: str,
        is_issue: bool,
        confidence: float,
        category: str,
        severity: str,
        summary: str
    ) -> None:
        """Log the parsed analysis result.
        
        Args:
            request_id: Unique identifier matching the request.
            post_id: Reddit post ID.
            is_issue: Whether an issue was detected.
            confidence: Confidence score.
            category: Issue category.
            severity: Issue severity.
            summary: Issue summary.
        """
        entry = {
            "type": "analysis_result",
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "post_id": post_id,
            "is_issue": is_issue,
            "confidence": confidence,
            "category": category,
            "severity": severity,
            "summary": summary,
        }
        
        self._write_log_entry(entry)
        self.logger.info(
            f"Analysis Result [{request_id[:8]}]: "
            f"post_id={post_id}, is_issue={is_issue}, "
            f"confidence={confidence:.2f}, category={category}"
        )

    def get_logs_for_date(self, date: datetime) -> list:
        """Retrieve all log entries for a specific date.
        
        Args:
            date: The date to retrieve logs for.
            
        Returns:
            List of log entry dictionaries.
        """
        date_str = date.strftime("%Y-%m-%d")
        log_path = self.log_dir / f"llm_log_{date_str}.jsonl"
        
        if not log_path.exists():
            return []
        
        entries = []
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        
        return entries

    def get_request_chain(self, request_id: str, date: datetime) -> list:
        """Get all log entries for a specific request ID.
        
        Args:
            request_id: The request ID to look up.
            date: The date the request was made.
            
        Returns:
            List of log entries for this request.
        """
        all_logs = self.get_logs_for_date(date)
        return [
            entry for entry in all_logs 
            if entry.get("request_id") == request_id
        ]

    def close(self) -> None:
        """Close any open file handles."""
        if self._current_file_handle:
            self._current_file_handle.close()
            self._current_file_handle = None

    def __enter__(self) -> "LLMLogger":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit context manager and close resources."""
        self.close()
        return False
