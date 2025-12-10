"""Configuration management for the ICM Manager."""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

@dataclass
class AzureOpenAIConfig:
    """Configuration for Azure OpenAI.
    
    Attributes:
        endpoint: Azure OpenAI endpoint URL.
        api_key: Azure OpenAI API key.
        api_version: API version to use.
        deployment_name: Name of the model deployment.
        max_tokens: Maximum tokens for responses.
        temperature: Temperature for generation.
    """
    endpoint: str = ""
    api_key: str = ""
    api_version: str = "2024-02-01"
    deployment_name: str = "gpt-5.1"
    max_tokens: int = 1000
    temperature: float = 0.1

    @classmethod
    def from_env(cls) -> "AzureOpenAIConfig":
        """Create config from environment variables."""
        return cls(
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.1"),
            max_tokens=int(os.getenv("AZURE_OPENAI_MAX_TOKENS", "1000")),
            temperature=float(os.getenv("AZURE_OPENAI_TEMPERATURE", "0.1")),
        )

    def validate(self) -> List[str]:
        """Validate the configuration.
        
        Returns:
            List of validation error messages, empty if valid.
        """
        errors = []
        if not self.endpoint:
            errors.append("Azure OpenAI endpoint is required")
        if not self.api_key:
            errors.append("Azure OpenAI API key is required")
        if not self.deployment_name:
            errors.append("Azure OpenAI deployment name is required")
        return errors

@dataclass
class StorageConfig:
    """Configuration for storage paths.
    
    Attributes:
        data_dir: Directory for database files.
        logs_dir: Directory for log files.
        db_filename: Name of the SQLite database file.
    """
    data_dir: str = "data"
    logs_dir: str = "logs"
    db_filename: str = "post_tracker.db"

    @classmethod
    def from_env(cls) -> "StorageConfig":
        """Create config from environment variables."""
        return cls(
            data_dir=os.getenv("ICM_DATA_DIR", "data"),
            logs_dir=os.getenv("ICM_LOGS_DIR", "logs"),
            db_filename=os.getenv("ICM_DB_FILENAME", "post_tracker.db"),
        )

    @property
    def db_path(self) -> str:
        """Get the full path to the database file."""
        return str(Path(self.data_dir) / self.db_filename)

    def ensure_directories(self) -> None:
        """Create storage directories if they don't exist."""
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.logs_dir).mkdir(parents=True, exist_ok=True)

@dataclass
class PipelineSettings:
    """Settings for the issue detection pipeline.
    
    Attributes:
        subreddits: List of subreddits to monitor.
        posts_per_subreddit: Maximum posts to fetch per subreddit.
        time_filter: Time filter for fetching posts.
        min_confidence: Minimum confidence threshold for creating ICMs.
        min_severity: Minimum severity level for creating ICMs.
        include_rising: Whether to include rising posts.
        include_hot: Whether to include hot posts.
        skip_low_engagement: Skip posts with low engagement.
        min_score: Minimum post score to analyze.
        min_comments: Minimum number of comments to analyze.
    """
    subreddits: List[str] = field(default_factory=lambda: ["xbox"])
    posts_per_subreddit: int = 100
    time_filter: str = "hour"
    min_confidence: float = 0.7
    min_severity: str = "medium"
    include_rising: bool = True
    include_hot: bool = True
    skip_low_engagement: bool = False
    min_score: int = 0
    min_comments: int = 0

    @classmethod
    def from_env(cls) -> "PipelineSettings":
        """Create settings from environment variables."""
        subreddits_str = os.getenv("ICM_SUBREDDITS", "xbox")
        subreddits = [s.strip() for s in subreddits_str.split(",") if s.strip()]
        
        return cls(
            subreddits=subreddits,
            posts_per_subreddit=int(os.getenv("ICM_POSTS_PER_SUBREDDIT", "100")),
            time_filter=os.getenv("ICM_TIME_FILTER", "hour"),
            min_confidence=float(os.getenv("ICM_MIN_CONFIDENCE", "0.7")),
            min_severity=os.getenv("ICM_MIN_SEVERITY", "medium"),
            include_rising=os.getenv("ICM_INCLUDE_RISING", "true").lower() == "true",
            include_hot=os.getenv("ICM_INCLUDE_HOT", "true").lower() == "true",
            skip_low_engagement=os.getenv("ICM_SKIP_LOW_ENGAGEMENT", "false").lower() == "true",
            min_score=int(os.getenv("ICM_MIN_SCORE", "0")),
            min_comments=int(os.getenv("ICM_MIN_COMMENTS", "0")),
        )

@dataclass
class Config:
    """Main configuration container.
    
    Attributes:
        azure_openai: Azure OpenAI configuration.
        storage: Storage configuration.
        pipeline: Pipeline settings.
        log_to_console: Whether to log to console.
        log_level: Logging level.
    """
    azure_openai: AzureOpenAIConfig = field(default_factory=AzureOpenAIConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    pipeline: PipelineSettings = field(default_factory=PipelineSettings)
    log_to_console: bool = True
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables."""
        return cls(
            azure_openai=AzureOpenAIConfig.from_env(),
            storage=StorageConfig.from_env(),
            pipeline=PipelineSettings.from_env(),
            log_to_console=os.getenv("ICM_LOG_TO_CONSOLE", "true").lower() == "true",
            log_level=os.getenv("ICM_LOG_LEVEL", "INFO"),
        )

    def validate(self) -> List[str]:
        """Validate the entire configuration.
        
        Returns:
            List of validation error messages, empty if valid.
        """
        errors = []
        errors.extend(self.azure_openai.validate())
        
        # Validate pipeline settings
        if not self.pipeline.subreddits:
            errors.append("At least one subreddit must be configured")
        
        if self.pipeline.min_confidence < 0 or self.pipeline.min_confidence > 1:
            errors.append("min_confidence must be between 0 and 1")
        
        valid_severities = ["low", "medium", "high", "critical"]
        if self.pipeline.min_severity not in valid_severities:
            errors.append(f"min_severity must be one of: {valid_severities}")
        
        return errors

    def setup(self) -> None:
        """Perform initial setup (create directories, etc.)."""
        self.storage.ensure_directories()


def load_config() -> Config:
    """Load configuration from environment variables.
    
    Returns:
        Configured Config instance.
    """
    config = Config.from_env()
    config.setup()
    return config


def get_sample_env_file() -> str:
    """Get a sample .env file content for reference.
    
    Returns:
        Sample .env file content as a string.
    """
    return """# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_DEPLOYMENT=gpt-5.1
AZURE_OPENAI_MAX_TOKENS=1000
AZURE_OPENAI_TEMPERATURE=0.1

# Storage Configuration
ICM_DATA_DIR=data
ICM_LOGS_DIR=logs
ICM_DB_FILENAME=post_tracker.db

# Pipeline Settings
ICM_SUBREDDITS=xbox,xboxone,XboxSeriesX
ICM_POSTS_PER_SUBREDDIT=100
ICM_TIME_FILTER=hour
ICM_MIN_CONFIDENCE=0.7
ICM_MIN_SEVERITY=medium
ICM_INCLUDE_RISING=true
ICM_INCLUDE_HOT=true
ICM_SKIP_LOW_ENGAGEMENT=false
ICM_MIN_SCORE=0
ICM_MIN_COMMENTS=0

# Logging
ICM_LOG_TO_CONSOLE=true
ICM_LOG_LEVEL=INFO
"""
