# Xbox Reddit ICM Manager

Automated issue detection and ICM (Incident/Case Management) creation from Xbox-related Reddit posts.

## Overview

This utility monitors Xbox-related subreddits to detect user-reported issues and automatically creates ICMs for Microsoft investigation. It uses an LLM to analyze posts and comment threads to determine if users are experiencing problems.

## Features

- **Reddit Monitoring**: Tracks new and trending posts from Xbox subreddits
- **LLM-Powered Analysis**: Uses Azure OpenAI (GPT-5.1) to detect issues from post content
- **Automatic ICM Creation**: Creates tickets for Microsoft investigation
- **Duplicate Detection**: Avoids creating duplicate ICMs for the same issue
- **Post Tracking**: Remembers analyzed posts to avoid reprocessing
- **Comprehensive Logging**: Logs all LLM inputs/outputs for auditing

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Reddit Client  │────▶│   LLM Analyzer  │────▶│   ICM Manager   │
│   (IRedditClient)│     │  (ILLMAnalyzer) │     │  (IICMManager)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │                       │
         │                       ▼                       │
         │              ┌─────────────────┐              │
         │              │   LLM Logger    │              │
         │              └─────────────────┘              │
         │                                               │
         └──────────────────────┬────────────────────────┘
                                ▼
                       ┌─────────────────┐
                       │  Post Tracker   │
                       │ (IPostTracker)  │
                       └─────────────────┘
```

All components are interface-based, allowing implementations to be swapped.

## Project Structure

```
IcmManager/
├── src/
│   ├── models/           # Data models (RedditPost, IssueAnalysis, etc.)
│   ├── interfaces/       # Abstract interfaces
│   ├── analyzers/        # LLM analyzer implementations
│   ├── tracking/         # Post tracker implementations
│   ├── logging/          # LLM logging
│   ├── pipeline/         # Main pipeline orchestrator
│   └── config.py         # Configuration management
├── tests/                # Unit tests
├── logs/                 # LLM interaction logs
├── data/                 # SQLite database
├── requirements.txt
├── pytest.ini
└── README.md
```

## Installation

```bash
# Clone the repository
git clone https://github.com/agorevski/IcmManager.git
cd IcmManager

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Create a `.env` file with your configuration:

```bash
# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_DEPLOYMENT=gpt-5.1

# Pipeline Settings
ICM_SUBREDDITS=xbox,xboxone,XboxSeriesX
ICM_MIN_CONFIDENCE=0.7
ICM_MIN_SEVERITY=medium
```

See `src/config.py` for all available configuration options.

## Usage

### Basic Usage

```python
from src.config import load_config
from src.pipeline.issue_detector import IssueDetectorPipeline, PipelineConfig
from src.analyzers.azure_openai_analyzer import AzureOpenAIAnalyzer
from src.tracking.sqlite_tracker import SQLitePostTracker
from src.logging.llm_logger import LLMLogger

# Load configuration
config = load_config()

# Initialize components
logger = LLMLogger(log_dir=config.storage.logs_dir)
analyzer = AzureOpenAIAnalyzer(
    azure_endpoint=config.azure_openai.endpoint,
    api_key=config.azure_openai.api_key,
    deployment_name=config.azure_openai.deployment_name,
    logger=logger,
)
tracker = SQLitePostTracker(db_path=config.storage.db_path)

# You need to provide your own implementations:
reddit_client = YourRedditClient()  # Implements IRedditClient
icm_manager = YourICMManager()      # Implements IICMManager

# Create and run pipeline
pipeline = IssueDetectorPipeline(
    reddit_client=reddit_client,
    llm_analyzer=analyzer,
    icm_manager=icm_manager,
    post_tracker=tracker,
)

result = pipeline.run()
print(f"Analyzed {result.posts_analyzed} posts")
print(f"Detected {result.issues_detected} issues")
print(f"Created {result.icms_created} ICMs")
```

### Custom Pipeline Configuration

```python
from src.pipeline.issue_detector import PipelineConfig

config = PipelineConfig(
    subreddits=["xbox", "xboxone", "XboxSeriesX"],
    posts_per_subreddit=50,
    time_filter="hour",
    min_confidence=0.8,
    min_severity="high",
    include_rising=True,
    include_hot=True,
)

pipeline = IssueDetectorPipeline(..., config=config)
```

## Interfaces

### IRedditClient

Implement this interface to provide Reddit data:

```python
from src.interfaces.reddit_client import IRedditClient

class MyRedditClient(IRedditClient):
    def get_recent_posts(self, subreddit, limit, time_filter):
        # Fetch posts from Reddit
        pass
    
    def get_post_by_id(self, post_id):
        pass
    
    def get_hot_posts(self, subreddit, limit):
        pass
    
    def get_rising_posts(self, subreddit, limit):
        pass
```

### IICMManager

Implement this interface to connect to your ICM system:

```python
from src.interfaces.icm_manager import IICMManager

class MyICMManager(IICMManager):
    def get_current_issues(self):
        # Return list of open ICMs
        pass
    
    def create_new_icm(self, title, description, severity, source_url, category, tags):
        # Create a new ICM
        pass
    
    def get_issue_by_id(self, issue_id):
        pass
    
    def update_issue_status(self, issue_id, status):
        pass
    
    def add_comment_to_issue(self, issue_id, comment):
        pass
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_pipeline.py

# Run specific test
pytest tests/test_pipeline.py::TestIssueDetectorPipeline::test_run_detects_issue_and_creates_icm
```

## Issue Categories

The analyzer categorizes issues into:

| Category | Description |
|----------|-------------|
| connectivity | Network, Xbox Live, online gaming issues |
| performance | Lag, frame drops, slow loading |
| game_crash | Games crashing, freezing, not launching |
| account | Sign-in, profile, account-related issues |
| purchase | Store, payment, subscription issues |
| update | System update, game update problems |
| hardware | Controller, console hardware issues |
| game_pass | Game Pass specific issues |
| cloud_gaming | Cloud gaming/streaming issues |
| social | Friends list, party chat, messaging issues |
| other | Issues that don't fit other categories |

## Severity Levels

| Level | Description |
|-------|-------------|
| low | Minor inconvenience, workaround available |
| medium | Significant impact but not blocking |
| high | Major functionality broken |
| critical | Widespread outage or data loss |

## Logs

LLM interactions are logged to `logs/llm_log_YYYY-MM-DD.jsonl` in JSON Lines format:

```json
{"type": "request", "request_id": "...", "model": "gpt-5.1", "prompt": "...", ...}
{"type": "response", "request_id": "...", "response": "...", "tokens_used": 150, ...}
{"type": "analysis_result", "request_id": "...", "is_issue": true, ...}
```

## Database

Analyzed posts are stored in a SQLite database at `data/post_tracker.db`. The schema includes:

- `post_id`: Reddit post ID
- `subreddit`: Source subreddit
- `analyzed_at`: Timestamp of analysis
- `is_issue`: Whether an issue was detected
- `confidence`, `severity`, `category`: Analysis results
- `icm_created`, `icm_id`: ICM creation status

## License

Internal Microsoft use only.
