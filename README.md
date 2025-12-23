# Xbox Reddit ICM Manager

Automated issue detection and ICM (Incident/Case Management) creation from Xbox-related Reddit posts.

## Overview

This utility monitors Xbox-related subreddits to detect user-reported issues and automatically creates ICMs for Microsoft investigation. It uses an LLM to analyze posts and comment threads to determine if users are experiencing problems.

## Features

- **Reddit Monitoring**: Tracks new and trending posts from Xbox subreddits
- **LLM-Powered Analysis**: Uses Azure OpenAI to detect issues from post content
- **Automatic ICM Creation**: Creates tickets for Microsoft investigation
- **Duplicate Detection**: Avoids creating duplicate ICMs for the same issue
- **Post Tracking**: Remembers analyzed posts to avoid reprocessing
- **Comprehensive Logging**: Logs all LLM inputs/outputs for auditing

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Reddit Client  │────>│   LLM Analyzer  │────>│   ICM Manager   │
│  (IRedditClient)│     │  (ILLMAnalyzer) │     │  (IICMManager)  │
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
├── .github/
│   └── workflows/        # GitHub Actions CI/CD
│       └── tests.yml     # Test and lint workflow
├── src/
│   ├── models/           # Data models (RedditPost, IssueAnalysis, etc.)
│   ├── interfaces/       # Abstract interfaces
│   ├── analyzers/        # LLM analyzer implementations
│   │   ├── azure_openai_analyzer.py
│   │   └── prompts.yaml  # LLM prompts (editable without code changes)
│   ├── evaluation/       # Prompt evaluation framework
│   │   ├── evaluator.py  # Core evaluation with Precision/Recall/F1
│   │   ├── prompt_manager.py  # Prompt version management
│   │   ├── runner.py     # CLI interface
│   │   └── README.md     # Detailed documentation
│   ├── tracking/         # Post tracker implementations
│   ├── logging/          # LLM logging
│   ├── pipeline/         # Main pipeline orchestrator
│   └── config.py         # Configuration management
├── tests/
│   ├── test_assets/      # Test fixture data (JSON)
│   ├── evaluation_datasets/  # Labeled datasets for prompt evaluation
│   ├── conftest.py       # Pytest fixtures and mocks
│   ├── test_models.py    # Data model tests
│   ├── test_tracker.py   # Post tracker tests
│   ├── test_pipeline.py  # Pipeline orchestrator tests
│   ├── test_e2e_classification.py  # E2E tests with real Azure OpenAI
│   ├── test_evaluation.py  # Evaluation framework tests
│   └── .env.example      # Example test environment config
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
```

See `src/config.py` for all available configuration options and their defaults.

## Usage

### Basic Usage

```python
from src.config import load_config
from src.pipeline.issue_detector import IssueDetectorPipeline, PipelineConfig
from src.analyzers.azure_openai_analyzer import AzureOpenAIAnalyzer
from src.tracking.sqlite_tracker import SQLitePostTracker
from src.llm_logging.llm_logger import LLMLogger

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

### Running Unit Tests

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

### Running End-to-End Tests

E2E tests use real Azure OpenAI to verify classification accuracy. They require valid credentials:

```bash
# Copy the example env file and fill in your credentials
cp tests/.env.example tests/.env
# Edit tests/.env with your Azure OpenAI credentials

# Run E2E tests
pytest tests/test_e2e_classification.py -v
```

E2E tests will be automatically skipped if Azure credentials are not configured.

### Test Assets

Sample Reddit post data for testing is stored in `tests/test_assets/`:
- `icm_worthy_xbox_live_signin.json` - An outage post that should trigger ICM creation
- `non_icm_worthy_storage_question.json` - A question post that should not trigger ICM creation

## CI/CD

The project uses GitHub Actions for continuous integration. The workflow (`.github/workflows/tests.yml`) runs:

- **Tests**: Runs the full pytest suite across Python 3.10, 3.11, 3.12, and 3.13
- **Coverage**: Generates and uploads test coverage reports to Codecov
- **Linting**: Checks code quality with ruff (non-blocking)

The workflow runs on push/PR to `main`, `master`, and `develop` branches.

## Prompt Evaluation Framework

The `src/evaluation/` module provides a comprehensive framework for testing and fine-tuning LLM prompts used in issue detection.

### Key Capabilities

- **Metrics-Based Evaluation**: Measure prompt quality using Precision, Recall, and F1 scores for issue detection, category classification, and severity classification
- **Prompt Version Management**: Save, load, and compare multiple prompt versions for A/B testing
- **Automated Reports**: Generate detailed Markdown or JSON reports with confusion matrices and failure analysis
- **Optimization Suggestions**: Analyze failures and get automated suggestions for prompt improvements

### Quick Start

```bash
# Evaluate the current prompt against a golden dataset
python -m src.evaluation.runner evaluate \
  --version current \
  --dataset tests/evaluation_datasets/golden_set_v1.json

# Compare multiple prompt versions
python -m src.evaluation.runner compare \
  --versions current,v1_baseline,v2_improved \
  --dataset tests/evaluation_datasets/golden_set_v1.json

# List available prompt versions
python -m src.evaluation.runner list-versions

# Validate a dataset file
python -m src.evaluation.runner validate \
  --dataset tests/evaluation_datasets/golden_set_v1.json
```

### Evaluation Datasets

Labeled test datasets are stored in `tests/evaluation_datasets/`. Each dataset contains test cases with:
- Reddit posts (title, body, comments)
- Ground truth labels (`expected_is_issue`, `expected_category`, `expected_severity`)
- Optional edge case annotations

See `src/evaluation/README.md` for complete documentation including dataset format, metrics interpretation, and prompt improvement workflows.

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
