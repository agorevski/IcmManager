"""End-to-end tests for post classification using real LLM analyzer.

These tests verify that the Azure OpenAI analyzer correctly classifies
real-world Reddit posts as ICM-worthy or not.

To run these tests, you need Azure OpenAI credentials configured via
environment variables or a .env file in the tests directory:
- AZURE_OPENAI_ENDPOINT
- AZURE_OPENAI_API_KEY
- AZURE_OPENAI_DEPLOYMENT (optional, defaults to gpt-5.1)
"""

import json
import os
import pytest
from pathlib import Path

from dotenv import load_dotenv

from src.models.reddit_data import RedditPost
from src.models.issue import IssueAnalysis
from src.analyzers.azure_openai_analyzer import AzureOpenAIAnalyzer
from src.config import AzureOpenAIConfig

# Path to test assets and .env file
TEST_DIR = Path(__file__).parent
TEST_ASSETS_DIR = TEST_DIR / "test_assets"

# Load .env file from tests directory
load_dotenv(TEST_DIR / ".env")


def has_azure_credentials() -> bool:
    """Check if Azure OpenAI credentials are available.

    Checks for the presence of required environment variables for
    Azure OpenAI API authentication.

    Returns:
        bool: True if both AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY
            environment variables are set, False otherwise.
    """
    return bool(
        os.getenv("AZURE_OPENAI_ENDPOINT") and 
        os.getenv("AZURE_OPENAI_API_KEY")
    )


# Skip all tests in this module if no Azure credentials
pytestmark = pytest.mark.skipif(
    not has_azure_credentials(),
    reason="Azure OpenAI credentials not configured"
)


@pytest.fixture(scope="module")
def azure_analyzer() -> AzureOpenAIAnalyzer:
    """Create Azure OpenAI analyzer from environment config.

    Creates a module-scoped fixture that initializes an AzureOpenAIAnalyzer
    instance using configuration loaded from environment variables.

    Returns:
        AzureOpenAIAnalyzer: Configured analyzer instance ready for post analysis.
    """
    config = AzureOpenAIConfig.from_env()
    return AzureOpenAIAnalyzer(
        azure_endpoint=config.endpoint,
        api_key=config.api_key,
        api_version=config.api_version,
        deployment_name=config.deployment_name,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )


@pytest.fixture
def icm_worthy_post() -> RedditPost:
    """Load the ICM-worthy Xbox Live sign-in outage post.

    Loads a test fixture representing a Reddit post about an Xbox Live
    sign-in outage that should be classified as ICM-worthy.

    Returns:
        RedditPost: A RedditPost instance containing the test post data.
    """
    json_path = TEST_ASSETS_DIR / "icm_worthy_xbox_live_signin.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return RedditPost.from_dict(data)


@pytest.fixture
def non_icm_worthy_post() -> RedditPost:
    """Load the non-ICM-worthy storage expansion question post.

    Loads a test fixture representing a Reddit post asking about storage
    expansion options that should NOT be classified as ICM-worthy.

    Returns:
        RedditPost: A RedditPost instance containing the test post data.
    """
    json_path = TEST_ASSETS_DIR / "non_icm_worthy_storage_question.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return RedditPost.from_dict(data)


class TestE2EClassification:
    """End-to-end tests for post classification."""

    def test_icm_worthy_post_is_classified_as_issue(
        self,
        azure_analyzer: AzureOpenAIAnalyzer,
        icm_worthy_post: RedditPost,
    ):
        """Test that the Xbox Live sign-in outage post is classified as ICM-worthy.

        This post describes a widespread Xbox Live sign-in outage affecting many users.
        It should be classified as:
        - is_issue: True
        - category: 'account' or 'connectivity' (sign-in/Xbox Live related)
        - severity: 'high' or 'critical' (widespread outage)
        - confidence: >= 0.7 (high confidence)

        Args:
            azure_analyzer: The Azure OpenAI analyzer fixture.
            icm_worthy_post: The ICM-worthy test post fixture.
        """
        # Analyze the post
        analysis = azure_analyzer.analyze_post(icm_worthy_post)
        
        # Verify it's classified as an issue
        assert analysis.is_issue is True, (
            f"Expected ICM-worthy post to be classified as an issue. "
            f"Got: is_issue={analysis.is_issue}, summary={analysis.summary}"
        )
        
        # Verify confidence is reasonably high
        assert analysis.confidence >= 0.7, (
            f"Expected confidence >= 0.7 for clear outage post. "
            f"Got: {analysis.confidence}"
        )
        
        # Verify category is appropriate (account or connectivity for sign-in issues)
        valid_categories = ["account", "connectivity"]
        assert analysis.category in valid_categories, (
            f"Expected category to be one of {valid_categories} for sign-in issue. "
            f"Got: {analysis.category}"
        )
        
        # Verify severity is high or critical for widespread outage
        valid_severities = ["high", "critical"]
        assert analysis.severity in valid_severities, (
            f"Expected severity to be one of {valid_severities} for widespread outage. "
            f"Got: {analysis.severity}"
        )
        
        # Verify summary mentions key aspects
        summary_lower = analysis.summary.lower()
        assert any(keyword in summary_lower for keyword in ["sign", "login", "xbox live", "account"]), (
            f"Expected summary to mention sign-in/login/Xbox Live issue. "
            f"Got: {analysis.summary}"
        )

    def test_non_icm_worthy_post_is_not_classified_as_issue(
        self,
        azure_analyzer: AzureOpenAIAnalyzer,
        non_icm_worthy_post: RedditPost,
    ):
        """Test that the storage expansion question is NOT classified as ICM-worthy.

        This post is just a user asking for product recommendations about
        storage expansion options. There is no technical issue, bug, or
        service problem being reported.

        Args:
            azure_analyzer: The Azure OpenAI analyzer fixture.
            non_icm_worthy_post: The non-ICM-worthy test post fixture.
        """
        # Analyze the post
        analysis = azure_analyzer.analyze_post(non_icm_worthy_post)
        
        # Verify it's NOT classified as an issue
        assert analysis.is_issue is False, (
            f"Expected non-ICM-worthy post (product question) to NOT be classified as an issue. "
            f"Got: is_issue={analysis.is_issue}, summary={analysis.summary}"
        )

    def test_icm_worthy_post_has_affected_users_estimate(
        self,
        azure_analyzer: AzureOpenAIAnalyzer,
        icm_worthy_post: RedditPost,
    ):
        """Test that ICM-worthy posts include an affected users estimate.

        The Xbox Live outage post has high engagement (510 upvotes, 820 comments)
        and many users reporting the same issue in comments, so the affected
        users estimate should be significant.

        Args:
            azure_analyzer: The Azure OpenAI analyzer fixture.
            icm_worthy_post: The ICM-worthy test post fixture.
        """
        analysis = azure_analyzer.analyze_post(icm_worthy_post)
        
        # Should be classified as an issue first
        assert analysis.is_issue is True
        
        # Should have a reasonable affected users estimate (> 1)
        assert analysis.affected_users_estimate > 1, (
            f"Expected affected_users_estimate > 1 for widespread outage. "
            f"Got: {analysis.affected_users_estimate}"
        )

    def test_icm_worthy_post_has_relevant_keywords(
        self,
        azure_analyzer: AzureOpenAIAnalyzer,
        icm_worthy_post: RedditPost,
    ):
        """Test that ICM-worthy posts include relevant keywords.

        Verifies that the analysis extracts meaningful keywords from the post
        content that relate to the Xbox Live sign-in issue.

        Args:
            azure_analyzer: The Azure OpenAI analyzer fixture.
            icm_worthy_post: The ICM-worthy test post fixture.
        """
        analysis = azure_analyzer.analyze_post(icm_worthy_post)
        
        # Should be classified as an issue
        assert analysis.is_issue is True
        
        # Should have some keywords
        assert len(analysis.keywords) > 0, (
            f"Expected keywords for ICM-worthy post. Got empty list."
        )
        
        # Keywords should contain at least one relevant term
        keywords_lower = [k.lower() for k in analysis.keywords]
        relevant_terms = ["xbox", "live", "sign", "login", "account", "outage", "down"]
        has_relevant = any(
            any(term in kw for term in relevant_terms) 
            for kw in keywords_lower
        )
        assert has_relevant, (
            f"Expected keywords to contain relevant terms like {relevant_terms}. "
            f"Got: {analysis.keywords}"
        )


class TestE2EAnalysisResponse:
    """Tests for analysis response structure and validity."""

    def test_analysis_returns_valid_issue_analysis(
        self,
        azure_analyzer: AzureOpenAIAnalyzer,
        icm_worthy_post: RedditPost,
    ):
        """Test that analysis returns a valid IssueAnalysis object.

        Verifies that the analyzer returns a properly structured IssueAnalysis
        instance with all required fields populated with valid values.

        Args:
            azure_analyzer: The Azure OpenAI analyzer fixture.
            icm_worthy_post: The ICM-worthy test post fixture.
        """
        analysis = azure_analyzer.analyze_post(icm_worthy_post)
        
        # Should be an IssueAnalysis instance
        assert isinstance(analysis, IssueAnalysis)
        
        # Should have all required fields with valid values
        assert isinstance(analysis.is_issue, bool)
        assert 0.0 <= analysis.confidence <= 1.0
        assert isinstance(analysis.summary, str)
        assert len(analysis.summary) > 0
        assert isinstance(analysis.category, str)
        assert isinstance(analysis.severity, str)
        assert isinstance(analysis.keywords, list)

    def test_analysis_has_raw_response(
        self,
        azure_analyzer: AzureOpenAIAnalyzer,
        icm_worthy_post: RedditPost,
    ):
        """Test that analysis includes the raw LLM response for debugging.

        Verifies that the IssueAnalysis object contains the raw JSON response
        from the LLM, which is useful for debugging and transparency.

        Args:
            azure_analyzer: The Azure OpenAI analyzer fixture.
            icm_worthy_post: The ICM-worthy test post fixture.
        """
        analysis = azure_analyzer.analyze_post(icm_worthy_post)
        
        # Should have raw response
        assert analysis.raw_response is not None
        assert len(analysis.raw_response) > 0
        
        # Raw response should be valid JSON
        try:
            parsed = json.loads(analysis.raw_response)
            assert isinstance(parsed, dict)
        except json.JSONDecodeError:
            pytest.fail(f"raw_response is not valid JSON: {analysis.raw_response}")
