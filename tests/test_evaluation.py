"""Unit tests for the evaluation framework."""

import json
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.models.reddit_data import RedditPost, RedditComment
from src.models.issue import IssueAnalysis
from src.evaluation.models import (
    LabeledTestCase,
    PredictionResult,
    FailureCase,
    EvaluationDataset,
)
from src.evaluation.metrics import (
    ClassificationMetrics,
    CategoryMetrics,
    EvaluationResult,
    ComparisonResult,
)
from src.evaluation.evaluator import PromptEvaluator

# Test fixtures

@pytest.fixture
def sample_reddit_post() -> RedditPost:
    """Create a sample Reddit post for testing."""
    return RedditPost(
        id="test_post_001",
        subreddit="xbox",
        title="Can't sign in to Xbox Live",
        body="Having issues signing in since this morning.",
        author="TestUser",
        created_utc=datetime(2024, 1, 15, 10, 0, 0),
        url="https://reddit.com/r/xbox/test",
        score=100,
        num_comments=50,
        comments=[
            RedditComment(
                id="c1",
                post_id="test_post_001",
                body="Same issue here!",
                author="Commenter1",
                created_utc=datetime(2024, 1, 15, 10, 5, 0),
                score=10,
            )
        ],
    )

@pytest.fixture
def positive_test_case(sample_reddit_post) -> LabeledTestCase:
    """Create a positive (issue) test case."""
    return LabeledTestCase(
        test_case_id="tc_001",
        post=sample_reddit_post,
        expected_is_issue=True,
        expected_category="connectivity",
        expected_severity="high",
        expected_min_confidence=0.7,
        description="Xbox Live sign-in issue",
    )

@pytest.fixture
def negative_test_case() -> LabeledTestCase:
    """Create a negative (non-issue) test case."""
    post = RedditPost(
        id="test_post_002",
        subreddit="xbox",
        title="What games are you playing?",
        body="Just curious about everyone's weekend plans.",
        author="CasualGamer",
        created_utc=datetime(2024, 1, 15, 12, 0, 0),
        url="https://reddit.com/r/xbox/test2",
        score=50,
        num_comments=100,
    )
    return LabeledTestCase(
        test_case_id="tc_002",
        post=post,
        expected_is_issue=False,
        expected_category=None,
        expected_severity=None,
        description="General discussion post",
    )

@pytest.fixture
def correct_analysis() -> IssueAnalysis:
    """Create a correct issue analysis."""
    return IssueAnalysis(
        is_issue=True,
        confidence=0.9,
        summary="Xbox Live sign-in issue affecting users",
        category="connectivity",
        severity="high",
        keywords=["xbox live", "sign in"],
    )

@pytest.fixture
def incorrect_analysis() -> IssueAnalysis:
    """Create an incorrect issue analysis (false negative)."""
    return IssueAnalysis(
        is_issue=False,
        confidence=0.8,
        summary="No issue detected",
        category="other",
        severity="low",
    )


class TestLabeledTestCase:
    """Tests for LabeledTestCase model."""

    def test_to_dict(self, positive_test_case):
        """Test serialization to dictionary."""
        data = positive_test_case.to_dict()
        
        assert data["test_case_id"] == "tc_001"
        assert data["expected_is_issue"] is True
        assert data["expected_category"] == "connectivity"
        assert data["expected_severity"] == "high"
        assert "post" in data

    def test_from_dict(self, positive_test_case):
        """Test deserialization from dictionary."""
        data = positive_test_case.to_dict()
        restored = LabeledTestCase.from_dict(data)
        
        assert restored.test_case_id == positive_test_case.test_case_id
        assert restored.expected_is_issue == positive_test_case.expected_is_issue
        assert restored.expected_category == positive_test_case.expected_category
        assert restored.post.id == positive_test_case.post.id


class TestPredictionResult:
    """Tests for PredictionResult model."""

    def test_true_positive(self, positive_test_case, correct_analysis):
        """Test true positive detection."""
        pred = PredictionResult(
            test_case=positive_test_case,
            analysis=correct_analysis,
            is_issue_correct=True,
            category_correct=True,
            severity_correct=True,
            confidence_acceptable=True,
        )
        
        assert pred.is_true_positive is True
        assert pred.is_false_positive is False
        assert pred.is_true_negative is False
        assert pred.is_false_negative is False
        assert pred.is_fully_correct is True

    def test_false_negative(self, positive_test_case, incorrect_analysis):
        """Test false negative detection."""
        pred = PredictionResult(
            test_case=positive_test_case,
            analysis=incorrect_analysis,
            is_issue_correct=False,
            confidence_acceptable=True,
        )
        
        assert pred.is_true_positive is False
        assert pred.is_false_positive is False
        assert pred.is_true_negative is False
        assert pred.is_false_negative is True
        assert pred.is_fully_correct is False

    def test_false_positive(self, negative_test_case, correct_analysis):
        """Test false positive detection."""
        pred = PredictionResult(
            test_case=negative_test_case,
            analysis=correct_analysis,  # Says it's an issue when it's not
            is_issue_correct=False,
            confidence_acceptable=True,
        )
        
        assert pred.is_true_positive is False
        assert pred.is_false_positive is True
        assert pred.is_true_negative is False
        assert pred.is_false_negative is False

    def test_wrong_category_not_fully_correct(self, positive_test_case, correct_analysis):
        """Test that wrong category means not fully correct."""
        pred = PredictionResult(
            test_case=positive_test_case,
            analysis=correct_analysis,
            is_issue_correct=True,
            category_correct=False,  # Wrong category
            severity_correct=True,
            confidence_acceptable=True,
        )
        
        assert pred.is_fully_correct is False


class TestFailureCase:
    """Tests for FailureCase model."""

    def test_from_correct_prediction_returns_none(self, positive_test_case, correct_analysis):
        """Test that correct predictions don't create failure cases."""
        pred = PredictionResult(
            test_case=positive_test_case,
            analysis=correct_analysis,
            is_issue_correct=True,
            category_correct=True,
            severity_correct=True,
            confidence_acceptable=True,
        )
        
        failure = FailureCase.from_prediction(pred)
        assert failure is None

    def test_false_positive_failure(self, negative_test_case, correct_analysis):
        """Test false positive failure case creation."""
        pred = PredictionResult(
            test_case=negative_test_case,
            analysis=correct_analysis,
            is_issue_correct=False,
            confidence_acceptable=True,
        )
        
        failure = FailureCase.from_prediction(pred)
        assert failure is not None
        assert failure.failure_type == FailureCase.FALSE_POSITIVE

    def test_false_negative_failure(self, positive_test_case, incorrect_analysis):
        """Test false negative failure case creation."""
        pred = PredictionResult(
            test_case=positive_test_case,
            analysis=incorrect_analysis,
            is_issue_correct=False,
            confidence_acceptable=True,
        )
        
        failure = FailureCase.from_prediction(pred)
        assert failure is not None
        assert failure.failure_type == FailureCase.FALSE_NEGATIVE


class TestClassificationMetrics:
    """Tests for ClassificationMetrics calculations."""

    def test_perfect_classification(self):
        """Test metrics for perfect classification."""
        metrics = ClassificationMetrics(tp=10, tn=10, fp=0, fn=0)
        
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_score == 1.0
        assert metrics.accuracy == 1.0

    def test_precision_calculation(self):
        """Test precision calculation."""
        metrics = ClassificationMetrics(tp=8, tn=5, fp=2, fn=5)
        
        # Precision = TP / (TP + FP) = 8 / 10 = 0.8
        assert metrics.precision == 0.8

    def test_recall_calculation(self):
        """Test recall calculation."""
        metrics = ClassificationMetrics(tp=8, tn=5, fp=2, fn=2)
        
        # Recall = TP / (TP + FN) = 8 / 10 = 0.8
        assert metrics.recall == 0.8

    def test_f1_calculation(self):
        """Test F1 score calculation."""
        metrics = ClassificationMetrics(tp=8, tn=5, fp=2, fn=2)
        
        # F1 = 2 * P * R / (P + R)
        # P = 8/10 = 0.8, R = 8/10 = 0.8
        # F1 = 2 * 0.8 * 0.8 / 1.6 = 0.8
        assert abs(metrics.f1_score - 0.8) < 1e-10

    def test_zero_division_handling(self):
        """Test that zero division is handled."""
        metrics = ClassificationMetrics(tp=0, tn=10, fp=0, fn=0)
        
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1_score == 0.0

    def test_from_predictions(self, positive_test_case, negative_test_case, correct_analysis, incorrect_analysis):
        """Test creating metrics from predictions."""
        predictions = [
            PredictionResult(
                test_case=positive_test_case,
                analysis=correct_analysis,
                is_issue_correct=True,
            ),  # True positive
            PredictionResult(
                test_case=negative_test_case,
                analysis=IssueAnalysis.no_issue(),
                is_issue_correct=True,
            ),  # True negative
        ]
        
        metrics = ClassificationMetrics.from_predictions(predictions)
        
        assert metrics.tp == 1
        assert metrics.tn == 1
        assert metrics.fp == 0
        assert metrics.fn == 0


class TestCategoryMetrics:
    """Tests for CategoryMetrics calculations."""

    def test_add_prediction(self):
        """Test adding predictions to confusion matrix."""
        metrics = CategoryMetrics()
        metrics.add_prediction("connectivity", "connectivity")
        metrics.add_prediction("connectivity", "account")
        metrics.add_prediction("account", "account")
        
        assert metrics.confusion_matrix["connectivity"]["connectivity"] == 1
        assert metrics.confusion_matrix["connectivity"]["account"] == 1
        assert metrics.confusion_matrix["account"]["account"] == 1
        assert "connectivity" in metrics.labels
        assert "account" in metrics.labels

    def test_perfect_accuracy(self):
        """Test accuracy with perfect predictions."""
        metrics = CategoryMetrics()
        for _ in range(5):
            metrics.add_prediction("connectivity", "connectivity")
        for _ in range(5):
            metrics.add_prediction("account", "account")
        
        assert metrics.accuracy == 1.0

    def test_macro_f1(self):
        """Test macro F1 calculation."""
        metrics = CategoryMetrics()
        # All correct for category A
        for _ in range(10):
            metrics.add_prediction("A", "A")
        # All correct for category B
        for _ in range(10):
            metrics.add_prediction("B", "B")
        
        assert metrics.macro_f1 == 1.0


class TestEvaluationDataset:
    """Tests for EvaluationDataset."""

    def test_load_from_file(self, tmp_path):
        """Test loading dataset from JSON file."""
        dataset_data = {
            "version": "1.0",
            "description": "Test dataset",
            "created_at": "2024-01-15T00:00:00",
            "confidence_threshold": 0.7,
            "test_cases": [
                {
                    "test_case_id": "tc_001",
                    "description": "Test case",
                    "expected_is_issue": True,
                    "expected_category": "connectivity",
                    "expected_severity": "high",
                    "post": {
                        "id": "post_001",
                        "subreddit": "xbox",
                        "title": "Test",
                        "body": "Test body",
                        "author": "TestUser",
                        "created_utc": "2024-01-15T10:00:00",
                        "url": "https://reddit.com/test",
                        "score": 10,
                        "num_comments": 5,
                    }
                }
            ]
        }
        
        dataset_path = tmp_path / "test_dataset.json"
        with open(dataset_path, "w") as f:
            json.dump(dataset_data, f)
        
        dataset = EvaluationDataset.load(dataset_path)
        
        assert dataset.version == "1.0"
        assert dataset.description == "Test dataset"
        assert len(dataset.test_cases) == 1
        assert dataset.test_cases[0].test_case_id == "tc_001"

    def test_positive_negative_counts(self, positive_test_case, negative_test_case):
        """Test counting positive and negative cases."""
        dataset = EvaluationDataset(
            version="1.0",
            description="Test",
            created_at=datetime.utcnow(),
            confidence_threshold=0.7,
            test_cases=[positive_test_case, negative_test_case],
        )
        
        assert dataset.positive_count == 1
        assert dataset.negative_count == 1

    def test_category_distribution(self, positive_test_case, negative_test_case):
        """Test category distribution calculation."""
        dataset = EvaluationDataset(
            version="1.0",
            description="Test",
            created_at=datetime.utcnow(),
            confidence_threshold=0.7,
            test_cases=[positive_test_case, negative_test_case],
        )
        
        dist = dataset.category_distribution
        assert dist.get("connectivity") == 1


class TestPromptEvaluator:
    """Tests for PromptEvaluator."""

    def test_evaluate_with_mock_analyzer(self, positive_test_case, correct_analysis):
        """Test evaluation with a mocked analyzer."""
        mock_analyzer = MagicMock()
        mock_analyzer.analyze_post.return_value = correct_analysis
        
        evaluator = PromptEvaluator(confidence_threshold=0.7)
        result = evaluator.evaluate(
            analyzer=mock_analyzer,
            test_cases=[positive_test_case],
            prompt_version="test_v1",
            dataset_name="test_dataset",
        )
        
        assert result.total_cases == 1
        assert result.issue_classification.tp == 1
        assert result.prompt_version == "test_v1"

    def test_category_matching_case_insensitive(self, positive_test_case, correct_analysis):
        """Test that category matching is case-insensitive."""
        # Create analysis with different case
        analysis = IssueAnalysis(
            is_issue=True,
            confidence=0.9,
            summary="Test",
            category="CONNECTIVITY",  # Uppercase
            severity="HIGH",
        )
        
        evaluator = PromptEvaluator()
        pred = evaluator._create_prediction(positive_test_case, analysis)
        
        assert pred.category_correct is True
        assert pred.severity_correct is True


class TestEvaluationResult:
    """Tests for EvaluationResult."""

    def test_combined_f1_calculation(self, positive_test_case, correct_analysis):
        """Test combined F1 calculation."""
        pred = PredictionResult(
            test_case=positive_test_case,
            analysis=correct_analysis,
            is_issue_correct=True,
            category_correct=True,
            severity_correct=True,
            confidence_acceptable=True,
        )
        
        result = EvaluationResult.compute(
            predictions=[pred],
            prompt_version="test",
            dataset_name="test",
        )
        
        # All F1s should be 1.0 for perfect prediction
        assert result.issue_classification.f1_score == 1.0
        assert result.combined_f1 > 0  # Will be average with category/severity

    def test_failure_breakdown(self, positive_test_case, negative_test_case, incorrect_analysis, correct_analysis):
        """Test failure breakdown calculation."""
        predictions = [
            PredictionResult(
                test_case=positive_test_case,
                analysis=incorrect_analysis,  # False negative
                is_issue_correct=False,
                confidence_acceptable=True,
            ),
            PredictionResult(
                test_case=negative_test_case,
                analysis=correct_analysis,  # False positive
                is_issue_correct=False,
                confidence_acceptable=True,
            ),
        ]
        
        result = EvaluationResult.compute(
            predictions=predictions,
            prompt_version="test",
            dataset_name="test",
        )
        
        assert len(result.failures) == 2
        assert FailureCase.FALSE_NEGATIVE in result.failure_breakdown
        assert FailureCase.FALSE_POSITIVE in result.failure_breakdown


class TestComparisonResult:
    """Tests for ComparisonResult."""

    def test_best_version_selection(self):
        """Test that best version is correctly identified."""
        # Create mock results with different F1 scores
        result1 = MagicMock()
        result1.combined_f1 = 0.8
        
        result2 = MagicMock()
        result2.combined_f1 = 0.9
        
        result3 = MagicMock()
        result3.combined_f1 = 0.7
        
        comparison = ComparisonResult(
            results={"v1": result1, "v2": result2, "v3": result3},
            dataset_name="test",
        )
        
        assert comparison.best_version == "v2"

    def test_rankings(self):
        """Test version rankings."""
        result1 = MagicMock()
        result1.combined_f1 = 0.8
        
        result2 = MagicMock()
        result2.combined_f1 = 0.9
        
        comparison = ComparisonResult(
            results={"v1": result1, "v2": result2},
            dataset_name="test",
        )
        
        rankings = comparison.rankings
        assert rankings[0][0] == "v2"  # v2 should be first (higher F1)
        assert rankings[1][0] == "v1"
