"""Data models for the prompt evaluation framework."""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any

from src.models.reddit_data import RedditPost
from src.models.issue import IssueAnalysis


@dataclass
class LabeledTestCase:
    """A Reddit post with ground truth labels for evaluation.
    
    Attributes:
        test_case_id: Unique identifier for this test case.
        post: The Reddit post to analyze.
        expected_is_issue: Ground truth - whether this is a real issue.
        expected_category: Expected category if it's an issue (e.g., "connectivity").
        expected_severity: Expected severity if it's an issue (e.g., "high").
        expected_min_confidence: Minimum acceptable confidence score.
        description: Human-readable description of why this case exists.
        edge_case_type: Type of edge case if applicable (e.g., "ambiguous", "borderline").
    """
    test_case_id: str
    post: RedditPost
    expected_is_issue: bool
    expected_category: Optional[str] = None
    expected_severity: Optional[str] = None
    expected_min_confidence: float = 0.7
    description: str = ""
    edge_case_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary containing all test case fields,
                with the post converted to its dictionary form.
        """
        return {
            "test_case_id": self.test_case_id,
            "post": self.post.to_dict(),
            "expected_is_issue": self.expected_is_issue,
            "expected_category": self.expected_category,
            "expected_severity": self.expected_severity,
            "expected_min_confidence": self.expected_min_confidence,
            "description": self.description,
            "edge_case_type": self.edge_case_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LabeledTestCase":
        """Create from dictionary representation.

        Args:
            data: Dictionary containing test case fields. Must include
                'test_case_id', 'post', and 'expected_is_issue'.

        Returns:
            LabeledTestCase: A new instance populated from the dictionary.
        """
        return cls(
            test_case_id=data["test_case_id"],
            post=RedditPost.from_dict(data["post"]),
            expected_is_issue=data["expected_is_issue"],
            expected_category=data.get("expected_category"),
            expected_severity=data.get("expected_severity"),
            expected_min_confidence=data.get("expected_min_confidence", 0.7),
            description=data.get("description", ""),
            edge_case_type=data.get("edge_case_type"),
        )


@dataclass
class PredictionResult:
    """Result of comparing a prediction against ground truth.
    
    Attributes:
        test_case: The labeled test case that was evaluated.
        analysis: The LLM's analysis result.
        is_issue_correct: Whether is_issue prediction matches ground truth.
        category_correct: Whether category matches (None if not applicable).
        severity_correct: Whether severity matches (None if not applicable).
        confidence_acceptable: Whether confidence meets the minimum threshold.
    """
    test_case: LabeledTestCase
    analysis: IssueAnalysis
    is_issue_correct: bool
    category_correct: Optional[bool] = None
    severity_correct: Optional[bool] = None
    confidence_acceptable: bool = True

    @property
    def is_fully_correct(self) -> bool:
        """Check if all applicable predictions are correct.

        Returns:
            bool: True if is_issue is correct, confidence is acceptable,
                and (for issues) category and severity are correct.
        """
        if not self.is_issue_correct:
            return False
        if not self.confidence_acceptable:
            return False
        # Only check category/severity if this is supposed to be an issue
        if self.test_case.expected_is_issue:
            if self.category_correct is False:
                return False
            if self.severity_correct is False:
                return False
        return True

    @property
    def is_true_positive(self) -> bool:
        """Check if this is a true positive (correctly identified issue).

        Returns:
            bool: True if both expected and predicted are issues.
        """
        return self.test_case.expected_is_issue and self.analysis.is_issue

    @property
    def is_true_negative(self) -> bool:
        """Check if this is a true negative (correctly identified non-issue).

        Returns:
            bool: True if both expected and predicted are non-issues.
        """
        return not self.test_case.expected_is_issue and not self.analysis.is_issue

    @property
    def is_false_positive(self) -> bool:
        """Check if this is a false positive (incorrectly flagged as issue).

        Returns:
            bool: True if expected is non-issue but predicted is issue.
        """
        return not self.test_case.expected_is_issue and self.analysis.is_issue

    @property
    def is_false_negative(self) -> bool:
        """Check if this is a false negative (missed a real issue).

        Returns:
            bool: True if expected is issue but predicted is non-issue.
        """
        return self.test_case.expected_is_issue and not self.analysis.is_issue

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary containing prediction details including
                expected vs predicted values, correctness flags, and summary.
        """
        return {
            "test_case_id": self.test_case.test_case_id,
            "expected_is_issue": self.test_case.expected_is_issue,
            "predicted_is_issue": self.analysis.is_issue,
            "is_issue_correct": self.is_issue_correct,
            "expected_category": self.test_case.expected_category,
            "predicted_category": self.analysis.category,
            "category_correct": self.category_correct,
            "expected_severity": self.test_case.expected_severity,
            "predicted_severity": self.analysis.severity,
            "severity_correct": self.severity_correct,
            "confidence": self.analysis.confidence,
            "confidence_acceptable": self.confidence_acceptable,
            "is_fully_correct": self.is_fully_correct,
            "summary": self.analysis.summary,
        }


@dataclass
class FailureCase:
    """Detailed analysis of a prediction failure.
    
    Attributes:
        prediction: The prediction result that failed.
        failure_type: Type of failure (false_positive, false_negative, wrong_category, etc.).
        error_details: Human-readable description of the failure.
        potential_causes: List of potential reasons for the failure.
    """
    prediction: PredictionResult
    failure_type: str
    error_details: str
    potential_causes: List[str] = field(default_factory=list)

    # Failure type constants
    FALSE_POSITIVE = "false_positive"
    FALSE_NEGATIVE = "false_negative"
    WRONG_CATEGORY = "wrong_category"
    WRONG_SEVERITY = "wrong_severity"
    LOW_CONFIDENCE = "low_confidence"

    @classmethod
    def from_prediction(cls, prediction: PredictionResult) -> Optional["FailureCase"]:
        """Create a FailureCase from a prediction if it failed.

        Args:
            prediction: The prediction result to analyze for failures.

        Returns:
            Optional[FailureCase]: A FailureCase with failure type, details,
                and potential causes if the prediction failed; None if correct.
        """
        if prediction.is_fully_correct:
            return None

        # Determine failure type and details
        if prediction.is_false_positive:
            return cls(
                prediction=prediction,
                failure_type=cls.FALSE_POSITIVE,
                error_details=(
                    f"Incorrectly classified as issue. "
                    f"Summary: {prediction.analysis.summary}"
                ),
                potential_causes=[
                    "Prompt may be too sensitive to certain keywords",
                    "Post contains complaint-like language but isn't a technical issue",
                ],
            )
        elif prediction.is_false_negative:
            return cls(
                prediction=prediction,
                failure_type=cls.FALSE_NEGATIVE,
                error_details=(
                    f"Missed a real issue. "
                    f"Expected category: {prediction.test_case.expected_category}, "
                    f"severity: {prediction.test_case.expected_severity}"
                ),
                potential_causes=[
                    "Issue description may be too subtle",
                    "Prompt may need better examples of this issue type",
                ],
            )
        elif not prediction.confidence_acceptable:
            return cls(
                prediction=prediction,
                failure_type=cls.LOW_CONFIDENCE,
                error_details=(
                    f"Confidence {prediction.analysis.confidence:.2f} below threshold "
                    f"{prediction.test_case.expected_min_confidence:.2f}"
                ),
                potential_causes=[
                    "Post may be ambiguous",
                    "Prompt may need clearer decision criteria",
                ],
            )
        elif prediction.category_correct is False:
            return cls(
                prediction=prediction,
                failure_type=cls.WRONG_CATEGORY,
                error_details=(
                    f"Wrong category: predicted '{prediction.analysis.category}', "
                    f"expected '{prediction.test_case.expected_category}'"
                ),
                potential_causes=[
                    "Category definitions may overlap",
                    "Prompt needs better category discrimination",
                ],
            )
        elif prediction.severity_correct is False:
            return cls(
                prediction=prediction,
                failure_type=cls.WRONG_SEVERITY,
                error_details=(
                    f"Wrong severity: predicted '{prediction.analysis.severity}', "
                    f"expected '{prediction.test_case.expected_severity}'"
                ),
                potential_causes=[
                    "Severity criteria may be unclear",
                    "Prompt needs better severity examples",
                ],
            )
        
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary containing failure details including
                test case ID, failure type, error details, and causes.
        """
        return {
            "test_case_id": self.prediction.test_case.test_case_id,
            "failure_type": self.failure_type,
            "error_details": self.error_details,
            "potential_causes": self.potential_causes,
            "prediction": self.prediction.to_dict(),
        }


@dataclass
class EvaluationDataset:
    """A collection of labeled test cases for evaluation.
    
    Attributes:
        version: Dataset version string.
        description: Description of the dataset.
        created_at: When the dataset was created.
        confidence_threshold: Default confidence threshold for evaluation.
        test_cases: List of labeled test cases.
    """
    version: str
    description: str
    created_at: datetime
    confidence_threshold: float
    test_cases: List[LabeledTestCase]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary containing dataset metadata and
                all test cases converted to their dictionary form.
        """
        return {
            "version": self.version,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "confidence_threshold": self.confidence_threshold,
            "test_cases": [tc.to_dict() for tc in self.test_cases],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationDataset":
        """Create from dictionary representation.

        Args:
            data: Dictionary containing dataset fields. Test cases should
                be provided as a list of dictionaries under 'test_cases'.

        Returns:
            EvaluationDataset: A new instance populated from the dictionary.
        """
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now(timezone.utc)

        return cls(
            version=data.get("version", "1.0"),
            description=data.get("description", ""),
            created_at=created_at,
            confidence_threshold=data.get("confidence_threshold", 0.7),
            test_cases=[
                LabeledTestCase.from_dict(tc) for tc in data.get("test_cases", [])
            ],
        )

    @classmethod
    def load(cls, path: Path | str) -> "EvaluationDataset":
        """Load dataset from a JSON file.

        Args:
            path: Path to the JSON file containing the dataset.

        Returns:
            EvaluationDataset: A new instance loaded from the file.
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def save(self, path: Path | str) -> None:
        """Save dataset to a JSON file.

        Args:
            path: Path where the JSON file will be saved.
                Parent directories are created if they don't exist.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def filter_by_edge_case_type(self, edge_case_type: Optional[str]) -> "EvaluationDataset":
        """Filter test cases by edge case type.

        Args:
            edge_case_type: The edge case type to filter by (e.g., "ambiguous").
                Use None to get cases without an edge case type.

        Returns:
            EvaluationDataset: A new dataset containing only matching cases.
        """
        filtered = [
            tc for tc in self.test_cases
            if tc.edge_case_type == edge_case_type
        ]
        return EvaluationDataset(
            version=self.version,
            description=f"{self.description} (filtered: {edge_case_type})",
            created_at=self.created_at,
            confidence_threshold=self.confidence_threshold,
            test_cases=filtered,
        )

    def filter_positive_cases(self) -> "EvaluationDataset":
        """Filter to only positive cases (expected issues).

        Returns:
            EvaluationDataset: A new dataset containing only cases where
                expected_is_issue is True.
        """
        filtered = [tc for tc in self.test_cases if tc.expected_is_issue]
        return EvaluationDataset(
            version=self.version,
            description=f"{self.description} (positive cases only)",
            created_at=self.created_at,
            confidence_threshold=self.confidence_threshold,
            test_cases=filtered,
        )

    def filter_negative_cases(self) -> "EvaluationDataset":
        """Filter to only negative cases (non-issues).

        Returns:
            EvaluationDataset: A new dataset containing only cases where
                expected_is_issue is False.
        """
        filtered = [tc for tc in self.test_cases if not tc.expected_is_issue]
        return EvaluationDataset(
            version=self.version,
            description=f"{self.description} (negative cases only)",
            created_at=self.created_at,
            confidence_threshold=self.confidence_threshold,
            test_cases=filtered,
        )

    @property
    def positive_count(self) -> int:
        """Count of positive (issue) cases.

        Returns:
            int: Number of test cases where expected_is_issue is True.
        """
        return sum(1 for tc in self.test_cases if tc.expected_is_issue)

    @property
    def negative_count(self) -> int:
        """Count of negative (non-issue) cases.

        Returns:
            int: Number of test cases where expected_is_issue is False.
        """
        return sum(1 for tc in self.test_cases if not tc.expected_is_issue)

    @property
    def category_distribution(self) -> Dict[str, int]:
        """Distribution of categories in positive cases.

        Returns:
            Dict[str, int]: Mapping of category names to their counts
                across all positive test cases.
        """
        dist: Dict[str, int] = {}
        for tc in self.test_cases:
            if tc.expected_is_issue and tc.expected_category:
                dist[tc.expected_category] = dist.get(tc.expected_category, 0) + 1
        return dist

    @property
    def severity_distribution(self) -> Dict[str, int]:
        """Distribution of severities in positive cases.

        Returns:
            Dict[str, int]: Mapping of severity levels to their counts
                across all positive test cases.
        """
        dist: Dict[str, int] = {}
        for tc in self.test_cases:
            if tc.expected_is_issue and tc.expected_severity:
                dist[tc.expected_severity] = dist.get(tc.expected_severity, 0) + 1
        return dist
