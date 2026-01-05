"""Metrics calculations for prompt evaluation."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any

from src.evaluation.models import PredictionResult, FailureCase

@dataclass
class ClassificationMetrics:
    """Binary classification metrics for is_issue detection.
    
    Attributes:
        tp: True positives (correctly identified issues).
        tn: True negatives (correctly identified non-issues).
        fp: False positives (incorrectly flagged as issues).
        fn: False negatives (missed real issues).
    """
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def total(self) -> int:
        """Calculate the total number of samples.

        Returns:
            int: Sum of true positives, true negatives, false positives,
                and false negatives.
        """
        return self.tp + self.tn + self.fp + self.fn

    @property
    def precision(self) -> float:
        """Calculate precision metric.

        Precision measures how many predicted issues are real issues.
        Calculated as TP / (TP + FP).

        Returns:
            float: Precision value between 0.0 and 1.0. Returns 0.0 if
                there are no positive predictions.
        """
        denominator = self.tp + self.fp
        return self.tp / denominator if denominator > 0 else 0.0

    @property
    def recall(self) -> float:
        """Calculate recall metric.

        Recall measures how many real issues were correctly identified.
        Calculated as TP / (TP + FN).

        Returns:
            float: Recall value between 0.0 and 1.0. Returns 0.0 if
                there are no actual positive cases.
        """
        denominator = self.tp + self.fn
        return self.tp / denominator if denominator > 0 else 0.0

    @property
    def f1_score(self) -> float:
        """Calculate F1 score.

        F1 score is the harmonic mean of precision and recall, providing
        a single metric that balances both concerns.

        Returns:
            float: F1 score between 0.0 and 1.0. Returns 0.0 if both
                precision and recall are zero.
        """
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        """Calculate accuracy metric.

        Accuracy measures the proportion of correct predictions.
        Calculated as (TP + TN) / Total.

        Returns:
            float: Accuracy value between 0.0 and 1.0. Returns 0.0 if
                there are no samples.
        """
        return (self.tp + self.tn) / self.total if self.total > 0 else 0.0

    @property
    def specificity(self) -> float:
        """Calculate specificity metric.

        Specificity measures how many non-issues were correctly identified.
        Calculated as TN / (TN + FP).

        Returns:
            float: Specificity value between 0.0 and 1.0. Returns 0.0 if
                there are no actual negative cases.
        """
        denominator = self.tn + self.fp
        return self.tn / denominator if denominator > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary containing all metric values with
                computed metrics rounded to 4 decimal places.
        """
        return {
            "tp": self.tp,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "total": self.total,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "accuracy": round(self.accuracy, 4),
            "specificity": round(self.specificity, 4),
        }

    @classmethod
    def from_predictions(cls, predictions: List[PredictionResult]) -> "ClassificationMetrics":
        """Calculate classification metrics from a list of predictions.

        Args:
            predictions: List of PredictionResult objects containing
                actual and predicted values.

        Returns:
            ClassificationMetrics: A new instance with computed confusion
                matrix values (tp, tn, fp, fn).
        """
        tp = sum(1 for p in predictions if p.is_true_positive)
        tn = sum(1 for p in predictions if p.is_true_negative)
        fp = sum(1 for p in predictions if p.is_false_positive)
        fn = sum(1 for p in predictions if p.is_false_negative)
        return cls(tp=tp, tn=tn, fp=fp, fn=fn)

@dataclass
class CategoryMetrics:
    """Multi-class metrics for category or severity classification.
    
    Attributes:
        confusion_matrix: Nested dict of actual -> predicted -> count.
        labels: List of all category/severity labels.
    """
    confusion_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)
    labels: List[str] = field(default_factory=list)

    def add_prediction(self, actual: str, predicted: str) -> None:
        """Add a prediction to the confusion matrix.

        Updates the confusion matrix with a new actual/predicted pair and
        tracks any new labels encountered.

        Args:
            actual: The actual (ground truth) label.
            predicted: The predicted label from the model.
        """
        if actual not in self.confusion_matrix:
            self.confusion_matrix[actual] = {}
        if predicted not in self.confusion_matrix[actual]:
            self.confusion_matrix[actual][predicted] = 0
        self.confusion_matrix[actual][predicted] += 1

        # Track all labels
        if actual not in self.labels:
            self.labels.append(actual)
        if predicted not in self.labels:
            self.labels.append(predicted)

    def per_class_metrics(self) -> Dict[str, ClassificationMetrics]:
        """Calculate binary classification metrics for each class.

        Computes one-vs-rest classification metrics for each label in the
        confusion matrix.

        Returns:
            Dict[str, ClassificationMetrics]: Dictionary mapping each label
                to its binary classification metrics (tp, tn, fp, fn).
        """
        result = {}
        for label in self.labels:
            tp = self.confusion_matrix.get(label, {}).get(label, 0)
            
            # FP: predicted as this label but was actually something else
            fp = sum(
                self.confusion_matrix.get(actual, {}).get(label, 0)
                for actual in self.labels if actual != label
            )
            
            # FN: was actually this label but predicted as something else
            fn = sum(
                count for pred, count in self.confusion_matrix.get(label, {}).items()
                if pred != label
            )
            
            # TN: not this label and not predicted as this label
            tn = sum(
                count for actual in self.labels if actual != label
                for pred, count in self.confusion_matrix.get(actual, {}).items()
                if pred != label
            )
            
            result[label] = ClassificationMetrics(tp=tp, tn=tn, fp=fp, fn=fn)
        
        return result

    @property
    def macro_f1(self) -> float:
        """Calculate macro-averaged F1 score.

        Computes the unweighted average of F1 scores across all classes,
        treating each class equally regardless of frequency.

        Returns:
            float: Macro F1 score between 0.0 and 1.0. Returns 0.0 if
                there are no classes.
        """
        per_class = self.per_class_metrics()
        if not per_class:
            return 0.0
        f1_scores = [m.f1_score for m in per_class.values()]
        return sum(f1_scores) / len(f1_scores)

    @property
    def weighted_f1(self) -> float:
        """Calculate weighted-average F1 score.

        Computes the F1 score weighted by the frequency of each class,
        giving more importance to classes with more samples.

        Returns:
            float: Weighted F1 score between 0.0 and 1.0. Returns 0.0 if
                there are no classes or no samples.
        """
        per_class = self.per_class_metrics()
        if not per_class:
            return 0.0
        
        # Get total samples per class
        class_counts = {}
        for label in self.labels:
            class_counts[label] = sum(self.confusion_matrix.get(label, {}).values())
        
        total = sum(class_counts.values())
        if total == 0:
            return 0.0
        
        weighted_sum = sum(
            per_class[label].f1_score * class_counts[label]
            for label in self.labels
        )
        return weighted_sum / total

    @property
    def accuracy(self) -> float:
        """Calculate overall accuracy.

        Computes the proportion of correct predictions across all classes.

        Returns:
            float: Accuracy value between 0.0 and 1.0. Returns 0.0 if
                there are no predictions.
        """
        correct = sum(
            self.confusion_matrix.get(label, {}).get(label, 0)
            for label in self.labels
        )
        total = sum(
            count for actual_dict in self.confusion_matrix.values()
            for count in actual_dict.values()
        )
        return correct / total if total > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert category metrics to a dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary containing confusion matrix, labels,
                per-class metrics, and aggregate scores rounded to 4 decimal
                places.
        """
        return {
            "confusion_matrix": self.confusion_matrix,
            "labels": self.labels,
            "per_class_metrics": {
                label: metrics.to_dict()
                for label, metrics in self.per_class_metrics().items()
            },
            "macro_f1": round(self.macro_f1, 4),
            "weighted_f1": round(self.weighted_f1, 4),
            "accuracy": round(self.accuracy, 4),
        }

    @classmethod
    def from_predictions(
        cls,
        predictions: List[PredictionResult],
        metric_type: str = "category"  # or "severity"
    ) -> "CategoryMetrics":
        """Calculate category or severity metrics from predictions.

        Only considers predictions where the test case expected an issue
        (expected_is_issue=True) and the prediction identified it as an
        issue (is_issue=True).

        Args:
            predictions: List of PredictionResult objects to evaluate.
            metric_type: Type of metric to calculate. Either "category"
                or "severity". Defaults to "category".

        Returns:
            CategoryMetrics: A new instance with the confusion matrix
                populated from the filtered predictions.
        """
        metrics = cls()
        
        for pred in predictions:
            # Only evaluate category/severity for true issues that were detected
            if not pred.test_case.expected_is_issue:
                continue
            if not pred.analysis.is_issue:
                continue
            
            if metric_type == "category":
                actual = pred.test_case.expected_category
                predicted = pred.analysis.category
            else:  # severity
                actual = pred.test_case.expected_severity
                predicted = pred.analysis.severity
            
            if actual and predicted:
                metrics.add_prediction(actual, predicted)
        
        return metrics

@dataclass
class EvaluationResult:
    """Complete evaluation results for a prompt version.
    
    Attributes:
        prompt_version: Identifier for the prompt version evaluated.
        dataset_name: Name of the dataset used.
        timestamp: When the evaluation was run.
        total_cases: Total number of test cases evaluated.
        issue_classification: Binary classification metrics for is_issue.
        category_metrics: Multi-class metrics for category.
        severity_metrics: Multi-class metrics for severity.
        avg_confidence_true_positives: Average confidence for true positives.
        avg_confidence_false_positives: Average confidence for false positives.
        avg_confidence_true_negatives: Average confidence for true negatives.
        avg_confidence_false_negatives: Average confidence for false negatives.
        predictions: All prediction results.
        failures: All failure cases.
    """
    prompt_version: str
    dataset_name: str
    timestamp: datetime
    total_cases: int
    
    # Core metrics
    issue_classification: ClassificationMetrics
    category_metrics: CategoryMetrics
    severity_metrics: CategoryMetrics
    
    # Confidence analysis
    avg_confidence_true_positives: float = 0.0
    avg_confidence_false_positives: float = 0.0
    avg_confidence_true_negatives: float = 0.0
    avg_confidence_false_negatives: float = 0.0
    
    # Detailed results
    predictions: List[PredictionResult] = field(default_factory=list)
    failures: List[FailureCase] = field(default_factory=list)

    @property
    def combined_f1(self) -> float:
        """Calculate combined F1 score.

        Computes the equally-weighted average of issue classification F1,
        category macro F1, and severity macro F1 scores.

        Returns:
            float: Combined F1 score between 0.0 and 1.0.
        """
        issue_f1 = self.issue_classification.f1_score
        category_f1 = self.category_metrics.macro_f1
        severity_f1 = self.severity_metrics.macro_f1
        return (issue_f1 + category_f1 + severity_f1) / 3

    @property
    def pass_rate(self) -> float:
        """Calculate the pass rate.

        Computes the percentage of test cases that passed all checks
        (is_issue, category, and severity all correct).

        Returns:
            float: Pass rate between 0.0 and 1.0. Returns 0.0 if there
                are no predictions.
        """
        if not self.predictions:
            return 0.0
        correct = sum(1 for p in self.predictions if p.is_fully_correct)
        return correct / len(self.predictions)

    @property
    def failure_rate(self) -> float:
        """Calculate the failure rate.

        Computes the percentage of test cases that failed at least one check.

        Returns:
            float: Failure rate between 0.0 and 1.0.
        """
        return 1.0 - self.pass_rate

    @property
    def failure_breakdown(self) -> Dict[str, int]:
        """Get breakdown of failures by type.

        Counts the number of failures for each failure type.

        Returns:
            Dict[str, int]: Dictionary mapping failure type names to
                their occurrence counts.
        """
        breakdown: Dict[str, int] = {}
        for failure in self.failures:
            breakdown[failure.failure_type] = breakdown.get(failure.failure_type, 0) + 1
        return breakdown

    def to_dict(self) -> Dict[str, Any]:
        """Convert evaluation result to a dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary containing all evaluation data
                including metrics, predictions, failures, and metadata.
                Floating point values are rounded to 4 decimal places.
        """
        return {
            "prompt_version": self.prompt_version,
            "dataset_name": self.dataset_name,
            "timestamp": self.timestamp.isoformat(),
            "total_cases": self.total_cases,
            "issue_classification": self.issue_classification.to_dict(),
            "category_metrics": self.category_metrics.to_dict(),
            "severity_metrics": self.severity_metrics.to_dict(),
            "combined_f1": round(self.combined_f1, 4),
            "pass_rate": round(self.pass_rate, 4),
            "failure_rate": round(self.failure_rate, 4),
            "failure_breakdown": self.failure_breakdown,
            "confidence_analysis": {
                "avg_confidence_true_positives": round(self.avg_confidence_true_positives, 4),
                "avg_confidence_false_positives": round(self.avg_confidence_false_positives, 4),
                "avg_confidence_true_negatives": round(self.avg_confidence_true_negatives, 4),
                "avg_confidence_false_negatives": round(self.avg_confidence_false_negatives, 4),
            },
            "predictions": [p.to_dict() for p in self.predictions],
            "failures": [f.to_dict() for f in self.failures],
        }

    @classmethod
    def compute(
        cls,
        predictions: List[PredictionResult],
        prompt_version: str,
        dataset_name: str,
    ) -> "EvaluationResult":
        """Compute complete evaluation result from predictions.

        Calculates all metrics including classification, category, severity,
        and confidence analysis from the provided predictions.

        Args:
            predictions: List of PredictionResult objects to evaluate.
            prompt_version: Identifier for the prompt version being evaluated.
            dataset_name: Name of the dataset used for evaluation.

        Returns:
            EvaluationResult: A complete evaluation result with all metrics
                and failure analysis.
        """
        # Calculate all metrics
        issue_classification = ClassificationMetrics.from_predictions(predictions)
        category_metrics = CategoryMetrics.from_predictions(predictions, "category")
        severity_metrics = CategoryMetrics.from_predictions(predictions, "severity")
        
        # Calculate confidence averages
        tp_confidences = [p.analysis.confidence for p in predictions if p.is_true_positive]
        fp_confidences = [p.analysis.confidence for p in predictions if p.is_false_positive]
        tn_confidences = [p.analysis.confidence for p in predictions if p.is_true_negative]
        fn_confidences = [p.analysis.confidence for p in predictions if p.is_false_negative]
        
        avg_conf_tp = sum(tp_confidences) / len(tp_confidences) if tp_confidences else 0.0
        avg_conf_fp = sum(fp_confidences) / len(fp_confidences) if fp_confidences else 0.0
        avg_conf_tn = sum(tn_confidences) / len(tn_confidences) if tn_confidences else 0.0
        avg_conf_fn = sum(fn_confidences) / len(fn_confidences) if fn_confidences else 0.0
        
        # Collect failures
        failures = []
        for pred in predictions:
            failure = FailureCase.from_prediction(pred)
            if failure:
                failures.append(failure)
        
        return cls(
            prompt_version=prompt_version,
            dataset_name=dataset_name,
            timestamp=datetime.now(timezone.utc),
            total_cases=len(predictions),
            issue_classification=issue_classification,
            category_metrics=category_metrics,
            severity_metrics=severity_metrics,
            avg_confidence_true_positives=avg_conf_tp,
            avg_confidence_false_positives=avg_conf_fp,
            avg_confidence_true_negatives=avg_conf_tn,
            avg_confidence_false_negatives=avg_conf_fn,
            predictions=predictions,
            failures=failures,
        )

@dataclass
class ComparisonResult:
    """Result of comparing multiple prompt versions.
    
    Attributes:
        results: Dictionary of prompt version to evaluation result.
        dataset_name: Name of the dataset used for comparison.
        best_version: The prompt version with the highest combined F1.
    """
    results: Dict[str, EvaluationResult]
    dataset_name: str

    @property
    def best_version(self) -> Optional[str]:
        """Get the version with the highest combined F1 score.

        Returns:
            Optional[str]: The prompt version identifier with the best
                combined F1 score, or None if no results exist.
        """
        if not self.results:
            return None
        return max(self.results.keys(), key=lambda v: self.results[v].combined_f1)

    @property
    def rankings(self) -> List[tuple]:
        """Get versions ranked by combined F1 score.

        Returns:
            List[tuple]: List of (version, combined_f1) tuples sorted by
                combined F1 score in descending order (highest first).
        """
        return sorted(
            [(v, r.combined_f1) for v, r in self.results.items()],
            key=lambda x: x[1],
            reverse=True
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert comparison result to a dictionary representation.

        Returns:
            Dict[str, Any]: Dictionary containing dataset name, best version,
                rankings, and full results for each version compared.
        """
        return {
            "dataset_name": self.dataset_name,
            "best_version": self.best_version,
            "rankings": self.rankings,
            "results": {v: r.to_dict() for v, r in self.results.items()},
        }
