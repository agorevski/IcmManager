"""Metrics calculations for prompt evaluation."""

from dataclasses import dataclass, field
from datetime import datetime
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
        """Total number of samples."""
        return self.tp + self.tn + self.fp + self.fn

    @property
    def precision(self) -> float:
        """Precision: TP / (TP + FP). How many predicted issues are real."""
        denominator = self.tp + self.fp
        return self.tp / denominator if denominator > 0 else 0.0

    @property
    def recall(self) -> float:
        """Recall: TP / (TP + FN). How many real issues were found."""
        denominator = self.tp + self.fn
        return self.tp / denominator if denominator > 0 else 0.0

    @property
    def f1_score(self) -> float:
        """F1 Score: Harmonic mean of precision and recall."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def accuracy(self) -> float:
        """Accuracy: (TP + TN) / Total."""
        return (self.tp + self.tn) / self.total if self.total > 0 else 0.0

    @property
    def specificity(self) -> float:
        """Specificity: TN / (TN + FP). How many non-issues were correctly identified."""
        denominator = self.tn + self.fp
        return self.tn / denominator if denominator > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
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
        """Calculate metrics from a list of predictions."""
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
        """Add a prediction to the confusion matrix."""
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
        """Calculate binary classification metrics for each class."""
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
        """Macro F1: Average F1 across all classes (unweighted)."""
        per_class = self.per_class_metrics()
        if not per_class:
            return 0.0
        f1_scores = [m.f1_score for m in per_class.values()]
        return sum(f1_scores) / len(f1_scores)

    @property
    def weighted_f1(self) -> float:
        """Weighted F1: F1 weighted by class frequency."""
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
        """Overall accuracy: correct predictions / total predictions."""
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
        """Convert to dictionary representation."""
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
        """Calculate category/severity metrics from predictions.
        
        Only considers predictions where:
        - The test case expected an issue (expected_is_issue=True)
        - The prediction identified it as an issue (is_issue=True)
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
        """Combined F1: Average of issue, category, and severity F1 scores.
        
        Equally weighted as per requirements.
        """
        issue_f1 = self.issue_classification.f1_score
        category_f1 = self.category_metrics.macro_f1
        severity_f1 = self.severity_metrics.macro_f1
        return (issue_f1 + category_f1 + severity_f1) / 3

    @property
    def pass_rate(self) -> float:
        """Percentage of test cases that passed all checks."""
        if not self.predictions:
            return 0.0
        correct = sum(1 for p in self.predictions if p.is_fully_correct)
        return correct / len(self.predictions)

    @property
    def failure_rate(self) -> float:
        """Percentage of test cases that failed."""
        return 1.0 - self.pass_rate

    @property
    def failure_breakdown(self) -> Dict[str, int]:
        """Breakdown of failures by type."""
        breakdown: Dict[str, int] = {}
        for failure in self.failures:
            breakdown[failure.failure_type] = breakdown.get(failure.failure_type, 0) + 1
        return breakdown

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
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
        """Compute evaluation result from predictions."""
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
            timestamp=datetime.utcnow(),
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
        """Get the version with the highest combined F1."""
        if not self.results:
            return None
        return max(self.results.keys(), key=lambda v: self.results[v].combined_f1)

    @property
    def rankings(self) -> List[tuple]:
        """Get versions ranked by combined F1 (highest first)."""
        return sorted(
            [(v, r.combined_f1) for v, r in self.results.items()],
            key=lambda x: x[1],
            reverse=True
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "dataset_name": self.dataset_name,
            "best_version": self.best_version,
            "rankings": self.rankings,
            "results": {v: r.to_dict() for v, r in self.results.items()},
        }
