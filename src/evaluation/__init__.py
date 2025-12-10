"""Prompt evaluation framework for testing and fine-tuning LLM prompts.

This package provides tools for:
- Evaluating prompt quality using Precision/Recall/F1 metrics
- Managing multiple prompt versions for A/B testing
- Generating detailed evaluation reports
- Automated prompt optimization suggestions

Example usage:
    from src.evaluation import PromptEvaluator, EvaluationDataset
    
    dataset = EvaluationDataset.load("tests/evaluation_datasets/golden_set_v1.json")
    evaluator = PromptEvaluator(confidence_threshold=0.7)
    result = evaluator.evaluate(analyzer, dataset.test_cases, "v1.0")
    print(f"F1 Score: {result.issue_classification.f1_score:.2f}")
"""

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
)
from src.evaluation.evaluator import (
    IPromptEvaluator,
    PromptEvaluator,
)
from src.evaluation.prompt_manager import PromptVersionManager
from src.evaluation.reports import ReportGenerator

__all__ = [
    # Models
    "LabeledTestCase",
    "PredictionResult",
    "FailureCase",
    "EvaluationDataset",
    # Metrics
    "ClassificationMetrics",
    "CategoryMetrics",
    "EvaluationResult",
    # Evaluator
    "IPromptEvaluator",
    "PromptEvaluator",
    # Management
    "PromptVersionManager",
    "ReportGenerator",
]
