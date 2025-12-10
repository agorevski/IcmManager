"""Report generation for evaluation results."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

from src.evaluation.metrics import EvaluationResult, ComparisonResult, ClassificationMetrics

# Default directory for metrics summaries (relative to this module)
DEFAULT_METRICS_DIR = Path(__file__).parent / "metrics"

class ReportGenerator:
    """Generates evaluation reports in various formats.
    
    Supports Markdown and JSON output formats for evaluation results
    and prompt version comparisons.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize the report generator.
        
        Args:
            output_dir: Directory for output files. Defaults to current directory.
        """
        self.output_dir = output_dir or Path(".")

    def generate_markdown_report(
        self,
        result: EvaluationResult,
        include_failures: bool = True,
        include_predictions: bool = False,
    ) -> str:
        """Generate a Markdown report from evaluation results.
        
        Args:
            result: The evaluation result to report on.
            include_failures: Whether to include failure details.
            include_predictions: Whether to include all predictions.
            
        Returns:
            Markdown formatted string.
        """
        lines = []
        
        # Header
        lines.append(f"# Prompt Evaluation Report: {result.prompt_version}")
        lines.append("")
        lines.append(f"**Dataset:** {result.dataset_name}")
        lines.append(f"**Evaluated:** {result.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append(f"**Total Cases:** {result.total_cases}")
        lines.append("")
        
        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Combined F1 | **{result.combined_f1:.4f}** |")
        lines.append(f"| Pass Rate | {result.pass_rate:.2%} |")
        lines.append(f"| Failure Rate | {result.failure_rate:.2%} |")
        lines.append("")
        
        # Issue Classification Metrics
        lines.append("## Issue Classification (is_issue)")
        lines.append("")
        lines.append(self._format_classification_metrics_table(result.issue_classification))
        lines.append("")
        
        # Confusion Matrix
        lines.append("### Confusion Matrix")
        lines.append("")
        lines.append("```")
        lines.append("                 Predicted")
        lines.append("              Issue    Not Issue")
        lines.append(f"Actual Issue    {result.issue_classification.tp:4d}       {result.issue_classification.fn:4d}")
        lines.append(f"   Not Issue    {result.issue_classification.fp:4d}       {result.issue_classification.tn:4d}")
        lines.append("```")
        lines.append("")
        
        # Category Metrics
        lines.append("## Category Classification")
        lines.append("")
        if result.category_metrics.labels:
            lines.append(self._format_multiclass_metrics_table(result.category_metrics, "Category"))
        else:
            lines.append("*No category data available (no true positives)*")
        lines.append("")
        
        # Severity Metrics
        lines.append("## Severity Classification")
        lines.append("")
        if result.severity_metrics.labels:
            lines.append(self._format_multiclass_metrics_table(result.severity_metrics, "Severity"))
        else:
            lines.append("*No severity data available (no true positives)*")
        lines.append("")
        
        # Confidence Analysis
        lines.append("## Confidence Analysis")
        lines.append("")
        lines.append("| Classification | Avg Confidence |")
        lines.append("|----------------|----------------|")
        lines.append(f"| True Positives | {result.avg_confidence_true_positives:.4f} |")
        lines.append(f"| False Positives | {result.avg_confidence_false_positives:.4f} |")
        lines.append(f"| True Negatives | {result.avg_confidence_true_negatives:.4f} |")
        lines.append(f"| False Negatives | {result.avg_confidence_false_negatives:.4f} |")
        lines.append("")
        
        # Failure Analysis
        if include_failures and result.failures:
            lines.append("## Failure Analysis")
            lines.append("")
            lines.append(f"**Total Failures:** {len(result.failures)}")
            lines.append("")
            
            # Breakdown by type
            lines.append("### Failures by Type")
            lines.append("")
            lines.append("| Type | Count |")
            lines.append("|------|-------|")
            for failure_type, count in sorted(result.failure_breakdown.items()):
                lines.append(f"| {failure_type} | {count} |")
            lines.append("")
            
            # Individual failures
            lines.append("### Failure Details")
            lines.append("")
            for i, failure in enumerate(result.failures[:20], 1):  # Limit to 20
                lines.append(f"#### {i}. {failure.prediction.test_case.test_case_id}")
                lines.append(f"- **Type:** {failure.failure_type}")
                lines.append(f"- **Details:** {failure.error_details}")
                if failure.potential_causes:
                    lines.append(f"- **Potential Causes:**")
                    for cause in failure.potential_causes:
                        lines.append(f"  - {cause}")
                lines.append("")
            
            if len(result.failures) > 20:
                lines.append(f"*... and {len(result.failures) - 20} more failures*")
                lines.append("")
        
        # Predictions (optional)
        if include_predictions:
            lines.append("## All Predictions")
            lines.append("")
            lines.append("| Test Case | Expected | Predicted | Correct | Confidence |")
            lines.append("|-----------|----------|-----------|---------|------------|")
            for pred in result.predictions:
                expected = "Issue" if pred.test_case.expected_is_issue else "Not Issue"
                predicted = "Issue" if pred.analysis.is_issue else "Not Issue"
                correct = "✓" if pred.is_issue_correct else "✗"
                lines.append(
                    f"| {pred.test_case.test_case_id} | {expected} | {predicted} | "
                    f"{correct} | {pred.analysis.confidence:.2f} |"
                )
            lines.append("")
        
        return "\n".join(lines)

    def generate_json_report(
        self,
        result: EvaluationResult,
        include_predictions: bool = True,
    ) -> str:
        """Generate a JSON report from evaluation results.
        
        Args:
            result: The evaluation result to report on.
            include_predictions: Whether to include all predictions.
            
        Returns:
            JSON formatted string.
        """
        data = result.to_dict()
        if not include_predictions:
            data.pop("predictions", None)
        return json.dumps(data, indent=2, default=str)

    def generate_comparison_report(
        self,
        comparison: ComparisonResult,
        format: str = "markdown",
    ) -> str:
        """Generate a report comparing multiple prompt versions.
        
        Args:
            comparison: The comparison result.
            format: Output format ("markdown" or "json").
            
        Returns:
            Formatted report string.
        """
        if format == "json":
            return json.dumps(comparison.to_dict(), indent=2, default=str)
        
        lines = []
        
        # Header
        lines.append("# Prompt Version Comparison Report")
        lines.append("")
        lines.append(f"**Dataset:** {comparison.dataset_name}")
        lines.append(f"**Best Version:** {comparison.best_version}")
        lines.append("")
        
        # Rankings
        lines.append("## Rankings")
        lines.append("")
        lines.append("| Rank | Version | Combined F1 | Issue F1 | Category F1 | Severity F1 |")
        lines.append("|------|---------|-------------|----------|-------------|-------------|")
        
        for rank, (version, combined_f1) in enumerate(comparison.rankings, 1):
            result = comparison.results[version]
            lines.append(
                f"| {rank} | {version} | **{combined_f1:.4f}** | "
                f"{result.issue_classification.f1_score:.4f} | "
                f"{result.category_metrics.macro_f1:.4f} | "
                f"{result.severity_metrics.macro_f1:.4f} |"
            )
        lines.append("")
        
        # Detailed comparison
        lines.append("## Detailed Metrics")
        lines.append("")
        
        for version, result in comparison.results.items():
            lines.append(f"### {version}")
            lines.append("")
            lines.append(f"- **Pass Rate:** {result.pass_rate:.2%}")
            lines.append(f"- **Precision:** {result.issue_classification.precision:.4f}")
            lines.append(f"- **Recall:** {result.issue_classification.recall:.4f}")
            lines.append(f"- **Failures:** {len(result.failures)}")
            
            if result.failure_breakdown:
                lines.append("- **Failure Breakdown:**")
                for ftype, count in sorted(result.failure_breakdown.items()):
                    lines.append(f"  - {ftype}: {count}")
            lines.append("")
        
        return "\n".join(lines)

    def save_report(
        self,
        content: str,
        filename: str,
        subdir: Optional[str] = None,
    ) -> Path:
        """Save a report to a file.
        
        Args:
            content: Report content.
            filename: Output filename.
            subdir: Optional subdirectory within output_dir.
            
        Returns:
            Path to the saved file.
        """
        output_path = self.output_dir
        if subdir:
            output_path = output_path / subdir
        
        output_path.mkdir(parents=True, exist_ok=True)
        
        file_path = output_path / filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        return file_path

    def save_evaluation_report(
        self,
        result: EvaluationResult,
        format: str = "markdown",
        include_failures: bool = True,
    ) -> Path:
        """Save an evaluation report with automatic naming.
        
        Args:
            result: The evaluation result.
            format: Output format ("markdown" or "json").
            include_failures: Whether to include failure details.
            
        Returns:
            Path to the saved file.
        """
        timestamp = result.timestamp.strftime("%Y%m%d_%H%M%S")
        version_safe = result.prompt_version.replace("/", "_").replace("\\", "_")
        
        if format == "markdown":
            content = self.generate_markdown_report(result, include_failures)
            filename = f"eval_{version_safe}_{timestamp}.md"
        else:
            content = self.generate_json_report(result)
            filename = f"eval_{version_safe}_{timestamp}.json"
        
        return self.save_report(content, filename, subdir="evaluation_reports")

    def _format_classification_metrics_table(self, metrics: ClassificationMetrics) -> str:
        """Format classification metrics as a Markdown table."""
        lines = [
            "| Metric | Value |",
            "|--------|-------|",
            f"| Precision | {metrics.precision:.4f} |",
            f"| Recall | {metrics.recall:.4f} |",
            f"| F1 Score | **{metrics.f1_score:.4f}** |",
            f"| Accuracy | {metrics.accuracy:.4f} |",
            f"| Specificity | {metrics.specificity:.4f} |",
            f"| True Positives | {metrics.tp} |",
            f"| True Negatives | {metrics.tn} |",
            f"| False Positives | {metrics.fp} |",
            f"| False Negatives | {metrics.fn} |",
        ]
        return "\n".join(lines)

    def _format_multiclass_metrics_table(self, metrics, label: str) -> str:
        """Format multi-class metrics as a Markdown table."""
        lines = [
            f"**Macro F1:** {metrics.macro_f1:.4f}",
            f"**Weighted F1:** {metrics.weighted_f1:.4f}",
            f"**Accuracy:** {metrics.accuracy:.4f}",
            "",
            f"| {label} | Precision | Recall | F1 |",
            "|---------|-----------|--------|-----|",
        ]
        
        per_class = metrics.per_class_metrics()
        for class_label in sorted(metrics.labels):
            m = per_class.get(class_label, ClassificationMetrics())
            lines.append(
                f"| {class_label} | {m.precision:.4f} | {m.recall:.4f} | {m.f1_score:.4f} |"
            )
        
        return "\n".join(lines)


def generate_metrics_summary(
    result: EvaluationResult,
    dataset_filename: str,
) -> str:
    """Generate a concise metrics summary markdown report.
    
    Args:
        result: The evaluation result.
        dataset_filename: Name of the dataset file used.
        
    Returns:
        Markdown formatted string with Precision/Recall/F1 metrics.
    """
    lines = []
    
    # Header
    lines.append(f"# Evaluation Metrics: {result.prompt_version}")
    lines.append("")
    lines.append(f"**Dataset:** {dataset_filename}")
    lines.append(f"**Evaluated:** {result.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"**Total Cases:** {result.total_cases}")
    lines.append("")
    
    # Issue Detection Metrics
    lines.append("## Issue Detection")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Precision | {result.issue_classification.precision:.4f} |")
    lines.append(f"| Recall | {result.issue_classification.recall:.4f} |")
    lines.append(f"| F1 Score | {result.issue_classification.f1_score:.4f} |")
    lines.append("")
    
    # Category Classification Metrics
    lines.append("## Category Classification")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Macro F1 | {result.category_metrics.macro_f1:.4f} |")
    lines.append("")
    
    # Severity Classification Metrics
    lines.append("## Severity Classification")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Macro F1 | {result.severity_metrics.macro_f1:.4f} |")
    lines.append("")
    
    # Combined Metrics
    lines.append("## Combined")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Combined F1 | {result.combined_f1:.4f} |")
    lines.append("")
    
    return "\n".join(lines)


def save_metrics_summary(
    result: EvaluationResult,
    dataset_filename: str,
    metrics_dir: Optional[Path] = None,
) -> Path:
    """Save metrics summary to evaluation/metrics/<prompt_version>.md.
    
    Args:
        result: The evaluation result.
        dataset_filename: Name of the dataset file used.
        metrics_dir: Directory for metrics files. Defaults to evaluation/metrics.
        
    Returns:
        Path to the saved file.
    """
    metrics_dir = metrics_dir or DEFAULT_METRICS_DIR
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate safe filename from prompt version
    version_safe = result.prompt_version.replace("/", "_").replace("\\", "_")
    filename = f"{version_safe}.md"
    
    # Generate content
    content = generate_metrics_summary(result, dataset_filename)
    
    # Save file
    file_path = metrics_dir / filename
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    return file_path


def print_quick_summary(result: EvaluationResult) -> None:
    """Print a quick summary of evaluation results to console.
    
    Args:
        result: The evaluation result.
    """
    print(f"\n{'='*60}")
    print(f"Evaluation Results: {result.prompt_version}")
    print(f"{'='*60}")
    print(f"Dataset: {result.dataset_name}")
    print(f"Total Cases: {result.total_cases}")
    print(f"")
    print(f"Issue Detection:")
    print(f"  Precision: {result.issue_classification.precision:.4f}")
    print(f"  Recall:    {result.issue_classification.recall:.4f}")
    print(f"  F1:        {result.issue_classification.f1_score:.4f}")
    print(f"")
    print(f"Category F1: {result.category_metrics.macro_f1:.4f}")
    print(f"Severity F1: {result.severity_metrics.macro_f1:.4f}")
    print(f"Combined F1: {result.combined_f1:.4f}")
    print(f"")
    print(f"Pass Rate: {result.pass_rate:.2%}")
    print(f"Failures:  {len(result.failures)}")
    
    if result.failure_breakdown:
        print(f"")
        print(f"Failure Breakdown:")
        for ftype, count in sorted(result.failure_breakdown.items()):
            print(f"  {ftype}: {count}")
    
    print(f"{'='*60}\n")
