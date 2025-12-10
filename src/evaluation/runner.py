"""CLI runner for prompt evaluation."""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from src.config import AzureOpenAIConfig
from src.evaluation.models import EvaluationDataset
from src.evaluation.evaluator import PromptEvaluator, BatchPromptEvaluator
from src.evaluation.prompt_manager import PromptVersionManager
from src.evaluation.reports import ReportGenerator, print_quick_summary

class EvaluationRunner:
    """Orchestrates prompt evaluation runs.
    
    Provides a high-level interface for running evaluations,
    comparing prompt versions, and generating reports.
    """

    def __init__(
        self,
        config: Optional[AzureOpenAIConfig] = None,
        confidence_threshold: float = 0.7,
        verbose: bool = False,
        output_dir: Optional[Path] = None,
    ):
        """Initialize the evaluation runner.
        
        Args:
            config: Azure OpenAI configuration. If None, loads from environment.
            confidence_threshold: Minimum confidence threshold for evaluation.
            verbose: Whether to print progress during evaluation.
            output_dir: Directory for output reports.
        """
        self.config = config
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose
        self.output_dir = output_dir or Path("results")
        
        self.evaluator = PromptEvaluator(
            confidence_threshold=confidence_threshold,
            verbose=verbose,
        )
        self.prompt_manager = PromptVersionManager()
        self.report_generator = ReportGenerator(output_dir=self.output_dir)

    def _get_config(self) -> AzureOpenAIConfig:
        """Get or load Azure OpenAI configuration."""
        if self.config:
            return self.config
        return AzureOpenAIConfig.from_env()

    def run_evaluation(
        self,
        prompt_version: str,
        dataset_path: Path,
        output_format: str = "markdown",
        save_report: bool = True,
    ):
        """Run evaluation for a single prompt version.
        
        Args:
            prompt_version: Version to evaluate ("current" or version name).
            dataset_path: Path to the evaluation dataset JSON.
            output_format: Report format ("markdown" or "json").
            save_report: Whether to save the report to a file.
            
        Returns:
            EvaluationResult with metrics and predictions.
        """
        if self.verbose:
            print(f"Loading dataset from {dataset_path}...")
        
        # Load dataset
        dataset = EvaluationDataset.load(dataset_path)
        
        if self.verbose:
            print(f"Dataset: {dataset.description}")
            print(f"Test cases: {len(dataset.test_cases)}")
            print(f"Positive: {dataset.positive_count}, Negative: {dataset.negative_count}")
        
        # Create analyzer for the prompt version
        config = self._get_config()
        analyzer = self.prompt_manager.create_analyzer_for_version(
            prompt_version, config
        )
        
        if self.verbose:
            print(f"\nEvaluating prompt version: {prompt_version}")
            print(f"Using model: {analyzer.get_model_name()}")
        
        # Run evaluation
        result = self.evaluator.evaluate_dataset(
            analyzer=analyzer,
            dataset=dataset,
            prompt_version=prompt_version,
        )
        
        # Print summary
        print_quick_summary(result)
        
        # Save report if requested
        if save_report:
            report_path = self.report_generator.save_evaluation_report(
                result=result,
                format=output_format,
                include_failures=True,
            )
            print(f"Report saved to: {report_path}")
        
        return result

    def run_comparison(
        self,
        prompt_versions: List[str],
        dataset_path: Path,
        output_format: str = "markdown",
        save_report: bool = True,
    ):
        """Compare multiple prompt versions.
        
        Args:
            prompt_versions: List of versions to compare.
            dataset_path: Path to the evaluation dataset JSON.
            output_format: Report format ("markdown" or "json").
            save_report: Whether to save the report to a file.
            
        Returns:
            ComparisonResult with rankings and metrics.
        """
        if self.verbose:
            print(f"Loading dataset from {dataset_path}...")
        
        # Load dataset
        dataset = EvaluationDataset.load(dataset_path)
        
        if self.verbose:
            print(f"Dataset: {dataset.description}")
            print(f"Comparing {len(prompt_versions)} prompt versions")
        
        # Create analyzers for each version
        config = self._get_config()
        analyzers = {}
        for version in prompt_versions:
            if self.verbose:
                print(f"Loading prompt version: {version}")
            analyzers[version] = self.prompt_manager.create_analyzer_for_version(
                version, config
            )
        
        # Run comparison
        comparison = self.evaluator.compare(
            analyzers=analyzers,
            test_cases=dataset.test_cases,
            dataset_name=f"{dataset.description} v{dataset.version}",
        )
        
        # Print rankings
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        print(f"\nBest Version: {comparison.best_version}")
        print("\nRankings:")
        for rank, (version, f1) in enumerate(comparison.rankings, 1):
            print(f"  {rank}. {version}: F1={f1:.4f}")
        print("="*60 + "\n")
        
        # Save report if requested
        if save_report:
            report_content = self.report_generator.generate_comparison_report(
                comparison, format=output_format
            )
            filename = f"comparison_{'_'.join(prompt_versions)}.{'md' if output_format == 'markdown' else 'json'}"
            report_path = self.report_generator.save_report(
                report_content, filename, subdir="comparison_reports"
            )
            print(f"Comparison report saved to: {report_path}")
        
        return comparison

    def list_versions(self) -> List[str]:
        """List available prompt versions."""
        versions = self.prompt_manager.list_versions()
        versions.insert(0, "current")  # Always include current
        return versions

    def validate_dataset(self, dataset_path: Path) -> bool:
        """Validate a dataset file.
        
        Args:
            dataset_path: Path to the dataset JSON.
            
        Returns:
            True if valid, False otherwise.
        """
        try:
            dataset = EvaluationDataset.load(dataset_path)
            print(f"Dataset is valid!")
            print(f"  Version: {dataset.version}")
            print(f"  Description: {dataset.description}")
            print(f"  Test cases: {len(dataset.test_cases)}")
            print(f"  Positive cases: {dataset.positive_count}")
            print(f"  Negative cases: {dataset.negative_count}")
            print(f"  Categories: {list(dataset.category_distribution.keys())}")
            print(f"  Severities: {list(dataset.severity_distribution.keys())}")
            return True
        except Exception as e:
            print(f"Dataset validation failed: {e}")
            return False

def main():
    """CLI entry point for prompt evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate LLM prompts for Xbox issue detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate current prompt
  python -m src.evaluation.runner evaluate --version current --dataset tests/evaluation_datasets/golden_set_v1.json

  # Compare multiple versions
  python -m src.evaluation.runner compare --versions current,v1_baseline,v2_improved --dataset tests/evaluation_datasets/golden_set_v1.json

  # List available versions
  python -m src.evaluation.runner list-versions

  # Validate a dataset
  python -m src.evaluation.runner validate --dataset tests/evaluation_datasets/golden_set_v1.json
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a single prompt version")
    eval_parser.add_argument(
        "--version", "-v",
        required=True,
        help="Prompt version to evaluate ('current' or version name)"
    )
    eval_parser.add_argument(
        "--dataset", "-d",
        required=True,
        type=Path,
        help="Path to evaluation dataset JSON"
    )
    eval_parser.add_argument(
        "--format", "-f",
        choices=["markdown", "json"],
        default="markdown",
        help="Output report format (default: markdown)"
    )
    eval_parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("results"),
        help="Output directory for reports (default: results/)"
    )
    eval_parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum confidence threshold (default: 0.7)"
    )
    eval_parser.add_argument(
        "--min-f1",
        type=float,
        help="Minimum F1 score to pass (for CI/CD)"
    )
    eval_parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save the report to a file"
    )
    eval_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress"
    )
    eval_parser.add_argument(
        "--env-file",
        type=Path,
        help="Path to .env file with Azure credentials"
    )
    
    # compare command
    compare_parser = subparsers.add_parser("compare", help="Compare multiple prompt versions")
    compare_parser.add_argument(
        "--versions", "-v",
        required=True,
        help="Comma-separated list of versions to compare"
    )
    compare_parser.add_argument(
        "--dataset", "-d",
        required=True,
        type=Path,
        help="Path to evaluation dataset JSON"
    )
    compare_parser.add_argument(
        "--format", "-f",
        choices=["markdown", "json"],
        default="markdown",
        help="Output report format (default: markdown)"
    )
    compare_parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("results"),
        help="Output directory for reports (default: results/)"
    )
    compare_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress"
    )
    compare_parser.add_argument(
        "--env-file",
        type=Path,
        help="Path to .env file with Azure credentials"
    )
    
    # list-versions command
    list_parser = subparsers.add_parser("list-versions", help="List available prompt versions")
    
    # validate command
    validate_parser = subparsers.add_parser("validate", help="Validate a dataset file")
    validate_parser.add_argument(
        "--dataset", "-d",
        required=True,
        type=Path,
        help="Path to dataset JSON to validate"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Load environment variables
    if hasattr(args, 'env_file') and args.env_file:
        load_dotenv(args.env_file)
    else:
        load_dotenv()
    
    # Execute command
    if args.command == "evaluate":
        runner = EvaluationRunner(
            confidence_threshold=args.confidence_threshold,
            verbose=args.verbose,
            output_dir=args.output,
        )
        
        result = runner.run_evaluation(
            prompt_version=args.version,
            dataset_path=args.dataset,
            output_format=args.format,
            save_report=not args.no_save,
        )
        
        # Check minimum F1 if specified (for CI/CD)
        if args.min_f1 is not None:
            if result.combined_f1 < args.min_f1:
                print(f"\nFAILED: Combined F1 {result.combined_f1:.4f} is below minimum {args.min_f1:.4f}")
                sys.exit(1)
            else:
                print(f"\nPASSED: Combined F1 {result.combined_f1:.4f} meets minimum {args.min_f1:.4f}")
    
    elif args.command == "compare":
        versions = [v.strip() for v in args.versions.split(",")]
        
        runner = EvaluationRunner(
            verbose=args.verbose,
            output_dir=args.output,
        )
        
        runner.run_comparison(
            prompt_versions=versions,
            dataset_path=args.dataset,
            output_format=args.format,
        )
    
    elif args.command == "list-versions":
        manager = PromptVersionManager()
        versions = manager.list_versions()
        
        print("Available prompt versions:")
        print("  current (production prompt)")
        for version in versions:
            metadata = manager.get_version_metadata(version)
            description = metadata.get("description", "")
            print(f"  {version}" + (f" - {description}" if description else ""))
    
    elif args.command == "validate":
        runner = EvaluationRunner()
        valid = runner.validate_dataset(args.dataset)
        sys.exit(0 if valid else 1)

if __name__ == "__main__":
    main()
