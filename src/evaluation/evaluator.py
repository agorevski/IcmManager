"""Core evaluator for prompt quality assessment."""

from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from src.interfaces.llm_analyzer import ILLMAnalyzer
from src.models.issue import IssueAnalysis
from src.evaluation.models import (
    LabeledTestCase,
    PredictionResult,
    EvaluationDataset,
)
from src.evaluation.metrics import EvaluationResult, ComparisonResult


class IPromptEvaluator(ABC):
    """Abstract interface for prompt evaluation.
    
    Implementations can support different evaluation strategies,
    such as sequential evaluation, parallel evaluation, or
    evaluation with different LLM backends.
    """

    @abstractmethod
    def evaluate(
        self,
        analyzer: ILLMAnalyzer,
        test_cases: List[LabeledTestCase],
        prompt_version: str,
        dataset_name: str = "unknown",
    ) -> EvaluationResult:
        """Evaluate an analyzer against a set of test cases.
        
        Args:
            analyzer: The LLM analyzer to evaluate.
            test_cases: List of labeled test cases.
            prompt_version: Identifier for the prompt version.
            dataset_name: Name of the dataset being used.
            
        Returns:
            Complete evaluation results with metrics.
        """
        pass

    @abstractmethod
    def evaluate_dataset(
        self,
        analyzer: ILLMAnalyzer,
        dataset: EvaluationDataset,
        prompt_version: str,
    ) -> EvaluationResult:
        """Evaluate an analyzer against a full dataset.
        
        Args:
            analyzer: The LLM analyzer to evaluate.
            dataset: The evaluation dataset.
            prompt_version: Identifier for the prompt version.
            
        Returns:
            Complete evaluation results with metrics.
        """
        pass

    @abstractmethod
    def compare(
        self,
        analyzers: dict,
        test_cases: List[LabeledTestCase],
        dataset_name: str = "unknown",
    ) -> ComparisonResult:
        """Compare multiple analyzers/prompt versions.
        
        Args:
            analyzers: Dictionary mapping version names to analyzers.
            test_cases: List of labeled test cases.
            dataset_name: Name of the dataset being used.
            
        Returns:
            Comparison results with rankings.
        """
        pass


class PromptEvaluator(IPromptEvaluator):
    """Standard implementation of prompt evaluation.
    
    Evaluates LLM analyzers against labeled test cases and computes
    precision, recall, F1, and other metrics for issue detection,
    category classification, and severity classification.
    
    Attributes:
        confidence_threshold: Minimum confidence for acceptable predictions.
        verbose: Whether to print progress during evaluation.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        verbose: bool = False,
    ):
        """Initialize the evaluator.
        
        Args:
            confidence_threshold: Minimum acceptable confidence score.
            verbose: Whether to print progress during evaluation.
        """
        self.confidence_threshold = confidence_threshold
        self.verbose = verbose

    def evaluate(
        self,
        analyzer: ILLMAnalyzer,
        test_cases: List[LabeledTestCase],
        prompt_version: str,
        dataset_name: str = "unknown",
    ) -> EvaluationResult:
        """Evaluate an analyzer against a set of test cases."""
        predictions = []
        
        for i, test_case in enumerate(test_cases):
            if self.verbose:
                print(f"Evaluating {i+1}/{len(test_cases)}: {test_case.test_case_id}")
            
            # Get the analyzer's prediction
            analysis = analyzer.analyze_post(test_case.post)
            
            # Create prediction result with correctness checks
            prediction = self._create_prediction(test_case, analysis)
            predictions.append(prediction)
        
        # Compute and return metrics
        return EvaluationResult.compute(
            predictions=predictions,
            prompt_version=prompt_version,
            dataset_name=dataset_name,
        )

    def evaluate_dataset(
        self,
        analyzer: ILLMAnalyzer,
        dataset: EvaluationDataset,
        prompt_version: str,
    ) -> EvaluationResult:
        """Evaluate an analyzer against a full dataset."""
        # Use the dataset's confidence threshold if not overridden
        original_threshold = self.confidence_threshold
        if dataset.confidence_threshold:
            self.confidence_threshold = dataset.confidence_threshold
        
        try:
            return self.evaluate(
                analyzer=analyzer,
                test_cases=dataset.test_cases,
                prompt_version=prompt_version,
                dataset_name=f"{dataset.description} v{dataset.version}",
            )
        finally:
            self.confidence_threshold = original_threshold

    def compare(
        self,
        analyzers: dict,
        test_cases: List[LabeledTestCase],
        dataset_name: str = "unknown",
    ) -> ComparisonResult:
        """Compare multiple analyzers/prompt versions."""
        results = {}
        
        for version, analyzer in analyzers.items():
            if self.verbose:
                print(f"\n=== Evaluating version: {version} ===")
            
            result = self.evaluate(
                analyzer=analyzer,
                test_cases=test_cases,
                prompt_version=version,
                dataset_name=dataset_name,
            )
            results[version] = result
        
        return ComparisonResult(
            results=results,
            dataset_name=dataset_name,
        )

    def _create_prediction(
        self,
        test_case: LabeledTestCase,
        analysis: IssueAnalysis,
    ) -> PredictionResult:
        """Create a prediction result from a test case and analysis.
        
        Args:
            test_case: The labeled test case.
            analysis: The LLM's analysis result.
            
        Returns:
            PredictionResult with correctness flags.
        """
        # Check if is_issue prediction matches
        is_issue_correct = test_case.expected_is_issue == analysis.is_issue
        
        # Check confidence threshold
        min_conf = test_case.expected_min_confidence or self.confidence_threshold
        confidence_acceptable = analysis.confidence >= min_conf
        
        # Category and severity are only checked if this is a true issue
        category_correct: Optional[bool] = None
        severity_correct: Optional[bool] = None
        
        if test_case.expected_is_issue and analysis.is_issue:
            # Check category match (case-insensitive)
            if test_case.expected_category:
                category_correct = (
                    test_case.expected_category.lower() == analysis.category.lower()
                )
            
            # Check severity match (case-insensitive)
            if test_case.expected_severity:
                severity_correct = (
                    test_case.expected_severity.lower() == analysis.severity.lower()
                )
        
        return PredictionResult(
            test_case=test_case,
            analysis=analysis,
            is_issue_correct=is_issue_correct,
            category_correct=category_correct,
            severity_correct=severity_correct,
            confidence_acceptable=confidence_acceptable,
        )


class ParallelPromptEvaluator(PromptEvaluator):
    """Evaluator that processes test cases in parallel using threads.
    
    Uses ThreadPoolExecutor to run LLM calls concurrently, which can
    significantly speed up evaluation for large datasets.
    
    Attributes:
        max_workers: Maximum number of concurrent LLM calls.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        verbose: bool = False,
        max_workers: int = 5,
    ):
        """Initialize the parallel evaluator.
        
        Args:
            confidence_threshold: Minimum acceptable confidence score.
            verbose: Whether to print progress during evaluation.
            max_workers: Maximum number of concurrent threads (default: 5).
        """
        super().__init__(confidence_threshold, verbose)
        self.max_workers = max_workers

    def evaluate(
        self,
        analyzer: ILLMAnalyzer,
        test_cases: List[LabeledTestCase],
        prompt_version: str,
        dataset_name: str = "unknown",
    ) -> EvaluationResult:
        """Evaluate using parallel processing with ThreadPoolExecutor."""
        # Store results with their original indices to preserve order
        indexed_predictions: List[tuple] = []
        total = len(test_cases)
        completed = 0
        
        def process_test_case(index: int, test_case: LabeledTestCase):
            """Process a single test case and return indexed result."""
            analysis = analyzer.analyze_post(test_case.post)
            prediction = self._create_prediction(test_case, analysis)
            return (index, prediction)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(process_test_case, i, tc): i 
                for i, tc in enumerate(test_cases)
            }
            
            # Collect results as they complete
            for future in as_completed(futures):
                index, prediction = future.result()
                indexed_predictions.append((index, prediction))
                completed += 1
                
                if self.verbose:
                    test_case_id = test_cases[index].test_case_id
                    print(f"Completed {completed}/{total}: {test_case_id}")
        
        # Sort by original index to preserve order
        indexed_predictions.sort(key=lambda x: x[0])
        predictions = [pred for _, pred in indexed_predictions]
        
        # Compute and return metrics
        return EvaluationResult.compute(
            predictions=predictions,
            prompt_version=prompt_version,
            dataset_name=dataset_name,
        )


class BatchPromptEvaluator(PromptEvaluator):
    """Evaluator that uses batch processing for efficiency.
    
    Uses the analyzer's batch processing capability when available
    for faster evaluation of large datasets.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        verbose: bool = False,
        batch_size: int = 10,
    ):
        """Initialize the batch evaluator.
        
        Args:
            confidence_threshold: Minimum acceptable confidence score.
            verbose: Whether to print progress during evaluation.
            batch_size: Number of posts to process in each batch.
        """
        super().__init__(confidence_threshold, verbose)
        self.batch_size = batch_size

    def evaluate(
        self,
        analyzer: ILLMAnalyzer,
        test_cases: List[LabeledTestCase],
        prompt_version: str,
        dataset_name: str = "unknown",
    ) -> EvaluationResult:
        """Evaluate using batch processing."""
        predictions = []
        
        # Process in batches
        for batch_start in range(0, len(test_cases), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(test_cases))
            batch = test_cases[batch_start:batch_end]
            
            if self.verbose:
                print(f"Processing batch {batch_start//self.batch_size + 1}: "
                      f"cases {batch_start+1}-{batch_end}")
            
            # Extract posts for batch processing
            posts = [tc.post for tc in batch]
            
            # Get batch predictions
            analyses = analyzer.analyze_posts_batch(posts)
            
            # Create prediction results
            for test_case, analysis in zip(batch, analyses):
                prediction = self._create_prediction(test_case, analysis)
                predictions.append(prediction)
        
        return EvaluationResult.compute(
            predictions=predictions,
            prompt_version=prompt_version,
            dataset_name=dataset_name,
        )
