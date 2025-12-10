"""Automated prompt optimization based on failure analysis."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

import yaml

from src.interfaces.llm_analyzer import ILLMAnalyzer
from src.evaluation.models import FailureCase
from src.evaluation.metrics import EvaluationResult

# Path to the optimizer prompts
OPTIMIZER_PROMPTS_FILE = Path(__file__).parent / "suggestion_prompts.yaml"

@dataclass
class PromptSuggestion:
    """A single suggestion for improving the prompt.
    
    Attributes:
        suggestion_type: Type of suggestion (e.g., "category_clarification").
        description: Human-readable description of the suggestion.
        current_text: The current prompt text that should be changed.
        suggested_text: The suggested replacement text.
        rationale: Why this change would help.
        confidence: Confidence in this suggestion (0.0-1.0).
        affected_failures: List of failure IDs this would address.
    """
    suggestion_type: str
    description: str
    current_text: Optional[str] = None
    suggested_text: Optional[str] = None
    rationale: str = ""
    confidence: float = 0.5
    affected_failures: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "suggestion_type": self.suggestion_type,
            "description": self.description,
            "current_text": self.current_text,
            "suggested_text": self.suggested_text,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "affected_failures": self.affected_failures,
        }

@dataclass
class PromptOptimizationResult:
    """Results of prompt optimization analysis.
    
    Attributes:
        failure_patterns: Identified patterns in the failures.
        suggestions: List of prompt improvement suggestions.
        category_clarifications: Specific category definition improvements.
        severity_clarifications: Specific severity definition improvements.
        overall_analysis: High-level analysis of the prompt's performance.
        estimated_improvement: Estimated F1 improvement if suggestions applied.
    """
    failure_patterns: List[str]
    suggestions: List[PromptSuggestion]
    category_clarifications: Dict[str, str]
    severity_clarifications: Dict[str, str]
    overall_analysis: str
    estimated_improvement: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "failure_patterns": self.failure_patterns,
            "suggestions": [s.to_dict() for s in self.suggestions],
            "category_clarifications": self.category_clarifications,
            "severity_clarifications": self.severity_clarifications,
            "overall_analysis": self.overall_analysis,
            "estimated_improvement": self.estimated_improvement,
        }

    def to_markdown(self) -> str:
        """Generate a Markdown report of the optimization results."""
        lines = []
        
        lines.append("# Prompt Optimization Suggestions")
        lines.append("")
        
        lines.append("## Overall Analysis")
        lines.append("")
        lines.append(self.overall_analysis)
        lines.append("")
        
        if self.failure_patterns:
            lines.append("## Identified Failure Patterns")
            lines.append("")
            for i, pattern in enumerate(self.failure_patterns, 1):
                lines.append(f"{i}. {pattern}")
            lines.append("")
        
        if self.suggestions:
            lines.append("## Suggestions")
            lines.append("")
            for i, suggestion in enumerate(self.suggestions, 1):
                lines.append(f"### {i}. {suggestion.description}")
                lines.append("")
                lines.append(f"**Type:** {suggestion.suggestion_type}")
                lines.append(f"**Confidence:** {suggestion.confidence:.0%}")
                lines.append("")
                lines.append(f"**Rationale:** {suggestion.rationale}")
                if suggestion.current_text:
                    lines.append("")
                    lines.append("**Current:**")
                    lines.append(f"```\n{suggestion.current_text}\n```")
                if suggestion.suggested_text:
                    lines.append("")
                    lines.append("**Suggested:**")
                    lines.append(f"```\n{suggestion.suggested_text}\n```")
                lines.append("")
        
        if self.category_clarifications:
            lines.append("## Category Clarifications")
            lines.append("")
            for category, clarification in self.category_clarifications.items():
                lines.append(f"### {category}")
                lines.append(clarification)
                lines.append("")
        
        if self.severity_clarifications:
            lines.append("## Severity Clarifications")
            lines.append("")
            for severity, clarification in self.severity_clarifications.items():
                lines.append(f"### {severity}")
                lines.append(clarification)
                lines.append("")
        
        lines.append(f"## Estimated Improvement: {self.estimated_improvement:.1%}")
        
        return "\n".join(lines)

class PromptOptimizer:
    """Analyzes evaluation failures and suggests prompt improvements.
    
    Uses an LLM to analyze failure patterns and generate specific
    suggestions for improving the prompt's performance.
    """

    def __init__(
        self,
        analyzer: ILLMAnalyzer,
        prompts_file: Optional[Path] = None,
    ):
        """Initialize the prompt optimizer.
        
        Args:
            analyzer: LLM analyzer to use for generating suggestions.
            prompts_file: Path to the optimizer prompts YAML file.
        """
        self.analyzer = analyzer
        self.prompts = self._load_prompts(prompts_file or OPTIMIZER_PROMPTS_FILE)

    def _load_prompts(self, prompts_file: Path) -> Dict[str, str]:
        """Load optimizer prompts from YAML file."""
        if not prompts_file.exists():
            return {
                "analysis_prompt": self._get_default_analysis_prompt(),
                "suggestion_prompt": self._get_default_suggestion_prompt(),
            }
        
        with open(prompts_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def analyze_failures(
        self,
        result: EvaluationResult,
        current_prompt: Dict[str, str],
    ) -> PromptOptimizationResult:
        """Analyze evaluation failures and generate improvement suggestions.
        
        Args:
            result: Evaluation result containing failures.
            current_prompt: Current prompt configuration.
            
        Returns:
            PromptOptimizationResult with suggestions.
        """
        if not result.failures:
            return PromptOptimizationResult(
                failure_patterns=[],
                suggestions=[],
                category_clarifications={},
                severity_clarifications={},
                overall_analysis="No failures to analyze. The prompt is performing well!",
                estimated_improvement=0.0,
            )
        
        # Analyze failure patterns
        patterns = self._identify_patterns(result.failures)
        
        # Generate suggestions based on patterns
        suggestions = self._generate_suggestions(
            failures=result.failures,
            patterns=patterns,
            current_prompt=current_prompt,
            result=result,
        )
        
        # Generate category-specific clarifications
        category_clarifications = self._generate_category_clarifications(
            result.failures, current_prompt
        )
        
        # Generate severity-specific clarifications
        severity_clarifications = self._generate_severity_clarifications(
            result.failures, current_prompt
        )
        
        # Generate overall analysis
        overall_analysis = self._generate_overall_analysis(result, patterns)
        
        # Estimate improvement
        estimated_improvement = self._estimate_improvement(suggestions, result)
        
        return PromptOptimizationResult(
            failure_patterns=patterns,
            suggestions=suggestions,
            category_clarifications=category_clarifications,
            severity_clarifications=severity_clarifications,
            overall_analysis=overall_analysis,
            estimated_improvement=estimated_improvement,
        )

    def _identify_patterns(self, failures: List[FailureCase]) -> List[str]:
        """Identify common patterns in failures."""
        patterns = []
        
        # Count failure types
        failure_type_counts = {}
        for failure in failures:
            failure_type_counts[failure.failure_type] = (
                failure_type_counts.get(failure.failure_type, 0) + 1
            )
        
        # Identify dominant failure types
        total = len(failures)
        for ftype, count in failure_type_counts.items():
            if count / total >= 0.2:  # 20% or more
                patterns.append(
                    f"High rate of {ftype} failures ({count}/{total} = {count/total:.0%})"
                )
        
        # Check for category confusion
        category_errors = [
            f for f in failures if f.failure_type == FailureCase.WRONG_CATEGORY
        ]
        if len(category_errors) >= 2:
            patterns.append(
                f"Category classification errors detected ({len(category_errors)} cases)"
            )
        
        # Check for severity confusion
        severity_errors = [
            f for f in failures if f.failure_type == FailureCase.WRONG_SEVERITY
        ]
        if len(severity_errors) >= 2:
            patterns.append(
                f"Severity classification errors detected ({len(severity_errors)} cases)"
            )
        
        # Check for low confidence issues
        low_conf_errors = [
            f for f in failures if f.failure_type == FailureCase.LOW_CONFIDENCE
        ]
        if low_conf_errors:
            patterns.append(
                f"Low confidence predictions detected ({len(low_conf_errors)} cases)"
            )
        
        return patterns

    def _generate_suggestions(
        self,
        failures: List[FailureCase],
        patterns: List[str],
        current_prompt: Dict[str, str],
        result: EvaluationResult,
    ) -> List[PromptSuggestion]:
        """Generate suggestions based on failure analysis."""
        suggestions = []
        
        # Check for false positives - prompt might be too sensitive
        fp_count = result.failure_breakdown.get(FailureCase.FALSE_POSITIVE, 0)
        if fp_count >= 2:
            suggestions.append(PromptSuggestion(
                suggestion_type="sensitivity_reduction",
                description="Reduce false positive rate",
                rationale=(
                    f"The prompt is incorrectly flagging {fp_count} non-issues as issues. "
                    "Consider adding stricter criteria for what constitutes a real issue."
                ),
                suggested_text=(
                    "Before classifying as an issue, verify that:\n"
                    "- The user is reporting a problem they experienced (not asking a question)\n"
                    "- The post describes a malfunction, error, or unexpected behavior\n"
                    "- It's not a feature request, discussion, or product recommendation"
                ),
                confidence=0.7,
                affected_failures=[
                    f.prediction.test_case.test_case_id for f in failures
                    if f.failure_type == FailureCase.FALSE_POSITIVE
                ],
            ))
        
        # Check for false negatives - prompt might be missing issues
        fn_count = result.failure_breakdown.get(FailureCase.FALSE_NEGATIVE, 0)
        if fn_count >= 2:
            suggestions.append(PromptSuggestion(
                suggestion_type="sensitivity_increase",
                description="Improve issue detection rate",
                rationale=(
                    f"The prompt is missing {fn_count} real issues. "
                    "Consider being more inclusive in issue detection."
                ),
                suggested_text=(
                    "Look for signs of issues even if not explicitly stated:\n"
                    "- Users mentioning errors, crashes, or unexpected behavior\n"
                    "- Multiple users in comments reporting the same problem\n"
                    "- References to things 'not working' or being 'broken'"
                ),
                confidence=0.7,
                affected_failures=[
                    f.prediction.test_case.test_case_id for f in failures
                    if f.failure_type == FailureCase.FALSE_NEGATIVE
                ],
            ))
        
        # Check for category errors
        category_errors = [
            f for f in failures if f.failure_type == FailureCase.WRONG_CATEGORY
        ]
        if category_errors:
            # Identify most confused categories
            confusion_pairs = {}
            for f in category_errors:
                expected = f.prediction.test_case.expected_category
                predicted = f.prediction.analysis.category
                pair = (expected, predicted)
                confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
            
            for (expected, predicted), count in confusion_pairs.items():
                if count >= 1:
                    suggestions.append(PromptSuggestion(
                        suggestion_type="category_distinction",
                        description=f"Clarify distinction between '{expected}' and '{predicted}'",
                        rationale=(
                            f"The prompt confused '{expected}' with '{predicted}' {count} time(s). "
                            "The category definitions may need clearer boundaries."
                        ),
                        confidence=0.6,
                        affected_failures=[
                            f.prediction.test_case.test_case_id for f in category_errors
                            if f.prediction.test_case.expected_category == expected
                        ],
                    ))
        
        # Check for severity errors
        severity_errors = [
            f for f in failures if f.failure_type == FailureCase.WRONG_SEVERITY
        ]
        if severity_errors:
            suggestions.append(PromptSuggestion(
                suggestion_type="severity_calibration",
                description="Improve severity calibration",
                rationale=(
                    f"The prompt incorrectly assessed severity {len(severity_errors)} time(s). "
                    "Consider adding more specific criteria for each severity level."
                ),
                suggested_text=(
                    "Severity guidelines:\n"
                    "- critical: Affects >100 users, complete service outage, data loss\n"
                    "- high: Affects >10 users, major feature broken, no workaround\n"
                    "- medium: Affects few users, workaround available, partial functionality\n"
                    "- low: Single user, minor inconvenience, easy workaround"
                ),
                confidence=0.6,
                affected_failures=[
                    f.prediction.test_case.test_case_id for f in severity_errors
                ],
            ))
        
        return suggestions

    def _generate_category_clarifications(
        self,
        failures: List[FailureCase],
        current_prompt: Dict[str, str],
    ) -> Dict[str, str]:
        """Generate clarifications for problematic categories."""
        clarifications = {}
        
        category_errors = [
            f for f in failures if f.failure_type == FailureCase.WRONG_CATEGORY
        ]
        
        confused_categories = set()
        for f in category_errors:
            confused_categories.add(f.prediction.test_case.expected_category)
            confused_categories.add(f.prediction.analysis.category)
        
        for category in confused_categories:
            if category == "connectivity":
                clarifications[category] = (
                    "Use 'connectivity' for: Network issues, Xbox Live service problems, "
                    "online multiplayer connection issues, NAT type problems. "
                    "NOT for account sign-in issues (use 'account')."
                )
            elif category == "account":
                clarifications[category] = (
                    "Use 'account' for: Sign-in problems, profile issues, "
                    "authentication errors, account-related errors. "
                    "If it's a broader Xbox Live outage affecting sign-in, use 'connectivity'."
                )
            elif category == "game_pass":
                clarifications[category] = (
                    "Use 'game_pass' for: Issues specific to Game Pass subscriptions, "
                    "Game Pass game downloads, Game Pass library access. "
                    "General download issues should be 'other' or appropriate category."
                )
        
        return clarifications

    def _generate_severity_clarifications(
        self,
        failures: List[FailureCase],
        current_prompt: Dict[str, str],
    ) -> Dict[str, str]:
        """Generate clarifications for severity levels."""
        clarifications = {}
        
        severity_errors = [
            f for f in failures if f.failure_type == FailureCase.WRONG_SEVERITY
        ]
        
        if severity_errors:
            clarifications["general"] = (
                "When assessing severity, consider:\n"
                "1. Number of affected users (check post score and comments)\n"
                "2. Impact on functionality (complete vs partial)\n"
                "3. Availability of workarounds\n"
                "4. Duration of the issue"
            )
        
        return clarifications

    def _generate_overall_analysis(
        self,
        result: EvaluationResult,
        patterns: List[str],
    ) -> str:
        """Generate an overall analysis of prompt performance."""
        lines = []
        
        lines.append(f"The prompt achieved a combined F1 score of {result.combined_f1:.2%}.")
        lines.append("")
        
        if result.issue_classification.precision < 0.8:
            lines.append(
                f"Precision ({result.issue_classification.precision:.2%}) indicates "
                "the prompt is flagging too many non-issues as issues."
            )
        
        if result.issue_classification.recall < 0.8:
            lines.append(
                f"Recall ({result.issue_classification.recall:.2%}) indicates "
                "the prompt is missing some real issues."
            )
        
        if patterns:
            lines.append("")
            lines.append("Key patterns identified:")
            for pattern in patterns:
                lines.append(f"- {pattern}")
        
        return "\n".join(lines)

    def _estimate_improvement(
        self,
        suggestions: List[PromptSuggestion],
        result: EvaluationResult,
    ) -> float:
        """Estimate potential F1 improvement from suggestions."""
        if not suggestions:
            return 0.0
        
        # Simple heuristic: each suggestion could improve by a small amount
        # weighted by confidence
        improvement = 0.0
        for suggestion in suggestions:
            # Each suggestion could fix some failures
            potential_fix_rate = len(suggestion.affected_failures) / max(result.total_cases, 1)
            improvement += potential_fix_rate * suggestion.confidence * 0.5
        
        # Cap at reasonable improvement
        return min(improvement, 0.15)

    def _get_default_analysis_prompt(self) -> str:
        """Get the default analysis prompt."""
        return """Analyze the following evaluation failures and identify patterns:

{failures}

Identify:
1. Common patterns in the failures
2. Categories that are often confused
3. Severity assessment issues
4. Whether the prompt is too sensitive or not sensitive enough

Respond with a structured analysis."""

    def _get_default_suggestion_prompt(self) -> str:
        """Get the default suggestion prompt."""
        return """Based on the following failure patterns and current prompt, suggest improvements:

Current Prompt:
{current_prompt}

Failure Patterns:
{patterns}

Generate specific, actionable suggestions for improving the prompt."""
