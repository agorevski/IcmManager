# Prompt Evaluation Framework

A comprehensive framework for testing and fine-tuning LLM prompts used in Xbox issue detection. This framework enables measuring prompt quality using Precision/Recall/F1 metrics, managing multiple prompt versions for A/B testing, and generating automated optimization suggestions.

## Quick Start

```bash
# Evaluate the current prompt
python -m src.evaluation.runner evaluate \
  --version current \
  --dataset tests/evaluation_datasets/golden_set_v1.json

# Compare multiple prompt versions
python -m src.evaluation.runner compare \
  --versions current,v1_baseline,v2_improved \
  --dataset tests/evaluation_datasets/golden_set_v1.json

# Validate a dataset
python -m src.evaluation.runner validate \
  --dataset tests/evaluation_datasets/golden_set_v1.json
```

## Architecture Overview

```
src/evaluation/
├── __init__.py           # Package exports
├── models.py             # Data models (LabeledTestCase, PredictionResult, etc.)
├── metrics.py            # Metric calculations (Precision, Recall, F1)
├── evaluator.py          # Core evaluation logic
├── prompt_manager.py     # Prompt version management
├── reports.py            # Report generation (Markdown, JSON)
├── runner.py             # CLI interface
├── prompt_optimizer.py   # Automated optimization suggestions
└── suggestion_prompts.yaml  # Prompts for optimizer
```

## Key Components

### 1. Evaluation Dataset (`models.py`)

Test cases consist of labeled Reddit posts with ground truth:

```python
from src.evaluation import EvaluationDataset, LabeledTestCase

# Load a dataset
dataset = EvaluationDataset.load("tests/evaluation_datasets/golden_set_v1.json")

# Access test cases
for test_case in dataset.test_cases:
    print(f"ID: {test_case.test_case_id}")
    print(f"Expected: is_issue={test_case.expected_is_issue}")
    print(f"Category: {test_case.expected_category}")
    print(f"Severity: {test_case.expected_severity}")
```

### Dataset JSON Format

```json
{
  "version": "1.0",
  "description": "Golden test set for Xbox issue classification",
  "confidence_threshold": 0.7,
  "test_cases": [
    {
      "test_case_id": "tc_001_xbox_live_signin",
      "description": "Xbox Live sign-in outage",
      "expected_is_issue": true,
      "expected_category": "connectivity",
      "expected_severity": "critical",
      "expected_min_confidence": 0.8,
      "edge_case_type": null,
      "post": {
        "id": "post_001",
        "subreddit": "xbox",
        "title": "Can't sign in to Xbox Live",
        "body": "...",
        "score": 510,
        "num_comments": 820,
        "comments": [...]
      }
    }
  ]
}
```

### 2. Metrics (`metrics.py`)

The framework calculates comprehensive metrics:

**Binary Classification (is_issue detection):**
- Precision: How many predicted issues are real
- Recall: How many real issues were found
- F1 Score: Harmonic mean of precision and recall
- Accuracy: Overall correctness
- Specificity: How many non-issues were correctly identified

**Multi-class Classification (category/severity):**
- Per-class Precision, Recall, F1
- Macro F1: Unweighted average across classes
- Weighted F1: Weighted by class frequency
- Confusion matrices

**Combined F1:**
The overall score is the average of:
- Issue detection F1
- Category classification macro F1
- Severity classification macro F1

### 3. Evaluator (`evaluator.py`)

```python
from src.evaluation import PromptEvaluator, EvaluationDataset
from src.analyzers.azure_openai_analyzer import AzureOpenAIAnalyzer

# Create evaluator
evaluator = PromptEvaluator(confidence_threshold=0.7, verbose=True)

# Load dataset and analyzer
dataset = EvaluationDataset.load("path/to/dataset.json")
analyzer = AzureOpenAIAnalyzer(...)

# Run evaluation
result = evaluator.evaluate_dataset(
    analyzer=analyzer,
    dataset=dataset,
    prompt_version="v1.0"
)

# Access results
print(f"Issue F1: {result.issue_classification.f1_score:.4f}")
print(f"Category F1: {result.category_metrics.macro_f1:.4f}")
print(f"Combined F1: {result.combined_f1:.4f}")
print(f"Failures: {len(result.failures)}")
```

### 4. Prompt Version Manager (`prompt_manager.py`)

Manage multiple prompt versions for A/B testing:

```python
from src.evaluation import PromptVersionManager

manager = PromptVersionManager()

# List available versions
versions = manager.list_versions()

# Save a new version
current_prompt = manager.get_current_prompt()
manager.save_version("v2_improved", current_prompt, "Added category examples")

# Create a variant with specific changes
manager.create_variant(
    base_version="current",
    new_version="v3_strict",
    changes={
        "system_prompt": "... modified prompt ..."
    },
    description="Stricter issue detection criteria"
)

# Promote a version to production
manager.promote_to_current("v3_strict", backup=True)
```

### 5. Report Generator (`reports.py`)

Generate detailed reports in Markdown or JSON:

```python
from src.evaluation import ReportGenerator

generator = ReportGenerator(output_dir=Path("results"))

# Generate Markdown report
markdown = generator.generate_markdown_report(result, include_failures=True)

# Save report
report_path = generator.save_evaluation_report(result, format="markdown")
```

### 6. Prompt Optimizer (`prompt_optimizer.py`)

Analyze failures and get improvement suggestions:

```python
from src.evaluation.prompt_optimizer import PromptOptimizer

optimizer = PromptOptimizer(analyzer=my_analyzer)

# Analyze evaluation results
optimization_result = optimizer.analyze_failures(
    result=evaluation_result,
    current_prompt=current_prompt_dict
)

# Get suggestions
for suggestion in optimization_result.suggestions:
    print(f"Type: {suggestion.suggestion_type}")
    print(f"Description: {suggestion.description}")
    print(f"Rationale: {suggestion.rationale}")
    print(f"Suggested: {suggestion.suggested_text}")
```

## CLI Commands

### Evaluate a Prompt Version

```bash
python -m src.evaluation.runner evaluate \
  --version current \
  --dataset tests/evaluation_datasets/golden_set_v1.json \
  --format markdown \
  --output results/ \
  --confidence-threshold 0.7 \
  --verbose
```

Options:
- `--version, -v`: Prompt version ("current" or version name)
- `--dataset, -d`: Path to evaluation dataset JSON
- `--format, -f`: Output format (markdown/json)
- `--output, -o`: Output directory
- `--confidence-threshold`: Minimum confidence threshold
- `--min-f1`: Minimum F1 for CI/CD pass (exits with error if below)
- `--verbose`: Print detailed progress

### Compare Prompt Versions

```bash
python -m src.evaluation.runner compare \
  --versions current,v1_baseline,v2_improved \
  --dataset tests/evaluation_datasets/golden_set_v1.json
```

### List Available Versions

```bash
python -m src.evaluation.runner list-versions
```

### Validate a Dataset

```bash
python -m src.evaluation.runner validate \
  --dataset tests/evaluation_datasets/golden_set_v1.json
```

## CI/CD Integration

Add to your GitHub Actions workflow:

```yaml
- name: Evaluate Prompt Quality
  run: |
    python -m src.evaluation.runner evaluate \
      --version current \
      --dataset tests/evaluation_datasets/golden_set_v1.json \
      --min-f1 0.80 \
      --no-save
  env:
    AZURE_OPENAI_ENDPOINT: ${{ secrets.AZURE_OPENAI_ENDPOINT }}
    AZURE_OPENAI_API_KEY: ${{ secrets.AZURE_OPENAI_API_KEY }}
```

This will fail the build if the combined F1 score falls below 0.80.

## Creating Test Cases

### From Existing Reddit Posts

1. Copy a post JSON from `tests/test_assets/`
2. Add ground truth labels
3. Include in the dataset

### Test Case Guidelines

- **Balance**: Include both positive (issue) and negative (non-issue) cases
- **Coverage**: Cover all categories and severity levels
- **Edge Cases**: Include ambiguous and borderline cases
- **Diversity**: Vary post lengths, comment counts, and formats

### Edge Case Types

Use `edge_case_type` to tag special cases:
- `"ambiguous"`: Posts that are hard to classify
- `"borderline"`: Posts on the boundary between categories
- `"subtle"`: Issues described indirectly
- `"verbose"`: Very long posts with buried issues
- `null`: Standard test cases

## Metrics Interpretation

### Good Performance
- **Combined F1 ≥ 0.85**: Excellent prompt performance
- **Issue F1 ≥ 0.90**: Great issue detection
- **Precision ≥ 0.85**: Few false positives
- **Recall ≥ 0.85**: Few missed issues

### Warning Signs
- **High FP rate**: Prompt too sensitive, add stricter criteria
- **High FN rate**: Prompt missing issues, add more detection patterns
- **Category confusion**: Clarify category definitions
- **Severity errors**: Add more specific severity criteria

## Workflow: Improving Prompts

1. **Evaluate** the current prompt against the golden dataset
2. **Analyze** failures to identify patterns
3. **Generate** suggestions using the optimizer
4. **Create** a new prompt version with improvements
5. **Compare** the new version against the baseline
6. **Promote** the better version to production

```bash
# Step 1: Evaluate
python -m src.evaluation.runner evaluate -v current -d golden_set.json

# Step 2-3: Review report and suggestions

# Step 4: Create new version (in Python)
manager = PromptVersionManager()
manager.create_variant("current", "v2_improved", {...})

# Step 5: Compare
python -m src.evaluation.runner compare -v current,v2_improved -d golden_set.json

# Step 6: Promote if better
manager.promote_to_current("v2_improved")
```

## API Reference

### EvaluationDataset
- `load(path)`: Load from JSON file
- `save(path)`: Save to JSON file
- `positive_count`: Number of issue cases
- `negative_count`: Number of non-issue cases
- `category_distribution`: Count per category
- `filter_positive_cases()`: Get only issue cases
- `filter_negative_cases()`: Get only non-issue cases

### EvaluationResult
- `issue_classification`: Binary metrics for is_issue
- `category_metrics`: Multi-class metrics for category
- `severity_metrics`: Multi-class metrics for severity
- `combined_f1`: Overall score
- `pass_rate`: Percentage fully correct
- `failures`: List of failure cases
- `failure_breakdown`: Count by failure type

### PromptVersionManager
- `list_versions()`: List all versions
- `load_version(name)`: Load a version
- `save_version(name, prompts)`: Save a version
- `create_variant(base, new, changes)`: Create modified version
- `promote_to_current(version)`: Set as production
- `diff_versions(a, b)`: Compare two versions
