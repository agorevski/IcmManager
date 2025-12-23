# Development Anti-Patterns Identified

This document catalogs development anti-patterns identified in the IcmManager repository, along with recommendations for improvement.

> **Note:** Items marked with ⏳ are pending (performance-related).

---

## 1. Sequential Processing Where Parallel is Appropriate ⏳

**Status:** Not fixed (performance optimization)

**Location:** `src/analyzers/azure_openai_analyzer.py`

**Issue:** The `analyze_posts_batch` method processes posts sequentially in a loop despite being named "batch."

```python
def analyze_posts_batch(self, posts: List[RedditPost]) -> List[IssueAnalysis]:
    """Analyze multiple posts in batch."""
    # For now, process sequentially
    # Could be optimized with async/parallel processing
    results = []
    for post in posts:
        results.append(self.analyze_post(post))
    return results
```

**Impact:** Significant performance degradation when processing multiple posts, as each LLM API call blocks the next.

**Recommendation:** Use `asyncio` with `aiohttp` or `concurrent.futures.ThreadPoolExecutor` to parallelize API calls. The `ParallelPromptEvaluator` class already demonstrates this pattern.

---

## 2. Deprecated `datetime.utcnow()` Usage

**Status:** Partially fixed

**Remaining Locations:**
- `src/evaluation/metrics.py`
- `src/evaluation/models.py`
- `src/evaluation/prompt_manager.py`

**Issue:** `datetime.utcnow()` is deprecated in Python 3.12+ and returns a naive datetime without timezone info.

**Recommendation:**
```python
from datetime import datetime, timezone

# Change from:
datetime.utcnow()

# To:
datetime.now(timezone.utc)
```

---

## 3. Swallowing Exceptions Silently

**Location:** `src/analyzers/azure_openai_analyzer.py`

**Issue:** Exceptions are caught and a default value returned without re-raising or providing visibility into the failure mode.

```python
except Exception as e:
    self.logger.log_error(request_id, e, {"post_id": post.id})
    # Return a safe default on error
    return IssueAnalysis.no_issue()
```

**Impact:** Silent failures can mask systematic issues. A post failing analysis looks identical to a post with no issue.

**Recommendation:** Consider raising custom exceptions or returning a distinct error state that callers can handle appropriately.

---

## 4. Broad Exception Catching

**Status:** Not fixed (requires careful analysis of exception types)

**Location:** `src/pipeline/issue_detector.py`

**Issue:** Catching bare `Exception` is too broad and can mask programming errors.

```python
except Exception as e:
    error_msg = f"Error processing post {post.id}: {e}"
    logger.error(error_msg)
    result.errors.append(error_msg)
```

**Recommendation:** Catch specific exceptions (e.g., `APIError`, `JSONDecodeError`, `TimeoutError`) and handle them appropriately.

---

## 5. N+1 Query Pattern in Post Filtering ⏳

**Status:** Not fixed (performance optimization)

**Location:** `src/pipeline/issue_detector.py`

**Issue:** The `_filter_new_posts` method calls `is_analyzed()` for each post individually.

```python
def _filter_new_posts(self, posts: List[RedditPost]) -> List[RedditPost]:
    """Filter out posts that have already been analyzed."""
    new_posts = []
    for post in posts:
        if not self.post_tracker.is_analyzed(post.id):
            new_posts.append(post)
    return new_posts
```

**Impact:** For 100 posts, this executes 100 database queries instead of one.

**Recommendation:** Add a batch method to the `IPostTracker` interface:

```python
def are_analyzed(self, post_ids: List[str]) -> Dict[str, bool]:
    """Check multiple posts at once."""
    pass
```

---

## 6. SQL Injection via String Formatting

**Location:** `src/tracking/sqlite_tracker.py` (lines 212-215, 218-221, 224-227, 232-240)

**Issue:** Dynamic SQL query construction using f-strings with `base_where` variable:

```python
base_where = "WHERE 1=1"
# ...
cursor = conn.execute(
    f"SELECT COUNT(*) FROM analyzed_posts {base_where}",
    params
)
```

**Mitigation:** While `base_where` is constructed internally (not from user input), this pattern is fragile and could become vulnerable if modified. Consider using a query builder or ORM.

---

## 13. No Retry Logic for External API Calls ⏳

**Status:** Not fixed (performance/reliability optimization)

**Location:** `src/analyzers/azure_openai_analyzer.py`

**Issue:** API calls to Azure OpenAI have no retry logic for transient failures (rate limits, network issues, etc.).

**Recommendation:** Implement exponential backoff with retry:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def _call_llm(self, messages):
    return self.client.chat.completions.create(...)
```

---

## Summary

| Priority | Anti-Pattern | Status | Impact |
|----------|-------------|--------|--------|
| High | Sequential batch processing | ⏳ Pending | Performance |
| High | No retry logic for APIs | ⏳ Pending | Reliability |
| Medium | N+1 query pattern | ⏳ Pending | Performance |
| Low | Broad exception catching | Noted | Debuggability |
| Low | Silent exception swallowing | Noted | Debuggability |
