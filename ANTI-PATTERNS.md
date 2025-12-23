# Development Anti-Patterns Identified

This document catalogs development anti-patterns identified in the IcmManager repository, along with recommendations for improvement.

> **Note:** Items marked with ⏳ are pending (performance-related).


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
| High | No retry logic for APIs | ⏳ Pending | Reliability |
| Medium | N+1 query pattern | ⏳ Pending | Performance |
| Low | Broad exception catching | Noted | Debuggability |
