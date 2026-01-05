"""Tests for the Azure OpenAI analyzer, particularly batch processing."""

import pytest
import time
import threading
from datetime import datetime
from typing import List
from unittest.mock import Mock, patch, MagicMock

from src.models.reddit_data import RedditPost
from src.models.issue import IssueAnalysis
from src.analyzers.azure_openai_analyzer import AzureOpenAIAnalyzer


class TestAnalyzePostsBatch:
    """Tests for the analyze_posts_batch parallel processing."""

    def create_test_posts(self, count: int) -> List[RedditPost]:
        """Create a list of test posts for testing purposes.

        Args:
            count: The number of test posts to create.

        Returns:
            A list of RedditPost objects with sequential IDs and test data.
        """
        return [
            RedditPost(
                id=f"post_{i}",
                subreddit="xbox",
                title=f"Test post {i}",
                body=f"Test body {i}",
                author=f"user_{i}",
                created_utc=datetime(2024, 1, 15, 10, 0, 0),
                url=f"https://reddit.com/r/xbox/comments/post_{i}",
                score=10 * i,
                comments=[],
                num_comments=i,
            )
            for i in range(count)
        ]

    @patch('src.analyzers.azure_openai_analyzer.AzureOpenAI')
    @patch('src.analyzers.azure_openai_analyzer.load_prompts')
    def test_batch_returns_correct_number_of_results(
        self, mock_load_prompts, mock_azure_client
    ):
        """Test that batch processing returns one result per input post.

        Args:
            mock_load_prompts: Mock for the load_prompts function.
            mock_azure_client: Mock for the AzureOpenAI client.
        """
        # Setup mock prompts
        mock_load_prompts.return_value = {
            "system_prompt": "Test system prompt",
            "duplicate_check_prompt": "Test duplicate prompt",
        }
        
        # Setup mock API response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"is_issue": false, "confidence": 0.5, "summary": "No issue", "category": "other", "severity": "low"}'
        mock_response.usage.total_tokens = 100
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 50
        
        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_azure_client.return_value = mock_client_instance
        
        analyzer = AzureOpenAIAnalyzer(
            azure_endpoint="https://test.openai.azure.com",
            api_key="test-key",
        )
        
        posts = self.create_test_posts(5)
        results = analyzer.analyze_posts_batch(posts)
        
        assert len(results) == 5
        assert all(isinstance(r, IssueAnalysis) for r in results)

    @patch('src.analyzers.azure_openai_analyzer.AzureOpenAI')
    @patch('src.analyzers.azure_openai_analyzer.load_prompts')
    def test_batch_preserves_order(self, mock_load_prompts, mock_azure_client):
        """Test that batch processing preserves the order of results.

        Verifies that when posts are processed in parallel, the results
        are returned in the same order as the input posts.

        Args:
            mock_load_prompts: Mock for the load_prompts function.
            mock_azure_client: Mock for the AzureOpenAI client.
        """
        mock_load_prompts.return_value = {
            "system_prompt": "Test system prompt",
            "duplicate_check_prompt": "Test duplicate prompt",
        }
        
        # Create responses that include the post ID in the summary
        def create_response(post_id: str):
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = f'{{"is_issue": true, "confidence": 0.8, "summary": "Issue for {post_id}", "category": "other", "severity": "low"}}'
            mock_response.usage.total_tokens = 100
            mock_response.usage.prompt_tokens = 50
            mock_response.usage.completion_tokens = 50
            return mock_response
        
        # Track which post is being processed
        call_order = []
        
        def mock_create(*args, **kwargs):
            # Extract post ID from the user message
            messages = kwargs.get('messages', [])
            user_msg = messages[1]['content'] if len(messages) > 1 else ''
            # Find post_X pattern in the message
            for i in range(10):
                if f"post_{i}" in user_msg or f"Test post {i}" in user_msg:
                    call_order.append(i)
                    # Add small random delay to simulate API latency
                    time.sleep(0.01)
                    return create_response(f"post_{i}")
            return create_response("unknown")
        
        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.side_effect = mock_create
        mock_azure_client.return_value = mock_client_instance
        
        analyzer = AzureOpenAIAnalyzer(
            azure_endpoint="https://test.openai.azure.com",
            api_key="test-key",
        )
        
        posts = self.create_test_posts(5)
        results = analyzer.analyze_posts_batch(posts)
        
        # Verify results are in the correct order (matching input posts)
        for i, result in enumerate(results):
            assert f"post_{i}" in result.summary, f"Result {i} should be for post_{i}, got: {result.summary}"

    @patch('src.analyzers.azure_openai_analyzer.AzureOpenAI')
    @patch('src.analyzers.azure_openai_analyzer.load_prompts')
    def test_batch_runs_in_parallel(self, mock_load_prompts, mock_azure_client):
        """Test that batch processing actually runs in parallel.

        Verifies parallelism by measuring execution time. Sequential
        processing would take ~0.5s (5 * 0.1s), while parallel should
        complete in ~0.1-0.2s.

        Args:
            mock_load_prompts: Mock for the load_prompts function.
            mock_azure_client: Mock for the AzureOpenAI client.
        """
        mock_load_prompts.return_value = {
            "system_prompt": "Test system prompt",
            "duplicate_check_prompt": "Test duplicate prompt",
        }
        
        # Track concurrent execution
        active_threads = []
        max_concurrent = [0]
        lock = threading.Lock()
        
        def mock_create(*args, **kwargs):
            with lock:
                active_threads.append(threading.current_thread().ident)
                current_concurrent = len(set(active_threads))
                max_concurrent[0] = max(max_concurrent[0], current_concurrent)
            
            # Simulate API latency
            time.sleep(0.1)
            
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = '{"is_issue": false, "confidence": 0.5, "summary": "No issue", "category": "other", "severity": "low"}'
            mock_response.usage.total_tokens = 100
            mock_response.usage.prompt_tokens = 50
            mock_response.usage.completion_tokens = 50
            
            with lock:
                active_threads.remove(threading.current_thread().ident)
            
            return mock_response
        
        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.side_effect = mock_create
        mock_azure_client.return_value = mock_client_instance
        
        analyzer = AzureOpenAIAnalyzer(
            azure_endpoint="https://test.openai.azure.com",
            api_key="test-key",
        )
        
        posts = self.create_test_posts(5)
        
        start_time = time.time()
        results = analyzer.analyze_posts_batch(posts, max_workers=5)
        elapsed_time = time.time() - start_time
        
        # If running sequentially, it would take ~0.5 seconds (5 * 0.1s)
        # With parallelism, it should be closer to 0.1-0.2 seconds
        assert elapsed_time < 0.4, f"Batch processing took {elapsed_time}s, expected parallel execution"
        assert len(results) == 5

    @patch('src.analyzers.azure_openai_analyzer.AzureOpenAI')
    @patch('src.analyzers.azure_openai_analyzer.load_prompts')
    def test_batch_respects_max_workers(self, mock_load_prompts, mock_azure_client):
        """Test that max_workers parameter limits concurrency.

        Verifies that the number of concurrent API calls never exceeds
        the max_workers limit by tracking active threads during execution.

        Args:
            mock_load_prompts: Mock for the load_prompts function.
            mock_azure_client: Mock for the AzureOpenAI client.
        """
        mock_load_prompts.return_value = {
            "system_prompt": "Test system prompt",
            "duplicate_check_prompt": "Test duplicate prompt",
        }
        
        # Track maximum concurrent calls
        active_count = [0]
        max_concurrent = [0]
        lock = threading.Lock()
        
        def mock_create(*args, **kwargs):
            with lock:
                active_count[0] += 1
                max_concurrent[0] = max(max_concurrent[0], active_count[0])
            
            time.sleep(0.05)
            
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = '{"is_issue": false, "confidence": 0.5, "summary": "No issue", "category": "other", "severity": "low"}'
            mock_response.usage.total_tokens = 100
            mock_response.usage.prompt_tokens = 50
            mock_response.usage.completion_tokens = 50
            
            with lock:
                active_count[0] -= 1
            
            return mock_response
        
        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.side_effect = mock_create
        mock_azure_client.return_value = mock_client_instance
        
        analyzer = AzureOpenAIAnalyzer(
            azure_endpoint="https://test.openai.azure.com",
            api_key="test-key",
        )
        
        posts = self.create_test_posts(10)
        results = analyzer.analyze_posts_batch(posts, max_workers=2)
        
        assert len(results) == 10
        # Max concurrent should not exceed max_workers
        assert max_concurrent[0] <= 2, f"Max concurrent was {max_concurrent[0]}, expected <= 2"

    @patch('src.analyzers.azure_openai_analyzer.AzureOpenAI')
    @patch('src.analyzers.azure_openai_analyzer.load_prompts')
    def test_batch_empty_list(self, mock_load_prompts, mock_azure_client):
        """Test that batch processing handles empty list correctly.

        Verifies that passing an empty list of posts returns an empty
        list of results without errors.

        Args:
            mock_load_prompts: Mock for the load_prompts function.
            mock_azure_client: Mock for the AzureOpenAI client.
        """
        mock_load_prompts.return_value = {
            "system_prompt": "Test system prompt",
            "duplicate_check_prompt": "Test duplicate prompt",
        }
        
        mock_azure_client.return_value = MagicMock()
        
        analyzer = AzureOpenAIAnalyzer(
            azure_endpoint="https://test.openai.azure.com",
            api_key="test-key",
        )
        
        results = analyzer.analyze_posts_batch([])
        
        assert results == []

    @patch('src.analyzers.azure_openai_analyzer.AzureOpenAI')
    @patch('src.analyzers.azure_openai_analyzer.load_prompts')
    def test_batch_single_post(self, mock_load_prompts, mock_azure_client):
        """Test that batch processing works with a single post.

        Verifies that the batch processing correctly handles the edge
        case of processing just one post and returns the expected result.

        Args:
            mock_load_prompts: Mock for the load_prompts function.
            mock_azure_client: Mock for the AzureOpenAI client.
        """
        mock_load_prompts.return_value = {
            "system_prompt": "Test system prompt",
            "duplicate_check_prompt": "Test duplicate prompt",
        }
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"is_issue": true, "confidence": 0.9, "summary": "Single post issue", "category": "performance", "severity": "high"}'
        mock_response.usage.total_tokens = 100
        mock_response.usage.prompt_tokens = 50
        mock_response.usage.completion_tokens = 50
        
        mock_client_instance = MagicMock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_azure_client.return_value = mock_client_instance
        
        analyzer = AzureOpenAIAnalyzer(
            azure_endpoint="https://test.openai.azure.com",
            api_key="test-key",
        )
        
        posts = self.create_test_posts(1)
        results = analyzer.analyze_posts_batch(posts)
        
        assert len(results) == 1
        assert results[0].is_issue is True
        assert results[0].summary == "Single post issue"
