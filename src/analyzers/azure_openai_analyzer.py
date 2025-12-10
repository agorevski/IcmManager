"""Azure OpenAI implementation of the LLM analyzer interface."""

import json
import time
from pathlib import Path
from typing import List, Optional

import yaml
from openai import AzureOpenAI

from src.interfaces.llm_analyzer import ILLMAnalyzer
from src.models.reddit_data import RedditPost
from src.models.issue import IssueAnalysis, ICMIssue, IssueCategory, Severity
from src.logging.llm_logger import LLMLogger

# Path to the prompts YAML file
PROMPTS_FILE = Path(__file__).parent / "prompts.yaml"


def load_prompts(prompts_file: Path = PROMPTS_FILE) -> dict:
    """Load prompts from the YAML file.
    
    Args:
        prompts_file: Path to the prompts YAML file.
        
    Returns:
        Dictionary containing the prompts.
    """
    with open(prompts_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class AzureOpenAIAnalyzer(ILLMAnalyzer):
    """Azure OpenAI-based implementation for analyzing Reddit posts.
    
    Uses Azure OpenAI's GPT models to analyze posts and detect issues.
    Prompts are loaded from prompts.yaml for easy customization.
    
    Attributes:
        client: Azure OpenAI client instance.
        model: Model deployment name to use.
        logger: LLM logger for tracking requests/responses.
        prompts: Dictionary containing loaded prompts.
    """

    def __init__(
        self,
        azure_endpoint: str,
        api_key: str,
        api_version: str = "2024-02-01",
        deployment_name: str = "gpt-5.1",
        logger: Optional[LLMLogger] = None,
        max_tokens: int = 1000,
        temperature: float = 0.1,
        prompts_file: Optional[Path] = None
    ):
        """Initialize the Azure OpenAI analyzer.
        
        Args:
            azure_endpoint: Azure OpenAI endpoint URL.
            api_key: Azure OpenAI API key.
            api_version: API version to use.
            deployment_name: Name of the model deployment.
            logger: Optional LLM logger instance.
            max_tokens: Maximum tokens for responses.
            temperature: Temperature for generation (lower = more deterministic).
            prompts_file: Optional path to prompts YAML file.
        """
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        self.model = deployment_name
        self.logger = logger or LLMLogger()
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Load prompts from YAML file
        self.prompts = load_prompts(prompts_file or PROMPTS_FILE)
        self.system_prompt = self.prompts["system_prompt"]
        self.duplicate_check_prompt_template = self.prompts["duplicate_check_prompt"]

    def analyze_post(self, post: RedditPost) -> IssueAnalysis:
        """Analyze a Reddit post and its comments to detect issues."""
        request_id = self.logger.generate_request_id()
        
        # Build the prompt with post content
        user_content = self._build_post_prompt(post)
        
        # Log the request
        self.logger.log_request(
            request_id=request_id,
            model=self.model,
            prompt=user_content,
            context={"system_prompt": self.system_prompt},
            post_id=post.id,
            subreddit=post.subreddit,
        )
        
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_content},
                ],
                max_completion_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
            
            latency_ms = (time.time() - start_time) * 1000
            response_text = response.choices[0].message.content
            
            # Log the response
            self.logger.log_response(
                request_id=request_id,
                response=response_text,
                tokens_used=response.usage.total_tokens if response.usage else None,
                prompt_tokens=response.usage.prompt_tokens if response.usage else None,
                completion_tokens=response.usage.completion_tokens if response.usage else None,
                latency_ms=latency_ms,
                model=self.model,
            )
            
            # Parse the response
            analysis = self._parse_analysis_response(response_text, post)
            
            # Log the analysis result
            self.logger.log_analysis_result(
                request_id=request_id,
                post_id=post.id,
                is_issue=analysis.is_issue,
                confidence=analysis.confidence,
                category=analysis.category,
                severity=analysis.severity,
                summary=analysis.summary,
            )
            
            return analysis
            
        except Exception as e:
            self.logger.log_error(request_id, e, {"post_id": post.id})
            # Return a safe default on error
            return IssueAnalysis.no_issue()

    def analyze_posts_batch(self, posts: List[RedditPost]) -> List[IssueAnalysis]:
        """Analyze multiple posts in batch."""
        # For now, process sequentially
        # Could be optimized with async/parallel processing
        results = []
        for post in posts:
            results.append(self.analyze_post(post))
        return results

    def check_duplicate(
        self,
        analysis: IssueAnalysis,
        existing_issues: List[ICMIssue]
    ) -> bool:
        """Check if an analysis represents a duplicate of an existing issue."""
        if not existing_issues:
            return False
        
        request_id = self.logger.generate_request_id()
        
        # Format the issues for comparison
        new_issue_str = json.dumps({
            "summary": analysis.summary,
            "category": analysis.category,
            "severity": analysis.severity,
            "keywords": analysis.keywords,
        }, indent=2)
        
        existing_issues_str = json.dumps([
            {
                "id": issue.id,
                "title": issue.title,
                "category": issue.category,
                "description": issue.description[:500],  # Truncate for token efficiency
            }
            for issue in existing_issues[:20]  # Limit to recent issues
        ], indent=2)
        
        prompt = self.duplicate_check_prompt_template.format(
            new_issue=new_issue_str,
            existing_issues=existing_issues_str,
        )
        
        self.logger.log_request(
            request_id=request_id,
            model=self.model,
            prompt=prompt,
            context={"type": "duplicate_check"},
        )
        
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
            
            latency_ms = (time.time() - start_time) * 1000
            response_text = response.choices[0].message.content
            
            self.logger.log_response(
                request_id=request_id,
                response=response_text,
                tokens_used=response.usage.total_tokens if response.usage else None,
                latency_ms=latency_ms,
            )
            
            result = json.loads(response_text)
            return result.get("is_duplicate", False)
            
        except Exception as e:
            self.logger.log_error(request_id, e)
            # On error, assume not a duplicate (safer to create than miss)
            return False

    def get_model_name(self) -> str:
        """Get the name/identifier of the LLM model being used."""
        return self.model

    def _build_post_prompt(self, post: RedditPost) -> str:
        """Build the user prompt from a Reddit post."""
        parts = [
            f"Subreddit: r/{post.subreddit}",
            f"Title: {post.title}",
            f"Score: {post.score} | Comments: {post.num_comments}",
        ]
        
        if post.flair:
            parts.append(f"Flair: {post.flair}")
        
        if post.body:
            parts.append(f"\nPost Body:\n{post.body[:2000]}")  # Truncate long posts
        
        if post.comments:
            parts.append("\n--- Comments ---")
            comment_text = post.get_comment_thread_text(max_comments=30)
            parts.append(comment_text[:3000])  # Truncate long comment threads
        
        return "\n".join(parts)

    def _parse_analysis_response(
        self, 
        response_text: str, 
        post: RedditPost
    ) -> IssueAnalysis:
        """Parse the LLM response into an IssueAnalysis object."""
        try:
            data = json.loads(response_text)
            
            # Validate and normalize category
            category = data.get("category", "other").lower()
            valid_categories = [c.value for c in IssueCategory]
            if category not in valid_categories:
                category = "other"
            
            # Validate and normalize severity
            severity = data.get("severity", "low").lower()
            valid_severities = [s.value for s in Severity]
            if severity not in valid_severities:
                severity = "low"
            
            # Estimate affected users based on engagement if not provided
            affected_users = data.get("affected_users_estimate", 1)
            if affected_users < 1:
                # Estimate based on post engagement
                affected_users = max(1, post.score // 10 + post.num_comments // 5)
            
            return IssueAnalysis(
                is_issue=data.get("is_issue", False),
                confidence=min(1.0, max(0.0, data.get("confidence", 0.5))),
                summary=data.get("summary", "No summary provided"),
                category=category,
                severity=severity,
                affected_users_estimate=affected_users,
                keywords=data.get("keywords", []),
                raw_response=response_text,
            )
            
        except (json.JSONDecodeError, KeyError) as e:
            # Return no-issue on parse failure
            return IssueAnalysis(
                is_issue=False,
                confidence=0.0,
                summary=f"Failed to parse LLM response: {e}",
                category="other",
                severity="low",
                raw_response=response_text,
            )
