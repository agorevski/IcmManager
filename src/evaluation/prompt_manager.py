"""Prompt version management for A/B testing and iteration."""

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any

import yaml

from src.config import AzureOpenAIConfig
from src.analyzers.azure_openai_analyzer import AzureOpenAIAnalyzer

# Default paths
DEFAULT_PROMPTS_DIR = Path(__file__).parent.parent / "analyzers"
DEFAULT_VERSIONS_DIR = DEFAULT_PROMPTS_DIR / "prompt_versions"

class PromptVersionManager:
    """Manages multiple prompt versions for A/B testing and iteration.
    
    Allows loading, saving, and comparing different prompt versions
    to find the most effective prompts for issue detection.
    
    Attributes:
        prompts_dir: Base directory containing prompts.yaml.
        versions_dir: Directory containing versioned prompts.
    """

    def __init__(
        self,
        prompts_dir: Optional[Path] = None,
        versions_dir: Optional[Path] = None,
    ):
        """Initialize the prompt version manager.
        
        Args:
            prompts_dir: Base directory for prompts. Defaults to src/analyzers.
            versions_dir: Directory for versioned prompts. Defaults to src/analyzers/prompt_versions.
        """
        self.prompts_dir = prompts_dir or DEFAULT_PROMPTS_DIR
        self.versions_dir = versions_dir or DEFAULT_VERSIONS_DIR

    def get_current_prompt_path(self) -> Path:
        """Get path to the current production prompt.

        Returns:
            Path to the prompts.yaml file in the prompts directory.
        """
        return self.prompts_dir / "prompts.yaml"

    def get_current_prompt(self) -> Dict[str, Any]:
        """Load the current production prompts.yaml.
        
        Returns:
            Dictionary containing the prompts.
        """
        return self._load_yaml(self.get_current_prompt_path())

    def list_versions(self) -> List[str]:
        """List all available prompt versions.
        
        Returns:
            List of version names (without .yaml extension).
        """
        if not self.versions_dir.exists():
            return []
        
        versions = []
        for path in self.versions_dir.glob("*.yaml"):
            versions.append(path.stem)
        
        return sorted(versions)

    def version_exists(self, version: str) -> bool:
        """Check if a version exists.
        
        Args:
            version: Version name to check.
            
        Returns:
            True if the version exists.
        """
        if version == "current":
            return self.get_current_prompt_path().exists()
        return self._get_version_path(version).exists()

    def load_version(self, version: str) -> Dict[str, Any]:
        """Load a specific prompt version.
        
        Args:
            version: Version name (e.g., "v1_baseline") or "current" for production.
            
        Returns:
            Dictionary containing the prompts.
            
        Raises:
            FileNotFoundError: If the version doesn't exist.
        """
        if version == "current":
            return self.get_current_prompt()
        
        path = self._get_version_path(version)
        if not path.exists():
            raise FileNotFoundError(f"Prompt version '{version}' not found at {path}")
        
        return self._load_yaml(path)

    def save_version(
        self,
        version: str,
        prompts: Dict[str, Any],
        description: str = "",
        overwrite: bool = False,
    ) -> Path:
        """Save a new prompt version.
        
        Args:
            version: Version name (e.g., "v2_improved_categories").
            prompts: Dictionary containing the prompts.
            description: Optional description added as a comment.
            overwrite: Whether to overwrite existing version.
            
        Returns:
            Path to the saved version file.
            
        Raises:
            ValueError: If version exists and overwrite is False.
        """
        path = self._get_version_path(version)
        
        if path.exists() and not overwrite:
            raise ValueError(
                f"Version '{version}' already exists. Use overwrite=True to replace."
            )
        
        # Ensure directory exists
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        prompts_with_meta = prompts.copy()
        if "_metadata" not in prompts_with_meta:
            prompts_with_meta["_metadata"] = {}
        
        prompts_with_meta["_metadata"].update({
            "version": version,
            "description": description,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        
        self._save_yaml(path, prompts_with_meta)
        return path

    def create_variant(
        self,
        base_version: str,
        new_version: str,
        changes: Dict[str, str],
        description: str = "",
    ) -> Path:
        """Create a new version based on an existing one with specific changes.
        
        Args:
            base_version: Version to base the new one on.
            new_version: Name for the new version.
            changes: Dictionary of prompt keys to new values.
            description: Description of the changes.
            
        Returns:
            Path to the new version file.
        """
        # Load base version
        base_prompts = self.load_version(base_version)
        
        # Apply changes
        for key, value in changes.items():
            base_prompts[key] = value
        
        # Update metadata
        if "_metadata" not in base_prompts:
            base_prompts["_metadata"] = {}
        
        base_prompts["_metadata"]["based_on"] = base_version
        
        return self.save_version(new_version, base_prompts, description)

    def promote_to_current(self, version: str, backup: bool = True) -> None:
        """Promote a version to be the current production prompt.
        
        Args:
            version: Version to promote.
            backup: Whether to backup the current version first.
        """
        current_path = self.get_current_prompt_path()
        
        # Backup current if requested
        if backup and current_path.exists():
            backup_version = f"backup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
            current_prompts = self.get_current_prompt()
            self.save_version(backup_version, current_prompts, "Automatic backup before promotion")
        
        # Load and save the version as current
        new_prompts = self.load_version(version)
        
        # Remove metadata before saving as current
        if "_metadata" in new_prompts:
            del new_prompts["_metadata"]
        
        self._save_yaml(current_path, new_prompts)

    def delete_version(self, version: str) -> None:
        """Delete a prompt version.
        
        Args:
            version: Version to delete.
            
        Raises:
            ValueError: If trying to delete "current".
            FileNotFoundError: If version doesn't exist.
        """
        if version == "current":
            raise ValueError("Cannot delete the current production prompt")
        
        path = self._get_version_path(version)
        if not path.exists():
            raise FileNotFoundError(f"Version '{version}' not found")
        
        path.unlink()

    def get_version_metadata(self, version: str) -> Dict[str, Any]:
        """Get metadata for a version.
        
        Args:
            version: Version to get metadata for.
            
        Returns:
            Metadata dictionary.
        """
        prompts = self.load_version(version)
        return prompts.get("_metadata", {})

    def create_analyzer_for_version(
        self,
        version: str,
        config: AzureOpenAIConfig,
    ) -> AzureOpenAIAnalyzer:
        """Create an analyzer using a specific prompt version.
        
        Args:
            version: Prompt version to use.
            config: Azure OpenAI configuration.
            
        Returns:
            Configured AzureOpenAIAnalyzer.
        """
        # Save the version to a temporary file
        prompts = self.load_version(version)
        
        # Remove metadata for analyzer
        if "_metadata" in prompts:
            prompts = {k: v for k, v in prompts.items() if k != "_metadata"}
        
        # Create a temporary prompts file
        temp_dir = self.versions_dir / "_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / f"{version}_temp.yaml"
        
        self._save_yaml(temp_path, prompts)
        
        try:
            return AzureOpenAIAnalyzer(
                azure_endpoint=config.endpoint,
                api_key=config.api_key,
                api_version=config.api_version,
                deployment_name=config.deployment_name,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                prompts_file=temp_path,
            )
        finally:
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()
            # Remove temp dir if empty
            if temp_dir.exists() and not any(temp_dir.iterdir()):
                temp_dir.rmdir()

    def diff_versions(
        self,
        version_a: str,
        version_b: str,
    ) -> Dict[str, Dict[str, str]]:
        """Compare two prompt versions and show differences.
        
        Args:
            version_a: First version.
            version_b: Second version.
            
        Returns:
            Dictionary with keys that differ, showing both values.
        """
        prompts_a = self.load_version(version_a)
        prompts_b = self.load_version(version_b)
        
        # Get all keys (excluding metadata)
        keys_a = set(k for k in prompts_a.keys() if not k.startswith("_"))
        keys_b = set(k for k in prompts_b.keys() if not k.startswith("_"))
        all_keys = keys_a | keys_b
        
        diff = {}
        for key in all_keys:
            val_a = prompts_a.get(key)
            val_b = prompts_b.get(key)
            
            if val_a != val_b:
                diff[key] = {
                    version_a: val_a,
                    version_b: val_b,
                }
        
        return diff

    def _get_version_path(self, version: str) -> Path:
        """Get the path for a version file.

        Args:
            version: Version name to get path for.

        Returns:
            Path to the version YAML file.
        """
        return self.versions_dir / f"{version}.yaml"

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load a YAML file.

        Args:
            path: Path to the YAML file to load.

        Returns:
            Dictionary containing the parsed YAML content.
        """
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _save_yaml(self, path: Path, data: Dict[str, Any]) -> None:
        """Save data to a YAML file.

        Args:
            path: Path to the YAML file to save.
            data: Dictionary data to save as YAML.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
