"""
client.py: Langtune Server Client

Client SDK for communicating with Langtrain server for heavy computation tasks.
"""

import os
import time
import json
import logging
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

# Default API base URL
DEFAULT_API_BASE = "https://api.langtrain.xyz/v1"


class JobStatus(Enum):
    """Fine-tuning job status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class FineTuneJob:
    """Represents a fine-tuning job."""
    id: str
    status: JobStatus
    model: str
    created_at: str
    updated_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    result_url: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None


@dataclass
class Model:
    """Represents an available model."""
    id: str
    name: str
    description: str
    parameters: int
    context_length: int
    supports_finetuning: bool


@dataclass
class Agent:
    """Represents an AI agent."""
    id: str
    name: str
    workspace_id: str
    description: Optional[str] = None
    model_id: Optional[str] = None
    config: Optional[Dict[str, Any]] = None
    is_active: bool = True
    created_at: Optional[str] = None


@dataclass
class AgentRun:
    """Represents an agent execution run."""
    id: str
    status: str
    agent_id: str
    input: Dict[str, Any]
    output: Optional[Dict[str, Any]] = None
    token_usage: Optional[Dict[str, int]] = None
    latency_ms: Optional[int] = None
    created_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None


@dataclass
class UsageRecord:
    """Represents workspace usage for a period."""
    tokens_used: int
    tokens_limit: int
    finetune_jobs_used: int
    finetune_jobs_limit: int
    agent_runs_used: int
    agent_runs_limit: int
    period_start: str
    period_end: Optional[str] = None


@dataclass
class Plan:
    """Represents a billing plan."""
    id: Optional[str]
    name: str
    code: str
    billing_period: str
    limits: Dict[str, int]


class LangtuneClient:
    """
    Client for Langtrain API.
    
    Handles authentication and communication with the server for:
    - Fine-tuning jobs
    - Text generation
    - Model management
    
    Example:
        >>> client = LangtuneClient()
        >>> job = client.create_finetune_job(
        ...     training_data="path/to/data.jsonl",
        ...     model="llama-7b"
        ... )
        >>> client.wait_for_job(job.id)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 300
    ):
        """
        Initialize the client.
        
        Args:
            api_key: API key (defaults to LANGTUNE_API_KEY env var)
            base_url: API base URL (defaults to https://api.langtrain.xyz/v1)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.environ.get("LANGTUNE_API_KEY")
        self.base_url = (base_url or os.environ.get("LANGTUNE_API_BASE") or DEFAULT_API_BASE).rstrip("/")
        self.timeout = timeout
        
        self._session = None
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "langtune-python/0.1"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    # ==================== API Key Validation ====================
    
    def validate(self) -> Dict[str, Any]:
        """
        Validate the API key and return plan/feature info.
        
        Returns:
            dict with:
                - valid: bool
                - plan: str (free, pro, enterprise)
                - features: list of feature names
                - limits: dict of limits
                - workspace_id: str
                
        Raises:
            APIError: If validation fails
        """
        if not self.api_key:
            return {"valid": False, "error": "No API key configured"}
        
        try:
            response = self._request("POST", "/auth/api-keys/validate", {"api_key": self.api_key})
            return response
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def is_valid(self) -> bool:
        """Check if API key is valid."""
        result = self.validate()
        return result.get("valid", False)
    
    def get_features(self) -> List[str]:
        """Get list of available features for current plan."""
        result = self.validate()
        return result.get("features", [])
    
    def has_feature(self, feature: str) -> bool:
        """Check if a specific feature is available."""
        return feature in self.get_features()
    
    def get_limits(self) -> Dict[str, int]:
        """Get current plan limits."""
        result = self.validate()
        return result.get("limits", {})
    
    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        files: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make an API request."""
        try:
            import requests
        except ImportError:
            raise ImportError("requests library required. Install with: pip install requests")
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            if files:
                # Multipart form data
                response = requests.request(
                    method,
                    url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    files=files,
                    data=data,
                    timeout=self.timeout
                )
            else:
                response = requests.request(
                    method,
                    url,
                    headers=self._get_headers(),
                    json=data,
                    timeout=self.timeout
                )
            
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.HTTPError as e:
            error_msg = str(e)
            try:
                error_data = e.response.json()
                error_msg = error_data.get("error", {}).get("message", str(e))
            except:
                pass
            raise APIError(f"API error: {error_msg}")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Connection error: {e}")
    
    # ==================== Fine-tuning ====================
    
    def create_finetune_job(
        self,
        training_file: str,
        model: str = "llama-7b",
        training_method: str = "qlora",
        validation_file: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        suffix: Optional[str] = None,
        sft_config: Optional[Dict[str, Any]] = None,
        dpo_config: Optional[Dict[str, Any]] = None,
        rlhf_config: Optional[Dict[str, Any]] = None,
    ) -> FineTuneJob:
        """
        Create a fine-tuning job.
        
        Args:
            training_file: Path to training data (JSONL format)
            model: Base model to fine-tune
            training_method: Training method - one of:
                - "sft" (Supervised Fine-Tuning)
                - "dpo" (Direct Preference Optimization)
                - "rlhf" (Reinforcement Learning from Human Feedback)
                - "lora" (LoRA adapters)
                - "qlora" (Quantized LoRA, default)
            validation_file: Optional validation data
            hyperparameters: Training hyperparameters
            suffix: Suffix for the fine-tuned model name
            sft_config: SFT-specific config (packing, dataset_text_field)
            dpo_config: DPO-specific config (beta, loss_type)
            rlhf_config: RLHF-specific config (reward_model, ppo_epochs)
            
        Returns:
            FineTuneJob object
        """
        # Upload training file first
        training_file_id = self._upload_file(training_file, "fine-tune")
        
        data = {
            "training_file": training_file_id,
            "model": model,
            "training_method": training_method
        }
        
        if validation_file:
            val_file_id = self._upload_file(validation_file, "fine-tune")
            data["validation_file"] = val_file_id
        
        if hyperparameters:
            data["hyperparameters"] = hyperparameters
        
        if suffix:
            data["suffix"] = suffix
        
        # Method-specific configs
        if sft_config and training_method == "sft":
            data["sft_config"] = sft_config
        if dpo_config and training_method == "dpo":
            data["dpo_config"] = dpo_config
        if rlhf_config and training_method == "rlhf":
            data["rlhf_config"] = rlhf_config
        
        response = self._request("POST", "/fine-tuning/jobs", data)
        return self._parse_job(response)
    
    def get_finetune_job(self, job_id: str) -> FineTuneJob:
        """Get fine-tuning job status."""
        response = self._request("GET", f"/fine-tuning/jobs/{job_id}")
        return self._parse_job(response)
    
    def list_finetune_jobs(self, limit: int = 10) -> List[FineTuneJob]:
        """List fine-tuning jobs."""
        response = self._request("GET", f"/fine-tuning/jobs?limit={limit}")
        return [self._parse_job(j) for j in response.get("data", [])]
    
    def cancel_finetune_job(self, job_id: str) -> FineTuneJob:
        """Cancel a fine-tuning job."""
        response = self._request("POST", f"/fine-tuning/jobs/{job_id}/cancel")
        return self._parse_job(response)
    
    def wait_for_job(
        self,
        job_id: str,
        poll_interval: int = 30,
        timeout: Optional[int] = None,
        callback: Optional[callable] = None
    ) -> FineTuneJob:
        """
        Wait for a job to complete.
        
        Args:
            job_id: Job ID
            poll_interval: Seconds between status checks
            timeout: Maximum wait time (None for no limit)
            callback: Optional callback function(job) called on each poll
            
        Returns:
            Completed job
        """
        start_time = time.time()
        
        while True:
            job = self.get_finetune_job(job_id)
            
            if callback:
                callback(job)
            
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                return job
            
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Job {job_id} did not complete within {timeout}s")
            
            logger.info(f"Job {job_id} status: {job.status.value}")
            time.sleep(poll_interval)
    
    def _parse_job(self, data: Dict) -> FineTuneJob:
        """Parse job response."""
        return FineTuneJob(
            id=data["id"],
            status=JobStatus(data.get("status", "pending")),
            model=data.get("model", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at"),
            completed_at=data.get("finished_at"),
            error=data.get("error", {}).get("message") if data.get("error") else None,
            result_url=data.get("result_files", [None])[0] if data.get("result_files") else None,
            metrics=data.get("metrics")
        )
    
    # ==================== Files ====================
    
    def _upload_file(self, file_path: str, purpose: str = "fine-tune") -> str:
        """Upload a file and return file ID."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(path, "rb") as f:
            response = self._request(
                "POST",
                "/files",
                data={"purpose": purpose},
                files={"file": (path.name, f)}
            )
        
        return response["id"]
    
    # ==================== Generation ====================
    
    def generate(
        self,
        prompt: str,
        model: str = "llama-7b",
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None
    ) -> str:
        """
        Generate text completion.
        
        Args:
            prompt: Input prompt
            model: Model to use
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences
            
        Returns:
            Generated text
        """
        data = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p
        }
        if stop:
            data["stop"] = stop
        
        response = self._request("POST", "/completions", data)
        return response["choices"][0]["text"]
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "llama-7b-chat",
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> str:
        """
        Chat completion.
        
        Args:
            messages: List of {"role": "user/assistant", "content": "..."}
            model: Model to use
            max_tokens: Maximum tokens
            temperature: Sampling temperature
            
        Returns:
            Assistant response
        """
        data = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        response = self._request("POST", "/chat/completions", data)
        return response["choices"][0]["message"]["content"]
    
    # ==================== Models ====================
    
    def list_models(self) -> List[Model]:
        """List available models."""
        response = self._request("GET", "/models")
        return [
            Model(
                id=m["id"],
                name=m.get("name", m["id"]),
                description=m.get("description", ""),
                parameters=m.get("parameters", 0),
                context_length=m.get("context_length", 4096),
                supports_finetuning=m.get("supports_finetuning", False)
            )
            for m in response.get("data", [])
        ]
    
    def get_model(self, model_id: str) -> Model:
        """Get model details."""
        response = self._request("GET", f"/models/{model_id}")
        return Model(
            id=response["id"],
            name=response.get("name", response["id"]),
            description=response.get("description", ""),
            parameters=response.get("parameters", 0),
            context_length=response.get("context_length", 4096),
            supports_finetuning=response.get("supports_finetuning", False)
        )

    # ==================== Agents ====================
    
    def create_agent(
        self,
        workspace_id: str,
        name: str,
        model_id: Optional[str] = None,
        description: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Agent:
        """
        Create a new agent.
        
        Args:
            workspace_id: Workspace ID
            name: Agent name
            model_id: Optional model ID to use
            description: Agent description
            config: Agent configuration (system_prompt, temperature, tools, etc.)
            
        Returns:
            Agent object
        """
        data = {
            "name": name,
            "description": description or "",
            "model_id": model_id,
            "config": config or {"system_prompt": "You are a helpful assistant.", "temperature": 0.7}
        }
        
        response = self._request("POST", f"/workspaces/{workspace_id}/agents", data)
        return self._parse_agent(response)
    
    def list_agents(self, workspace_id: str) -> List[Agent]:
        """List agents in a workspace."""
        response = self._request("GET", f"/workspaces/{workspace_id}/agents")
        return [self._parse_agent(a) for a in response.get("data", [])]
    
    def get_agent(self, agent_id: str) -> Agent:
        """Get agent details."""
        response = self._request("GET", f"/agents/{agent_id}")
        return self._parse_agent(response)
    
    def update_agent(
        self,
        agent_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        model_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Agent:
        """Update an agent."""
        data = {}
        if name is not None:
            data["name"] = name
        if description is not None:
            data["description"] = description
        if model_id is not None:
            data["model_id"] = model_id
        if config is not None:
            data["config"] = config
        
        response = self._request("PATCH", f"/agents/{agent_id}", data)
        return self._parse_agent(response)
    
    def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent (soft delete)."""
        response = self._request("DELETE", f"/agents/{agent_id}")
        return response.get("success", False)
    
    def run_agent(
        self,
        agent_id: str,
        messages: List[Dict[str, str]],
        params: Optional[Dict[str, Any]] = None
    ) -> AgentRun:
        """
        Execute an agent run.
        
        Args:
            agent_id: Agent ID
            messages: List of {"role": "user/assistant", "content": "..."}
            params: Optional additional parameters
            
        Returns:
            AgentRun with output
        """
        data = {
            "input": {
                "messages": messages,
                **(params or {})
            }
        }
        
        response = self._request("POST", f"/agents/{agent_id}/runs", data)
        return self._parse_agent_run(response)
    
    def list_agent_runs(
        self,
        agent_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[AgentRun]:
        """List runs for an agent."""
        response = self._request("GET", f"/agents/{agent_id}/runs?limit={limit}&offset={offset}")
        return [self._parse_agent_run(r) for r in response.get("data", [])]
    
    def _parse_agent(self, data: Dict) -> Agent:
        """Parse agent response."""
        return Agent(
            id=data["id"],
            name=data.get("name", ""),
            workspace_id=data.get("workspace_id", ""),
            description=data.get("description"),
            model_id=data.get("model_id"),
            config=data.get("config"),
            is_active=data.get("is_active", True),
            created_at=data.get("created_at")
        )
    
    def _parse_agent_run(self, data: Dict) -> AgentRun:
        """Parse agent run response."""
        return AgentRun(
            id=data["id"],
            status=data.get("status", "unknown"),
            agent_id=data.get("agent_id", ""),
            input=data.get("input", {}),
            output=data.get("output"),
            token_usage=data.get("token_usage"),
            latency_ms=data.get("latency_ms"),
            created_at=data.get("created_at"),
            finished_at=data.get("finished_at"),
            error=data.get("error")
        )
    
    # ==================== Billing & Usage ====================
    
    def get_usage(self, workspace_id: str) -> UsageRecord:
        """
        Get current usage for a workspace.
        
        Args:
            workspace_id: Workspace ID
            
        Returns:
            UsageRecord with current usage and limits
        """
        response = self._request("GET", f"/billing/usage?workspace_id={workspace_id}")
        return UsageRecord(
            tokens_used=response.get("tokens", {}).get("used", 0),
            tokens_limit=response.get("tokens", {}).get("limit", 0),
            finetune_jobs_used=response.get("finetune_jobs", {}).get("used", 0),
            finetune_jobs_limit=response.get("finetune_jobs", {}).get("limit", 0),
            agent_runs_used=response.get("agent_runs", {}).get("used", 0),
            agent_runs_limit=response.get("agent_runs", {}).get("limit", 0),
            period_start=response.get("period", {}).get("start", ""),
            period_end=response.get("period", {}).get("end")
        )
    
    def get_plan(self, workspace_id: str) -> Plan:
        """
        Get current plan for a workspace.
        
        Args:
            workspace_id: Workspace ID
            
        Returns:
            Plan with limits
        """
        response = self._request("GET", f"/billing/plan?workspace_id={workspace_id}")
        plan_data = response.get("plan", {})
        return Plan(
            id=plan_data.get("id"),
            name=plan_data.get("name", "Free"),
            code=plan_data.get("code", "free"),
            billing_period=plan_data.get("billing_period", "lifetime"),
            limits=response.get("limits", {})
        )
    
    # ==================== Workspace Finetune Jobs ====================
    
    def create_workspace_finetune_job(
        self,
        workspace_id: str,
        base_model: str,
        dataset_id: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> FineTuneJob:
        """
        Create a finetune job in a workspace.
        
        Args:
            workspace_id: Workspace ID
            base_model: Base model name (e.g., "Llama-3-8B")
            dataset_id: Dataset ID
            name: Optional name for the finetuned model
            config: Training configuration (epochs, lr, batch_size, etc.)
            
        Returns:
            FineTuneJob object
        """
        data = {
            "base_model": base_model,
            "dataset_id": dataset_id,
            "name": name,
            "config": config or {}
        }
        
        response = self._request("POST", f"/workspaces/{workspace_id}/finetune-jobs", data)
        return self._parse_job(response)
    
    def list_workspace_finetune_jobs(self, workspace_id: str) -> List[FineTuneJob]:
        """List finetune jobs in a workspace."""
        response = self._request("GET", f"/workspaces/{workspace_id}/finetune-jobs")
        return [self._parse_job(j) for j in response.get("data", [])]


class APIError(Exception):
    """API error."""
    pass


# Convenience function
def get_client(api_key: Optional[str] = None) -> LangtuneClient:
    """Get a configured client instance."""
    return LangtuneClient(api_key=api_key)
