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
DEFAULT_API_BASE = "https://api.langtrain.ai/v1"


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
            base_url: API base URL (defaults to https://api.langtrain.ai/v1)
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
        validation_file: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        suffix: Optional[str] = None
    ) -> FineTuneJob:
        """
        Create a fine-tuning job.
        
        Args:
            training_file: Path to training data (JSONL format)
            model: Base model to fine-tune
            validation_file: Optional validation data
            hyperparameters: Training hyperparameters
            suffix: Suffix for the fine-tuned model name
            
        Returns:
            FineTuneJob object
        """
        # Upload training file first
        training_file_id = self._upload_file(training_file, "fine-tune")
        
        data = {
            "training_file": training_file_id,
            "model": model
        }
        
        if validation_file:
            val_file_id = self._upload_file(validation_file, "fine-tune")
            data["validation_file"] = val_file_id
        
        if hyperparameters:
            data["hyperparameters"] = hyperparameters
        
        if suffix:
            data["suffix"] = suffix
        
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


class APIError(Exception):
    """API error."""
    pass


# Convenience function
def get_client(api_key: Optional[str] = None) -> LangtuneClient:
    """Get a configured client instance."""
    return LangtuneClient(api_key=api_key)
