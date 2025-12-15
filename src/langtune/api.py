"""
api.py: High-level API for Langtune

Provides simple, user-friendly functions that work both locally and via server.
"""

import os
from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def finetune(
    training_data: Union[str, List[str]],
    model: str = "llama-7b",
    validation_data: Optional[Union[str, List[str]]] = None,
    use_server: bool = True,
    # Hyperparameters
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    lora_rank: int = 16,
    # Local options
    preset: str = "small",
    output_dir: str = "./output",
    # Server options
    wait: bool = True,
    **kwargs
) -> Union[str, Any]:
    """
    Fine-tune a language model.
    
    By default, runs on Langtrain server. Set use_server=False for local training.
    
    Args:
        training_data: Path to training data or list of texts
        model: Model to fine-tune (server: 'llama-7b', local: preset name)
        validation_data: Optional validation data
        use_server: Use server for training (default: True)
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        lora_rank: LoRA rank
        preset: Local model preset (tiny/small/base/large)
        output_dir: Output directory for local training
        wait: Wait for server job to complete
        **kwargs: Additional arguments
        
    Returns:
        Server: Fine-tuned model ID or Job object
        Local: Trained model
        
    Examples:
        # Server-side fine-tuning
        >>> from langtune import finetune
        >>> model_id = finetune("data.jsonl", model="llama-7b")
        
        # Local fine-tuning
        >>> model = finetune("data.txt", use_server=False, preset="small")
    """
    if use_server:
        return _finetune_server(
            training_data=training_data,
            model=model,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            lora_rank=lora_rank,
            wait=wait,
            **kwargs
        )
    else:
        return _finetune_local(
            training_data=training_data,
            validation_data=validation_data,
            preset=preset,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            lora_rank=lora_rank,
            output_dir=output_dir,
            **kwargs
        )


def _finetune_server(
    training_data: str,
    model: str,
    validation_data: Optional[str],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    lora_rank: int,
    wait: bool,
    **kwargs
):
    """Server-side fine-tuning."""
    from .client import LangtuneClient
    
    client = LangtuneClient()
    
    hyperparameters = {
        "n_epochs": epochs,
        "batch_size": batch_size,
        "learning_rate_multiplier": learning_rate,
        "lora_rank": lora_rank
    }
    
    job = client.create_finetune_job(
        training_file=training_data,
        model=model,
        validation_file=validation_data,
        hyperparameters=hyperparameters
    )
    
    logger.info(f"Created fine-tuning job: {job.id}")
    
    if wait:
        job = client.wait_for_job(job.id)
        if job.error:
            raise RuntimeError(f"Fine-tuning failed: {job.error}")
        logger.info(f"Fine-tuning complete! Model: {job.result_url}")
        return job.result_url
    
    return job


def _finetune_local(
    training_data: Union[str, List[str]],
    validation_data: Optional[Union[str, List[str]]],
    preset: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    lora_rank: int,
    output_dir: str,
    **kwargs
):
    """Local fine-tuning."""
    from .finetune import finetune as local_finetune
    
    return local_finetune(
        train_data=training_data,
        val_data=validation_data,
        preset=preset,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        lora_rank=lora_rank,
        output_dir=output_dir,
        **kwargs
    )


def generate(
    prompt: str,
    model: str = "llama-7b",
    max_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    use_server: bool = True,
    **kwargs
) -> str:
    """
    Generate text completion.
    
    Args:
        prompt: Input prompt
        model: Model to use
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        use_server: Use server for generation
        **kwargs: Additional arguments
        
    Returns:
        Generated text
        
    Example:
        >>> from langtune import generate
        >>> text = generate("Once upon a time", model="llama-7b")
    """
    if use_server:
        from .client import LangtuneClient
        client = LangtuneClient()
        return client.generate(
            prompt=prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
    else:
        raise NotImplementedError("Local generation requires a loaded model. Use TextGenerator.")


def chat(
    messages: List[Dict[str, str]],
    model: str = "llama-7b-chat",
    max_tokens: int = 256,
    temperature: float = 0.7,
    **kwargs
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
        
    Example:
        >>> from langtune import chat
        >>> response = chat([{"role": "user", "content": "Hello!"}])
    """
    from .client import LangtuneClient
    client = LangtuneClient()
    return client.chat(
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        **kwargs
    )


def list_models() -> List[Dict[str, Any]]:
    """
    List available models.
    
    Returns:
        List of model info dicts
    """
    from .client import LangtuneClient
    client = LangtuneClient()
    models = client.list_models()
    return [
        {
            "id": m.id,
            "name": m.name,
            "description": m.description,
            "parameters": m.parameters,
            "supports_finetuning": m.supports_finetuning
        }
        for m in models
    ]


def list_jobs(limit: int = 10) -> List[Dict[str, Any]]:
    """
    List fine-tuning jobs.
    
    Args:
        limit: Maximum number of jobs to return
        
    Returns:
        List of job info dicts
    """
    from .client import LangtuneClient
    client = LangtuneClient()
    jobs = client.list_finetune_jobs(limit=limit)
    return [
        {
            "id": j.id,
            "status": j.status.value,
            "model": j.model,
            "created_at": j.created_at,
            "error": j.error
        }
        for j in jobs
    ]


def get_job(job_id: str) -> Dict[str, Any]:
    """
    Get fine-tuning job status.
    
    Args:
        job_id: Job ID
        
    Returns:
        Job info dict
    """
    from .client import LangtuneClient
    client = LangtuneClient()
    job = client.get_finetune_job(job_id)
    return {
        "id": job.id,
        "status": job.status.value,
        "model": job.model,
        "created_at": job.created_at,
        "completed_at": job.completed_at,
        "error": job.error,
        "result_url": job.result_url,
        "metrics": job.metrics
    }


def cancel_job(job_id: str) -> Dict[str, Any]:
    """
    Cancel a fine-tuning job.
    
    Args:
        job_id: Job ID
        
    Returns:
        Updated job info
    """
    from .client import LangtuneClient
    client = LangtuneClient()
    job = client.cancel_finetune_job(job_id)
    return {"id": job.id, "status": job.status.value}
