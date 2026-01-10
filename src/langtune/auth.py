"""
auth.py: API Key authentication and usage tracking for Langtune

Users must obtain an API key from https://langtrain.xyz to use this package.
"""

import os
import sys
import json
import hashlib
import time
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

# Try to import requests for API calls
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Try to import rich for beautiful output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

# API configuration
API_BASE_URL = "https://api.langtrain.xyz"
AUTH_ENDPOINT = f"{API_BASE_URL}/v1/auth/verify"
USAGE_ENDPOINT = f"{API_BASE_URL}/v1/usage"

# Config paths
CONFIG_DIR = Path.home() / ".langtune"
CONFIG_FILE = CONFIG_DIR / "config.json"
CACHE_FILE = CONFIG_DIR / ".auth_cache"

# Environment variable names
API_KEY_ENV = "LANGTUNE_API_KEY"


class AuthenticationError(Exception):
    """Raised when API key authentication fails."""
    pass


class UsageLimitError(Exception):
    """Raised when usage limit is exceeded."""
    pass


def _get_config_dir() -> Path:
    """Get or create the config directory."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return CONFIG_DIR


def _load_config() -> Dict[str, Any]:
    """Load configuration from file."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save_config(config: Dict[str, Any]) -> None:
    """Save configuration to file."""
    _get_config_dir()
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


def _load_auth_cache() -> Dict[str, Any]:
    """Load cached authentication data."""
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def _save_auth_cache(cache: Dict[str, Any]) -> None:
    """Save authentication cache."""
    _get_config_dir()
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)


def get_api_key() -> Optional[str]:
    """
    Get the API key from environment or config file.
    
    Priority:
    1. LANGTUNE_API_KEY environment variable
    2. Config file (~/.langtune/config.json)
    """
    # Check environment variable first
    api_key = os.environ.get(API_KEY_ENV)
    if api_key:
        return api_key
    
    # Check config file
    config = _load_config()
    return config.get("api_key")


def set_api_key(api_key: str) -> None:
    """Save API key to config file."""
    config = _load_config()
    config["api_key"] = api_key
    _save_config(config)
    
    if RICH_AVAILABLE:
        console.print("[green]✓[/] API key saved to ~/.langtune/config.json")
    else:
        print("✓ API key saved to ~/.langtune/config.json")


def _hash_key(api_key: str) -> str:
    """Hash API key for cache lookup."""
    return hashlib.sha256(api_key.encode()).hexdigest()[:16]


def verify_api_key(api_key: str, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Verify API key with the Langtrain API.
    
    Returns user info and usage limits on success.
    Raises AuthenticationError on failure.
    """
    # Check cache first (valid for 1 hour)
    cache = _load_auth_cache()
    key_hash = _hash_key(api_key)
    
    if not force_refresh and key_hash in cache:
        cached_data = cache[key_hash]
        cache_time = cached_data.get("cached_at", 0)
        if time.time() - cache_time < 3600:  # 1 hour cache
            return cached_data.get("data", {})
    
    # For now, simulate API verification (offline mode)
    # In production, this would make an actual API call
    if not REQUESTS_AVAILABLE:
        # Offline verification - accept keys that match pattern
        if api_key.startswith("lt_") and len(api_key) >= 32:
            user_data = {
                "valid": True,
                "user_id": key_hash,
                "plan": "free",
                "usage": {
                    "tokens_used": 0,
                    "tokens_limit": 100000,
                    "requests_used": 0,
                    "requests_limit": 1000
                },
                "offline_mode": True
            }
            # Cache the result
            cache[key_hash] = {
                "cached_at": time.time(),
                "data": user_data
            }
            _save_auth_cache(cache)
            return user_data
        else:
            raise AuthenticationError(
                "Invalid API key format. Keys should start with 'lt_' and be at least 32 characters.\n"
                "Get your API key at: https://app.langtrain.xyz"
            )
    
    # Make API call to verify key
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.post(AUTH_ENDPOINT, headers=headers, timeout=10)
        
        if response.status_code == 200:
            user_data = response.json()
            # Cache the result
            cache[key_hash] = {
                "cached_at": time.time(),
                "data": user_data
            }
            _save_auth_cache(cache)
            return user_data
        elif response.status_code == 401:
            raise AuthenticationError(
                "Invalid API key. Please check your key at: https://app.langtrain.xyz"
            )
        elif response.status_code == 403:
            raise UsageLimitError(
                "API key is valid but access is denied. Your subscription may have expired.\n"
                "Manage your subscription at: https://billing.langtrain.xyz"
            )
        else:
            raise AuthenticationError(
                f"Authentication failed with status {response.status_code}. "
                "Please try again or contact support."
            )
    except requests.exceptions.RequestException as e:
        # If we can't reach the API, use cached data if available
        if key_hash in cache:
            return cache[key_hash].get("data", {})
        raise AuthenticationError(
            f"Could not verify API key: {e}\n"
            "Please check your internet connection."
        )


def check_usage(api_key: str) -> Dict[str, Any]:
    """Check current usage against limits."""
    user_data = verify_api_key(api_key)
    usage = user_data.get("usage", {})
    
    tokens_used = usage.get("tokens_used", 0)
    tokens_limit = usage.get("tokens_limit", 100000)
    requests_used = usage.get("requests_used", 0)
    requests_limit = usage.get("requests_limit", 1000)
    
    if tokens_used >= tokens_limit:
        raise UsageLimitError(
            f"Token limit exceeded ({tokens_used:,}/{tokens_limit:,}).\n"
            "Upgrade your plan at: https://billing.langtrain.xyz"
        )
    
    if requests_used >= requests_limit:
        raise UsageLimitError(
            f"Request limit exceeded ({requests_used:,}/{requests_limit:,}).\n"
            "Upgrade your plan at: https://billing.langtrain.xyz"
        )
    
    return {
        "tokens_used": tokens_used,
        "tokens_limit": tokens_limit,
        "tokens_remaining": tokens_limit - tokens_used,
        "requests_used": requests_used,
        "requests_limit": requests_limit,
        "requests_remaining": requests_limit - requests_used,
        "plan": user_data.get("plan", "free")
    }


def require_auth(func):
    """Decorator to require API key authentication for a function."""
    def wrapper(*args, **kwargs):
        api_key = get_api_key()
        
        if not api_key:
            _print_auth_required()
            sys.exit(1)
        
        try:
            verify_api_key(api_key)
        except AuthenticationError as e:
            _print_auth_error(str(e))
            sys.exit(1)
        except UsageLimitError as e:
            _print_usage_error(str(e))
            sys.exit(1)
        
        return func(*args, **kwargs)
    
    return wrapper


def _print_auth_required():
    """Print authentication required message."""
    if RICH_AVAILABLE:
        text = Text()
        text.append("\n🔐 ", style="")
        text.append("API Key Required\n\n", style="bold red")
        text.append("Langtune requires an API key to run. Get your free key at:\n", style="")
        text.append("https://app.langtrain.xyz\n\n", style="blue underline")
        text.append("Once you have your key, authenticate using:\n\n", style="")
        text.append("  langtune auth login\n\n", style="cyan")
        text.append("Or set the environment variable:\n\n", style="")
        text.append(f"  export {API_KEY_ENV}=lt_your_api_key_here\n", style="cyan")
        
        panel = Panel(text, title="[bold]Authentication Required[/]", border_style="red", box=box.ROUNDED)
        console.print(panel)
    else:
        print(f"""
🔐 API Key Required

Langtune requires an API key to run. Get your free key at:
https://app.langtrain.xyz

Once you have your key, authenticate using:

  langtune auth login

Or set the environment variable:

  export {API_KEY_ENV}=lt_your_api_key_here
""")


def _print_auth_error(message: str):
    """Print authentication error message."""
    if RICH_AVAILABLE:
        console.print(f"\n[bold red]❌ Authentication Error[/]\n")
        console.print(f"[red]{message}[/]\n")
    else:
        print(f"\n❌ Authentication Error\n")
        print(f"{message}\n")


def _print_usage_error(message: str):
    """Print usage limit error message."""
    if RICH_AVAILABLE:
        console.print(f"\n[bold yellow]⚠️ Usage Limit Reached[/]\n")
        console.print(f"[yellow]{message}[/]\n")
    else:
        print(f"\n⚠️ Usage Limit Reached\n")
        print(f"{message}\n")


def print_usage_info():
    """Print current usage information."""
    api_key = get_api_key()
    
    if not api_key:
        _print_auth_required()
        return
    
    try:
        usage = check_usage(api_key)
        
        if RICH_AVAILABLE:
            from rich.table import Table
            
            table = Table(title="Langtune Usage", box=box.ROUNDED, title_style="bold cyan")
            table.add_column("Metric", style="cyan", no_wrap=True)
            table.add_column("Used", style="white", justify="right")
            table.add_column("Limit", style="white", justify="right")
            table.add_column("Remaining", style="green", justify="right")
            
            table.add_row(
                "Tokens",
                f"{usage['tokens_used']:,}",
                f"{usage['tokens_limit']:,}",
                f"{usage['tokens_remaining']:,}"
            )
            table.add_row(
                "Requests",
                f"{usage['requests_used']:,}",
                f"{usage['requests_limit']:,}",
                f"{usage['requests_remaining']:,}"
            )
            
            console.print()
            console.print(f"[dim]Plan:[/] [bold]{usage['plan'].title()}[/]")
            console.print(table)
            console.print()
            console.print("[dim]Manage your plan at:[/] [blue underline]https://billing.langtrain.xyz[/]\n")
        else:
            print(f"\nPlan: {usage['plan'].title()}")
            print(f"Tokens: {usage['tokens_used']:,} / {usage['tokens_limit']:,}")
            print(f"Requests: {usage['requests_used']:,} / {usage['requests_limit']:,}")
            print(f"\nManage your plan at: https://billing.langtrain.xyz\n")
    
    except (AuthenticationError, UsageLimitError) as e:
        if RICH_AVAILABLE:
            console.print(f"[red]{e}[/]")
        else:
            print(str(e))


def interactive_login():
    """Interactive login flow."""
    if RICH_AVAILABLE:
        console.print("\n[bold cyan]🔐 Langtune Authentication[/]\n")
        console.print("Get your API key at: [blue underline]https://app.langtrain.xyz[/]\n")
        api_key = console.input("[bold]Enter your API key:[/] ")
    else:
        print("\n🔐 Langtune Authentication\n")
        print("Get your API key at: https://app.langtrain.xyz\n")
        api_key = input("Enter your API key: ")
    
    api_key = api_key.strip()
    
    if not api_key:
        if RICH_AVAILABLE:
            console.print("[red]No API key entered.[/]")
        else:
            print("No API key entered.")
        return False
    
    try:
        user_data = verify_api_key(api_key, force_refresh=True)
        set_api_key(api_key)
        
        if RICH_AVAILABLE:
            console.print(f"\n[bold green]✓ Authentication successful![/]")
            console.print(f"[dim]Plan:[/] [bold]{user_data.get('plan', 'free').title()}[/]")
            console.print(f"\n[dim]You're ready to use Langtune. Run[/] [cyan]langtune info[/] [dim]to get started.[/]\n")
        else:
            print(f"\n✓ Authentication successful!")
            print(f"Plan: {user_data.get('plan', 'free').title()}")
            print(f"\nYou're ready to use Langtune. Run 'langtune info' to get started.\n")
        
        return True
    
    except AuthenticationError as e:
        _print_auth_error(str(e))
        return False


def logout():
    """Remove stored API key."""
    config = _load_config()
    if "api_key" in config:
        del config["api_key"]
        _save_config(config)
    
    # Clear cache
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()
    
    if RICH_AVAILABLE:
        console.print("[green]✓[/] Logged out successfully. API key removed.")
    else:
        print("✓ Logged out successfully. API key removed.")
