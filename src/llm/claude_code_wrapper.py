"""
Claude Code CLI Wrapper for AI Hedge Fund.

This wrapper integrates Claude Code CLI (subscription-based) with the AI Hedge Fund,
allowing you to use your Claude Pro Max subscription instead of pay-per-use API.

Features:
- Uses Claude Code CLI (no API key needed, uses subscription)
- Supports one-shot queries and interactive sessions
- Integrates with existing hedge fund infrastructure
- Provides financial analysis tools via prompt engineering

Prerequisites:
1. Install Claude Code CLI:
   npm install -g @anthropic-ai/claude-code
   # OR
   brew install --cask claude-code
   # OR
   curl -fsSL https://claude.ai/install.sh | bash

2. Authenticate (one-time):
   claude /login
   # Opens browser to authenticate with your Claude.ai subscription

Usage:
    from src.llm.claude_code_wrapper import ClaudeCodeAnalyst

    analyst = ClaudeCodeAnalyst()
    result = analyst.analyze("Analyze AAPL fundamentals")
    print(result)
"""

import subprocess
import json
import os
import shutil
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ClaudeCodeConfig:
    """Configuration for Claude Code CLI wrapper."""
    cli_path: Optional[str] = None  # Auto-detect if None
    model_name: Optional[str] = None  # Claude model to use (e.g., "sonnet", "haiku", "opus") or None for default cli model
    timeout: int = 60  # Seconds
    max_retries: int = 3
    working_dir: Optional[str] = None


class ClaudeCodeCLIError(Exception):
    """Base exception for Claude Code CLI errors."""
    pass


class ClaudeCodeNotFoundError(ClaudeCodeCLIError):
    """Raised when Claude Code CLI is not installed."""
    pass


class ClaudeCodeAuthError(ClaudeCodeCLIError):
    """Raised when authentication is required."""
    pass


class ClaudeCodeWrapper:
    """
    Wrapper for Claude Code CLI that uses subscription instead of API key.

    This class provides a Python interface to Claude Code CLI, allowing you to:
    - Run one-shot analyses using your subscription
    - Integrate Claude into Python workflows
    - Avoid pay-per-use API costs

    Example:
        >>> wrapper = ClaudeCodeWrapper()
        >>> response = wrapper.query("What is 2+2?")
        >>> print(response)
    """

    def __init__(self, config: Optional[ClaudeCodeConfig] = None):
        """
        Initialize Claude Code CLI wrapper.

        Args:
            config: Configuration options
        """
        self.config = config or ClaudeCodeConfig()

        # Find Claude Code CLI
        self.cli_path = self._find_cli()

        # Verify authentication
        self._check_auth()

    def _find_cli(self) -> str:
        """
        Find Claude Code CLI executable.

        Returns:
            Path to CLI executable

        Raises:
            ClaudeCodeNotFoundError: If CLI is not found
        """
        if self.config.cli_path:
            if not os.path.exists(self.config.cli_path):
                raise ClaudeCodeNotFoundError(
                    f"Claude Code CLI not found at {self.config.cli_path}"
                )
            return self.config.cli_path

        # Try to find in PATH
        cli_path = shutil.which("claude")
        if cli_path:
            return cli_path

        # Common installation paths
        common_paths = [
            "/usr/local/bin/claude",
            "/opt/homebrew/bin/claude",
            os.path.expanduser("~/.local/bin/claude"),
        ]

        for path in common_paths:
            if os.path.exists(path):
                return path

        raise ClaudeCodeNotFoundError(
            "Claude Code CLI not found. Install it with:\n"
            "  npm install -g @anthropic-ai/claude-code\n"
            "  OR\n"
            "  brew install --cask claude-code\n"
            "  OR\n"
            "  curl -fsSL https://claude.ai/install.sh | bash"
        )

    def _check_auth(self) -> None:
        """
        Check if user is authenticated.

        Raises:
            ClaudeCodeAuthError: If authentication is required
        """
        # Try a simple query to check auth
        try:
            # Use -p flag for prompt-only mode
            result = subprocess.run(
                [self.cli_path, "-p", "test"],
                capture_output=True,
                text=True,
                timeout=10
            )

            # Check for auth errors in output
            if "login" in result.stderr.lower() or "authenticate" in result.stderr.lower():
                raise ClaudeCodeAuthError(
                    "Authentication required. Run:\n"
                    f"  {self.cli_path} /login"
                )

        except subprocess.TimeoutExpired:
            # Timeout is acceptable for auth check
            pass
        except FileNotFoundError:
            raise ClaudeCodeNotFoundError(f"CLI not found at {self.cli_path}")

    def query(
        self,
        prompt: str,
        timeout: Optional[int] = None,
        working_dir: Optional[str] = None,
        model_name: Optional[str] = None
    ) -> str:
        """
        Execute a one-shot query to Claude Code.

        Args:
            prompt: The prompt/question to send
            timeout: Timeout in seconds (default: from config)
            working_dir: Working directory for command (default: current dir)
            model_name: Model to use (overrides config, e.g., "claude-3-5-sonnet-latest")

        Returns:
            Claude's response as text

        Raises:
            ClaudeCodeCLIError: If command fails
            subprocess.TimeoutExpired: If command times out
        """
        timeout = timeout or self.config.timeout
        working_dir = working_dir or self.config.working_dir or os.getcwd()
        model = model_name or self.config.model_name

        try:
            # Build command with optional model flag
            cmd = [self.cli_path]

            # Add model flag if specified
            # Claude Code uses --model with aliases: sonnet, haiku, opus
            if model:
                cmd.extend(["--model", model])

            # Add prompt
            cmd.extend(["-p", prompt])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=working_dir
            )

            if result.returncode != 0:
                error_msg = f"Claude Code CLI failed with code {result.returncode}"
                if result.stdout:
                    error_msg += f"\nstdout: {result.stdout}"
                if result.stderr:
                    error_msg += f"\nstderr: {result.stderr}"
                raise ClaudeCodeCLIError(error_msg)

            return result.stdout.strip()

        except subprocess.TimeoutExpired:
            raise ClaudeCodeCLIError(
                f"Query timed out after {timeout} seconds"
            )
        except FileNotFoundError:
            raise ClaudeCodeNotFoundError(
                f"Claude Code CLI not found at {self.cli_path}"
            )

    def query_with_context(
        self,
        prompt: str,
        context: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Query Claude with additional context.

        Args:
            prompt: The main question
            context: Additional context to include
            **kwargs: Additional arguments for query()

        Returns:
            Claude's response
        """
        full_prompt = prompt
        if context:
            full_prompt = f"""Context:
{context}

Question:
{prompt}"""

        return self.query(full_prompt, **kwargs)


class ClaudeCodeAnalyst(ClaudeCodeWrapper):
    """
    Specialized analyst using Claude Code for financial analysis.

    This class extends ClaudeCodeWrapper with domain-specific methods
    for stock analysis and investment decisions.

    Example:
        >>> analyst = ClaudeCodeAnalyst()
        >>> analysis = analyst.analyze_stock("AAPL")
        >>> print(analysis)
    """

    def analyze_stock(
        self,
        ticker: str,
        financial_data: Optional[Dict[str, Any]] = None,
        analyst_persona: str = "warren_buffett"
    ) -> str:
        """
        Analyze a stock using Claude Code.

        Args:
            ticker: Stock ticker symbol
            financial_data: Optional financial metrics and data
            analyst_persona: Which investor persona to use

        Returns:
            Investment analysis and recommendation
        """
        # Persona prompts
        personas = {
            "warren_buffett": """You are Warren Buffett analyzing this stock. Focus on:
- Sustainable competitive advantage (moat)
- Quality of management
- Consistent profitability
- Intrinsic value with margin of safety
- Long-term earnings power""",

            "charlie_munger": """You are Charlie Munger analyzing this stock. Focus on:
- Quality of the business model
- Rational pricing vs. value
- Management quality and integrity
- Long-term competitive position
- Avoiding permanent capital loss""",

            "technical": """You are a technical analyst. Focus on:
- Price trends and momentum
- Support and resistance levels
- Volume patterns
- Chart patterns
- Entry/exit signals""",

            "fundamental": """You are a fundamental analyst. Focus on:
- Financial statement analysis
- Revenue and earnings quality
- Cash flow generation
- Balance sheet strength
- Valuation multiples"""
        }

        persona_prompt = personas.get(analyst_persona, personas["warren_buffett"])

        # Build analysis prompt
        prompt = f"""{persona_prompt}

Analyze {ticker} and provide:
1. Key strengths and weaknesses
2. Valuation assessment (overvalued/fairly valued/undervalued)
3. Major risk factors
4. Final recommendation (Strong Buy/Buy/Hold/Sell/Strong Sell) with conviction level (1-10)

Keep the analysis concise but insightful."""

        # Add financial data if provided
        context = None
        if financial_data:
            context = f"Financial Data for {ticker}:\n{json.dumps(financial_data, indent=2)}"

        return self.query_with_context(prompt, context)

    def compare_stocks(
        self,
        tickers: List[str],
        criteria: Optional[List[str]] = None
    ) -> str:
        """
        Compare multiple stocks.

        Args:
            tickers: List of ticker symbols
            criteria: Specific criteria to compare (optional)

        Returns:
            Comparative analysis
        """
        criteria_str = ""
        if criteria:
            criteria_str = f"\nFocus on these criteria: {', '.join(criteria)}"

        prompt = f"""Compare these stocks: {', '.join(tickers)}

Provide a comparison table and analysis covering:
1. Valuation metrics
2. Growth prospects
3. Profitability
4. Financial health
5. Risk factors
{criteria_str}

Rank them from best to worst investment opportunity and explain why."""

        return self.query(prompt)

    def generate_trading_signal(
        self,
        ticker: str,
        current_price: float,
        financial_metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate a trading signal for a stock.

        Args:
            ticker: Stock ticker
            current_price: Current stock price
            financial_metrics: Optional financial data

        Returns:
            Dict with signal, confidence, and reasoning
        """
        context = f"Current price: ${current_price}"
        if financial_metrics:
            context += f"\n\nFinancial Metrics:\n{json.dumps(financial_metrics, indent=2)}"

        prompt = f"""Based on the data provided, generate a trading signal for {ticker}.

Respond ONLY with a JSON object in this exact format:
{{
  "signal": "buy" | "sell" | "hold",
  "confidence": <number 0-100>,
  "reasoning": "<brief explanation>",
  "target_price": <number or null>,
  "stop_loss": <number or null>
}}"""

        response = self.query_with_context(prompt, context)

        # Try to extract JSON from response
        try:
            # Find JSON in response (Claude might add text before/after)
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: return raw response
        return {
            "signal": "hold",
            "confidence": 0,
            "reasoning": response,
            "target_price": None,
            "stop_loss": None
        }


def check_claude_code_installed() -> bool:
    """
    Check if Claude Code CLI is installed.

    Returns:
        True if installed, False otherwise
    """
    return shutil.which("claude") is not None


def install_instructions() -> str:
    """
    Get installation instructions for Claude Code.

    Returns:
        Formatted installation instructions
    """
    return """
To install Claude Code CLI:

Option 1 - NPM (requires Node.js 18+):
  npm install -g @anthropic-ai/claude-code

Option 2 - Homebrew (macOS/Linux):
  brew install --cask claude-code

Option 3 - curl (macOS/Linux/WSL):
  curl -fsSL https://claude.ai/install.sh | bash

Option 4 - PowerShell (Windows):
  irm https://claude.ai/install.ps1 | iex

After installation, authenticate with:
  claude /login

This will open your browser to log in with your Claude.ai subscription.
"""
