"""
LangChain-compatible adapter for Claude Code CLI.

This adapter makes Claude Code CLI compatible with the existing LangChain-based
infrastructure, allowing seamless integration with the AI Hedge Fund agents.
"""

from typing import Any, List, Optional, Iterator
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from pydantic import BaseModel, Field, ConfigDict, PrivateAttr
from typing_extensions import Literal
import json

from src.llm.claude_code_wrapper import ClaudeCodeWrapper, ClaudeCodeConfig


class ClaudeCodeAdapter(BaseChatModel):
    """
    LangChain-compatible adapter for Claude Code CLI.

    This adapter allows Claude Code (subscription-based) to work with the existing
    LangChain/LangGraph infrastructure without any code changes in the agents.

    Example:
        >>> from src.llm.claude_code_adapter import ClaudeCodeAdapter
        >>> model = ClaudeCodeAdapter()
        >>> result = model.invoke([HumanMessage(content="Hello")])
    """

    wrapper: Optional[ClaudeCodeWrapper] = Field(default=None, exclude=True)
    timeout: int = Field(default=60)
    cli_path: Optional[str] = Field(default=None)

    # Private attribute to track structured output schema
    _structured_output_schema: Optional[type[BaseModel]] = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.wrapper is None:
            config = ClaudeCodeConfig(
                timeout=self.timeout,
                cli_path=self.cli_path
            )
            self.wrapper = ClaudeCodeWrapper(config=config)

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "claude-code-cli"

    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """
        Convert LangChain messages to a single prompt string for CLI.

        Args:
            messages: List of LangChain messages

        Returns:
            Combined prompt string
        """
        prompt_parts = []
        system_prompts = []

        for message in messages:
            content = message.content

            if isinstance(message, SystemMessage):
                system_prompts.append(content)
            elif isinstance(message, HumanMessage):
                prompt_parts.append(content)
            elif isinstance(message, AIMessage):
                # Include previous AI responses for context
                prompt_parts.append(f"Previous response: {content}")

        # Combine system prompts and user prompts
        full_prompt = ""

        if system_prompts:
            full_prompt += "System Instructions:\n"
            full_prompt += "\n".join(system_prompts)
            full_prompt += "\n\n"

        if prompt_parts:
            full_prompt += "\n\n".join(prompt_parts)

        return full_prompt

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Generate a response using Claude Code CLI.

        Args:
            messages: List of messages to process
            stop: Stop sequences (not used for CLI)
            run_manager: Callback manager
            **kwargs: Additional arguments

        Returns:
            ChatResult with the response
        """
        # Convert messages to prompt
        prompt = self._convert_messages_to_prompt(messages)

        # If structured output is requested, add JSON instruction
        if self._structured_output_schema:
            schema = self._structured_output_schema
            schema_json = schema.model_json_schema()

            prompt += f"""

IMPORTANT: Respond with ONLY a JSON object matching this exact schema:
{json.dumps(schema_json, indent=2)}

Return ONLY the JSON, optionally wrapped in ```json ``` markdown.
Do not include explanations before or after the JSON."""

        # Call Claude Code CLI
        try:
            response_text = self.wrapper.query(
                prompt,
                timeout=kwargs.get("timeout", self.timeout)
            )
        except Exception as e:
            # Return error as response
            response_text = f"Error calling Claude Code: {e}"

        # Create AIMessage with response
        message = AIMessage(content=response_text)

        # Create generation
        generation = ChatGeneration(
            message=message,
            generation_info={"finish_reason": "stop"}
        )

        return ChatResult(generations=[generation])

    def invoke(
        self,
        input: List[BaseMessage],
        config: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Invoke the model. If structured output is enabled, return Pydantic model.
        Otherwise, return AIMessage.

        Args:
            input: List of messages
            config: Optional configuration
            **kwargs: Additional arguments

        Returns:
            Pydantic model if structured output is enabled, otherwise AIMessage
        """
        # Call parent invoke to get AIMessage
        result = super().invoke(input, config, **kwargs)

        # If structured output is enabled, parse and return Pydantic model
        if self._structured_output_schema:
            try:
                # Result is an AIMessage, get its content
                response_text = result.content if hasattr(result, 'content') else str(result)
                return self._parse_structured_output(response_text, self._structured_output_schema)
            except Exception as e:
                # Fallback: return default instance of the model
                print(f"Warning: Failed to parse structured output: {e}")
                return self._create_default_model(self._structured_output_schema)

        return result

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """
        Stream is not supported by CLI adapter.
        Falls back to non-streaming generation.
        """
        # Claude Code CLI doesn't support streaming, so we just return the full response
        result = self._generate(messages, stop, run_manager, **kwargs)
        if result.generations:
            yield ChatGenerationChunk(
                message=result.generations[0].message,
                generation_info=result.generations[0].generation_info
            )

    def _parse_structured_output(self, response_text: str, schema: type[BaseModel]) -> BaseModel:
        """
        Parse JSON from Claude's response and convert to Pydantic model.

        Args:
            response_text: Raw response from Claude Code
            schema: Pydantic model class

        Returns:
            Instance of the Pydantic model

        Raises:
            ValueError: If JSON cannot be parsed
        """
        # Try to extract JSON from markdown code blocks
        json_str = response_text.strip()

        # Check for markdown JSON block
        if "```json" in json_str:
            start = json_str.find("```json") + 7
            end = json_str.find("```", start)
            if end > start:
                json_str = json_str[start:end].strip()
        elif "```" in json_str:
            # Generic code block
            start = json_str.find("```") + 3
            end = json_str.find("```", start)
            if end > start:
                json_str = json_str[start:end].strip()

        # Try to find JSON object boundaries if no code block
        if not json_str.startswith("{"):
            start = json_str.find("{")
            if start >= 0:
                # Find matching closing brace
                end = json_str.rfind("}")
                if end > start:
                    json_str = json_str[start:end+1]

        # Parse JSON and create model
        try:
            data = json.loads(json_str)
            return schema(**data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}\nResponse: {json_str[:200]}")
        except Exception as e:
            raise ValueError(f"Failed to create Pydantic model: {e}")

    def _create_default_model(self, schema: type[BaseModel]) -> BaseModel:
        """
        Create a default instance of the Pydantic model with safe defaults.

        Args:
            schema: Pydantic model class

        Returns:
            Default instance of the model
        """
        default_values = {}

        for field_name, field_info in schema.model_fields.items():
            annotation = field_info.annotation

            # Handle Literal types (like "bullish" | "bearish" | "neutral")
            if hasattr(annotation, "__origin__"):
                if annotation.__origin__ is Literal:
                    # Use first allowed value
                    default_values[field_name] = annotation.__args__[0]
                    continue

            # Handle basic types
            if annotation == str or annotation == Optional[str]:
                default_values[field_name] = "Error in analysis, using default"
            elif annotation == int or annotation == Optional[int]:
                default_values[field_name] = 0
            elif annotation == float or annotation == Optional[float]:
                default_values[field_name] = 0.0
            elif annotation == bool or annotation == Optional[bool]:
                default_values[field_name] = False
            else:
                default_values[field_name] = None

        return schema(**default_values)

    def with_structured_output(self, schema: Any, **kwargs) -> "ClaudeCodeAdapter":
        """
        Return a model configured to output structured data.

        For Claude Code, we add instructions to the prompt to output JSON
        matching the schema, then parse the response into the Pydantic model.

        Args:
            schema: Pydantic model class
            **kwargs: Additional arguments

        Returns:
            New ClaudeCodeAdapter configured for structured output
        """
        # Create a new instance with the same config but different schema
        new_adapter = ClaudeCodeAdapter(
            timeout=self.timeout,
            cli_path=self.cli_path
        )
        new_adapter._structured_output_schema = schema
        new_adapter.wrapper = self.wrapper  # Reuse the wrapper

        return new_adapter

    def bind_tools(self, tools: List[Any], **kwargs) -> "ClaudeCodeAdapter":
        """
        Bind tools to the model.

        Claude Code CLI doesn't support tool binding directly,
        but we can include tool descriptions in the prompt.
        """
        # Return self for compatibility
        return self


def create_claude_code_adapter(
    timeout: int = 60,
    cli_path: Optional[str] = None
) -> ClaudeCodeAdapter:
    """
    Factory function to create a Claude Code adapter.

    Args:
        timeout: Timeout for CLI queries in seconds
        cli_path: Custom path to claude CLI (optional)

    Returns:
        Configured ClaudeCodeAdapter
    """
    return ClaudeCodeAdapter(
        timeout=timeout,
        cli_path=cli_path
    )
