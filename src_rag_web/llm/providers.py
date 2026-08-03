"""
LLM provider implementations for RAG query system.

Supports Groq, OpenAI, Anthropic, and Ollama providers.
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class BaseLLMProvider:
    """Base class for LLM providers."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize LLM provider.

        Args:
            config: LLM configuration dictionary
        """
        self.config = config
        self.model = config.get("model")
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url")
        # Only set if defined in config (None = not set)
        self.temperature = config.get("temperature")
        self.top_p = config.get("top_p")
        # Prefer max_completion_tokens over max_tokens (for newer OpenAI models)
        self.max_completion_tokens = config.get("max_completion_tokens")
        self.max_tokens = config.get("max_tokens")
        self.timeout = config.get("timeout", 300)  # Default 5 minutes for GPT-5 reasoning
        self.enable_cot = config.get("enable_cot", False)
        self.enable_streaming = config.get("enable_streaming", False)

    def supports_json_mode(self) -> bool:
        """
        Check if this model supports response_format: {type: "json_object"}.

        This is an OpenAI-specific parameter that forces the model to return valid JSON.
        Not all models/providers support this feature.

        Returns:
            True if the model supports JSON mode, False otherwise

        Note:
            Override in subclasses to provide model-specific detection.
            Base implementation returns False (conservative default).
        """
        return False

    async def generate(self, prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        """
        Generate completion from prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        raise NotImplementedError

    async def close(self):
        """Close provider resources. Override in subclasses if needed."""
        pass


class OpenAICompatibleProvider(BaseLLMProvider):
    """Provider for OpenAI-compatible APIs (OpenAI, Groq, etc.)."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize OpenAI-compatible provider.

        Args:
            config: LLM configuration dictionary
        """
        super().__init__(config)

        # Check for API key
        if not self.api_key:
            provider = config.get("provider", "openai")
            env_var = f"{provider.upper()}_API_KEY"
            self.api_key = os.getenv(env_var) or os.getenv("LLM_API_KEY")
            if not self.api_key:
                if self.base_url:
                    # Local and self-hosted OpenAI-compatible endpoints (vLLM,
                    # LM Studio, Ollama's /v1, an unauthenticated proxy) take no
                    # auth — but AsyncOpenAI REFUSES to construct without an
                    # api_key and raises before the first request, which would
                    # surface as an opaque failure on the first query. Supply a
                    # placeholder; the endpoint ignores it.
                    logger.info(
                        f"No API key configured; sending a placeholder to {self.base_url} "
                        f"(set BIOGNOSIA_LLM_API_KEY if the endpoint requires auth)."
                    )
                    self.api_key = "not-needed"
                else:
                    logger.warning(
                        f"No API key found for {provider}. Set BIOGNOSIA_LLM_API_KEY, "
                        f"{env_var} or LLM_API_KEY."
                    )

        # Initialize client lazily (will be created on first use)
        self._client = None

    async def generate(self, prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        """
        Generate completion using OpenAI-compatible API.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        try:
            # Import openai client
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai") from None

        # Create or reuse client
        if self._client is None:
            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url

            self._client = AsyncOpenAI(**client_kwargs)

        client = self._client

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Parameter handling: config takes precedence over kwargs
        # If config doesn't define a parameter (None), it won't be passed to API
        # If config defines it, allow kwargs to override
        response_format = kwargs.get("response_format")  # Get response_format if provided

        # Temperature: only use if config defines it
        if self.temperature is not None:
            temperature = kwargs.get("temperature", self.temperature)
        else:
            temperature = None  # Config says don't pass temperature

        # Top-p: only use if config defines it
        if self.top_p is not None:
            top_p = kwargs.get("top_p", self.top_p)
        else:
            top_p = None  # Config says don't pass top_p

        # Token limits: respect config's parameter choice
        # If config defines max_completion_tokens, ALWAYS use it (even if caller passes max_tokens)
        # This ensures compatibility with newer models like GPT-5
        if self.max_completion_tokens is not None:
            # Config uses max_completion_tokens - prefer this parameter
            if "max_completion_tokens" in kwargs:
                max_completion_tokens = kwargs["max_completion_tokens"]
            elif "max_tokens" in kwargs:
                # Caller passed max_tokens but config uses max_completion_tokens
                # Convert to max_completion_tokens
                max_completion_tokens = kwargs["max_tokens"]
            else:
                max_completion_tokens = self.max_completion_tokens
            max_tokens = None
        elif self.max_tokens is not None:
            # Config uses max_tokens - use this parameter
            if "max_tokens" in kwargs:
                max_tokens = kwargs["max_tokens"]
            elif "max_completion_tokens" in kwargs:
                # Caller passed max_completion_tokens but config uses max_tokens
                # Convert to max_tokens
                max_tokens = kwargs["max_completion_tokens"]
            else:
                max_tokens = self.max_tokens
            max_completion_tokens = None
        else:
            # Config doesn't define either - use whatever caller provided
            max_completion_tokens = kwargs.get("max_completion_tokens")
            max_tokens = kwargs.get("max_tokens")
            # Prefer max_completion_tokens if both provided
            if max_completion_tokens is not None:
                max_tokens = None

        try:
            logger.debug(
                f"Calling LLM: model={self.model}, "
                f"temperature={temperature}, "
                f"max_completion_tokens={max_completion_tokens}, "
                f"max_tokens={max_tokens}"
            )

            # Build API call parameters
            api_params = {
                "model": self.model,
                "messages": messages,
                "timeout": self.timeout,
            }

            # Only add parameters if they are defined (not None)
            if temperature is not None:
                api_params["temperature"] = temperature

            if top_p is not None:
                api_params["top_p"] = top_p

            # Prefer max_completion_tokens over max_tokens
            if max_completion_tokens is not None:
                api_params["max_completion_tokens"] = max_completion_tokens
            elif max_tokens is not None:
                api_params["max_tokens"] = max_tokens

            # Add response_format if provided (for JSON mode)
            if response_format:
                api_params["response_format"] = response_format

            # Log full JSON request at INFO level
            import json

            # Create a loggable version (exclude timeout as it's not sent to API)
            request_json = {
                "model": api_params["model"],
                "messages": api_params["messages"],
            }
            if "temperature" in api_params:
                request_json["temperature"] = api_params["temperature"]
            if "max_tokens" in api_params:
                request_json["max_tokens"] = api_params["max_tokens"]
            if "max_completion_tokens" in api_params:
                request_json["max_completion_tokens"] = api_params["max_completion_tokens"]
            if "top_p" in api_params:
                request_json["top_p"] = api_params["top_p"]
            if response_format:
                request_json["response_format"] = response_format

            # Format JSON and replace \n in content strings with actual newlines for readability
            json_str = json.dumps(request_json, indent=2, ensure_ascii=False)
            # Replace escaped newlines in content strings with actual newlines
            # This makes prompts more readable in log files
            import re

            json_str = re.sub(r"\\n", "\n", json_str)

            # DEBUG, not INFO: `messages` carries the whole built context —
            # entities, relationships and full chunk text, tens to hundreds of
            # KB per call, several calls per query — plus the user's question.
            # Logging that by default floods the log and copies retrieved
            # content into it. Raise the level to DEBUG to inspect prompts.
            logger.debug(f"LLM API request:\n{json_str}")
            logger.info(f"LLM API request: model={self.model}")

            # Call API
            try:
                response = await client.chat.completions.create(**api_params)
            except Exception as json_mode_error:
                # Groq enforces response_format={"type": "json_object"} server
                # side and rejects anything whose top level is not an object.
                # Parts of the pipeline legitimately ask for a JSON *array*
                # (query expansion says so in as many words) and the model obeys
                # them, so the request fails even though the generated text was
                # exactly what the caller wanted — it comes back as
                # json_validate_failed, carrying the good output in
                # `failed_generation`.
                #
                # Retry once without the parameter. The callers have a fallback
                # of their own, but it only fires on errors that mention
                # "response_format", which this one never does; without this the
                # call fails every time for as long as JSON mode is on.
                if "response_format" not in api_params:
                    raise
                error_text = str(json_mode_error).lower()
                if "json_validate_failed" not in error_text and "response_format" not in error_text:
                    raise
                logger.warning(
                    f"Endpoint rejected JSON mode ({type(json_mode_error).__name__}); "
                    f"retrying the request without response_format."
                )
                api_params.pop("response_format")
                response = await client.chat.completions.create(**api_params)

            # Debug: Log response details for troubleshooting GPT-5
            logger.info(f"API response finish_reason: {response.choices[0].finish_reason}")
            if hasattr(response, "usage") and response.usage:
                comp_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens
                logger.info(f"API response usage: completion_tokens={comp_tokens}, total_tokens={total_tokens}")
                if hasattr(response.usage, "completion_tokens_details"):
                    details = response.usage.completion_tokens_details
                    if details:
                        reasoning_tokens = details.reasoning_tokens
                        logger.info(f"API response token details: reasoning_tokens={reasoning_tokens}")
            if hasattr(response.choices[0].message, "refusal"):
                logger.info(f"API response refusal: {response.choices[0].message.refusal}")

            # Extract text
            generated_text = response.choices[0].message.content
            content_preview = generated_text[:200] if generated_text else "EMPTY/None"
            logger.info(f"Extracted content (first 200 chars): {content_preview!r}")

            # Check for empty response (refusal or content filtering)
            if generated_text is None or generated_text == "":
                # Check for refusal
                msg = response.choices[0].message
                refusal = msg.refusal if hasattr(msg, "refusal") else None
                if refusal:
                    logger.error(f"LLM refused to respond: {refusal}")
                    raise ValueError(f"LLM refused to respond: {refusal}")
                else:
                    logger.warning("LLM returned empty content (possibly content filtered)")
                    # Return empty string rather than None to avoid downstream errors
                    generated_text = ""

            # Remove think tags if CoT is disabled
            if not self.enable_cot and generated_text:
                from ..utils.helpers import remove_think_tags

                generated_text = remove_think_tags(generated_text)

            logger.debug(
                f"LLM response received: {len(generated_text) if generated_text else 0} chars, "
                f"usage={response.usage.total_tokens if hasattr(response, 'usage') else 'N/A'} tokens"
            )

            return generated_text

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

    def supports_json_mode(self) -> bool:
        """
        Check if this model supports response_format: {type: "json_object"}.

        Returns:
            True if the model supports JSON mode, False otherwise

        Note:
            Detection is based on model name and provider type.
            - OpenAI: GPT-4o, GPT-4-turbo, GPT-3.5-turbo (with version >= 1106) support it
            - OpenAI: GPT-4 base does NOT support it
            - Groq: Supports JSON mode for most models (OpenAI-compatible)
            - Grok: Grok-2 and later support it
        """
        if not self.model:
            return False

        model_lower = self.model.lower()
        provider = self.config.get("provider", "").lower()

        # OpenAI models that support JSON mode
        if "gpt-4o" in model_lower:
            return True
        if "gpt-4-turbo" in model_lower:
            return True
        if "gpt-3.5-turbo" in model_lower:
            # Only newer versions (1106+) support JSON mode
            if "1106" in model_lower or "0125" in model_lower:
                return True
            return False

        # GPT-4 base does NOT support JSON mode
        if "gpt-4" in model_lower and "turbo" not in model_lower and "o" not in model_lower:
            return False

        # Grok models (xAI)
        if "grok-2" in model_lower or "grok-3" in model_lower:
            return True

        # Groq provider (inference service) - OpenAI-compatible, assume support
        if provider == "groq":
            return True

        # OpenAI provider - if we haven't matched above, be conservative
        if provider == "openai":
            return False

        # Default: conservative - don't use JSON mode unless we're sure
        return False

    async def generate_with_tools(
        self,
        prompt: str,
        system_prompt: str | None = None,
        tools: list[dict] | None = None,
        **kwargs
    ) -> dict[str, Any]:
        """
        Generate completion with tool/function calling support.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            tools: List of tool definitions (OpenAI format)
            **kwargs: Additional generation parameters

        Returns:
            Dictionary with:
                - content: Text response (may be None if tools called)
                - tool_calls: List of tool calls (if any)
                - finish_reason: Reason for completion
        """
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai") from None

        # Create or reuse client
        if self._client is None:
            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            self._client = AsyncOpenAI(**client_kwargs)

        client = self._client

        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Build API parameters
        api_params = {
            "model": self.model,
            "messages": messages,
            "timeout": self.timeout,
        }

        # Add tools if provided
        if tools:
            api_params["tools"] = tools
            api_params["tool_choice"] = "auto"

        # Add temperature (use 0.0 for routing decisions)
        temperature = kwargs.get("temperature", self.temperature)
        if temperature is not None:
            api_params["temperature"] = temperature

        # Add token limits: prefer max_completion_tokens over max_tokens (for GPT-5, o1, etc.)
        # Same logic as generate() method
        if self.max_completion_tokens is not None:
            max_completion_tokens = kwargs.get("max_completion_tokens", self.max_completion_tokens)
            if max_completion_tokens:
                api_params["max_completion_tokens"] = max_completion_tokens
        elif self.max_tokens is not None:
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            if max_tokens:
                api_params["max_tokens"] = max_tokens
        else:
            # Config doesn't define either - use whatever caller provided
            max_completion_tokens = kwargs.get("max_completion_tokens")
            max_tokens = kwargs.get("max_tokens")
            if max_completion_tokens:
                api_params["max_completion_tokens"] = max_completion_tokens
            elif max_tokens:
                api_params["max_tokens"] = max_tokens

        try:
            logger.debug(f"Calling LLM with tools: model={self.model}, num_tools={len(tools) if tools else 0}")

            # Call API
            response = await client.chat.completions.create(**api_params)

            # Extract response
            message = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            # Parse tool calls
            tool_calls = []
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tc in message.tool_calls:
                    tool_calls.append({
                        "id": tc.id,
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    })

            logger.debug(
                f"LLM response: content={len(message.content) if message.content else 0} chars, "
                f"tool_calls={len(tool_calls)}, finish_reason={finish_reason}"
            )

            return {
                "content": message.content,
                "tool_calls": tool_calls,
                "finish_reason": finish_reason
            }

        except Exception as e:
            logger.error(f"LLM generation with tools failed: {e}")
            raise

    async def close(self):
        """Close the OpenAI client if it exists."""
        if self._client is not None:
            try:
                await self._client.close()
                logger.debug("OpenAI client closed")
            except Exception as e:
                logger.warning(f"Error closing OpenAI client: {e}")
            finally:
                self._client = None


class AnthropicProvider(BaseLLMProvider):
    """Provider for Anthropic Claude API."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize Anthropic provider.

        Args:
            config: LLM configuration dictionary
        """
        super().__init__(config)

        # Check for API key
        if not self.api_key:
            self.api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("LLM_API_KEY")
            if not self.api_key:
                logger.warning(
                    "No API key found for Anthropic. Set ANTHROPIC_API_KEY or LLM_API_KEY environment variable."
                )

    async def generate(self, prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        """
        Generate completion using Anthropic API.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        try:
            # Import anthropic client
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic") from None

        # Create client
        client = AsyncAnthropic(api_key=self.api_key)

        # Override defaults with kwargs
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        # Handle temperature and top_p
        # Anthropic API constraint: Cannot set both temperature AND top_p
        # Only include parameters if they are not None
        # Priority: temperature > top_p (if both are set in config, prefer temperature)
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)

        try:
            logger.debug(
                f"Calling Anthropic: model={self.model}, temperature={temperature}, "
                f"max_tokens={max_tokens}, top_p={top_p}"
            )

            # Build API call parameters (only include non-None values)
            kwargs_api = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "timeout": self.timeout,
            }

            # Add system prompt if provided
            if system_prompt:
                kwargs_api["system"] = system_prompt

            # Add temperature or top_p (prefer temperature if both are set)
            # Anthropic requires exactly one of these, not both
            if temperature is not None:
                kwargs_api["temperature"] = temperature
            elif top_p is not None:
                kwargs_api["top_p"] = top_p

            # Log full JSON request at INFO level
            import json

            # Create a loggable version (exclude timeout as it's not sent to API)
            # Only include parameters that are actually in the API call
            request_json = {
                "model": kwargs_api["model"],
                "messages": kwargs_api["messages"],
                "max_tokens": kwargs_api["max_tokens"],
            }
            if "temperature" in kwargs_api:
                request_json["temperature"] = kwargs_api["temperature"]
            if "top_p" in kwargs_api:
                request_json["top_p"] = kwargs_api["top_p"]
            if "system" in kwargs_api:
                request_json["system"] = kwargs_api["system"]

            # Format JSON and replace \n with actual newlines for readability
            json_str = json.dumps(request_json, indent=2, ensure_ascii=False)
            import re

            json_str = re.sub(r"\\n", "\n", json_str)

            # DEBUG, not INFO: `messages` carries the whole built context —
            # entities, relationships and full chunk text, tens to hundreds of
            # KB per call, several calls per query — plus the user's question.
            # Logging that by default floods the log and copies retrieved
            # content into it. Raise the level to DEBUG to inspect prompts.
            logger.debug(f"LLM API request:\n{json_str}")
            logger.info(f"LLM API request: model={self.model}")

            # Call API
            response = await client.messages.create(**kwargs_api)

            # Extract text
            generated_text = response.content[0].text

            # Remove think tags if CoT is disabled
            if not self.enable_cot:
                from ..utils.helpers import remove_think_tags

                generated_text = remove_think_tags(generated_text)

            usage_tokens = (
                response.usage.input_tokens + response.usage.output_tokens if hasattr(response, "usage") else "N/A"
            )
            logger.debug(f"Anthropic response received: {len(generated_text)} chars, usage={usage_tokens} tokens")

            return generated_text

        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}")
            raise

    def supports_json_mode(self) -> bool:
        """
        Anthropic Claude models do not support OpenAI's response_format parameter.

        Returns:
            False - Claude uses tool calling or prompt engineering for JSON output
        """
        return False

    async def generate_with_tools(
        self,
        prompt: str,
        system_prompt: str | None = None,
        tools: list[dict] | None = None,
        **kwargs
    ) -> dict[str, Any]:
        """
        Generate completion with tool/function calling support.

        Converts OpenAI tool format to Anthropic format.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            tools: List of tool definitions (OpenAI format)
            **kwargs: Additional generation parameters

        Returns:
            Dictionary with:
                - content: Text response (may be None if tools called)
                - tool_calls: List of tool calls (if any)
                - finish_reason: Reason for completion
        """
        try:
            from anthropic import AsyncAnthropic
        except ImportError:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic") from None

        # Create client
        client = AsyncAnthropic(api_key=self.api_key)

        # Convert OpenAI tools to Anthropic format
        anthropic_tools = []
        if tools:
            for tool in tools:
                if tool.get("type") == "function":
                    func = tool.get("function", {})
                    anthropic_tools.append({
                        "name": func.get("name"),
                        "description": func.get("description"),
                        "input_schema": func.get("parameters", {})
                    })

        # Build API parameters
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)

        api_params = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "timeout": self.timeout,
        }

        # Add system prompt if provided
        if system_prompt:
            api_params["system"] = system_prompt

        # Add tools if provided
        if anthropic_tools:
            api_params["tools"] = anthropic_tools

        # Add temperature or top_p (prefer temperature)
        if temperature is not None:
            api_params["temperature"] = temperature
        elif top_p is not None:
            api_params["top_p"] = top_p

        try:
            logger.debug(f"Calling Anthropic with tools: model={self.model}, num_tools={len(anthropic_tools)}")

            # Call API
            response = await client.messages.create(**api_params)

            # Extract response
            content_text = ""
            tool_calls = []

            for block in response.content:
                if block.type == "text":
                    content_text += block.text
                elif block.type == "tool_use":
                    import json
                    tool_calls.append({
                        "id": block.id,
                        "name": block.name,
                        "arguments": json.dumps(block.input)
                    })

            finish_reason = response.stop_reason

            logger.debug(
                f"Anthropic response: content={len(content_text)} chars, "
                f"tool_calls={len(tool_calls)}, stop_reason={finish_reason}"
            )

            return {
                "content": content_text if content_text else None,
                "tool_calls": tool_calls,
                "finish_reason": finish_reason
            }

        except Exception as e:
            logger.error(f"Anthropic generation with tools failed: {e}")
            raise


class OllamaProvider(BaseLLMProvider):
    """Provider for Ollama local models."""

    def __init__(self, config: dict[str, Any]):
        """
        Initialize Ollama provider.

        Args:
            config: LLM configuration dictionary
        """
        super().__init__(config)
        self.base_url = self.base_url or "http://localhost:11434"

    async def generate(self, prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        """
        Generate completion using Ollama API.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        try:
            import aiohttp
        except ImportError:
            raise ImportError("aiohttp package not installed. Install with: pip install aiohttp") from None

        # Override defaults with kwargs
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        # Build request
        url = f"{self.base_url}/api/generate"
        data = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "num_predict": max_tokens,
            "stream": False,
        }
        if system_prompt:
            data["system"] = system_prompt

        try:
            logger.debug(f"Calling Ollama: model={self.model}, temperature={temperature}, max_tokens={max_tokens}")

            # Log full JSON request at INFO level
            import json

            # Format JSON and replace \n with actual newlines for readability
            json_str = json.dumps(data, indent=2, ensure_ascii=False)
            import re

            json_str = re.sub(r"\\n", "\n", json_str)

            # DEBUG, not INFO: `messages` carries the whole built context —
            # entities, relationships and full chunk text, tens to hundreds of
            # KB per call, several calls per query — plus the user's question.
            # Logging that by default floods the log and copies retrieved
            # content into it. Raise the level to DEBUG to inspect prompts.
            logger.debug(f"LLM API request:\n{json_str}")
            logger.info(f"LLM API request: model={self.model}")

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data, timeout=aiohttp.ClientTimeout(total=self.timeout)) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Ollama API error: {response.status} - {error_text}")
                        raise RuntimeError(f"Ollama API error: {response.status}")

                    result = await response.json()
                    generated_text = result.get("response", "")

                    # Remove think tags if CoT is disabled
                    if not self.enable_cot:
                        from ..utils.helpers import remove_think_tags

                        generated_text = remove_think_tags(generated_text)

                    logger.debug(f"Ollama response received: {len(generated_text)} chars")

                    return generated_text

        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise

    def supports_json_mode(self) -> bool:
        """
        Ollama models do not support OpenAI's response_format parameter.

        Returns:
            False - Ollama uses local models with different API
        """
        return False


def create_llm_provider(config: dict[str, Any]) -> BaseLLMProvider:
    """
    Create LLM provider based on configuration.

    Args:
        config: LLM configuration dictionary

    Returns:
        LLM provider instance
    """
    provider = config.get("provider", "openai").lower()

    if provider in ["openai", "groq"]:
        return OpenAICompatibleProvider(config)
    elif provider == "anthropic":
        return AnthropicProvider(config)
    elif provider == "ollama":
        return OllamaProvider(config)
    else:
        logger.error(f"Unknown LLM provider: {provider}")
        raise ValueError(f"Unknown LLM provider: {provider}")


async def generate_llm_response(
    prompt: str,
    llm_provider: BaseLLMProvider,
    system_prompt: str | None = None,
    hashing_kv=None,
    mode: str = "query",
    cache_type: str = "llm",
    enable_cache: bool = True,
    **kwargs,
) -> str:
    """
    Generate LLM response with error handling and caching support.

    Args:
        prompt: User prompt
        llm_provider: LLM provider instance
        system_prompt: Optional system prompt
        hashing_kv: Optional key-value storage for caching
        mode: Cache mode (default, query, etc.)
        cache_type: Type of cache (llm, query, etc.)
        enable_cache: Whether to use caching
        **kwargs: Additional generation parameters

    Returns:
        Generated text
    """
    from ..utils.cache import CacheData, handle_cache, save_to_cache
    from ..utils.helpers import compute_args_hash

    # Try cache first if enabled
    if enable_cache and hashing_kv is not None:
        # Compute hash from prompt and system_prompt
        cache_key_content = f"{prompt}|||{system_prompt or ''}"
        args_hash = compute_args_hash(cache_key_content)

        # Check cache
        cached = await handle_cache(
            hashing_kv=hashing_kv,
            args_hash=args_hash,
            prompt=prompt,
            mode=mode,
            cache_type=cache_type,
        )

        if cached:
            content, timestamp = cached
            logger.debug(f"Using cached LLM response ({len(content)} chars)")
            return content

    # Generate new response
    try:
        response = await llm_provider.generate(prompt=prompt, system_prompt=system_prompt, **kwargs)

        # Save to cache if enabled
        if enable_cache and hashing_kv is not None:
            cache_key_content = f"{prompt}|||{system_prompt or ''}"
            args_hash = compute_args_hash(cache_key_content)

            cache_data = CacheData(
                args_hash=args_hash,
                content=response,
                prompt=prompt,
                mode=mode,
                cache_type=cache_type,
            )

            await save_to_cache(hashing_kv, cache_data)

        return response

    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        # Re-raise to fail the entire pipeline instead of continuing with error message
        raise
