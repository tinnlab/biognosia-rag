"""External API LLM provider for per-request LLM endpoints."""

import json
import logging
import re
from typing import Any

import aiohttp

from .providers import BaseLLMProvider

logger = logging.getLogger(__name__)


class ExternalAPIProvider(BaseLLMProvider):
    """
    LLM provider that calls an external API endpoint with Bearer token auth.

    Each request provides its own endpoint URL and JWT token.
    The API format is: POST {endpoint} with {"prompt": "..."} body.
    """

    def __init__(self, endpoint: str, token: str, timeout: int = 300):
        """
        Initialize external API provider.

        Args:
            endpoint: Full URL to POST prompts to
            token: JWT/Bearer token for authorization
            timeout: Request timeout in seconds
        """
        # Initialize base with minimal config
        super().__init__({"timeout": timeout})
        self.endpoint = endpoint
        self.token = token
        self._session: aiohttp.ClientSession | None = None

    def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def generate(self, prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        """
        Generate response by calling external API endpoint.

        Combines system_prompt and prompt into a single prompt string,
        then POSTs {"prompt": combined} to the endpoint with Bearer auth.

        Args:
            prompt: The main prompt text
            system_prompt: Optional system prompt (prepended to prompt)
            **kwargs: Additional parameters (ignored for external API)

        Returns:
            Generated text response
        """
        # Combine system prompt and prompt
        if system_prompt:
            combined_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            combined_prompt = prompt

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }
        payload = {"prompt": combined_prompt}

        logger.info(f"Calling external LLM API: {self.endpoint} (prompt length: {len(combined_prompt)})")

        session = self._get_session()
        try:
            async with session.post(self.endpoint, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                raw_text = await resp.text()
                logger.info(f"External LLM API response status: {resp.status}, length: {len(raw_text)}")

                # Try JSON parse first, fall back to raw text
                try:
                    data = json.loads(raw_text)
                    if isinstance(data, dict):
                        # Try common response fields
                        for field in ("response", "text", "content", "output", "result"):
                            if field in data:
                                return str(data[field])
                        # If no known field, return the whole JSON as string
                        return raw_text
                except (json.JSONDecodeError, ValueError):
                    pass

                return raw_text

        except aiohttp.ClientResponseError as e:
            logger.error(f"External LLM API HTTP error: {e.status} {e.message}")
            raise
        except aiohttp.ClientError as e:
            logger.error(f"External LLM API connection error: {e}")
            raise
        except Exception as e:
            logger.error(f"External LLM API unexpected error: {e}")
            raise

    async def generate_with_tools(
        self,
        prompt: str,
        system_prompt: str | None = None,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Simulate tool calling via prompt-based approach.

        Embeds tool descriptions in the prompt and parses the LLM response
        to detect which tool (if any) the model wants to call.

        Returns:
            Dictionary with content, tool_calls, and finish_reason.
        """
        # Build tool description block
        tool_lines = []
        tool_names = []
        if tools:
            tool_lines.append("Available tools (respond with EXACTLY one tool call or answer directly):")
            tool_lines.append("")
            for tool in tools:
                func = tool.get("function", {})
                name = func.get("name", "")
                desc = func.get("description", "")
                tool_names.append(name)
                tool_lines.append(f"  TOOL: {name}")
                tool_lines.append(f"  Description: {desc}")
                tool_lines.append("")
            tool_lines.append(
                "To call a tool, respond with ONLY a line: CALL: <tool_name>"
            )
            tool_lines.append(
                "If you can answer directly without tools, just provide your answer."
            )

        tool_block = "\n".join(tool_lines)

        # Combine into a single prompt
        parts = []
        if system_prompt:
            parts.append(system_prompt)
        if tool_block:
            parts.append(tool_block)
        parts.append(prompt)
        combined = "\n\n".join(parts)

        # Call the external API
        response_text = await self.generate(combined)

        # Parse response for tool calls
        tool_calls = []
        content = response_text

        # Look for "CALL: tool_name" pattern
        call_match = re.search(r"CALL:\s*(\S+)", response_text)
        if call_match:
            called_tool = call_match.group(1).strip()
            if called_tool in tool_names:
                tool_calls.append({
                    "name": called_tool,
                    "arguments": {"query": prompt},
                })
                content = None
                logger.info(f"External API tool call detected: {called_tool}")

        # Also check if the response mentions tool names directly (fallback)
        if not tool_calls:
            for name in tool_names:
                if name in response_text and "retrieve" in name:
                    # Check for patterns like "I would call retrieve_entities"
                    pattern = rf"(?:call|use|invoke|choose)\s+{re.escape(name)}"
                    if re.search(pattern, response_text, re.IGNORECASE):
                        tool_calls.append({
                            "name": name,
                            "arguments": {"query": prompt},
                        })
                        content = None
                        logger.info(f"External API tool call inferred: {name}")
                        break

        finish_reason = "tool_calls" if tool_calls else "stop"
        return {
            "content": content,
            "tool_calls": tool_calls,
            "finish_reason": finish_reason,
        }

    def supports_json_mode(self) -> bool:
        """External API does not support JSON mode."""
        return False

    async def close(self):
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
