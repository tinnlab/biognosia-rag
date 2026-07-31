"""MCP sampling LLM provider: generation is delegated to the MCP client."""

import asyncio
import logging
import re

from .providers import BaseLLMProvider

logger = logging.getLogger(__name__)

# Matches both <thinking>…</thinking> and <think>…</think> reasoning blocks that
# some chat models emit. A no-op when the model emits neither.
_THINKING_RE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL)


class ZieeSamplingProvider(BaseLLMProvider):
    """
    LLM provider backed by MCP sampling.

    Instead of calling an LLM API directly, this provider sends
    sampling/createMessage requests back to the connected MCP client over the
    active SSE connection. The client calls its own model and returns the
    result, so the server needs no API key of its own.

    Used when MCP_LLM_MODE=sampling (the default). In standalone mode the
    server builds a direct provider from the [llm] config instead.

    Each instance is scoped to a single tool call: it holds the session_id
    for routing and the sse_queue for sending sampling requests out.
    """

    def __init__(self, session_id: str, sse_queue: asyncio.Queue):
        super().__init__({})
        self.session_id = session_id
        self.sse_queue = sse_queue

    async def generate(self, prompt: str, system_prompt: str | None = None, **kwargs) -> str:
        """
        Generate a response via MCP sampling.

        Converts the prompt string into MCP message format and sends it to the
        client through the active SSE channel, then waits for the result.
        """
        # Deferred import to avoid circular dependency with mcp_route
        from src_rag_web.mcp_route import ask_llm

        messages = [{"role": "user", "content": {"type": "text", "text": prompt}}]

        logger.info(f"[ZieeProvider] Requesting LLM via MCP sampling (session={self.session_id})")

        result = await ask_llm(
            session_id=self.session_id,
            sse_queue=self.sse_queue,
            messages=messages,
            max_tokens=kwargs.get("max_tokens", 8192),
            system_prompt=system_prompt,
        )
        # Strip extended thinking blocks the model may include
        result = _THINKING_RE.sub("", result).strip()
        return result

    async def generate_with_tools(
        self,
        prompt: str,
        system_prompt: str | None = None,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> dict:
        """
        Simulate function calling via a text-based routing prompt.

        Sends a short routing request to the client's chat model asking which tool to
        call, parses the one-word response, and returns the structure that
        auto_mode._execute_tool_calls() expects.
        """
        from src_rag_web.mcp_route import ask_llm

        routing_system = (
            "You are a routing assistant. Given a user question and a list of tools, "
            "respond with ONLY the name of the tool to call, or 'none' if no tool is needed.\n"
            "Available tools:\n"
            + "\n".join(
                f"- {t['function']['name']}: {t['function']['description']}"
                for t in (tools or [])
            )
            + "\n\nRespond with exactly one word: the tool name, or 'none'."
        )

        logger.info(f"[ZieeProvider] Routing query via MCP sampling (session={self.session_id})")

        raw = await ask_llm(
            session_id=self.session_id,
            sse_queue=self.sse_queue,
            messages=[{"role": "user", "content": {"type": "text", "text": prompt}}],
            max_tokens=50,
            system_prompt=routing_system,
        )
        raw = _THINKING_RE.sub("", raw).strip().lower()

        known_tools = {t["function"]["name"] for t in (tools or [])}
        chosen = raw if raw in known_tools else (
            "retrieve_full_context" if "retrieve_full_context" in known_tools else None
        )

        logger.info(f"[ZieeProvider] Routing decision: {chosen!r} (raw={raw!r})")

        return {
            "tool_calls": [{"name": chosen, "arguments": {"query": prompt}}] if chosen else [],
            "content": "",
            "finish_reason": "tool_calls" if chosen else "stop",
        }

    def supports_json_mode(self) -> bool:
        return False
