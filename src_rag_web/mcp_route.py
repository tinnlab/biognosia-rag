"""MCP route: the query_rag tool, and the sampling round-trip with the client."""

import asyncio
import hmac
import json
import logging
import re
import os
import time
import uuid
from collections import OrderedDict, defaultdict
from typing import Any, AsyncGenerator

from fastapi import APIRouter, Request
from fastapi.responses import Response, StreamingResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# Protocol versions this server accepts.
# We echo the client's requested version when it is one of these, otherwise fall
# back to MCP_PROTOCOL_VERSION_FALLBACK.
SUPPORTED_PROTOCOL_VERSIONS = frozenset(
    {"2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05", "2024-10-07"}
)
_DEFAULT_PROTOCOL_VERSION = "2025-11-25"
MCP_PROTOCOL_VERSION_FALLBACK = os.environ.get(
    "MCP_PROTOCOL_VERSION_FALLBACK", _DEFAULT_PROTOCOL_VERSION
).strip()
# Never negotiate an empty/unsupported version — a missing protocolVersion is a
# hard failure for the client.
if MCP_PROTOCOL_VERSION_FALLBACK not in SUPPORTED_PROTOCOL_VERSIONS:
    MCP_PROTOCOL_VERSION_FALLBACK = _DEFAULT_PROTOCOL_VERSION

# Optional shared-secret auth. When set, every /mcp request (including the
# sampling result POST-back from the client) must carry `Authorization: Bearer <token>`.
_AUTH_TOKEN = os.environ.get("MCP_AUTH_TOKEN", "").strip()

# Optional directory for the full-payload response dumps. Empty = disabled.
_RESPONSE_LOG_DIR = os.environ.get("MCP_RESPONSE_LOG_DIR", "").strip()


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


# Reject request bodies larger than this (bytes) before buffering/parsing them.
_MAX_BODY_BYTES = _int_env("MCP_MAX_BODY_BYTES", 1_048_576)  # 1 MiB
# Cap concurrent in-flight tool calls (each drives the full GPU/DB pipeline).
# 0 or negative = unlimited.
_MAX_CONCURRENT_TOOL_CALLS = _int_env("MCP_MAX_CONCURRENT_TOOL_CALLS", 8)
# Bound on the number of tracked (issued) sessions to prevent memory growth.
_MAX_TRACKED_SESSIONS = 4096

_SESSION_ID_RE = re.compile(r"\A[0-9a-fA-F-]{1,64}\Z")


# Test mode: skip RAGApp and answer query_rag with a minimal sampling round-trip.
def _test_mode() -> bool:
    return os.environ.get("MCP_TEST_MODE", "").strip() in ("1", "true", "yes", "on")


def _standalone_llm() -> bool:
    """True when this server calls the LLM itself instead of asking the client.

    Read from the environment on each call (rather than captured in a module
    constant) so it stays overridable in tests. See _llm_mode() in mcp_server.
    """
    return os.environ.get("MCP_LLM_MODE", "").strip().lower() == "standalone"


def _check_auth(request: Request) -> "Response | None":
    """Return a 401 Response if auth is enabled and the bearer token is wrong."""
    if not _AUTH_TOKEN:
        return None
    header = request.headers.get("authorization", "")
    # Constant-time comparison so the shared secret can't be recovered by timing.
    if hmac.compare_digest(header, f"Bearer {_AUTH_TOKEN}"):
        return None
    return Response(
        content=json.dumps({
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32001, "message": "Unauthorized"},
        }),
        media_type="application/json",
        status_code=401,
    )

# ── Shared state ─────────────────────────────────────────────────────────────
# Pending sampling futures, keyed by "session_id:req_id".
# Per-session atomic counters ensure unique req_ids even across concurrent
# tool calls from the same session.
_pending_sampling: dict[str, asyncio.Future] = {}
_session_req_counters: dict[str, int] = defaultdict(int)
_pending_lock = asyncio.Lock()

# Session ids we issued at initialize (bounded LRU). Non-initialize requests must
# carry one of these, so an attacker can't grow server state with random ids or
# inject sampling replies into a victim's session.
_issued_sessions: "OrderedDict[str, None]" = OrderedDict()
_active_tool_calls = 0


def _register_session(session_id: str) -> None:
    _issued_sessions[session_id] = None
    _issued_sessions.move_to_end(session_id)
    while len(_issued_sessions) > _MAX_TRACKED_SESSIONS:
        old, _ = _issued_sessions.popitem(last=False)
        _session_req_counters.pop(old, None)


def _known_session(session_id: str) -> bool:
    return session_id in _issued_sessions


def _ts() -> str:
    t = time.time()
    return f"{int(t // 3600 % 24):02}:{int(t // 60 % 60):02}:{int(t % 60):02}.{int((t % 1) * 1000):03}"


def _sse(data: dict) -> str:
    # Plain data-only SSE format — no "event:" line
    return f"data: {json.dumps(data)}\n\n"


def _ok(data: dict, session_id: str) -> Response:
    return Response(
        content=json.dumps(data),
        media_type="application/json",
        headers={"mcp-session-id": session_id},
    )


# ── ask_llm: used by ZieeSamplingProvider ────────────────────────────────────

async def ask_llm(
    session_id: str,
    sse_queue: asyncio.Queue,
    messages: list[dict],
    max_tokens: int = 500,
    system_prompt: str | None = None,
) -> str:
    """
    Send a sampling/createMessage request to the client via SSE and await the result.

    The req_id is generated atomically per session so that multiple concurrent
    tool calls from the same session never produce colliding IDs.
    """
    loop = asyncio.get_event_loop()

    async with _pending_lock:
        _session_req_counters[session_id] += 1
        req_id = _session_req_counters[session_id]
        future: asyncio.Future = loop.create_future()
        _pending_sampling[f"{session_id}:{req_id}"] = future

    payload = {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": "sampling/createMessage",
        "params": {
            "messages": messages,
            "maxTokens": max_tokens,
            **({"systemPrompt": system_prompt} if system_prompt else {}),
        },
    }

    logger.info(f"[MCP {_ts()}] → sampling/createMessage id={req_id} session={session_id}")
    await sse_queue.put(payload)

    try:
        result = await asyncio.wait_for(future, timeout=600.0)
    except asyncio.TimeoutError:
        logger.warning(f"[MCP {_ts()}] Sampling id={req_id} session={session_id} timed out")
        return "[LLM request timed out]"
    finally:
        async with _pending_lock:
            _pending_sampling.pop(f"{session_id}:{req_id}", None)

    if result is None:
        logger.warning(f"[MCP {_ts()}] Sampling id={req_id} session={session_id}: result is None")
        return ""
    # Clients return generation errors as a normal result with stopReason "error"
    # (not a JSON-RPC error). Surface it so failures aren't silently swallowed.
    if result.get("stopReason") == "error":
        logger.error(f"[MCP {_ts()}] Sampling id={req_id} session={session_id} returned stopReason=error: {result!r}")
    content = result.get("content", {})
    text = content.get("text", "") if isinstance(content, dict) else str(content)
    if not text:
        logger.warning(f"[MCP {_ts()}] Sampling id={req_id} session={session_id}: empty text in result={result!r}")
    return text


# ── MCP handlers ─────────────────────────────────────────────────────────────

def _handle_initialize(body: dict) -> Response:
    # Always generate a fresh UUID — never reuse across calls.
    # Parallel sessions each get their own unique ID so their futures never collide.
    new_session_id = str(uuid.uuid4())
    _register_session(new_session_id)

    # Echo the client's requested protocol version when we recognise it;
    # otherwise negotiate down to a supported fallback.
    requested = (body.get("params") or {}).get("protocolVersion")
    protocol_version = (
        requested if requested in SUPPORTED_PROTOCOL_VERSIONS
        else MCP_PROTOCOL_VERSION_FALLBACK
    )

    # `sampling` is advertised only when we will actually drive the CLIENT's
    # LLM. In standalone mode this server never sends sampling/createMessage, so
    # claiming the capability would tell a client to implement a callback that
    # is exactly what standalone mode exists to avoid. It also makes a single
    # `initialize` request the cheapest way to see which mode a deployment
    # booted in. Test mode keeps it: the canned pipeline does a real round-trip.
    capabilities: dict[str, dict] = {"tools": {}}
    if _test_mode() or not _standalone_llm():
        capabilities["sampling"] = {}

    return Response(
        content=json.dumps({
            "jsonrpc": "2.0",
            "id": body.get("id", 1),
            "result": {
                "protocolVersion": protocol_version,
                "capabilities": capabilities,
                "serverInfo": {"name": "biognosia-rag", "version": "1.0.0"},
            },
        }),
        media_type="application/json",
        headers={"mcp-session-id": new_session_id},
    )


def _handle_tools_list(body: dict, session_id: str) -> Response:
    return _ok({
        "jsonrpc": "2.0",
        "id": body.get("id", 1),
        "result": {
            "tools": [{
                "name": "query_rag",
                "description": (
                    "Use this tool when the user asks about a SPECIFIC biological subject - a particular "
                    "gene, protein, metabolite, pathway, network, cell type or disease - or about their own "
                    "pathway/omics/enrichment results, or about what the published literature reports. Call "
                    "it even when you already know the answer, and even when the question is phrased 'what "
                    "is X'. "
                    "Do NOT call it for a GENERAL concept, method or term that names no specific biological "
                    "subject (e.g. 'what is pathway analysis?', 'what is systems biology?', 'how does "
                    "enrichment analysis work?') - answer those yourself, without the tool. The knowledge "
                    "base holds research papers, not definitions, so searching it for a definition returns "
                    "nothing or an inconsistent answer. "
                    "Do NOT call it for greetings, small talk, or questions outside systems biology. "
                    "Pass the user's full question in a single call — don't break a multi-part or comparative "
                    "question into several calls; the knowledge base handles complex, multi-part, and comparative "
                    "questions best in one query, and splitting produces fragmented, inconsistent answers."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "The user's complete question, copied verbatim, no matter how long or complex. "
                                "Include all sub-questions, comparisons, gene/protein lists, and context in one string. "
                                "Example: 'How do feedback loops and crosstalk between MAPK and PI3K/AKT contribute to "
                                "drug resistance, and how can network-based modeling identify therapeutic targets?'"
                            ),
                        },
                    },
                    "required": ["query"],
                },
            }],
        },
    }, session_id)


async def _handle_sampling_result(body: dict, session_id: str) -> Response:
    """
    The client delivers the LLM result for a pending sampling request.
    Body has no 'method' — just 'id' and 'result'.
    """
    req_id = body.get("id")
    result = body.get("result")
    logger.info(f"[MCP {_ts()}] ← sampling result id={req_id} session={session_id}")

    key = f"{session_id}:{req_id}"
    async with _pending_lock:
        future = _pending_sampling.get(key)

    if future and not future.done():
        future.set_result(result)
    else:
        logger.warning(f"[MCP {_ts()}] No pending future for id={req_id} session={session_id}")

    # The sampling reply is a pure JSON-RPC response — ack with 202 + no body.
    return Response(status_code=202, headers={"mcp-session-id": session_id})


def _strip_leading_headers(text: str) -> str:
    """Strip leading-only markdown headers; convert mid-text headers to plain text lines."""
    lines = text.split("\n")
    # 1. Skip lines that are purely headers at the start
    start = 0
    while start < len(lines) and re.match(r"^#{1,6}\s+", lines[start].strip()):
        start += 1
    # 2. For remaining lines, strip '#' prefix from any header line (keep the text)
    body_lines = []
    for line in lines[start:]:
        m = re.match(r"^#{1,6}\s+(.*)", line.strip())
        if m:
            body_lines.append(m.group(1))  # keep header text, drop the '#'s
        else:
            body_lines.append(line)
    result = "\n".join(body_lines).strip()
    return result if result else text


def _author_names(authors: Any) -> list[str]:
    """Coerce a bibtex authors field into display names.

    Mongo records are not uniform: authors may be a list of strings, a list of
    dicts ({"name": ...} or {"family": ..., "given": ...}), or a bare string.
    Anything unrecognized is skipped — a footnote missing its author list beats
    a TypeError that turns the whole answer into an internal server error.
    """
    if isinstance(authors, str):
        return [authors.strip()] if authors.strip() else []
    if not isinstance(authors, list):
        return []

    names = []
    for author in authors:
        if isinstance(author, str):
            name = author.strip()
        elif isinstance(author, dict):
            name = str(author.get("name") or "").strip()
            if not name:
                family = str(author.get("family") or "").strip()
                given = str(author.get("given") or "").strip()
                name = f"{family} {given}".strip()
        else:
            name = ""
        if name:
            names.append(name)
    return names


def _journal_name(journal: Any) -> str:
    """Journal may be a {"name": ...} dict or a bare string, depending on the record."""
    if isinstance(journal, dict):
        return str(journal.get("name") or "")
    if isinstance(journal, str):
        return journal.strip()
    return ""


def _format_footnote_definition(label: str, meta: dict, chunk_text: str) -> str:
    """Format a citation as a markdown footnote definition.

    `label` is the footnote IDENTIFIER, a string — "3" for a paper cited via a
    single chunk, "1-2" for the 2nd cited chunk of paper 1 (shown to the reader
    as "1.2"). It is NOT the number the reader sees: GFM renumbers footnotes
    sequentially by first reference and the label survives only in the anchor
    id/href. A renderer reads it back out of there to display the
    hierarchy (see _assign_footnote_labels).
    """
    body = _strip_leading_headers(chunk_text)
    bibtex = meta.get("bibtex_json")
    if not isinstance(bibtex, dict):
        bibtex = {}
    authors = _author_names(bibtex.get("authors"))
    year = bibtex.get("year") or ""
    journal = _journal_name(meta.get("journal"))
    title = str(meta.get("title") or bibtex.get("title") or "").strip()

    # Build the reference header: Authors (Year). Title. *Journal*. DOI
    # The title MUST be included — without it the LLM tends to invent a
    # plausible-but-wrong title from the chunk text (the metadata below has it).
    parts = []
    if authors:
        parts.append(", ".join(authors))
    if year:
        parts.append(f"({year})")
    header = " ".join(parts)
    if title:
        header = f"{header}. {title.rstrip('.')}." if header else f"{title.rstrip('.')}."
    if journal:
        header += f" *{journal}*"
    if meta.get("doi"):
        header += f". https://doi.org/{meta['doi']}"

    # Multi-line footnote: continuation lines indented 4 spaces
    quoted = "\n    ".join(
        f"> {line}" if line.strip() else ">"
        for line in body.split("\n")
    )

    if header:
        return f"[^{label}]: {header}\n    {quoted}"
    else:
        return f"[^{label}]: {quoted}"


def _label_sort_key(label: str) -> tuple[int, ...]:
    """Sort key for a footnote label: "1-10" sorts AFTER "1-2", not before.

    Component-wise numeric, so a flat "2" orders between "1-3" and "3-1".
    """
    return tuple(int(part) for part in label.split("-"))


def _assign_footnote_labels(cited: list[tuple[str, str]]) -> dict[str, str]:
    """Map each cited marker to its footnote label, grouping chunks by paper.

    `cited` is [(marker, paper_id), ...] in order of first appearance, already
    filtered to markers that resolved to real metadata.

    Papers are numbered P by first appearance. A paper cited through a SINGLE
    chunk keeps a FLAT label ("2"): the hierarchy only appears where it carries
    information, so an answer whose papers each contribute one chunk still reads
    "1, 2, 3" exactly as it always has. A paper cited through several chunks
    labels them "P-1", "P-2", … in first-appearance order, which is what lets
    the renderer merge them into ONE bibliographic entry with nested excerpts
    (issue #167). The two forms mix freely within one answer.

    Mixing does mean the renderer cannot just trust GFM's numbering: GFM
    renumbers footnotes sequentially by first reference, so a flat "2" sitting
    after "1-1" and "1-2" occupies ordinal position 3 and would DISPLAY as 3.
    The renderer handles that — on a paper-grouped answer it shows every label
    verbatim (`stampFootnoteRefLabels` in rehypeGroupPaperFootnotes.ts) — but a
    reader of this function should know the ordinal and the label diverge here.

    The separator is a HYPHEN on the wire; the reader is shown "1.1". A dot here
    would be silently fatal: a renderer that splits a message into blocks
    before parsing, and a footnote definition labelled "[^1.1]:" makes that
    splitter cut the answer apart, so the references end up in different blocks
    from the markers that point at them and NOTHING renders as a footnote. Every
    other separator keeps the answer whole. Do not "tidy" this back to a dot.
    """
    papers: dict[str, list[str]] = {}
    for marker, paper_id in cited:
        papers.setdefault(paper_id, []).append(marker)

    labels: dict[str, str] = {}
    for paper_index, markers in enumerate(papers.values(), start=1):
        if len(markers) == 1:
            labels[markers[0]] = str(paper_index)
            continue
        for chunk_index, marker in enumerate(markers, start=1):
            labels[marker] = f"{paper_index}-{chunk_index}"
    return labels


async def _resolve_footnotes(response_text: str, raw_refs: list[dict], rag_app: Any) -> str:
    """
    Replace [chunk-...] markers in response_text with [^n] footnote references
    and append footnote definitions at the bottom of the text.

    Batch-fetches paper metadata from MongoDB (via rag_app.mongo_storage).
    Markers with no metadata are stripped. Returns the complete text with
    citations embedded as markdown footnotes.
    """
    # The prompt asks for ASCII brackets, but weaker models intermittently wrap
    # citation markers in full-width brackets instead (【…】 U+3010/U+3011, ［…］
    # U+FF3B/U+FF3D) — the marker regexes below then miss it and the raw marker
    # leaks to the user. Normalize first so every model behaves the same.
    # Deliberately scoped to brackets that actually wrap a chunk marker: an
    # unrelated 【…】 elsewhere in the answer is left alone.
    response_text = re.sub(
        r"[\[【［]\s*(chunk-[a-fA-F0-9]+-\d+)\s*[\]】］]", r"[\1]", response_text
    )
    # Same normalization for the short handles the prompt now hands out ([C1]).
    response_text = re.sub(
        r"[\[【［]\s*([cC]\d+(?:\s*[,;]\s*[cC]\d+)*)\s*[\]】］]", r"[\1]", response_text
    )

    # Short-handle markers -> chunk ids. The prompt presents chunks as [C1],
    # [C2], ... instead of their raw chunk-{40-hex}-{6-digit} id, because models
    # abbreviate a long hash (`chunk-b080ee8...-000003`) and the resulting marker
    # matches nothing below, leaking to the user. Rewriting handles here keeps
    # the whole footnote pipeline downstream unchanged — and the chunk-id path
    # stays in place for any answer that still emits a full id.
    handle_map = {
        str(ref["handle"]).upper(): ref.get("id")
        for ref in raw_refs
        if ref.get("handle") and ref.get("id")
    }

    # Models sometimes group handles in one bracket: [C1, C2] -> [C1][C2]
    def _split_handle_group(m: re.Match) -> str:
        return "".join(f"[{h.upper()}]" for h in re.findall(r"[cC]\d+", m.group(0)))

    response_text = re.sub(
        r"\[[cC]\d+(?:\s*[,;]\s*[cC]\d+)+\]", _split_handle_group, response_text
    )

    def _replace_handle(m: re.Match) -> str:
        chunk_id = handle_map.get(m.group(1).upper())
        # An invented / out-of-range handle is dropped, same as an unresolvable
        # chunk marker — better a missing citation than a raw [C9] in the answer.
        return f"[{chunk_id}]" if chunk_id else ""

    response_text = re.sub(r"\[([cC]\d+)\]", _replace_handle, response_text)

    if not raw_refs:
        return re.sub(r"\[chunk-[a-fA-F0-9]+-\d+\]", "", response_text).strip()

    # Parse paper IDs from chunk IDs: chunk-{40-char-hex}-{6-digit-index}
    chunk_paper_map: dict[str, str] = {}
    for ref in raw_refs:
        # `or ""`, not a .get default: a chunk stored with id=None has the key,
        # so the default never fires and re.match(pattern, None) raises.
        chunk_id = ref.get("id") or ""
        match = re.match(r"chunk-([a-f0-9]{40})-\d+", chunk_id)
        if match:
            chunk_paper_map[chunk_id] = match.group(1)

    # Batch-fetch paper metadata from MongoDB
    paper_meta: dict[str, dict] = {}
    mongo = getattr(rag_app, "mongo_storage", None)
    if mongo is None:
        logger.warning("[MCP] mongo_storage is None — skipping metadata lookup (server restarted? motor installed? init error?)")
    else:
        unique_paper_ids = list(set(chunk_paper_map.values()))
        logger.info(f"[MCP] Looking up {len(unique_paper_ids)} paper IDs in MongoDB: {unique_paper_ids[:3]}{'...' if len(unique_paper_ids) > 3 else ''}")
        try:
            paper_meta = await mongo.get_papers_metadata(unique_paper_ids)
            logger.info(f"[MCP] MongoDB returned {len(paper_meta)}/{len(unique_paper_ids)} papers. Found IDs: {list(paper_meta.keys())[:3]}")
        except Exception as e:
            logger.warning(f"[MCP] MongoDB metadata lookup failed: {e}")

    # Build ref lookup by chunk_id for fast access to chunk content
    ref_by_id: dict[str, dict] = {(ref.get("id") or ""): ref for ref in raw_refs}

    # Find all [chunk-...] markers in order of first appearance (deduplicated)
    all_markers = list(dict.fromkeys(re.findall(r"\[chunk-[a-fA-F0-9]+-\d+\]", response_text)))

    # PASS 1 — resolve each marker to its paper, dropping the ones with no
    # metadata (exactly as before: a dropped marker consumes no index, so the
    # labels assigned in pass 2 stay dense).
    cited: list[tuple[str, str]] = []
    resolved: dict[str, tuple[dict, str]] = {}
    skipped = []

    for marker in all_markers:
        # Strip [ and ]. Lower-cased because a model may echo the hash in upper
        # case, while both maps are keyed by the DB's lower-case chunk id.
        chunk_id = marker[1:-1].lower()
        paper_id = chunk_paper_map.get(chunk_id)
        meta = paper_meta.get(paper_id) if paper_id else None

        if not meta:
            skipped.append(chunk_id)
            continue

        ref = ref_by_id.get(chunk_id)
        chunk_text = ref.get("content") or ref.get("chunk_content", "") if ref else ""

        cited.append((marker, paper_id))
        resolved[marker] = (meta, chunk_text)

    if skipped:
        logger.info(f"[MCP] Dropped {len(skipped)} footnotes with no metadata: {skipped[:3]}{'...' if len(skipped) > 3 else ''}")

    # PASS 2 — group the cited chunks by paper and label them. The value is a
    # STRING label ("1.2"), never an int: that is what carries the paper→excerpt
    # hierarchy through the markdown into the renderer.
    chunk_to_footnote: dict[str, str] = _assign_footnote_labels(cited)

    # Emit definitions grouped by paper so the raw markdown reads in the same
    # order the rendered reference list does (the renderer groups regardless).
    footnote_defs: list[str] = [
        _format_footnote_definition(chunk_to_footnote[marker], *resolved[marker])
        for marker in sorted(
            resolved, key=lambda mk: _label_sort_key(chunk_to_footnote[mk])
        )
    ]

    def replace_marker(m: re.Match) -> str:
        label = chunk_to_footnote.get(m.group(0))
        return f"[^{label}]" if label is not None else ""

    result = re.sub(r"\[chunk-[a-fA-F0-9]+-\d+\]", replace_marker, response_text).strip()

    # Normalize a run of co-located refs: sort ascending ([^3][^1] → [^1][^3],
    # [^1.10] after [^1.2]) and CLOSE UP any whitespace inside the run. The
    # renderer separates adjacent citation superscripts with a comma using an
    # adjacent-sibling CSS rule, which only holds if the markers really are
    # siblings — so `[^1] [^2]` must collapse to `[^1][^2]` here.
    def _sort_ref_group(m: re.Match) -> str:
        labels = sorted(re.findall(r"\[\^([\d-]+)\]", m.group(0)), key=_label_sort_key)
        return "".join(f"[^{s}]" for s in labels)

    result = re.sub(r"\[\^[\d-]+\](?:[ \t]*\[\^[\d-]+\])+", _sort_ref_group, result)

    if footnote_defs:
        result += "\n\n" + "\n\n".join(footnote_defs)

    return result


async def _handle_tool_call_sse(body: dict, session_id: str, rag_app: Any):
    global _active_tool_calls
    call_id = body.get("id", 1)
    args = body.get("params", {}).get("arguments", {})
    query = args.get("query", "")
    mode = args.get("mode", "auto")

    # Defensive routing guard: only query_rag is registered. Reject a wrong tool
    # name up front (before reserving resources or importing the provider).
    tool_name = (body.get("params") or {}).get("name")
    if tool_name is not None and tool_name != "query_rag":
        return _ok({
            "jsonrpc": "2.0",
            "id": call_id,
            "error": {"code": -32602, "message": f"Unknown tool: {tool_name}"},
        }, session_id)

    # Reserve a concurrency slot; refuse when the server is at capacity so a burst
    # of tool calls can't saturate the GPU/DBs/memory.
    async with _pending_lock:
        if 0 < _MAX_CONCURRENT_TOOL_CALLS <= _active_tool_calls:
            return _ok({
                "jsonrpc": "2.0",
                "id": call_id,
                "error": {"code": -32000, "message": "Server busy: max concurrent tool calls reached"},
            }, session_id)
        _active_tool_calls += 1

    logger.info(f"[MCP {_ts()}] SSE stream opening query={query!r} mode={mode} session={session_id}")

    sse_queue: asyncio.Queue = asyncio.Queue()

    async def _test_pipeline() -> dict:
        """Minimal canned sampling round-trip used when MCP_TEST_MODE is on.

        Exercises the real SSE + future-resolution machinery (via ask_llm) so we
        can validate the client handshake end-to-end without any DB or GPU.
        """
        prompt = os.environ.get("MCP_TEST_PROMPT") or (
            "This is a Biognosia MCP connectivity test. "
            "Reply with exactly the single word: PONG."
        )
        messages = [{"role": "user", "content": {"type": "text", "text": prompt}}]
        text = await ask_llm(session_id, sse_queue, messages, max_tokens=256)
        return {"response": text, "references": []}

    async def stream() -> AsyncGenerator[bytes, None]:
        global _active_tool_calls
        pipeline_task: asyncio.Task | None = None
        queue_task: asyncio.Task | None = None
        try:
            if _test_mode():
                pipeline_task = asyncio.create_task(_test_pipeline())
            elif _standalone_llm():
                # No per-request provider: rag_app.query() falls back to the
                # app-level provider built at startup, and its own guard raises
                # a clear error if that is somehow absent. The SSE queue simply
                # stays empty, so the loop below sends keepalives until the
                # pipeline finishes and then the single final frame.
                logger.info(
                    f"[MCP {_ts()}] standalone LLM: "
                    f"{type(rag_app.llm_provider).__name__} "
                    f"model={getattr(rag_app.llm_provider, 'model', None)!r} "
                    f"session={session_id}"
                )
                pipeline_task = asyncio.create_task(
                    rag_app.query(query_text=query, mode=mode)
                )
            else:
                # Deferred import to avoid circular dependency with ziee_provider.
                from src_rag_web.llm.ziee_provider import ZieeSamplingProvider
                provider = ZieeSamplingProvider(session_id=session_id, sse_queue=sse_queue)
                pipeline_task = asyncio.create_task(
                    rag_app.query(query_text=query, mode=mode, llm_provider=provider)
                )
            while True:
                queue_task = asyncio.create_task(sse_queue.get())
                done, _ = await asyncio.wait(
                    [queue_task, pipeline_task],
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=15.0,
                )

                if not done:
                    # 15s idle — no sampling request or pipeline completion yet.
                    # Cancel the pending queue reader (safe: queue is empty) and
                    # send an SSE comment to keep the connection alive.
                    queue_task.cancel()
                    yield b": keepalive\n\n"
                    continue

                if pipeline_task in done:
                    # Pipeline finished — cancel dangling queue reader, send final result
                    queue_task.cancel()
                    result = pipeline_task.result()
                    response_text = result.get("response") or ""
                    raw_refs = result.get("references") or []

                    response_text = await _resolve_footnotes(response_text, raw_refs, rag_app)

                    content_blocks: list[dict] = [
                        {
                            "type": "text",
                            "text": response_text,
                            "annotations": {"audience": ["user"]},
                        }
                    ]
                    final = {
                        "jsonrpc": "2.0",
                        "id": call_id,
                        "result": {
                            "content": content_blocks,
                            "isError": False,
                            "is_final_response": True,
                        },
                    }
                    logger.info(
                        f"[MCP {_ts()}] ══════════════ FINAL RESPONSE session={session_id} "
                        f"({len(response_text)} chars) ══════════════"
                    )
                    if _RESPONSE_LOG_DIR:
                        try:
                            ts_file = time.strftime("%Y%m%d_%H%M%S")
                            os.makedirs(_RESPONSE_LOG_DIR, exist_ok=True)
                            log_path = os.path.join(
                                _RESPONSE_LOG_DIR,
                                f"response_{ts_file}_{session_id[:8]}.txt",
                            )
                            with open(log_path, "w", encoding="utf-8") as f:
                                f.write(f"Session: {session_id}\n")
                                f.write(f"Query:   {query}\n")
                                f.write(f"Time:    {ts_file}\n")
                                f.write("=" * 60 + "\n\n")
                                f.write("FULL SSE PAYLOAD SENT TO CLIENT:\n")
                                f.write("-" * 60 + "\n")
                                f.write(json.dumps(final, indent=2, ensure_ascii=False))
                                f.write("\n")
                            logger.info(f"[MCP {_ts()}] Full payload saved to {log_path}")
                        except Exception as save_err:
                            logger.warning(f"[MCP {_ts()}] Failed to save response log: {save_err}")
                    yield _sse(final).encode()
                    return
                else:
                    # Sampling request ready — forward to the client via SSE
                    sampling_req = queue_task.result()
                    logger.info(
                        f"[MCP {_ts()}] → forwarding sampling id={sampling_req.get('id')} session={session_id}"
                    )
                    yield _sse(sampling_req).encode()

        except Exception as e:
            # Log the detail server-side; send only a generic message to the
            # client so internal state (e.g. DB connection strings) isn't leaked.
            logger.error(f"[MCP {_ts()}] Pipeline error session={session_id}: {e}", exc_info=True)
            yield _sse({
                "jsonrpc": "2.0",
                "id": call_id,
                "error": {"code": -32603, "message": "Internal server error"},
            }).encode()
        finally:
            # Runs on normal completion, error, or client disconnect
            # (GeneratorExit): cancel outstanding work and release the slot.
            # The decrement is a single atomic op (no await → no context switch),
            # so it needs no lock and is safe to run during generator cleanup.
            for t in (queue_task, pipeline_task):
                if t is not None and not t.done():
                    t.cancel()
            _active_tool_calls -= 1

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "mcp-session-id": session_id,
            "x-accel-buffering": "no",
            "cache-control": "no-cache",
        },
    )


# ── Main dispatcher ───────────────────────────────────────────────────────────

@router.post("/mcp")
async def mcp_endpoint(request: Request) -> Response:
    # Optional shared-secret auth applied uniformly to every /mcp request,
    # including initialize and the client's sampling result POST-back.
    auth_error = _check_auth(request)
    if auth_error is not None:
        return auth_error

    # Reject oversized bodies before buffering them into memory.
    clen = request.headers.get("content-length")
    if clen is not None:
        try:
            if int(clen) > _MAX_BODY_BYTES:
                return Response(status_code=413)
        except ValueError:
            return Response(status_code=400)

    try:
        body = await request.json()
    except Exception:
        return Response(status_code=400)

    method = body.get("method")

    # initialize generates its own session ID — no header required
    if method == "initialize":
        logger.info(f"[MCP {_ts()}] ← initialize")
        return _handle_initialize(body)

    # All other methods require a session id that WE issued at initialize.
    session_id = request.headers.get("mcp-session-id")
    if session_id is None or not _SESSION_ID_RE.match(session_id) or not _known_session(session_id):
        return Response(
            content=json.dumps({
                "jsonrpc": "2.0",
                "id": body.get("id"),
                "error": {"code": -32600, "message": "Missing or unknown mcp-session-id"},
            }),
            media_type="application/json",
            status_code=400,
        )

    logger.info(f"[MCP {_ts()}] ← {method or 'sampling_result'} session={session_id}")

    # Sampling result delivery: no 'method', but has 'result' and 'id'
    if method is None and "result" in body:
        return await _handle_sampling_result(body, session_id)

    if method == "notifications/initialized":
        # Notification (no response expected) — ack with 202 + no body.
        return Response(status_code=202, headers={"mcp-session-id": session_id})

    if method == "tools/list":
        return _handle_tools_list(body, session_id)

    if method == "tools/call":
        accept = request.headers.get("accept", "")
        if "text/event-stream" not in accept:
            return _ok({
                "jsonrpc": "2.0",
                "id": body.get("id", 1),
                "error": {"code": -32600, "message": "tools/call requires Accept: text/event-stream"},
            }, session_id)
        rag_app = request.app.state.rag_app
        return await _handle_tool_call_sse(body, session_id, rag_app)

    # Unknown method
    return Response(
        content=json.dumps({
            "jsonrpc": "2.0",
            "id": body.get("id"),
            "error": {"code": -32601, "message": f"Unknown method: {method}"},
        }),
        media_type="application/json",
        status_code=400,
    )
