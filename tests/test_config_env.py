"""Tests for environment-variable parameterization of the RAG config."""
import re
from pathlib import Path

import pytest

from src_rag_web.config import (
    PLACEHOLDER_RE,
    check_endpoint_placeholders,
    load_config,
    validate_llm_config,
)

CONF = Path(__file__).resolve().parents[1] / "config" / "rag-web.conf"

# Values that would be a leak if they ever appeared in the committed config.
# Deliberately expressed as PATTERNS rather than as the literal strings we are
# guarding against: a test that spells out a real password or internal address
# in order to assert its absence publishes it just as effectively as the file
# would have.
_LEAK_PATTERNS = [
    # RFC1918 / link-local addresses — an internal host that escaped into config.
    (r"\b(?:10|127)\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "private IPv4 address"),
    (r"\b192\.168\.\d{1,3}\.\d{1,3}\b", "private IPv4 address"),
    (r"\b172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}\b", "private IPv4 address"),
    # Absolute paths outside the project — a developer's or server's filesystem.
    (r"(?:^|[\s=\"'])/(?:home|Users|data|mnt|srv)/", "absolute host path"),
    # Common credential shapes.
    (r"\bsk-[A-Za-z0-9_-]{16,}", "API key"),
    (r"\bhf_[A-Za-z0-9]{16,}", "HuggingFace token"),
    (r"\bghp_[A-Za-z0-9]{20,}", "GitHub token"),
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "private key"),
    # A credential embedded in a connection URI, e.g. mongodb://user:pw@host.
    (r"://[^/\s:]+:[^/\s@]+@", "credential in a URI"),
]

# Values that are safe to see on the right-hand side of a secret-ish key.
_PLACEHOLDER_VALUES = {"", "changeme", "your-password-here", "<set-in-env>"}


def test_committed_conf_has_no_secrets():
    """The committed config must carry no credentials, keys or internal hosts.

    Endpoints and credentials come from the environment (see .env.example); the
    file in git holds only localhost defaults and placeholders.
    """
    findings = []
    for lineno, line in enumerate(CONF.read_text().splitlines(), start=1):
        if line.lstrip().startswith((";", "#")):
            continue  # comments may legitimately discuss example values
        for pattern, label in _LEAK_PATTERNS:
            if re.search(pattern, line):
                findings.append(f"{CONF.name}:{lineno}: possible {label}: {line.strip()!r}")
    assert not findings, "secrets in committed config:\n" + "\n".join(findings)


def test_committed_conf_has_no_populated_credential_fields():
    """`password = ...` / `api_key = ...` must be empty or an obvious placeholder."""
    findings = []
    for lineno, line in enumerate(CONF.read_text().splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith((";", "#")) or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        if key.strip().lower() not in {"password", "api_key", "token", "secret"}:
            continue
        # Strip a trailing inline comment before judging the value.
        value = value.split(";", 1)[0].split("#", 1)[0].strip()
        if value.lower() not in _PLACEHOLDER_VALUES:
            findings.append(f"{CONF.name}:{lineno}: {key.strip()} is set to a non-placeholder value")
    assert not findings, "credentials in committed config:\n" + "\n".join(findings)


def test_env_overrides_all_db_endpoints(monkeypatch):
    monkeypatch.setenv("BIOGNOSIA_MONGODB_URI", "mongodb://h:1/x")
    monkeypatch.setenv("BIOGNOSIA_MILVUS_HOST", "milvus.example")
    monkeypatch.setenv("BIOGNOSIA_MILVUS_PORT", "19531")
    monkeypatch.setenv("BIOGNOSIA_REDIS_HOST", "redis.example")
    monkeypatch.setenv("BIOGNOSIA_REDIS_PASSWORD", "rpw")
    monkeypatch.setenv("BIOGNOSIA_NEO4J_URI", "bolt://neo.example:7687")
    monkeypatch.setenv("BIOGNOSIA_NEO4J_PASSWORD", "npw")
    monkeypatch.setenv("BIOGNOSIA_ELASTICSEARCH_HOSTS", "http://es.example:9200")

    cfg = load_config(str(CONF))
    assert cfg["mongodb"]["uri"] == "mongodb://h:1/x"
    assert cfg["milvus"]["host"] == "milvus.example"
    assert cfg["milvus"]["port"] == 19531 and isinstance(cfg["milvus"]["port"], int)
    assert cfg["redis"]["host"] == "redis.example"
    assert cfg["redis"]["password"] == "rpw"
    assert cfg["neo4j"]["uri"] == "bolt://neo.example:7687"
    assert cfg["neo4j"]["password"] == "npw"
    assert cfg["elasticsearch"]["hosts"] == "http://es.example:9200"


def test_empty_env_does_not_clobber_default(monkeypatch):
    monkeypatch.setenv("BIOGNOSIA_MILVUS_HOST", "")
    cfg = load_config(str(CONF))
    # Falls back to the value in the committed conf (localhost), not "".
    assert cfg["milvus"]["host"] == "localhost"


def test_enabled_bool_and_device_shortcut(monkeypatch):
    monkeypatch.setenv("BIOGNOSIA_ELASTICSEARCH_ENABLED", "false")
    monkeypatch.setenv("BIOGNOSIA_EMBEDDING_DEVICE", "cuda:2")
    cfg = load_config(str(CONF))
    assert cfg["elasticsearch"]["enabled"] is False
    assert cfg["embedding"]["chunk_model_device"] == "cuda:2"
    assert cfg["embedding"]["label_model_device"] == "cuda:2"
    assert cfg["embedding"]["stage2_model_device"] == "cuda:2"


def test_specific_device_overrides_shortcut(monkeypatch):
    monkeypatch.setenv("BIOGNOSIA_EMBEDDING_DEVICE", "cuda:2")
    monkeypatch.setenv("BIOGNOSIA_EMBEDDING_CHUNK_DEVICE", "cuda:3")
    cfg = load_config(str(CONF))
    assert cfg["embedding"]["chunk_model_device"] == "cuda:3"
    assert cfg["embedding"]["label_model_device"] == "cuda:2"


def test_bad_int_env_is_ignored(monkeypatch):
    monkeypatch.setenv("BIOGNOSIA_MILVUS_PORT", "not-a-number")
    cfg = load_config(str(CONF))
    assert cfg["milvus"]["port"] == 19530  # committed default preserved


# ── LLM configuration ────────────────────────────────────────────────────────

def test_llm_section_absent_yields_usable_defaults():
    """The committed conf keeps [llm] commented out; the shape must still be sane."""
    cfg = load_config(str(CONF))["llm"]
    assert cfg["provider"] == "openai"
    assert cfg["model"] == ""
    assert cfg["base_url"] == ""
    assert cfg["timeout"] == 300
    # Optional params must stay ABSENT: present-but-None would be sent to the API.
    assert "temperature" not in cfg
    assert "top_p" not in cfg
    assert "max_tokens" not in cfg
    assert "max_completion_tokens" not in cfg


def test_env_overrides_llm_settings(monkeypatch):
    monkeypatch.setenv("BIOGNOSIA_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("BIOGNOSIA_LLM_MODEL", "some-model")
    monkeypatch.setenv("BIOGNOSIA_LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("BIOGNOSIA_LLM_API_KEY", "unit-test-value")
    monkeypatch.setenv("BIOGNOSIA_LLM_TIMEOUT", "120")
    monkeypatch.setenv("BIOGNOSIA_LLM_TEMPERATURE", "0.2")
    monkeypatch.setenv("BIOGNOSIA_LLM_MAX_TOKENS", "8192")
    monkeypatch.setenv("BIOGNOSIA_LLM_ENABLE_COT", "false")

    cfg = load_config(str(CONF))["llm"]
    assert cfg["provider"] == "anthropic"
    assert cfg["model"] == "some-model"
    assert cfg["base_url"] == "http://localhost:11434/v1"
    assert cfg["api_key"] == "unit-test-value"
    assert cfg["timeout"] == 120 and isinstance(cfg["timeout"], int)
    assert cfg["temperature"] == 0.2 and isinstance(cfg["temperature"], float)
    assert cfg["max_tokens"] == 8192 and isinstance(cfg["max_tokens"], int)
    assert cfg["enable_cot"] is False


def test_max_completion_tokens_env_supersedes_max_tokens(monkeypatch):
    """The provider always prefers max_completion_tokens, so max_tokens must go."""
    monkeypatch.setenv("BIOGNOSIA_LLM_MAX_TOKENS", "4096")
    monkeypatch.setenv("BIOGNOSIA_LLM_MAX_COMPLETION_TOKENS", "8192")
    cfg = load_config(str(CONF))["llm"]
    assert cfg["max_completion_tokens"] == 8192
    assert "max_tokens" not in cfg


def test_bad_float_llm_temperature_is_ignored(monkeypatch):
    monkeypatch.setenv("BIOGNOSIA_LLM_TEMPERATURE", "warm")
    assert "temperature" not in load_config(str(CONF))["llm"]


def test_empty_llm_env_does_not_clobber_default(monkeypatch):
    monkeypatch.setenv("BIOGNOSIA_LLM_PROVIDER", "")
    monkeypatch.setenv("BIOGNOSIA_LLM_MODEL", "")
    cfg = load_config(str(CONF))["llm"]
    assert cfg["provider"] == "openai"
    assert cfg["model"] == ""


def test_validate_llm_config_rejects_missing_model():
    with pytest.raises(RuntimeError, match="BIOGNOSIA_LLM_MODEL"):
        validate_llm_config({"provider": "openai", "base_url": "http://x/v1"})


def test_validate_llm_config_rejects_unknown_provider():
    with pytest.raises(RuntimeError, match="BIOGNOSIA_LLM_PROVIDER"):
        validate_llm_config({"provider": "vllm", "model": "m", "base_url": "http://x/v1"})


def test_validate_llm_config_rejects_hosted_api_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="BIOGNOSIA_LLM_BASE_URL"):
        validate_llm_config({"provider": "openai", "model": "gpt-4o-mini"})


def test_validate_llm_config_accepts_local_endpoint_without_key(monkeypatch):
    """A self-hosted endpoint needs no key; that must not be a startup error."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    validate_llm_config(
        {"provider": "openai", "model": "some-model", "base_url": "http://localhost:11434/v1"}
    )


def test_validate_llm_config_accepts_hosted_api_with_key():
    validate_llm_config({"provider": "openai", "model": "gpt-4o-mini", "api_key": "a-key"})


def test_validate_llm_config_rejects_unedited_api_key_placeholder():
    with pytest.raises(RuntimeError, match="placeholder"):
        validate_llm_config(
            {"provider": "openai", "model": "gpt-4o-mini", "api_key": "<your-llm-api-key>"}
        )


# ── Endpoint placeholder guard ───────────────────────────────────────────────

def test_placeholder_endpoint_is_rejected(monkeypatch):
    """An unedited .env must fail immediately, naming the variable."""
    monkeypatch.setenv("BIOGNOSIA_MILVUS_HOST", "<public-endpoint-tbd>")
    cfg = load_config(str(CONF))
    with pytest.raises(RuntimeError, match="BIOGNOSIA_MILVUS_HOST"):
        check_endpoint_placeholders(cfg)


def test_placeholder_in_numeric_var_is_rejected(monkeypatch):
    """A placeholder port is dropped by the int coercion, so check the env too.

    Without the environment check this silently leaves the config default in
    place and the failure resurfaces much later as a connection error.
    """
    monkeypatch.setenv("BIOGNOSIA_MILVUS_PORT", "<public-port-tbd>")
    cfg = load_config(str(CONF))
    assert cfg["milvus"]["port"] == 19530  # coercion discarded the placeholder
    with pytest.raises(RuntimeError, match="BIOGNOSIA_MILVUS_PORT"):
        check_endpoint_placeholders(cfg)


def test_filled_endpoints_pass_the_placeholder_check(monkeypatch):
    for var, value in [
        ("BIOGNOSIA_MILVUS_HOST", "milvus.example"),
        ("BIOGNOSIA_MILVUS_PORT", "19530"),
        ("BIOGNOSIA_NEO4J_URI", "bolt://neo.example:7687"),
        ("BIOGNOSIA_NEO4J_USERNAME", "reader"),
        ("BIOGNOSIA_NEO4J_PASSWORD", "a-password"),
        ("BIOGNOSIA_REDIS_HOST", "redis.example"),
        ("BIOGNOSIA_REDIS_PORT", "6379"),
        ("BIOGNOSIA_MONGODB_URI", "mongodb://mongo.example:27017"),
        ("BIOGNOSIA_ELASTICSEARCH_HOSTS", "http://es.example:9200"),
    ]:
        monkeypatch.setenv(var, value)
    check_endpoint_placeholders(load_config(str(CONF)))  # must not raise


def test_shipped_env_example_is_all_placeholders():
    """Guard against a real endpoint or credential leaking into .env.example."""
    env_example = CONF.resolve().parents[1] / ".env.example"
    required = [
        "BIOGNOSIA_MILVUS_HOST",
        "BIOGNOSIA_NEO4J_URI",
        "BIOGNOSIA_REDIS_HOST",
        "BIOGNOSIA_MONGODB_URI",
        "BIOGNOSIA_ELASTICSEARCH_HOSTS",
    ]
    values = {}
    for line in env_example.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        values.setdefault(key.strip(), value.strip())

    for key in required:
        assert key in values, f"{key} missing from .env.example"
        assert PLACEHOLDER_RE.search(values[key]), (
            f".env.example {key} is not a placeholder: {values[key]!r}"
        )
