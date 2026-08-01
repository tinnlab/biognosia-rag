"""Credentials must never reach a log.

Every store is configured with a user-supplied connection string, and several
clients accept credentials inside the URI itself. These tests pin the redaction
so a future edit to a log line cannot quietly start writing passwords out.
"""
import pathlib
import re

import pytest

from src_rag_web._redaction import redact_credentials

SECRET = "hunter2-should-never-be-logged"


@pytest.mark.parametrize(
    "uri",
    [
        f"http://user:{SECRET}@milvus.example:19530",        # pymilvus under RBAC
        f"mongodb://admin:{SECRET}@mongo.example:27017",
        f"bolt://neo4j:{SECRET}@neo.example:7687",
        f"http://elastic:{SECRET}@es.example:9200",
        f"redis://default:{SECRET}@redis.example:6379",
    ],
)
def test_password_is_removed_from_every_uri_scheme(uri):
    out = redact_credentials(uri)
    assert SECRET not in out
    assert "***@" in out


def test_host_survives_so_the_log_stays_useful():
    out = redact_credentials(f"http://user:{SECRET}@milvus.example:19530")
    assert out == "http://***@milvus.example:19530"


def test_uri_without_credentials_is_untouched():
    for uri in ("http://milvus.example:19530", "bolt://neo.example:7687"):
        assert redact_credentials(uri) == uri


def test_accepts_an_exception_not_just_a_string():
    """Call sites pass exceptions directly, since clients echo the URI back."""
    err = ConnectionError(f"failed to connect to http://user:{SECRET}@es.example:9200")
    out = redact_credentials(err)
    assert SECRET not in out
    assert "es.example:9200" in out


def test_match_cannot_run_past_one_uri_into_unrelated_text():
    """A bare '@' later in the message must not extend the redaction."""
    out = redact_credentials(
        f"http://user:{SECRET}@es.example:9200 reported by admin@example.com"
    )
    assert SECRET not in out
    assert "admin@example.com" in out


def test_redacts_each_uri_when_several_appear():
    out = redact_credentials(
        f"primary http://a:{SECRET}@one.example:9200 replica http://b:{SECRET}@two.example:9200"
    )
    assert SECRET not in out
    assert out.count("***@") == 2


# ── the call sites ───────────────────────────────────────────────────────────
# The storage adapters import heavy clients (pymilvus, neo4j, redis), so rather
# than import them, assert on the source: no log line may interpolate a raw
# connection variable.

SRC = pathlib.Path(__file__).resolve().parents[1] / "src_rag_web"

_RAW_URI_IN_LOG = re.compile(r"logger\.\w+\(\s*f?\"[^\"]*\{(uri|host|hosts)\}")


@pytest.mark.parametrize(
    "relative_path",
    [
        "storage/milvus_storage.py",
        "storage/neo4j_storage.py",
        "storage/redis_storage.py",
        "storage/mongo_storage.py",
    ],
)
def test_no_adapter_logs_a_raw_connection_variable(relative_path):
    text = (SRC / relative_path).read_text()
    offenders = _RAW_URI_IN_LOG.findall(text)
    assert not offenders, (
        f"{relative_path} interpolates a raw connection variable into a log line; "
        f"wrap it in redact_credentials()"
    )
