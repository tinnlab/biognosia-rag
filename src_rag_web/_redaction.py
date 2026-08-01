"""Credential redaction for anything that reaches a log.

Every store in this project is configured with a user-supplied connection
string, and several of the clients accept credentials inside the URI itself —
pymilvus takes ``user:password@host``, Elasticsearch takes
``http://user:password@host``. Logging one of those verbatim writes the
password into the application log in cleartext, so any log line that can carry
a URI goes through here first.

Kept at the package root, with no imports beyond the standard library, so the
storage adapters can use it without pulling in the heavier ``utils`` package.
"""

import re

# scheme://<userinfo>@  ->  scheme://***@
#
# The character class excludes '@' so the match stops at the FIRST one, meaning
# only the userinfo section is replaced and the host stays readable. It also
# excludes '/' and whitespace so a match cannot run past the end of one URI —
# which matters when redacting free text such as an exception message that
# happens to contain both a URI and an unrelated email address.
_CREDENTIALS_RE = re.compile(r"://[^/@\s]*@")


def redact_credentials(value: object) -> str:
    """Return ``value`` as a string with any URI userinfo replaced by ``***``.

    Accepts arbitrary objects (typically a URI string or an exception) so it can
    be dropped straight into an f-string at the call site.
    """
    return _CREDENTIALS_RE.sub("://***@", str(value))
