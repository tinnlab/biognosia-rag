"""Force MCP test mode before the app package is imported."""
import os

os.environ.setdefault("MCP_TEST_MODE", "1")
