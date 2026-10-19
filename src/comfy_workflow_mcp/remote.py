"""Serve the MCP server over Streamable HTTP so MCP hosts on other machines can call it.

stdio only works when the host runs the server as its own child process, so every machine needs the code, Python and
the ComfyUI settings. In HTTP mode one machine runs the server and any number of hosts connect to its URL
(e.g. ``https://comfy.example.com/mcp`` behind a tunnel or reverse proxy that does the authentication).

The server binds to 127.0.0.1 by default. The MCP SDK's DNS-rebinding protection then accepts only localhost Host
headers, so a request coming through a tunnel (Host: comfy.example.com) gets 421; the public host is added to the
allow list from ``public_url``. Results saved on the server are exposed under ``/files/`` and get a ``url`` next to
``saved_to``, because a remote host cannot open the server's local paths.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from urllib.parse import quote, urlparse

LOCAL_HOSTS = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
LOCAL_ORIGINS = ["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*"]
FILE_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".webm", ".mov", ".wav", ".mp3", ".flac", ".json", ".txt"}


def allowed(public_url: str = "", extra_hosts: list[str] | tuple[str, ...] = ()) -> tuple[list[str], list[str]]:
    """Host and Origin allow lists for the transport security check: localhost, the public URL's host, extra hosts."""
    hosts, origins = list(LOCAL_HOSTS), list(LOCAL_ORIGINS)
    if public_url:
        u = urlparse(public_url)
        hosts.append(u.netloc)
        origins.append(f"{u.scheme}://{u.netloc}")
    for h in extra_hosts:
        h = h.strip()
        if h and h not in hosts:
            hosts.append(h)
    return hosts, origins


def file_url(public_url: str, root: Path, path: Path) -> str:
    """Public URL of a saved result, or "" when there is no public URL or the file is outside ``root``."""
    if not public_url:
        return ""
    try:
        rel = Path(path).resolve().relative_to(Path(root).resolve())
    except ValueError:
        return ""
    return f"{public_url.rstrip('/')}/files/" + quote(rel.as_posix())


def servable(root: Path, rel: str) -> Path | None:
    """The file under ``root`` that ``/files/<rel>`` may return (known media types only, no escaping ``root``)."""
    root = Path(root).resolve()
    f = (root / rel).resolve()
    if f.is_file() and f.suffix.lower() in FILE_TYPES and f.is_relative_to(root):
        return f
    return None


def run_http(server, host: str = "127.0.0.1", port: int = 8765, public_url: str = "",
             extra_hosts: list[str] | tuple[str, ...] = ()) -> None:
    """Run ``server`` (mcp 1.x FastMCP or 2.x MCPServer) on ``http://host:port/mcp``. Blocks."""
    from mcp.server.transport_security import TransportSecuritySettings

    hosts, origins = allowed(public_url, extra_hosts)
    security = TransportSecuritySettings(enable_dns_rebinding_protection=True, allowed_hosts=hosts, allowed_origins=origins)
    http = getattr(server, "run_streamable_http_async", None)
    # Stateless: the server keeps no per-client session, so a restart, a long idle gap or a proxy in between does not break connected hosts.
    if http is not None and "host" in inspect.signature(http).parameters:  # mcp 2.x: options go to run()
        server.run(transport="streamable-http", host=host, port=port, transport_security=security, stateless_http=True)
    else:  # mcp 1.x: options live on server.settings
        server.settings.host, server.settings.port = host, port
        server.settings.transport_security = security
        server.settings.stateless_http = True
        server.run(transport="streamable-http")
