"""HTTP mode helpers (remote.py), offline."""
from pathlib import Path

from comfy_workflow_mcp import remote


def test_allowed_adds_public_host():
    hosts, origins = remote.allowed("https://comfy.example.com", ["10.0.0.5:8765"])
    assert "comfy.example.com" in hosts and "10.0.0.5:8765" in hosts and "127.0.0.1:*" in hosts
    assert "https://comfy.example.com" in origins


def test_allowed_local_only_without_public_url():
    hosts, origins = remote.allowed()
    assert hosts == remote.LOCAL_HOSTS and origins == remote.LOCAL_ORIGINS


def test_file_url(tmp_path: Path):
    f = tmp_path / "run 1" / "out.png"
    f.parent.mkdir()
    f.write_bytes(b"x")
    assert remote.file_url("https://c.example.com/", tmp_path, f) == "https://c.example.com/files/run%201/out.png"
    assert remote.file_url("", tmp_path, f) == ""
    assert remote.file_url("https://c.example.com", tmp_path / "run 1", tmp_path / "other.png") == ""


def test_servable_blocks_escape_and_unknown_types(tmp_path: Path):
    root = tmp_path / "out"
    root.mkdir()
    (root / "a.png").write_bytes(b"x")
    (root / "notes.exe").write_bytes(b"x")
    (tmp_path / "secret.png").write_bytes(b"x")
    assert remote.servable(root, "a.png") == (root / "a.png").resolve()
    assert remote.servable(root, "notes.exe") is None
    assert remote.servable(root, "../secret.png") is None
    assert remote.servable(root, "missing.png") is None


def test_server_registers_files_route():
    from comfy_workflow_mcp import server

    assert any(getattr(r, "path", "") == "/files/{path:path}" for r in server.mcp._custom_starlette_routes)
