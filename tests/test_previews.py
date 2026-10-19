"""Finished image results come back with preview images (a remote agent cannot open the saved path or a protected URL)."""
import json

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from comfy_workflow_mcp import server  # noqa: E402


def test_done_result_has_previews(tmp_path, monkeypatch):
    monkeypatch.setenv("COMFY_RETURN_PREVIEWS", "2")
    files = []
    for k in range(3):
        p = tmp_path / f"out_{k}.png"
        Image.new("RGB", (1600, 900), (k * 60, 80, 120)).save(p)
        files.append({"filename": p.name, "saved_to": str(p)})
    files.append({"filename": "clip.mp4", "saved_to": str(tmp_path / "clip.mp4")})
    blocks = server._result({"prompt_id": "x", "status": "done", "files": files})
    assert isinstance(blocks, list) and blocks[0].type == "text"
    assert json.loads(blocks[0].text)["status"] == "done"
    assert [b.type for b in blocks[1:]] == ["image", "image"]  # capped at COMFY_RETURN_PREVIEWS, video skipped
    assert blocks[1].mimeType == "image/jpeg"


def test_unfinished_or_disabled_is_plain_json(tmp_path, monkeypatch):
    assert json.loads(server._result({"prompt_id": "x", "status": "queued"}))["status"] == "queued"
    p = tmp_path / "a.png"
    Image.new("RGB", (64, 64)).save(p)
    monkeypatch.setenv("COMFY_RETURN_PREVIEWS", "0")
    assert isinstance(server._result({"status": "done", "files": [{"saved_to": str(p)}]}), str)
