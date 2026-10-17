"""MCP server: the workflows saved in a ComfyUI server become tools for any agent.

    list_workflows -> describe_workflow -> run_workflow -> get_job

Edit a workflow in the ComfyUI editor and save it: the next run uses the new version (it is loaded on every run).
Settings come from the environment (see client.py): COMFY_URL or VAST_API_KEY, plus
COMFY_DOWNLOAD_DIR (default ./comfy_outputs) and COMFY_MAX_WAIT (seconds one call may block, default 25,
because some MCP hosts drop calls after ~30 s; long jobs return a prompt_id to poll with get_job).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

try:  # mcp 2.x renamed FastMCP to MCPServer (same name / instructions / tool() / run())
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

from .client import ComfyClient, from_env
from .convert import ConvertError, apply_sets, describe, one_node, reroll_seeds, to_api

mcp = _Server("comfy-workflow", instructions=(
    "Runs image/video workflows saved in a ComfyUI server. Start with list_workflows, read describe_workflow for the inputs "
    "(address them as '<node id, title or class>.<input>', e.g. 'KSampler.seed'), then run_workflow. Runs return a prompt_id; "
    "call get_job(prompt_id, wait_seconds) until status is done, and it downloads the files."))

_client: ComfyClient | None = None


def client() -> ComfyClient:
    global _client
    if _client is None:
        _client = from_env()
    return _client


def _max_wait() -> float:
    return float(os.environ.get("COMFY_MAX_WAIT", "25"))


def _download_dir(d: str | None) -> Path:
    return Path(d or os.environ.get("COMFY_DOWNLOAD_DIR") or "comfy_outputs")


def _err(e: Exception) -> str:
    return json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False)


def _finish(c: ComfyClient, pid: str, job: dict, download_dir: str | None) -> dict:
    out = {"prompt_id": pid, **job}
    if job.get("status") == "done" and job.get("files"):
        d = _download_dir(download_dir)
        for f in out["files"]:
            if f.get("type", "output") == "output":
                f["saved_to"] = str(c.download(f, d).resolve())
    return out


@mcp.tool()
def server_status() -> str:
    """ComfyUI server: which server, version, GPU and free VRAM, queue length."""
    try:
        c = client()
        s = c.system_stats()
        dev = (s.get("devices") or [{}])[0]
        return json.dumps({"server": c.label, "comfyui": (s.get("system") or {}).get("comfyui_version"),
                           "gpu": dev.get("name"), "vram_free_gb": round((dev.get("vram_free") or 0) / 2**30, 1),
                           "vram_total_gb": round((dev.get("vram_total") or 0) / 2**30, 1), "queue": c.queue_status()}, ensure_ascii=False)
    except Exception as e:  # noqa: BLE001
        return _err(e)


@mcp.tool()
def list_workflows() -> str:
    """Names of the workflows saved in the ComfyUI editor (Workflows panel)."""
    try:
        return json.dumps(client().list_workflows(), ensure_ascii=False)
    except Exception as e:  # noqa: BLE001
        return _err(e)


@mcp.tool()
def describe_workflow(name: str) -> str:
    """The nodes of a saved workflow and the values you can change: per node its id, title, class and widget inputs.
    'image_input' nodes take an image (see run_workflow images), 'output' nodes produce the files,
    'rerolls_each_run' seeds get a new random value on each run unless you set them."""
    try:
        c = client()
        prompt, meta = to_api(c.get_workflow(name), c.node_spec)
        return json.dumps({"workflow": name, "nodes": describe(prompt, meta, c.node_spec)}, ensure_ascii=False)
    except Exception as e:  # noqa: BLE001
        return _err(e)


@mcp.tool()
def run_workflow(name: str, set: dict | None = None, images: dict | None = None, enable: list[str] | None = None,
                 bypass: list[str] | None = None, wait_seconds: float = 0, download_dir: str | None = None) -> str:
    """Run a saved workflow with your inputs.
    set: {"<node id|title|class>.<input>": value}, e.g. {"Positive.text": "a cat", "KSampler.seed": 7, "EmptyLatentImage.width": 1024}.
    images: {"<LoadImage node id|title>": local file path (uploaded) or a file name already on the server}.
    enable / bypass: node ids or titles to switch on or bypass for this run (e.g. an optional LoRA).
    wait_seconds: wait up to this long (capped by COMFY_MAX_WAIT) and download results; otherwise poll get_job."""
    try:
        c = client()
        ui = c.get_workflow(name)
        prompt, meta = to_api(ui, c.node_spec, enable or (), bypass or ())
        done = apply_sets(prompt, meta, set or {})
        for ref, src in (images or {}).items():
            nid = one_node(meta, ref)
            if "image" not in prompt[nid]["inputs"]:
                raise ConvertError(f"node {nid} ({meta[nid]['title']}) does not take an image")
            p = Path(str(src)).expanduser()
            if p.is_file():
                digest = hashlib.sha1(p.read_bytes()).hexdigest()[:12]
                server_name = c.upload_image(p, f"{p.stem}_{digest}{p.suffix}")
            else:
                server_name = str(src)
            prompt[nid]["inputs"]["image"] = server_name
            done.add((nid, "image"))
        reroll_seeds(prompt, meta, done)
        pid = c.queue_prompt(prompt, ui_workflow=ui if "nodes" in ui else None)
        wait = min(float(wait_seconds or 0), _max_wait())
        job = c.wait(pid, wait) if wait > 0 else {"status": "queued"}
        return json.dumps(_finish(c, pid, job, download_dir), ensure_ascii=False)
    except Exception as e:  # noqa: BLE001
        return _err(e)


@mcp.tool()
def get_job(prompt_id: str, wait_seconds: float = 0, download_dir: str | None = None) -> str:
    """Status of a run (queued | running | done | error). When done, output files are downloaded and 'saved_to' has the local path.
    wait_seconds waits up to that long (capped by COMFY_MAX_WAIT)."""
    try:
        c = client()
        wait = min(float(wait_seconds or 0), _max_wait())
        job = c.wait(prompt_id, wait) if wait > 0 else c.job(prompt_id)
        return json.dumps(_finish(c, prompt_id, job, download_dir), ensure_ascii=False)
    except Exception as e:  # noqa: BLE001
        return _err(e)


@mcp.tool()
def upload_image(path: str, name: str | None = None) -> str:
    """Upload a local image to the server's input folder; returns the name to use in run_workflow images."""
    try:
        return json.dumps({"name": client().upload_image(Path(path).expanduser(), name)}, ensure_ascii=False)
    except Exception as e:  # noqa: BLE001
        return _err(e)


@mcp.tool()
def list_models(folder: str | None = None) -> str:
    """Model folders on the server, or the files in one folder (e.g. 'checkpoints', 'loras', 'diffusion_models')."""
    try:
        return json.dumps(client().models(folder), ensure_ascii=False)
    except Exception as e:  # noqa: BLE001
        return _err(e)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
