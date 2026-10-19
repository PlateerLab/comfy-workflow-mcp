"""MCP server: the workflows saved in a ComfyUI server become tools for any agent.

    list_workflows -> describe_workflow -> run_workflow -> get_job

Edit a workflow in the ComfyUI editor and save it: the next run uses the new version (it is loaded on every run).
Settings come from the environment (see client.py): COMFY_URL or VAST_API_KEY, plus
COMFY_DOWNLOAD_DIR (default ./comfy_outputs), COMFY_MAX_WAIT (seconds one call may block: default 25 over stdio, because some MCP
hosts drop calls after ~30 s, and 85 in HTTP mode, under the ~100 s response limit of common tunnels; long jobs return a prompt_id to
poll with get_job) and COMFY_RETURN_PREVIEWS (finished image results also come back as up to this many preview images, default 4,
0 turns it off: a remote agent often cannot open the saved path or a protected URL, so it could not see what it made).

    comfy-workflow-mcp                                  stdio (the MCP host starts it)
    comfy-workflow-mcp --http [--port 8765] [--public-url https://comfy.example.com]   one server, many hosts (remote.py)
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

from starlette.responses import FileResponse, PlainTextResponse  # starlette comes with the mcp SDK

from . import remote
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


def _public_url() -> str:
    """Set in HTTP mode (--public-url or COMFY_MCP_PUBLIC_URL): results then also get a URL a remote host can open."""
    return os.environ.get("COMFY_MCP_PUBLIC_URL", "").rstrip("/")


PREVIEW_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


def _result(out: dict):
    """A run result: the JSON text, plus preview images of the saved image files when the run is done (COMFY_RETURN_PREVIEWS)."""
    text = json.dumps(out, ensure_ascii=False)
    n = int(os.environ.get("COMFY_RETURN_PREVIEWS", "4") or 0)
    files = [Path(f["saved_to"]) for f in out.get("files", []) if f.get("saved_to")]
    files = [f for f in files if f.suffix.lower() in PREVIEW_TYPES and f.is_file()][:max(n, 0)]
    if not files:
        return text
    import base64
    import io

    from mcp.types import ImageContent, TextContent
    blocks = [TextContent(type="text", text=text)]
    for f in files:
        try:
            from PIL import Image  # optional: shrink to a preview; without Pillow small files go as they are
            img = Image.open(f).convert("RGB")
            img.thumbnail((768, 768))
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=82)
            data, mime = buf.getvalue(), "image/jpeg"
        except ImportError:
            if f.stat().st_size > 1_500_000:
                continue
            data, mime = f.read_bytes(), PREVIEW_TYPES[f.suffix.lower()]
        blocks.append(ImageContent(type="image", data=base64.b64encode(data).decode(), mimeType=mime))
    return blocks


def _finish(c: ComfyClient, pid: str, job: dict, download_dir: str | None) -> dict:
    out = {"prompt_id": pid, **job}
    if job.get("status") == "done" and job.get("files"):
        d = _download_dir(download_dir)
        for f in out["files"]:
            if f.get("type", "output") == "output":
                saved = c.download(f, d).resolve()
                f["saved_to"] = str(saved)
                url = remote.file_url(_public_url(), _download_dir(None), saved)
                if url:
                    f["url"] = url
    return out


@mcp.custom_route("/files/{path:path}", methods=["GET"])
async def files(request):
    """Saved results for remote hosts (HTTP mode only; stdio never serves it). Only media under COMFY_DOWNLOAD_DIR."""
    f = remote.servable(_download_dir(None), request.path_params.get("path", ""))
    return FileResponse(f) if f else PlainTextResponse("not found", status_code=404)


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


def _queue(c: ComfyClient, name: str, sets: dict | None, images: dict | None, enable=None, bypass=None) -> str:
    """Load the saved workflow, apply inputs and images, queue it. Returns the prompt_id."""
    ui = c.get_workflow(name)
    prompt, meta = to_api(ui, c.node_spec, enable or (), bypass or ())
    done = apply_sets(prompt, meta, sets or {})
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
    return c.queue_prompt(prompt, ui_workflow=ui if "nodes" in ui else None)


@mcp.tool()
def run_workflow(name: str, set: dict | None = None, images: dict | None = None, enable: list[str] | None = None,
                 bypass: list[str] | None = None, wait_seconds: float = 0, download_dir: str | None = None):
    """Run a saved workflow with your inputs.
    set: {"<node id|title|class>.<input>": value}, e.g. {"Positive.text": "a cat", "KSampler.seed": 7, "EmptyLatentImage.width": 1024}.
    images: {"<LoadImage node id|title>": local file path (uploaded) or a file name already on the server}.
    enable / bypass: node ids or titles to switch on or bypass for this run (e.g. an optional LoRA).
    wait_seconds: wait up to this long (capped by COMFY_MAX_WAIT) and download results; otherwise poll get_job."""
    try:
        c = client()
        pid = _queue(c, name, set, images, enable, bypass)
        wait = min(float(wait_seconds or 0), _max_wait())
        job = c.wait(pid, wait) if wait > 0 else {"status": "queued"}
        return _result(_finish(c, pid, job, download_dir))
    except Exception as e:  # noqa: BLE001
        return _err(e)


_judged: dict = {}  # job id -> {"status", "log", "result"} for run_judged (runs in a background thread)


def _first_image(c: ComfyClient, pid: str, out_dir: Path, timeout: float = 1800) -> Path:
    job = c.wait(pid, timeout)
    if job.get("status") != "done":
        raise RuntimeError(f"run {pid} ended as {job.get('status')}: {job.get('error', '')}")
    files = [f for f in job.get("files", []) if f.get("type", "output") == "output"]
    if not files:
        raise RuntimeError(f"run {pid} produced no output file")
    return c.download(files[0], out_dir).resolve()


@mcp.tool()
def judge_image(path: str, rubric: str, description: str) -> str:
    """Judge one image with a rubric (path to a rubric JSON file, or the JSON itself) against a description of what it must show.
    Returns score, pass/fail, the problems found (criteria + focused checks) and the suggested fix. Needs JUDGE_URL and JUDGE_MODEL."""
    try:
        from .harness import Judge, Rubric, evaluate
        return json.dumps(evaluate(Judge(), path, description, Rubric.load(rubric)), ensure_ascii=False)
    except Exception as e:  # noqa: BLE001
        return _err(e)


@mcp.tool()
def run_judged(name: str, prompt_field: str, prompt: str, description: str, rubric: str, set: dict | None = None,
               images: dict | None = None, download_dir: str | None = None) -> str:
    """Run a saved workflow under the harness: generate -> judge with the rubric -> fix -> judge ..., keeping the best result.
    prompt_field: the input that holds the prompt, e.g. "Positive.text"; on a retry the judge's problems are appended to it.
    description: what the image must show (the judge checks against it). rubric: rubric JSON file path or JSON (see harness.py).
    Starts in the background and returns a job id at once; poll get_judged(job_id). Needs JUDGE_URL and JUDGE_MODEL."""
    import threading
    import uuid
    from .harness import Judge, Rubric, run_judged as loop
    try:
        rb, jd, c = Rubric.load(rubric), Judge(), client()
    except Exception as e:  # noqa: BLE001
        return _err(e)
    jid = uuid.uuid4().hex[:10]
    out = _download_dir(download_dir) / f"judged_{jid}"
    state = _judged[jid] = {"status": "running", "log": [], "result": None}

    def generate(note: str) -> Path:
        text = prompt + (f"\n\nTHE LAST ATTEMPT HAD THESE MISTAKES, AVOID THEM: {note}" if note else "")
        return _first_image(c, _queue(c, name, {**(set or {}), prompt_field: text}, images), out)

    def edit(img: Path, instruction: str) -> Path:
        fx = rb.fixes.get("edit") or {}
        sets = {fx["prompt"]: f"Fix only this, keep everything else the same: {instruction}"}
        return _first_image(c, _queue(c, fx["workflow"], sets, {fx["image"]: str(img)}), out)

    def work():
        try:
            state["result"] = loop(generate, edit if rb.fixes.get("edit") else None, jd, rb, description, state["log"].append)
            state["status"] = "done"
        except Exception as e:  # noqa: BLE001
            state["status"], state["error"] = "error", f"{type(e).__name__}: {e}"
    threading.Thread(target=work, daemon=True).start()
    return json.dumps({"job_id": jid, "status": "running", "poll": "get_judged"}, ensure_ascii=False)


@mcp.tool()
def get_judged(job_id: str) -> str:
    """Status of a run_judged job: running | done | error, the attempt log, and when done the best image, whether it passed,
    and every attempt's score and problems."""
    st = _judged.get(job_id)
    return json.dumps(st if st else {"error": f"unknown job {job_id}"}, ensure_ascii=False, default=str)


@mcp.tool()
def get_job(prompt_id: str, wait_seconds: float = 0, download_dir: str | None = None):
    """Status of a run (queued | running | done | error). When done, output files are downloaded ('saved_to' has the local path,
    'url' the public one in HTTP mode) and image results also come back as preview images. wait_seconds waits up to that long
    (capped by COMFY_MAX_WAIT)."""
    try:
        c = client()
        wait = min(float(wait_seconds or 0), _max_wait())
        job = c.wait(prompt_id, wait) if wait > 0 else c.job(prompt_id)
        return _result(_finish(c, prompt_id, job, download_dir))
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
    import argparse

    ap = argparse.ArgumentParser(prog="comfy-workflow-mcp", description="ComfyUI saved workflows as MCP tools.")
    ap.add_argument("--http", action="store_true", help="serve Streamable HTTP on /mcp instead of stdio, for hosts on other machines")
    ap.add_argument("--host", default="127.0.0.1", help="bind address in HTTP mode (default 127.0.0.1; put an authenticating proxy in front)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--public-url", default=os.environ.get("COMFY_MCP_PUBLIC_URL", ""),
                    help="public base URL behind the tunnel/proxy, e.g. https://comfy.example.com (allows that Host, adds result URLs)")
    ap.add_argument("--allow-host", action="append", default=[], help="extra Host header to accept (repeatable)")
    a = ap.parse_args()
    if not a.http:
        mcp.run()
        return
    if a.public_url:
        os.environ["COMFY_MCP_PUBLIC_URL"] = a.public_url
    os.environ.setdefault("COMFY_MAX_WAIT", "85")  # one call may block longer over HTTP (tunnels allow ~100 s), so fewer get_job polls
    remote.run_http(mcp, a.host, a.port, a.public_url, a.allow_host)


if __name__ == "__main__":
    main()
