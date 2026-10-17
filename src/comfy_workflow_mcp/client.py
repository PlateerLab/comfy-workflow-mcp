"""Small ComfyUI HTTP client (standard library only).

Connection is read from the environment:

* ``COMFY_URL``: base URL of a ComfyUI server, e.g. ``http://127.0.0.1:8188``.
  Optional auth: ``COMFY_USER`` + ``COMFY_PASSWORD`` (basic), ``COMFY_TOKEN`` (bearer)
  or ``COMFY_HEADERS`` (a JSON object of extra headers).
* or ``VAST_API_KEY``: find a running vast.ai instance and connect to its ComfyUI port.
  ``VAST_INSTANCE_ID`` picks one instance (default: the first running one that exposes the port),
  ``VAST_COMFY_PORT`` is the container port (default 8188). The vast.ai API is only read
  (``GET /instances``); nothing is created, stopped or deleted. Auth is the vast.ai template's
  basic auth: user ``vastai``, password = the instance's ``jupyter_token`` (override with
  ``VAST_COMFY_USER`` / ``COMFY_PASSWORD``).
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

VAST_API = "https://console.vast.ai/api/v0"
USER_AGENT = "comfy-workflow-mcp"


class ComfyError(RuntimeError):
    pass


class ComfyClient:
    def __init__(self, base_url: str, headers: dict | None = None, label: str = ""):
        self.base = base_url.rstrip("/")
        self.headers = dict(headers or {})
        self.label = label or self.base
        self._spec: dict[str, dict | None] = {}

    # -- low level -------------------------------------------------------
    def request(self, path: str, data=None, content_type: str = "application/json", timeout: float = 60, raw: bool = False):
        headers = {"User-Agent": USER_AGENT, **self.headers}
        body = None
        if data is not None:
            body = data if isinstance(data, bytes) else json.dumps(data).encode("utf-8")
            headers["Content-Type"] = content_type
        req = urllib.request.Request(self.base + path, data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = r.read()
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:2000]
            raise ComfyError(f"{e.code} {path}: {detail}") from None
        if raw:
            return payload
        return json.loads(payload.decode("utf-8")) if payload else None

    # -- server ----------------------------------------------------------
    def system_stats(self) -> dict:
        return self.request("/system_stats")

    def queue_status(self) -> dict:
        q = self.request("/queue")
        return {"running": len(q.get("queue_running", [])), "pending": len(q.get("queue_pending", []))}

    def node_spec(self, class_type: str) -> dict | None:
        """``/object_info/<class>`` (cached). None for frontend-only nodes such as notes and reroutes."""
        if class_type not in self._spec:
            try:
                info = self.request(f"/object_info/{urllib.parse.quote(class_type, safe='')}")
            except ComfyError:
                info = {}
            self._spec[class_type] = (info or {}).get(class_type)
        return self._spec[class_type]

    def models(self, folder: str | None = None) -> list:
        return self.request(f"/models/{urllib.parse.quote(folder, safe='')}" if folder else "/models") or []

    # -- saved workflows (user data) -------------------------------------
    def list_workflows(self) -> list[str]:
        items = self.request("/api/userdata?dir=workflows&recurse=true") or []
        return sorted(x for x in items if isinstance(x, str) and x.endswith(".json"))

    def get_workflow(self, name: str) -> dict:
        if not name.endswith(".json"):
            name += ".json"
        wf = self.request("/api/userdata/" + urllib.parse.quote("workflows/" + name, safe=""))
        return wf if isinstance(wf, dict) else json.loads(wf)

    # -- run -------------------------------------------------------------
    def upload_image(self, path: str | Path, name: str | None = None, overwrite: bool = True) -> str:
        p = Path(path)
        name = name or p.name
        mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
        bnd = uuid.uuid4().hex
        body = (f"--{bnd}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\nContent-Type: {mime}\r\n\r\n").encode() \
            + p.read_bytes() \
            + (f"\r\n--{bnd}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\n{'true' if overwrite else 'false'}\r\n--{bnd}--\r\n").encode()
        r = self.request("/upload/image", body, content_type=f"multipart/form-data; boundary={bnd}")
        return f"{r['subfolder']}/{r['name']}" if r.get("subfolder") else r["name"]

    def queue_prompt(self, prompt: dict, ui_workflow: dict | None = None, client_id: str = USER_AGENT) -> str:
        """Queue an API-format prompt. Passing the UI workflow stores it in the output PNG metadata,
        so the result opens as a canvas in the ComfyUI editor."""
        body = {"prompt": prompt, "client_id": client_id}
        if ui_workflow:
            body["extra_data"] = {"extra_pnginfo": {"workflow": ui_workflow}}
        r = self.request("/prompt", body)
        if r.get("node_errors"):
            raise ComfyError(f"node errors: {json.dumps(r['node_errors'])[:2000]}")
        return r["prompt_id"]

    def job(self, prompt_id: str) -> dict:
        """status: queued | running | done | error, plus output files when done."""
        h = (self.request(f"/history/{prompt_id}") or {}).get(prompt_id)
        if h:
            st = h.get("status") or {}
            files = [dict(img, node=nid) for nid, out in (h.get("outputs") or {}).items()
                     for key in ("images", "gifs", "videos", "audio") for img in out.get(key, [])]
            if st.get("status_str") == "error":
                msgs = [m[1] for m in st.get("messages", []) if m and m[0] == "execution_error"]
                err = msgs[0] if msgs else {}
                return {"status": "error", "error": {k: err.get(k) for k in ("node_id", "node_type", "exception_message") if k in err}, "files": files}
            if st.get("completed") or files:
                return {"status": "done", "files": files}
        q = self.request("/queue") or {}
        if any(item[1] == prompt_id for item in q.get("queue_running", [])):
            return {"status": "running"}
        if any(item[1] == prompt_id for item in q.get("queue_pending", [])):
            return {"status": "queued"}
        return {"status": "unknown"}

    def wait(self, prompt_id: str, seconds: float) -> dict:
        end = time.time() + max(0.0, seconds)
        while True:
            j = self.job(prompt_id)
            if j["status"] in ("done", "error") or time.time() >= end:
                return j
            time.sleep(1.5)

    def download(self, file: dict, out_dir: str | Path) -> Path:
        q = urllib.parse.urlencode({"filename": file["filename"], "subfolder": file.get("subfolder", ""), "type": file.get("type", "output")})
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        dst = out / file["filename"]
        dst.write_bytes(self.request("/view?" + q, raw=True, timeout=300))
        return dst


def _vast_instance(api_key: str, instance_id: str | None, port: str) -> dict:
    req = urllib.request.Request(f"{VAST_API}/instances/?owner=me", headers={"Authorization": f"Bearer {api_key}", "User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        instances = json.loads(r.read().decode("utf-8")).get("instances", [])
    if instance_id:
        instances = [i for i in instances if str(i.get("id")) == str(instance_id)]
    running = [i for i in instances if i.get("actual_status") == "running" and (i.get("ports") or {}).get(f"{port}/tcp")]
    if not running:
        raise ComfyError(f"no running vast.ai instance exposing port {port}" + (f" with id {instance_id}" if instance_id else ""))
    return running[0]


def from_env(env=os.environ) -> ComfyClient:
    if env.get("COMFY_URL"):
        headers = json.loads(env["COMFY_HEADERS"]) if env.get("COMFY_HEADERS") else {}
        if env.get("COMFY_TOKEN"):
            headers["Authorization"] = f"Bearer {env['COMFY_TOKEN']}"
        elif env.get("COMFY_USER"):
            headers["Authorization"] = "Basic " + base64.b64encode(f"{env['COMFY_USER']}:{env.get('COMFY_PASSWORD', '')}".encode()).decode()
        return ComfyClient(env["COMFY_URL"], headers)
    if env.get("VAST_API_KEY"):
        port = env.get("VAST_COMFY_PORT", "8188")
        i = _vast_instance(env["VAST_API_KEY"], env.get("VAST_INSTANCE_ID"), port)
        base = f"http://{i['public_ipaddr']}:{i['ports'][f'{port}/tcp'][0]['HostPort']}"
        password = env.get("COMFY_PASSWORD") or i.get("jupyter_token", "")
        auth = "Basic " + base64.b64encode(f"{env.get('VAST_COMFY_USER', 'vastai')}:{password}".encode()).decode()
        return ComfyClient(base, {"Authorization": auth}, label=f"vast.ai instance {i.get('id')} ({i.get('gpu_name')})")
    raise ComfyError("set COMFY_URL (any ComfyUI server) or VAST_API_KEY (a vast.ai instance)")
