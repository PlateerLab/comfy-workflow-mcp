# comfy-workflow-mcp

Run the workflows saved in a ComfyUI server as tools for any MCP agent (Claude Code, Claude Desktop, Cursor, your own agent).

Build and tune a workflow in the ComfyUI editor and save it. Agents can then list it, see its inputs, and run it with their own prompt, images and seed. Edit the workflow and save again: the next run uses the new version, because the saved file is loaded on every run.

Works with a local or remote ComfyUI, and with a [vast.ai](https://vast.ai) instance found from your API key.

> **Source-available, not open source.** The code is public to read and evaluate. Using, copying, modifying or redistributing it needs written permission from the owner ([jinsoo96](https://github.com/jinsoo96)); PlateerLab members have it within Plateer products. See [LICENSE](LICENSE).

## Tools

| Tool | What it does |
|---|---|
| `server_status` | Which server, ComfyUI version, GPU, free VRAM, queue length |
| `list_workflows` | Workflows saved in the editor |
| `describe_workflow(name)` | Each node's id, title, class and the values you can change; which nodes take images, which produce outputs, which seeds re-roll |
| `run_workflow(name, set, images, enable, bypass, wait_seconds)` | Run with your values. Returns a `prompt_id` (and the files if it finished within `wait_seconds`) |
| `get_job(prompt_id, wait_seconds)` | `queued` / `running` / `done` / `error`; when done, downloads the files and returns their local paths |
| `upload_image(path)` | Put a local image in the server's input folder |
| `list_models(folder)` | Model folders, or the files in one (`checkpoints`, `loras`, `diffusion_models` ...) |
| `judge_image(path, rubric, description)` | Score an image against a rubric and a description of what it must show (harness) |
| `run_judged(name, prompt_field, prompt, description, rubric, set, images)` | Run a workflow under the harness: generate, judge, fix, judge again, keep the best. Returns a job id |
| `get_judged(job_id)` | Progress and result of a `run_judged` job: best image, pass/fail, each attempt's score and problems |

Inputs are addressed as `<node id | title | class>.<input>`:

```json
{
  "name": "portrait.json",
  "set": {"Positive.text": "a red fox in the snow", "KSampler.seed": 7, "EmptyLatentImage.width": 1024},
  "images": {"Reference": "C:/pics/fox.png"},
  "enable": ["Turbo LoRA"],
  "wait_seconds": 25
}
```

A title wins over a class name, so give the nodes you want agents to touch a clear title in the editor (`Positive`, `Reference`). If a name matches more than one node, the error lists the ids to use instead. `images` takes a local file (uploaded for you) or a file name already on the server. `enable` / `bypass` switch nodes on or off for that run only. Seeds saved as "randomize" get a new value each run unless you set them.

## Harness: judge and retry

A generated image is often almost right: someone faces the camera instead of the person they talk to, there are three mugs instead of two, the left and right of a scene swap. The harness runs any saved workflow as a loop, so these get caught and fixed without anyone looking:

```
generate -> judge (rubric) -> fix -> judge -> ... -> keep the best attempt
```

The rules are data, not prompt text, so every agent that uses the workflow gets the same checks. A rubric has:

- **criteria**: scored 0 to 1 by an isolated vision judge against your description of the image (`{name, description, weight, hard}`, a pass threshold, a retry limit). A second judge pass drops problems that are guesses or say the same thing differently.
- **checks**: one focused question each. A long checklist misses things a single question catches.
- **fixes**: `regenerate` (new seed, the problems appended to the prompt), `mirror` (flip left-right for a swapped scene, no re-run), or `edit` (run your edit workflow with the problems as the instruction).

See [`examples/rubrics/story_panel.json`](examples/rubrics/story_panel.json) for comics and storyboards. Set `JUDGE_URL` (an OpenAI-compatible `/v1` endpoint), `JUDGE_MODEL` (a vision model) and, if needed, `JUDGE_API_KEY`, and install the extra: `pip install "comfy-workflow-mcp[harness]"`.

## Install

```bash
pip install git+https://github.com/PlateerLab/comfy-workflow-mcp
# or from a checkout
pip install -e .
```

Python 3.10+. The only dependency is the `mcp` SDK; the ComfyUI client uses the standard library.

## Configure

| Variable | |
|---|---|
| `COMFY_URL` | ComfyUI base URL, e.g. `http://127.0.0.1:8188` |
| `COMFY_USER` + `COMFY_PASSWORD` / `COMFY_TOKEN` / `COMFY_HEADERS` | Basic auth / bearer token / extra headers as JSON, if your server needs them |
| `VAST_API_KEY` | Instead of `COMFY_URL`: connect to a running vast.ai instance |
| `VAST_INSTANCE_ID` | Which instance (default: the first running one that exposes the port) |
| `VAST_COMFY_PORT` | ComfyUI's container port on the instance (default `8188`) |
| `COMFY_DOWNLOAD_DIR` | Where results are saved (default `./comfy_outputs`) |
| `COMFY_MAX_WAIT` | Longest a single call blocks, in seconds (default `25`; some MCP hosts drop calls after ~30 s) |

On vast.ai the ComfyUI template puts basic auth in front of the port: user `vastai`, password = the instance's Jupyter token. The server reads both from the vast.ai API, so the address and token stay right when you recreate the instance.

### Claude Code / Claude Desktop

```json
{
  "mcpServers": {
    "comfy": {
      "command": "comfy-workflow-mcp",
      "env": {"VAST_API_KEY": "...", "COMFY_DOWNLOAD_DIR": "D:/comfy_outputs"}
    }
  }
}
```

## How it works

ComfyUI saves workflows in the editor format (nodes, positional widget values, numbered links). The `/prompt` endpoint needs the API format (named inputs). The server converts one into the other on every run, using the server's own `/object_info` to name the widgets, so any installed custom node works without a mapping table.

The conversion handles bypassed nodes (links pass through), muted nodes, notes, Reroute, legacy PrimitiveNode values, the seed "control after generate" slot, and widgets converted to inputs. API-format files saved in the workflows folder run as they are. Subgraphs and group nodes are not supported yet: unpack them in the editor and save again.

The editor workflow is sent along with each run, so every output PNG opens as the exact canvas that made it.

## Security

* The vast.ai API key can control your whole account. This server only reads the instance list (`GET /instances`); it never creates, stops or deletes anything. The key and the instance token are not logged or returned by any tool.
* Agents can run any saved workflow and read the server's model and input lists. Point the server at a ComfyUI you are happy to share with the agent.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The tests cover the conversion offline with a hand-made `/object_info`.

## License

Source-available, all rights reserved: [LICENSE](LICENSE). Reading, evaluating and citing are allowed; any other use needs written permission from the owner (jinsoo96, wlstn010203@gmail.com, or an issue on this repository). [PlateerLab/comfy-workflow-mcp](https://github.com/PlateerLab/comfy-workflow-mcp) is the origin; [jinsoo96/js-comfy-workflow-mcp](https://github.com/jinsoo96/js-comfy-workflow-mcp) mirrors it automatically.
