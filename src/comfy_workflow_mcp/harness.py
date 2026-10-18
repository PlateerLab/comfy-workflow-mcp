"""Judge-and-retry harness for ComfyUI workflows.

Any saved workflow can run as: generate -> judge -> fix -> judge ... -> keep the best result.
The rules live in a rubric (JSON data), not in an agent prompt, so every agent gets the same checks:

    {
      "threshold": 0.8, "max_retries": 2, "verify": true,
      "criteria": [{"name": "facing", "description": "...", "weight": 1.0, "hard": true}],
      "checks":   [{"name": "faces_viewer", "question": "...JSON answer...", "key": "at_viewer",
                    "fail_when": "nonempty", "fix": "regenerate"}],
      "fixes":    {"edit": {"workflow": "edit.json", "image": "Image1", "prompt": "Positive.text"}}
    }

criteria  : scored 0..1 by an isolated vision judge against your description of the image (same shape as an agent
            harness judge_criteria). A hard criterion fails the run whenever a (verified) problem is reported for it.
checks    : one focused question each (a focused question catches what a long checklist misses). fail_when is
            "nonempty" (the answer key holds a non-empty list), "true", or {"equals": value}.
fix       : what to do when a check fails. "regenerate" (default): run again with a new seed and the problems appended
            to the prompt. "mirror": flip the image left-right (for a left/right swap; no re-run). "edit": run the
            rubric's edit workflow on the image with the problems as the instruction.

The judge is any OpenAI-compatible vision endpoint: JUDGE_URL (".../v1"), JUDGE_MODEL, JUDGE_API_KEY.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

VERIFY = ("You are a second, stricter reviewer. For each reported problem decide whether it is a real violation of the rule it cites, "
          "judging only from the image. Drop problems that are guesses, matters of taste, or say the same thing differently. "
          'Answer JSON only: {"keep": [indexes of the problems to keep, starting at 0]}')

SCORE = """You are an isolated judge checking one generated image against its description and a rubric.
Description of what the image must show:
{description}

Rubric (score each criterion 0.0 to 1.0; report only clear violations, each with the criterion name and one line to fix it):
{criteria}

Answer JSON only: {{"scores": {{"<criterion>": 0.0}}, "problems": [{{"criterion": "<name>", "fix": "one line"}}]}}"""


@dataclass
class Rubric:
    criteria: list = field(default_factory=list)
    checks: list = field(default_factory=list)
    fixes: dict = field(default_factory=dict)
    threshold: float = 0.8
    max_retries: int = 2
    verify: bool = True

    @classmethod
    def load(cls, src) -> "Rubric":
        """A path to a JSON file, a JSON string or a dict."""
        if isinstance(src, (str, Path)) and Path(str(src)).is_file():
            src = json.loads(Path(str(src)).read_text(encoding="utf-8"))
        elif isinstance(src, str):
            src = json.loads(src)
        known = {k: src[k] for k in ("criteria", "checks", "fixes", "threshold", "max_retries", "verify") if k in src}
        return cls(**known)


class Judge:
    """Isolated vision judge over an OpenAI-compatible chat endpoint."""

    def __init__(self, url: str | None = None, model: str | None = None, api_key: str | None = None, timeout: float = 120):
        self.url = (url or os.environ.get("JUDGE_URL") or "").rstrip("/")
        self.model = model or os.environ.get("JUDGE_MODEL") or ""
        self.key = api_key or os.environ.get("JUDGE_API_KEY") or "EMPTY"
        self.timeout = timeout
        if not self.url or not self.model:
            raise ValueError("set JUDGE_URL and JUDGE_MODEL (an OpenAI-compatible vision model) to use the harness")

    def ask(self, images: list, text: str, max_tokens: int = 600) -> dict:
        content = []
        for p in images:
            content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + _jpeg_b64(p)}})
        content.append({"type": "text", "text": text})
        body = {"model": self.model, "temperature": 0.0, "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": content}], "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(self.url + "/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            txt = json.loads(r.read().decode())["choices"][0]["message"]["content"]
        m = re.search(r"\{.*\}", txt, re.S)
        return json.loads(m.group()) if m else {}


def _jpeg_b64(path) -> str:
    from PIL import Image  # optional dependency: only the harness needs it
    img = Image.open(path).convert("RGB")
    img.thumbnail((1280, 1280))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def _failed(answer: dict, check: dict) -> list:
    v = answer.get(check.get("key", "answer"))
    rule = check.get("fail_when", "nonempty")
    if rule == "nonempty":
        return [str(x) for x in v] if isinstance(v, list) and v else []
    if rule == "true":
        return [check["name"]] if v is True else []
    if isinstance(rule, dict) and "equals" in rule:
        return [f"{check['name']}: {v}"] if v == rule["equals"] else []
    return []


def evaluate(judge: Judge, image, description: str, rubric: Rubric) -> dict:
    """Score one image. Returns {"score", "passed", "problems": [{criterion|check, fix}], "fix": strategy for the failures}."""
    problems, scores = [], {}
    if rubric.criteria:
        crit = json.dumps(rubric.criteria, ensure_ascii=False, indent=1)
        r = judge.ask([image], SCORE.format(description=description, criteria=crit), max_tokens=1200)
        found = [p for p in r.get("problems", []) if isinstance(p, dict)]
        if found and rubric.verify:
            listing = "\n".join(f"{k}. [{p.get('criterion')}] {p.get('fix')}" for k, p in enumerate(found))
            keep = judge.ask([image], f"Rubric:\n{crit}\n\nReported problems:\n{listing}\n\n{VERIFY}", max_tokens=300).get("keep", [])
            found = [found[k] for k in keep if isinstance(k, int) and 0 <= k < len(found)]
        names = {p.get("criterion") for p in found}
        for c in rubric.criteria:  # a criterion with no surviving problem scores 1.0
            scores[c["name"]] = float(r.get("scores", {}).get(c["name"], 1.0)) if c["name"] in names else 1.0
        problems += [{"criterion": p.get("criterion"), "fix": p.get("fix", "")} for p in found]
    fix = None
    for ch in rubric.checks:
        bad = _failed(judge.ask([image], ch["question"], max_tokens=300), ch)
        if bad:
            problems.append({"check": ch["name"], "fix": "; ".join(bad), "hard": ch.get("hard", True)})
            fix = fix or ch.get("fix", "regenerate")
    wsum = sum(c.get("weight", 1.0) for c in rubric.criteria) or 1.0
    score = sum(scores.get(c["name"], 1.0) * c.get("weight", 1.0) for c in rubric.criteria) / wsum if rubric.criteria else 1.0
    hard = [c["name"] for c in rubric.criteria if c.get("hard") and any(p.get("criterion") == c["name"] for p in problems)]
    hard += [p["check"] for p in problems if p.get("check") and p.get("hard", True)]
    return {"score": round(score, 3), "passed": score >= rubric.threshold and not hard, "hard_failed": hard,
            "problems": problems, "fix": fix or ("regenerate" if problems else None)}


def mirror(path) -> None:
    from PIL import Image, ImageOps
    p = Path(path)
    ImageOps.mirror(Image.open(p).convert("RGB")).save(p)


def run_judged(generate: Callable[[str], Path], edit: Callable[[Path, str], Path] | None, judge: Judge, rubric: Rubric,
               description: str, log: Callable[[str], None] = lambda m: None) -> dict:
    """The loop. generate(note) makes a new image (note = problems to avoid, appended to the prompt by the caller) and returns
    its path; edit(image, instruction) returns an edited image path. Keeps the best attempt by (passed, score, -problems)."""
    attempts, img, note = [], generate(""), ""
    for k in range(rubric.max_retries + 1):
        ev = evaluate(judge, img, description, rubric)
        attempts.append({"image": str(img), **ev})
        log(f"attempt {k}: score {ev['score']} {'pass' if ev['passed'] else 'fail ' + str(ev['hard_failed'])}")
        if ev["passed"] or k == rubric.max_retries:
            break
        note = "; ".join(p["fix"] for p in ev["problems"])[:800]
        if ev["fix"] == "mirror" and all(p.get("check") for p in ev["problems"]):
            mirror(img)  # left/right swap only: flip instead of a new roll
        elif ev["fix"] == "edit" and edit:
            img = edit(img, note)
        else:
            img = generate(note)
    best = max(attempts, key=lambda a: (a["passed"], a["score"], -len(a["problems"])))
    return {"passed": best["passed"], "image": best["image"], "attempts": attempts}
