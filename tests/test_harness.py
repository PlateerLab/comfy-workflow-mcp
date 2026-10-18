"""Offline tests for the judge-and-retry loop with a scripted judge (no ComfyUI, no model)."""
import json

import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from comfy_workflow_mcp.harness import Rubric, evaluate, run_judged  # noqa: E402


class ScriptedJudge:
    """Answers each question from a queue, keyed by a word in the question."""

    def __init__(self, answers):
        self.answers = answers
        self.asked = []

    def ask(self, images, text, max_tokens=600):
        self.asked.append(text)
        for key, queue in self.answers.items():
            if key in text and queue:
                return queue.pop(0)
        return {}


def img(tmp_path, name, color=(200, 100, 50)):
    p = tmp_path / name
    Image.new("RGB", (64, 32), color).save(p)
    return p


RUBRIC = Rubric.load({
    "threshold": 0.8, "max_retries": 2,
    "criteria": [{"name": "props", "description": "exactly two mugs", "weight": 1.0, "hard": True}],
    "checks": [{"name": "faces_viewer", "question": "CHECK-VIEWER", "key": "at_viewer", "fail_when": "nonempty", "fix": "regenerate"}],
})


def test_pass_first_try(tmp_path):
    j = ScriptedJudge({"isolated judge": [{"scores": {"props": 1.0}, "problems": []}], "CHECK-VIEWER": [{"at_viewer": []}]})
    ev = evaluate(j, img(tmp_path, "a.png"), "two mugs", RUBRIC)
    assert ev["passed"] and ev["score"] == 1.0 and not ev["problems"]


def test_verified_problem_fails_hard_criterion(tmp_path):
    j = ScriptedJudge({"isolated judge": [{"scores": {"props": 0.4}, "problems": [{"criterion": "props", "fix": "three mugs"}]}],
                       "stricter reviewer": [{"keep": [0]}], "CHECK-VIEWER": [{"at_viewer": []}]})
    ev = evaluate(j, img(tmp_path, "a.png"), "two mugs", RUBRIC)
    assert not ev["passed"] and ev["hard_failed"] == ["props"]


def test_verifier_drops_false_positive(tmp_path):
    j = ScriptedJudge({"isolated judge": [{"scores": {"props": 0.4}, "problems": [{"criterion": "props", "fix": "same thing said differently"}]}],
                       "stricter reviewer": [{"keep": []}], "CHECK-VIEWER": [{"at_viewer": []}]})
    ev = evaluate(j, img(tmp_path, "a.png"), "two mugs", RUBRIC)
    assert ev["passed"] and ev["score"] == 1.0


def test_loop_regenerates_with_note_and_keeps_best(tmp_path):
    made, notes = [], []

    def generate(note):
        notes.append(note)
        made.append(img(tmp_path, f"g{len(made)}.png"))
        return made[-1]
    j = ScriptedJudge({"isolated judge": [{"scores": {"props": 1.0}, "problems": []}] * 2,
                       "CHECK-VIEWER": [{"at_viewer": ["man in blue"]}, {"at_viewer": []}]})
    out = run_judged(generate, None, j, RUBRIC, "two mugs")
    assert out["passed"] and out["image"] == str(made[1])
    assert notes[0] == "" and "man in blue" in notes[1]


def test_mirror_fix_flips_instead_of_rerun(tmp_path):
    rb = Rubric.load({"max_retries": 1, "checks": [{"name": "sides", "question": "CHECK-SIDES", "key": "swapped", "fail_when": "true", "fix": "mirror"}]})
    p = tmp_path / "s.png"
    im = Image.new("RGB", (2, 1))
    im.putpixel((0, 0), (255, 0, 0))
    im.putpixel((1, 0), (0, 0, 255))
    im.save(p)
    calls = []

    def generate(note):
        calls.append(note)
        return p
    j = ScriptedJudge({"CHECK-SIDES": [{"swapped": True}, {"swapped": False}]})
    out = run_judged(generate, None, j, rb, "")
    assert out["passed"] and len(calls) == 1  # flipped, not regenerated
    assert Image.open(p).convert("RGB").getpixel((0, 0)) == (0, 0, 255)


def test_rubric_loads_json_string_and_file(tmp_path):
    data = {"threshold": 0.9, "criteria": [{"name": "x", "description": "y"}]}
    f = tmp_path / "r.json"
    f.write_text(json.dumps(data), encoding="utf-8")
    assert Rubric.load(json.dumps(data)).threshold == 0.9
    assert Rubric.load(str(f)).criteria[0]["name"] == "x"
