"""Offline tests for the editor -> API conversion with a hand-made /object_info."""
import random

import pytest

from comfy_workflow_mcp.convert import ConvertError, apply_sets, describe, reroll_seeds, to_api

SPECS = {
    "CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [["a.safetensors", "b.safetensors"]]}},
                               "input_order": {"required": ["ckpt_name"]}},
    "LoraLoader": {"input": {"required": {"model": ["MODEL"], "clip": ["CLIP"], "lora_name": [["l.safetensors"]],
                                          "strength_model": ["FLOAT", {}], "strength_clip": ["FLOAT", {}]}},
                   "input_order": {"required": ["model", "clip", "lora_name", "strength_model", "strength_clip"]}},
    "CLIPTextEncode": {"input": {"required": {"text": ["STRING", {"multiline": True}], "clip": ["CLIP"]}},
                       "input_order": {"required": ["text", "clip"]}},
    "EmptyLatentImage": {"input": {"required": {"width": ["INT", {}], "height": ["INT", {}], "batch_size": ["INT", {}]}},
                         "input_order": {"required": ["width", "height", "batch_size"]}},
    "KSampler": {"input": {"required": {"model": ["MODEL"], "seed": ["INT", {"control_after_generate": True}], "steps": ["INT", {}],
                                        "cfg": ["FLOAT", {}], "sampler_name": [["euler"]], "scheduler": [["normal"]],
                                        "positive": ["CONDITIONING"], "negative": ["CONDITIONING"], "latent_image": ["LATENT"],
                                        "denoise": ["FLOAT", {}]}},
                 "input_order": {"required": ["model", "seed", "steps", "cfg", "sampler_name", "scheduler", "positive", "negative",
                                              "latent_image", "denoise"]}},
    "VAEDecode": {"input": {"required": {"samples": ["LATENT"], "vae": ["VAE"]}}, "input_order": {"required": ["samples", "vae"]}},
    "SaveImage": {"input": {"required": {"images": ["IMAGE"], "filename_prefix": ["STRING", {}]}},
                  "input_order": {"required": ["images", "filename_prefix"]}, "output_node": True},
    "LoadImage": {"input": {"required": {"image": [["x.png"], {"image_upload": True}]}}, "input_order": {"required": ["image"]}},
}
spec = SPECS.get


class UI:
    """Tiny editor-format builder."""

    def __init__(self):
        self.nodes, self.links = [], []

    def node(self, typ, widgets=None, inputs=(), outputs=(), title=None, mode=0):
        n = {"id": len(self.nodes) + 1, "type": typ, "mode": mode, "widgets_values": widgets if widgets is not None else [],
             "inputs": [{"name": nm, "type": t, "link": None} for nm, t in inputs],
             "outputs": [{"name": nm, "type": t, "links": []} for nm, t in outputs]}
        if title:
            n["title"] = title
        self.nodes.append(n)
        return n

    def link(self, a, slot, b, name):
        lid = len(self.links) + 1
        bi = next(i for i, x in enumerate(b["inputs"]) if x["name"] == name)
        self.links.append([lid, a["id"], slot, b["id"], bi, a["outputs"][slot]["type"]])
        b["inputs"][bi]["link"] = lid

    def dump(self):
        return {"nodes": self.nodes, "links": self.links}


def build(lora_mode=4, seed_control="fixed"):
    u = UI()
    u.node("Note", ["hello"])
    ck = u.node("CheckpointLoaderSimple", ["a.safetensors"], outputs=[("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")])
    lo = u.node("LoraLoader", ["l.safetensors", 0.8, 0.8], inputs=[("model", "MODEL"), ("clip", "CLIP")],
                outputs=[("MODEL", "MODEL"), ("CLIP", "CLIP")], title="Optional LoRA", mode=lora_mode)
    u.link(ck, 0, lo, "model")
    u.link(ck, 1, lo, "clip")
    rr = u.node("Reroute", inputs=[("", "*")], outputs=[("CLIP", "CLIP")])
    u.link(lo, 1, rr, "")
    pos = u.node("CLIPTextEncode", ["a cat"], inputs=[("clip", "CLIP")], outputs=[("CONDITIONING", "CONDITIONING")], title="Positive")
    neg = u.node("CLIPTextEncode", ["blurry"], inputs=[("clip", "CLIP")], outputs=[("CONDITIONING", "CONDITIONING")], title="Negative")
    u.link(rr, 0, pos, "clip")
    u.link(rr, 0, neg, "clip")
    lat = u.node("EmptyLatentImage", [512, 768, 1], outputs=[("LATENT", "LATENT")])
    prim = u.node("PrimitiveNode", [1234, seed_control], outputs=[("INT", "INT")], title="Seed")
    ks = u.node("KSampler", [0, "randomize", 20, 7.0, "euler", "normal", 1.0],
                inputs=[("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("latent_image", "LATENT"),
                        ("seed", "INT")], outputs=[("LATENT", "LATENT")])
    ks["inputs"][-1]["widget"] = {"name": "seed"}
    u.link(lo, 0, ks, "model")
    u.link(pos, 0, ks, "positive")
    u.link(neg, 0, ks, "negative")
    u.link(lat, 0, ks, "latent_image")
    u.link(prim, 0, ks, "seed")
    dec = u.node("VAEDecode", inputs=[("samples", "LATENT"), ("vae", "VAE")], outputs=[("IMAGE", "IMAGE")])
    u.link(ks, 0, dec, "samples")
    u.link(ck, 2, dec, "vae")
    sv = u.node("SaveImage", ["out/run"], inputs=[("images", "IMAGE")], title="Save")
    u.link(dec, 0, sv, "images")
    muted = u.node("SaveImage", ["muted"], inputs=[("images", "IMAGE")], mode=2)
    u.link(dec, 0, muted, "images")
    load = u.node("LoadImage", ["x.png", "image"], outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")], title="Reference")
    return u.dump(), {"ck": ck, "lo": lo, "pos": pos, "ks": ks, "sv": sv, "muted": muted, "load": load}


def test_conversion_bypass_reroute_primitive_and_mute():
    ui, n = build()
    prompt, meta = to_api(ui, spec)
    ids = lambda k: str(n[k]["id"])  # noqa: E731
    assert {v["class_type"] for v in prompt.values()} == {"CheckpointLoaderSimple", "CLIPTextEncode", "EmptyLatentImage", "KSampler",
                                                          "VAEDecode", "SaveImage", "LoadImage"}
    assert ids("muted") not in prompt and ids("lo") not in prompt  # muted dropped, bypassed LoRA gone
    ks = prompt[ids("ks")]["inputs"]
    assert ks["model"] == [ids("ck"), 0]  # bypass passes MODEL through to the checkpoint
    assert prompt[ids("pos")]["inputs"]["clip"] == [ids("ck"), 1]  # Reroute -> bypassed LoRA -> checkpoint CLIP
    assert ks["seed"] == 1234  # PrimitiveNode value inlined over the widget
    assert (ks["steps"], ks["cfg"], ks["sampler_name"], ks["denoise"]) == (20, 7.0, "euler", 1.0)  # control slot skipped
    assert meta[ids("ks")]["controls"] == {"seed": "fixed"}  # the primitive's control, not the widget's
    assert prompt[ids("load")]["inputs"] == {"image": "x.png"}  # trailing upload widget ignored


def test_enable_bypassed_node():
    ui, n = build()
    prompt, _ = to_api(ui, spec, enable=["Optional LoRA"])
    lo = str(n["lo"]["id"])
    assert prompt[lo]["inputs"]["strength_model"] == 0.8
    assert prompt[str(n["ks"]["id"])]["inputs"]["model"] == [lo, 0]


def test_apply_sets_and_describe():
    ui, n = build()
    prompt, meta = to_api(ui, spec)
    done = apply_sets(prompt, meta, {"Positive.text": "a dog", "KSampler.steps": 8, f"{n['sv']['id']}.filename_prefix": "x"})
    assert prompt[str(n["pos"]["id"])]["inputs"]["text"] == "a dog"
    assert prompt[str(n["ks"]["id"])]["inputs"]["steps"] == 8
    assert len(done) == 3
    with pytest.raises(ConvertError, match="matches 2 nodes"):
        apply_sets(prompt, meta, {"CLIPTextEncode.text": "?"})
    with pytest.raises(ConvertError, match="no settable input"):
        apply_sets(prompt, meta, {"KSampler.model": "?"})
    d = {x["title"]: x for x in describe(prompt, meta, spec)}
    assert d["Save"]["output"] and d["Reference"]["image_input"] and "rerolls_each_run" not in d["KSampler"]
    assert "model" not in d["KSampler"]["inputs"]


def test_reroll_respects_fixed_and_explicit_seed():
    ui, n = build()
    prompt, meta = to_api(ui, spec)
    reroll_seeds(prompt, meta, keep=set())
    assert prompt[str(n["ks"]["id"])]["inputs"]["seed"] == 1234  # fixed stays
    ui, n = build(seed_control="randomize")
    prompt, meta = to_api(ui, spec)
    ks = str(n["ks"]["id"])
    random.seed(1)
    reroll_seeds(prompt, meta, keep=set())
    assert prompt[ks]["inputs"]["seed"] != 1234
    prompt[ks]["inputs"]["seed"] = 5
    reroll_seeds(prompt, meta, keep={(ks, "seed")})
    assert prompt[ks]["inputs"]["seed"] == 5


def test_api_format_passthrough_and_subgraph_error():
    api = {"3": {"class_type": "KSampler", "inputs": {"seed": 1}, "_meta": {"title": "Sampler"}}}
    prompt, meta = to_api(api, spec)
    assert prompt == api and meta["3"]["title"] == "Sampler"
    ui, _ = build()
    ui["definitions"] = {"subgraphs": [{"id": "abc-123"}]}
    ui["nodes"].append({"id": 99, "type": "abc-123", "mode": 0})
    with pytest.raises(ConvertError, match="subgraphs"):
        to_api(ui, spec)
