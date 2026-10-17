"""Saved ComfyUI workflows (editor format) -> API format, and addressing their inputs.

The editor saves nodes with positional ``widgets_values`` and numbered links. The ``/prompt`` endpoint
wants ``{node_id: {"class_type": ..., "inputs": {...}}}`` with named inputs. Positions are mapped to names
with the server's ``/object_info`` (``get_spec``), so any installed node works without a hand-written table.

Handled: bypassed nodes (mode 4, links pass through to the input of the same type), muted nodes (mode 2,
dropped), frontend-only nodes (notes, Reroute, legacy PrimitiveNode whose value is inlined), the seed
"control after generate" slot, widgets converted to inputs, and both link encodings (list and dict).
Not handled: subgraphs / group nodes (unpack them in the editor first).
"""
from __future__ import annotations

import copy
import random
from typing import Callable, Iterable

GetSpec = Callable[[str], "dict | None"]
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
RANDOM_CONTROLS = {"randomize", "increment", "decrement"}


class ConvertError(ValueError):
    pass


def is_api_format(wf) -> bool:
    return isinstance(wf, dict) and bool(wf) and "nodes" not in wf and all(isinstance(v, dict) and "class_type" in v for v in wf.values())


def widget_names(spec: dict) -> list:
    """Input names in ``widgets_values`` order. ``None`` marks the editor-only "control after generate" slot."""
    inp = spec.get("input") or {}
    order = spec.get("input_order") or {k: list((inp.get(k) or {}).keys()) for k in ("required", "optional")}
    names: list = []
    for section in ("required", "optional"):
        for name in order.get(section, []):
            v = (inp.get(section) or {}).get(name)
            t = v[0] if isinstance(v, list) and v else v
            if isinstance(t, list) or t in WIDGET_TYPES:
                names.append(name)
                opts = v[1] if isinstance(v, list) and len(v) > 1 and isinstance(v[1], dict) else {}
                if opts.get("control_after_generate"):
                    names.append(None)
    return names


def _links(ui: dict) -> dict:
    out = {}
    for lk in ui.get("links") or []:
        if isinstance(lk, dict):  # newer editor encoding
            out[lk["id"]] = (str(lk["origin_id"]), lk["origin_slot"], lk.get("type"))
        else:  # [id, origin_id, origin_slot, target_id, target_slot, type]
            out[lk[0]] = (str(lk[1]), lk[2], lk[5] if len(lk) > 5 else None)
    return out


def _check_supported(ui: dict) -> None:
    subgraphs = {s.get("id") for s in (ui.get("definitions") or {}).get("subgraphs", [])}
    bad = sorted({str(n.get("type")) for n in ui.get("nodes", [])
                  if n.get("type") in subgraphs or str(n.get("type", "")).startswith(("workflow>", "workflow/"))})
    if bad:
        raise ConvertError(f"subgraphs / group nodes are not supported yet ({', '.join(bad)}): unpack them in the editor and save again")


def _ui_refs(nodes: dict, ref: str) -> list[str]:
    ref = str(ref)
    if ref in nodes:
        return [ref]
    hits = [nid for nid, n in nodes.items() if n.get("title") == ref]
    return hits or [nid for nid, n in nodes.items() if n.get("type") == ref]


def to_api(ui: dict, get_spec: GetSpec, enable: Iterable[str] = (), bypass: Iterable[str] = ()):
    """Return ``(prompt, meta)``. ``meta[node_id] = {"class_type", "title", "controls"}`` where ``controls``
    maps a widget to its "control after generate" mode (``randomize``, ``fixed`` ...).
    ``enable`` / ``bypass``: node ids, titles or class types to switch on (mode 0) or bypass (mode 4) first."""
    if is_api_format(ui):
        prompt = copy.deepcopy(ui)
        meta = {nid: {"class_type": v["class_type"], "title": (v.get("_meta") or {}).get("title") or v["class_type"], "controls": {}}
                for nid, v in prompt.items()}
        return prompt, meta
    if "nodes" not in ui:
        raise ConvertError("not a ComfyUI workflow (no 'nodes')")
    _check_supported(ui)
    nodes = {str(n["id"]): copy.deepcopy(n) for n in ui["nodes"]}
    for mode, refs in ((0, enable), (4, bypass)):
        for ref in refs:
            ids = _ui_refs(nodes, ref)
            if not ids:
                raise ConvertError(f"no node '{ref}' to {'enable' if mode == 0 else 'bypass'}")
            for nid in ids:
                nodes[nid]["mode"] = mode
    links = _links(ui)

    def resolve(link_id, depth=0):
        """('link', [node_id, slot], None) | ('value', v, control) | None"""
        if depth > 64 or link_id not in links:
            return None
        origin, slot, typ = links[link_id]
        n = nodes.get(origin)
        if n is None:
            return None
        t = n.get("type")
        if t == "Reroute" or n.get("mode", 0) == 4:
            for i in n.get("inputs") or []:
                if i.get("link") is not None and (t == "Reroute" or typ is None or i.get("type") in (typ, "*")):
                    return resolve(i["link"], depth + 1)
            return None
        if t == "PrimitiveNode":  # its own "control after generate" (fixed/randomize) wins over the target widget's
            vals = n.get("widgets_values") or []
            return ("value", vals[0], vals[1] if len(vals) > 1 else None) if vals else None
        if n.get("mode", 0) != 0 or get_spec(t) is None:
            return None
        return ("link", [origin, slot], None)

    prompt, meta = {}, {}
    for nid, n in nodes.items():
        t = n.get("type")
        if n.get("mode", 0) != 0:
            continue
        spec = get_spec(t)
        if spec is None:  # notes, reroutes, primitives: editor only
            continue
        inputs, controls = {}, {}
        wv = n.get("widgets_values")
        if isinstance(wv, dict):
            known = {k for sec in ("required", "optional") for k in ((spec.get("input") or {}).get(sec) or {})}
            inputs.update({k: v for k, v in wv.items() if k in known})
        else:
            prev = None
            for name, val in zip(widget_names(spec), wv or []):
                if name is None:
                    if prev:
                        controls[prev] = val
                else:
                    inputs[name], prev = val, name
        for i in n.get("inputs") or []:
            if i.get("link") is None:
                continue
            r = resolve(i["link"])
            if r is not None:
                name = (i.get("widget") or {}).get("name") or i["name"]
                inputs[name] = r[1]
                if r[0] == "value":
                    controls.pop(name, None)
                    if r[2] is not None:
                        controls[name] = r[2]
        prompt[nid] = {"class_type": t, "inputs": inputs}
        meta[nid] = {"class_type": t, "title": n.get("title") or spec.get("display_name") or t, "controls": controls}
    return prompt, meta


def _is_link(v) -> bool:
    return isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and isinstance(v[1], int)


def describe(prompt: dict, meta: dict, get_spec: GetSpec) -> list[dict]:
    """What an agent can set: each node's widget values (links left out), which seeds re-roll, outputs, image inputs."""
    out = []
    for nid, v in prompt.items():
        spec = get_spec(v["class_type"]) or {}
        entry = {"id": nid, "title": meta[nid]["title"], "class_type": v["class_type"],
                 "inputs": {k: val for k, val in v["inputs"].items() if not _is_link(val)}}
        rolls = [k for k, c in meta[nid]["controls"].items() if c in RANDOM_CONTROLS]
        if rolls:
            entry["rerolls_each_run"] = rolls
        if spec.get("output_node"):
            entry["output"] = True
        if "image" in entry["inputs"] and v["class_type"].startswith("LoadImage"):
            entry["image_input"] = True
        out.append(entry)
    return out


def find_nodes(meta: dict, ref: str) -> list[str]:
    """Node id, exact title, or class type (in that order)."""
    ref = str(ref)
    if ref in meta:
        return [ref]
    hits = [nid for nid, m in meta.items() if m["title"] == ref]
    return hits or [nid for nid, m in meta.items() if m["class_type"] == ref]


def one_node(meta: dict, ref: str) -> str:
    ids = find_nodes(meta, ref)
    if not ids:
        raise ConvertError(f"no node '{ref}'. Nodes: " + ", ".join(f"{nid}:{m['title']}" for nid, m in meta.items()))
    if len(ids) > 1:
        raise ConvertError(f"'{ref}' matches {len(ids)} nodes ({', '.join(ids)}): use the node id")
    return ids[0]


def apply_sets(prompt: dict, meta: dict, sets: dict) -> set:
    """``{"<node id|title|class>.<input>": value}`` -> prompt. Returns the (node, input) pairs that were set."""
    done = set()
    for key, val in (sets or {}).items():
        ref, _, name = str(key).rpartition(".")
        if not ref or not name:
            raise ConvertError(f"'{key}': use '<node id, title or class>.<input name>', e.g. 'KSampler.seed'")
        nid = one_node(meta, ref)
        known = [k for k, v in prompt[nid]["inputs"].items() if not _is_link(v)]
        if name not in known:
            raise ConvertError(f"node {nid} ({meta[nid]['title']}) has no settable input '{name}'. Inputs: {', '.join(known)}")
        prompt[nid]["inputs"][name] = val
        done.add((nid, name))
    return done


def reroll_seeds(prompt: dict, meta: dict, keep: set) -> None:
    """Seeds saved with "randomize" get a new value each run, like pressing Queue in the editor."""
    for nid, m in meta.items():
        for widget, control in m["controls"].items():
            if control == "randomize" and (nid, widget) not in keep and nid in prompt:
                prompt[nid]["inputs"][widget] = random.randint(0, 2**50)
