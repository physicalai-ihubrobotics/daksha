"""LLM-independent checks: tool-call validation and joint-state sanity."""

import math

_TYPES = {"string": str, "number": (int, float), "integer": int, "boolean": bool}


def validate_tool_call(registry, name, args):
    """Return (ok, error). Rejects unknown tools, unknown/missing/mistyped args."""
    if name not in registry:
        return False, f"unknown tool '{name}'"
    if not isinstance(args, dict):
        return False, "args must be an object"
    spec = registry[name].args
    for key in args:
        if key not in spec:
            return False, f"tool '{name}' has no argument '{key}'"
    for key, (typ, required, _) in spec.items():
        if key not in args:
            if required:
                return False, f"tool '{name}' requires argument '{key}'"
            continue
        if not isinstance(args[key], _TYPES[typ]) or (
                typ in ("number", "integer") and isinstance(args[key], bool)):
            return False, f"argument '{key}' must be {typ}"
    return True, ""


def joint_limit_report(model, positions, margin_frac=0.05):
    """Classify each limited joint as ok / near_limit / out_of_limit / invalid."""
    report = {}
    for name in model.movable_joints():
        lim = model.joints[name]["limit"]
        if name not in positions or not lim or "lower" not in lim or "upper" not in lim:
            continue
        q = positions[name]
        lo, hi = lim["lower"], lim["upper"]
        if q is None or not math.isfinite(q):
            status = "invalid"
        elif q < lo or q > hi:
            status = "out_of_limit"
        elif min(q - lo, hi - q) < margin_frac * (hi - lo):
            status = "near_limit"
        else:
            status = "ok"
        report[name] = {"position": q, "lower": lo, "upper": hi, "status": status}
    return report
