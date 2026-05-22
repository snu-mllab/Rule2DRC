#!/usr/bin/env python3
from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Any


def _layer_indices(ly) -> list[int]:
    if hasattr(ly, "layer_indices"):
        return list(ly.layer_indices())
    if hasattr(ly, "layer_indexes"):
        return list(ly.layer_indexes())
    return []


def _layer_label(ly, li: int) -> str:
    info = ly.get_info(li)
    return f"{int(getattr(info, 'layer', 0))}/{int(getattr(info, 'datatype', 0))}"


def _pts(it, precision: int) -> list[list[float]]:
    return [[round(p.x, precision), round(p.y, precision)] for p in it]


def _poly_to_rec(poly, precision: int) -> tuple[dict[str, Any], int]:
    hull = _pts(poly.each_point_hull(), precision) if hasattr(poly, "each_point_hull") else _pts(poly.each_point(), precision)
    holes = []
    if hasattr(poly, "holes") and hasattr(poly, "each_point_hole"):
        for hi in range(int(poly.holes())):
            holes.append(_pts(poly.each_point_hole(hi), precision))
    v = len(hull) + sum(len(h) for h in holes)
    return {"hull": hull, "holes": holes}, v


def gds_to_dict(
    gds_path: Path,
    *,
    max_layers: int = 6,
    max_polys_per_layer: int = 6,
    max_vertices: int = 240,
    precision: int = 3,
) -> dict:
    import pya  # type: ignore

    ly = pya.Layout(True)
    ly.read(str(gds_path))

    tops = list(ly.top_cells())
    top = sorted(tops, key=lambda c: c.name)[0] if tops else ly.cell(0)
    if hasattr(top, "flatten"):
        top.flatten(True)

    bbox = top.dbbox()
    out = {
        "gds": gds_path.name,
        "topcell_bbox_um_xyxy": [
            round(bbox.left, precision), round(bbox.bottom, precision),
            round(bbox.right, precision), round(bbox.top, precision),
        ],
        "layers": [],
        "vertex_budget_used": 0,
    }

    per_layer: list[tuple[float, str, list[tuple[float, dict, int]]]] = []
    for li in _layer_indices(ly):
        label = _layer_label(ly, li)
        polys: list[tuple[float, dict, int]] = []
        for sh in top.shapes(li).each():
            if sh.is_polygon():
                poly = sh.dpolygon
                rec, v = _poly_to_rec(poly, precision)
                polys.append((float(poly.area()), rec, v))
            elif sh.is_box():
                b = sh.dbox
                hull = [
                    [round(b.left, precision), round(b.bottom, precision)],
                    [round(b.right, precision), round(b.bottom, precision)],
                    [round(b.right, precision), round(b.top, precision)],
                    [round(b.left, precision), round(b.top, precision)],
                ]
                area = float((b.right - b.left) * (b.top - b.bottom))
                polys.append((area, {"hull": hull, "holes": []}, len(hull)))
        if polys:
            polys.sort(key=lambda t: t[0], reverse=True)
            tot_area = sum(a for a, _r, _v in polys)
            per_layer.append((tot_area, label, polys))

    per_layer.sort(key=lambda t: t[0], reverse=True)

    used = 0
    for _tot_area, label, polys in per_layer[:max_layers]:
        kept = []
        for _area, rec, v in polys[:max_polys_per_layer]:
            if used + v > max_vertices:
                break
            used += v
            kept.append(rec)
        if kept:
            out["layers"].append({"layer": label, "polys": kept})

    out["vertex_budget_used"] = used
    return out


def gds_to_text(
    gds_path: Path,
    *,
    max_layers: int = 20,
    max_polys_per_layer: int = 20,
    max_vertices: int = 480,
    precision: int = 3,
) -> str:
    d = gds_to_dict(
        gds_path,
        max_layers=max_layers,
        max_polys_per_layer=max_polys_per_layer,
        max_vertices=max_vertices,
        precision=precision,
    )
    lines = []
    lines.append(f"gds: {d['gds']}")
    lines.append(f"topcell_bbox_um_xyxy: {d['topcell_bbox_um_xyxy']}")
    for layer in d["layers"]:
        lines.append(f"layer {layer['layer']}:")
        for i, poly in enumerate(layer["polys"]):
            hull = " ".join(f"({x},{y})" for x, y in poly["hull"])
            lines.append(f"  poly{i} hull: {hull}")
            for hi, hole in enumerate(poly.get("holes") or []):
                h = " ".join(f"({x},{y})" for x, y in hole)
                lines.append(f"    hole{hi}: {h}")
    lines.append(f"vertex_budget_used: {d['vertex_budget_used']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert a GDS into a bounded text summary (for LLM prompts).")
    ap.add_argument("gds", nargs="+")
    ap.add_argument("--max-layers", type=int, default=6)
    ap.add_argument("--max-polys-per-layer", type=int, default=6)
    ap.add_argument("--max-vertices", type=int, default=240)
    ap.add_argument("--precision", type=int, default=3)
    ap.add_argument("--json", action="store_true", help="Print JSON instead of formatted text.")
    args = ap.parse_args()

    for p in map(Path, args.gds):
        if args.json:
            print(json.dumps(gds_to_dict(
                p,
                max_layers=args.max_layers,
                max_polys_per_layer=args.max_polys_per_layer,
                max_vertices=args.max_vertices,
                precision=args.precision,
            ), ensure_ascii=True, indent=2))
        else:
            print(gds_to_text(
                p,
                max_layers=args.max_layers,
                max_polys_per_layer=args.max_polys_per_layer,
                max_vertices=args.max_vertices,
                precision=args.precision,
            ))


if __name__ == "__main__":
    main()
