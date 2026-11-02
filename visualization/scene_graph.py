"""Scene graph of predicted human-object interactions (NetworkX + Matplotlib).

Torch-free on purpose: it takes plain Python lists, so it is easy to test and
the Streamlit app can call it without touching model code.

Input contract
--------------
entities:     list of dicts {"idx": int, "label": str, "is_human": bool, "score": float}
interactions: list of dicts {"human": int, "object": int, "action": str, "prob": float}
"human" / "object" are node indices into `entities` (same indexing as the graph).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402

HUMAN_COLOR = "#2e9e5b"
OBJECT_COLOR = "#e8873a"


def build_scene_graph(entities: Sequence[Dict[str, Any]],
                      interactions: Sequence[Dict[str, Any]],
                      show_all_entities: bool = False) -> nx.MultiDiGraph:
    """Nodes = entities, edges = predicted interactions (one edge per action).

    By default only entities that take part in at least one predicted interaction
    are kept, so a crowded image does not produce an unreadable graph.
    """
    g = nx.MultiDiGraph()
    used = {i["human"] for i in interactions} | {i["object"] for i in interactions}
    by_idx = {e["idx"]: e for e in entities}
    for e in entities:
        if show_all_entities or e["idx"] in used:
            g.add_node(e["idx"], label=f'{e["label"]} #{e["idx"]}',
                       is_human=bool(e["is_human"]), score=float(e.get("score", 1.0)))
    for it in interactions:
        h, o = it["human"], it["object"]
        if h not in by_idx or o not in by_idx:
            raise ValueError(f"interaction refers to unknown entity: {it}")
        g.add_edge(h, o, action=str(it["action"]), prob=float(it["prob"]))
    return g


def interaction_table(interactions: Sequence[Dict[str, Any]],
                      entities: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rows for a table: Human, Relationship, Object, Confidence (sorted by confidence)."""
    by_idx = {e["idx"]: e for e in entities}
    rows = []
    for it in sorted(interactions, key=lambda x: -x["prob"]):
        h, o = by_idx[it["human"]], by_idx[it["object"]]
        rows.append({"Human": f'{h["label"]} #{h["idx"]}', "Relationship": it["action"],
                     "Object": f'{o["label"]} #{o["idx"]}', "Confidence": round(float(it["prob"]), 3)})
    return rows


def draw_scene_graph(g: nx.MultiDiGraph, title: Optional[str] = None,
                     figsize=(9.0, 6.0)) -> plt.Figure:
    """Humans in a left column, objects in a right column, labelled directed edges."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    if g.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No entities to draw", ha="center", va="center", transform=ax.transAxes)
        return fig

    humans = sorted(n for n, d in g.nodes(data=True) if d["is_human"])
    objects = sorted(n for n, d in g.nodes(data=True) if not d["is_human"])
    pos = {}
    for col, nodes, x in ((0, humans, -1.0), (1, objects, 1.0)):
        for k, n in enumerate(nodes):
            y = 0.0 if len(nodes) == 1 else 1.0 - 2.0 * k / (len(nodes) - 1)
            pos[n] = (x, y)

    nx.draw_networkx_nodes(g, pos, nodelist=humans, node_color=HUMAN_COLOR, node_size=1500, ax=ax)
    nx.draw_networkx_nodes(g, pos, nodelist=objects, node_color=OBJECT_COLOR, node_size=1500, ax=ax)
    nx.draw_networkx_labels(g, pos, labels={n: d["label"] for n, d in g.nodes(data=True)},
                            font_size=8, font_color="white", ax=ax)

    # One curved edge per (human, object, action); curvature separates parallel edges.
    seen: Dict[tuple, int] = {}
    for u, v, k, d in g.edges(keys=True, data=True):
        n_prev = seen.get((u, v), 0)
        seen[(u, v)] = n_prev + 1
        rad = 0.12 + 0.18 * n_prev
        width = 0.8 + 3.0 * d["prob"]
        nx.draw_networkx_edges(g, pos, edgelist=[(u, v, k)], width=width, alpha=0.8,
                               arrows=True, arrowsize=18, node_size=1500,
                               connectionstyle=f"arc3,rad={rad}", ax=ax)
        (x1, y1), (x2, y2) = pos[u], pos[v]
        # label near the middle of the arc
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        ax.text(mx - rad * dy * 0.5, my + rad * dx * 0.5, f'{d["action"]} {d["prob"]:.2f}',
                fontsize=8, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.7", alpha=0.9))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.3, 1.3)
    fig.tight_layout()
    return fig
