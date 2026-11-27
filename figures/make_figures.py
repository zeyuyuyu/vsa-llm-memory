"""Every figure of the rewritten manuscript, drawn by matplotlib from the files under figdata/ (see figdata.py).

    python3 make_figures.py                 # all six figures, then the plotted values
    python3 make_figures.py 2 S1            # a subset: 1 2 3 4 5 S1
    python3 make_figures.py --no-verify     # skip the re-check of transcribed values against their source files

    key   file                        panels
    1     figs/fig1_overview.pdf      a three couplings of a frozen language model (schematic.py)
                                      b a worked two-hop retrieval, computed here from the algebra (an illustration)
    2     figs/fig2_benchmarks.pdf    a BABILong   b BABILong per task at 1M and 10M tokens   c CLUTRR
    5     figs/fig5_oplogs.pdf        a three question sets   b paired outcomes against GPT-6.1 Sol   c past window by site
    3     figs/fig3_store.pdf         a store against prompt as the evidence grows   b capacity of one store   c sharding
    4     figs/fig4_mechanism.pdf     a three arenas   b imposed cross-talk   c bindings per level
    S1    figs/figS1_sweeps.pdf       a chain depth   b one projection read two ways   c facts held against dimension

No number is typed in the plotting code. Values come from the banked artifacts (figdata/<run>_logs/*.json) or, where no
JSON artifact holds them, from small files in which each value carries its source (figdata/oplogs.json,
figdata/benchmarks/*.json, figdata/constants.json, figdata/worked_example.json). Transcribed values are re-checked
against the source text on every run while the source files exist (verify_sources).

Each figure is a vector PDF at the design size (180 mm wide, nothing cropped). FIGURE_VALUES.json receives every
plotted number with its source. Conventions: one colour and one name per arm (figstyle.C, figstyle.NAME); seed dots
where an arm has values that depend on the seed, none for deterministic arms and single runs; no titles; no text under
7 pt (check() reports on every figure as it is saved).
"""
import csv
import json
import math
import os
import re
import sys

import numpy as np
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.transforms import Bbox

from figstyle import plt, MM, C, NAME, INK, MUTED, GRID, NEUTRAL, LABEL, SMALL, MIN_PT, LETTER
from figdata import J, D, same, accs, rows
import schematic

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
os.makedirs("figs", exist_ok=True)

VALUES_JSON = "FIGURE_VALUES.json"
REC = {}
DASH, DOT, DASHDOT = (0, (4, 2)), (0, (1, 1.6)), (0, (5, 1.6, 1, 1.6))
mean = lambda xs: float(np.mean(xs))


def rec(fig, panel, series, **kw):
    def clean(v):
        if isinstance(v, np.generic):
            return float(v)
        if isinstance(v, np.ndarray):
            return [clean(x) for x in v.tolist()]
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, dict):
            return {str(k): clean(x) for k, x in v.items()}
        return v
    REC.setdefault(fig, {}).setdefault(panel, []).append(dict(series=series, **{k: clean(v) for k, v in kw.items()}))


# ================================================================================================= layout and checks
def new_fig(w_mm, h_mm):
    fig = plt.figure(figsize=(w_mm * MM, h_mm * MM))
    fig._mm = (float(w_mm), float(h_mm))
    return fig


def axmm(fig, x, y, w, h, grid=True):
    W, H = fig._mm
    ax = fig.add_axes([x / W, y / H, w / W, h / H])
    if grid:
        ax.yaxis.grid(True, linewidth=0.5, color=GRID, zorder=0)
        ax.set_axisbelow(True)
    return ax


def letter(fig, s, x, y):
    W, H = fig._mm
    fig.text(x / W, y / H, s, fontsize=LETTER, fontweight="bold", va="top", ha="left", color="black")


def leg(fig, handles, x, y, ncol=1, loc="upper left", **kw):
    W, H = fig._mm
    return fig.legend(handles=handles, loc=loc, bbox_to_anchor=(x / W, y / H), ncol=ncol, fontsize=SMALL,
                      borderaxespad=0, borderpad=0, **kw)


def check(fig, name, boxes=()):
    """Smallest font, text outside the canvas, overlapping text boxes, schematic labels that leave their box.
    Collisions between text and data marks are not seen here: those are checked by eye on the rendered page."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ext, problems = [], []
    for t in fig.findobj(Text):
        ax = getattr(t, "axes", None)
        if not t.get_visible() or not t.get_text().strip() or (ax is not None and not ax.get_visible()):
            continue
        bb = t.get_window_extent(r)
        if bb.width > 0 and bb.height > 0:
            ext.append((t, bb))
    for t, _ in ext:
        if t.get_fontsize() < MIN_PT - 1e-6:
            problems.append("font %.2f pt < %.1f: %r" % (t.get_fontsize(), MIN_PT, t.get_text()[:30]))
    fb = fig.bbox
    for t, bb in ext:
        if bb.x0 < fb.x0 - 0.5 or bb.y0 < fb.y0 - 0.5 or bb.x1 > fb.x1 + 0.5 or bb.y1 > fb.y1 + 0.5:
            problems.append("outside the canvas: %r" % t.get_text()[:40])
    pad = 0.6
    for i in range(len(ext)):
        ti, a = ext[i]
        for j in range(i + 1, len(ext)):
            tj, b = ext[j]
            if a.x0 + pad < b.x1 and b.x0 + pad < a.x1 and a.y0 + pad < b.y1 and b.y0 + pad < a.y1:
                problems.append("text overlap: %r <-> %r" % (ti.get_text()[:30], tj.get_text()[:30]))
    for patch, t in boxes:
        pb, tb = patch.get_window_extent(r), t.get_window_extent(r)
        m = 0.5 / 25.4 * fig.dpi
        if tb.x0 < pb.x0 + m or tb.x1 > pb.x1 - m or tb.y0 < pb.y0 + m or tb.y1 > pb.y1 - m:
            problems.append("text does not fit its box: %r" % t.get_text()[:30])
    sizes = [t.get_fontsize() for t, _ in ext]
    print("  check %-16s %3d text objects, smallest %.2f pt, %s"
          % (name, len(ext), min(sizes), "no layout problem found" if not problems else "%d PROBLEM(S)" % len(problems)))
    for p in problems:
        print("      - " + p)
    return problems


def save(fig, name, boxes=()):
    check(fig, name, boxes)
    W, H = fig.get_size_inches()
    path = "figs/%s.pdf" % name
    fig.savefig(path, bbox_inches=Bbox.from_extents(0, 0, W, H), metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    msg = ""
    try:                                        # glyph sizes actually written (includes any mathtext), page size, rasters
        import fitz
        page = fitz.open(path)[0]
        sp = [s["size"] for b in page.get_text("dict")["blocks"] for l in b.get("lines", []) for s in l["spans"] if s["text"].strip()]
        msg = "page %.1f x %.1f mm, smallest glyph %.2f pt, raster images %d" % (
            page.rect.width / 72 * 25.4, page.rect.height / 72 * 25.4, min(sp), len(page.get_images()))
        assert min(sp) >= MIN_PT - 0.05, "glyph under %.1f pt in %s" % (MIN_PT, path)
    except ImportError:
        pass
    print("%s  %s" % (path, msg))


def bar_dots(ax, x, w, ys, ms=2.6):
    """Seed values on a bar: dark dots with a white edge, so that they also show on black bars."""
    ax.scatter(np.linspace(x - w * 0.24, x + w * 0.24, len(ys)), ys, s=ms ** 2, color=INK, zorder=4,
               edgecolors="white", linewidths=0.4)


def line_dots(ax, x, ys, col, alpha=0.45):
    xs = [x] * len(ys) if np.isscalar(x) else x
    ax.scatter(xs, ys, s=7, color=col, alpha=alpha, zorder=2, linewidths=0)


def grouped(ax, groups, arms, gap=0.22, dots=True):
    """Grouped bars. arms: [(key, label, colour, {group: per-seed list or scalar or None}, bar keywords)]."""
    n = len(arms)
    width = (1 - gap) / n
    xs = np.arange(len(groups))
    pos = {}
    for i, (key, lab, col, data, kw) in enumerate(arms):
        off = (i - (n - 1) / 2) * width
        for g, x in zip(groups, xs):
            pos[(key, g)] = x + off
            v = data.get(g)
            if v is None:
                continue
            ys = list(v) if isinstance(v, (list, tuple)) else [float(v)]
            k = dict(color=col, linewidth=0)
            k.update(kw)
            ax.bar(x + off, mean(ys), width * 0.9, zorder=2, **k)
            if dots and len(ys) > 1:
                bar_dots(ax, x + off, width, ys)
    ax.set_xticks(xs)
    ax.tick_params(axis="x", length=0)
    return pos, width


def ramp(hexcol, n, lo=0.38, hi=1.0, dark=0.62):
    """n shades of one hue, light to dark; the last-but-one is the colour itself."""
    c = np.array(matplotlib.colors.to_rgb(hexcol))
    out = []
    for i in range(n):
        if i < n - 1:
            f = lo + (hi - lo) * i / (n - 2)
            out.append(matplotlib.colors.to_hex(1 - f * (1 - c)))
        else:
            out.append(matplotlib.colors.to_hex(dark * c))
    return out


def inleg(ax, handles, loc, anchor, transform=None, **kw):
    """A key inside the panel, in a region that holds no data (the style of Figs 5 and 7: no detached legend blocks)."""
    k = dict(fontsize=SMALL, borderaxespad=0, borderpad=0, labelspacing=0.4, handletextpad=0.5, frameon=False)
    k.update(kw)
    return ax.legend(handles=handles, loc=loc, bbox_to_anchor=anchor, bbox_transform=transform or ax.transAxes, **k)


def tint(c, f):
    return matplotlib.colors.to_hex(1 - f * (1 - np.array(matplotlib.colors.to_rgb(c))))


# =========================================================================================== transcribed values: checks
def verify_sources():
    """Every value of figdata/oplogs.json and figdata/benchmarks/babilong_per_task.json carries the verbatim source
    line. Re-check: the line is in the file, the number's text is in the line (in the stated table cell where one is
    given), and the number equals the stored value. Files that are not on this machine are reported and skipped."""
    seen, bad, skipped = 0, [], set()
    texts = {}

    def text_of(path):
        if path not in texts:
            texts[path] = open(path).read() if os.path.exists(path) else None
        return texts[path]

    def visit(node, where):
        nonlocal seen
        if isinstance(node, dict):
            if "file" in node and ("line" in node or "lines" in node):
                src = text_of(node["file"])
                if src is None:
                    skipped.add(node["file"])
                else:
                    seen += 1
                    lines = node.get("lines", [node.get("line")])
                    if "\n".join(lines) not in src:
                        bad.append("%s: line not found verbatim in %s" % (where, node["file"]))
                    if "text" in node:
                        hay = lines[0]
                        if "cell" in node:
                            hay = [c.strip() for c in hay.strip().strip("|").split("|")][node["cell"]]
                        if node["text"] not in hay:
                            bad.append("%s: %r not in its source line" % (where, node["text"]))
                        if "value" in node:                 # the stored number is the printed one (a fraction, a per cent, or c/n)
                            num = node["text"].split("±")[0]
                            if "/" in num:
                                c, n = num.split()[-1].split("/")
                                got = [100.0 * int(c) / int(n)]
                            else:
                                got = [float(num), 100 * float(num)]
                            if min(abs(g - node["value"]) for g in got) > 1e-9:
                                bad.append("%s: value %s != text %s" % (where, node["value"], node["text"]))
                        if "sd" in node and abs(float(node["text"].split("±")[1]) - node["sd"]) > 1e-9:
                            bad.append("%s: s.d. %s != text %s" % (where, node["sd"], node["text"]))
            for k, v in node.items():
                visit(v, where + "/" + str(k))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                visit(v, where + "[%d]" % i)

    visit(J("oplogs.json"), "oplogs.json")
    visit(J("benchmarks/babilong_per_task.json"), "babilong_per_task.json")
    visit(J("gains.json"), "gains.json")
    visit(J("benchmarks/ruler.json"), "ruler.json")
    visit(J("benchmarks/b23.json"), "b23.json")
    visit(J("benchmarks/b26.json"), "b26.json")
    visit(J("b27.json"), "b27.json")
    visit(J("benchmarks/b24.json"), "b24.json")
    visit(J("b28.json"), "b28.json")
    visit(J("b33.json"), "b33.json")
    visit(J("b47_fig2bc.json")["label_free_points"], "b47_fig2bc.json label_free_points")
    visit(J("b47_fig2bc.json")["phi2_scratch_n800"], "b47_fig2bc.json phi2_scratch_n800")
    visit(J("b49_compression.json"), "b49_compression.json")
    visit(J("b60_compression.json"), "b60_compression.json")
    visit(J("b44.json"), "b44.json")
    visit(J("b31.json"), "b31.json")
    visit(J("b38.json"), "b38.json")
    # counts named in the paired-outcome sentences, in the order of the sentence
    for r in J("oplogs.json")["paired"]["rows"]:
        for block in (r, r.get("repaired_parse")):
            if block is None or text_of(r["file"]) is None:
                continue
            seen += 1
            nums = [int(x) for x in re.findall(r"(?:right|RM|history|frontier|wrong) (\d+)[,;]", " ".join(block.get("lines", r["lines"])))]
            if nums[:4] != [block[k] for k in ("both_right", "only_ours", "only_gpt", "both_wrong")]:
                bad.append("paired/%s: counts %s are not those of the source sentence %s" % (r["id"], [block[k] for k in ("both_right", "only_ours", "only_gpt", "both_wrong")], nums))
    # per-site question counts of the past-window set: recounted from the ids of the scored answers
    for s in J("oplogs.json")["per_site"]["sites"]:
        f = s["n"]["count_file"]
        if os.path.exists(f):
            seen += 1
            got = sum(1 for l in open(f) if json.loads(l)["id"].startswith(s["n"]["count_prefix"]))
            if got != s["n"]["value"]:
                bad.append("per_site/%s: n %d, counted %d" % (s["id"], s["n"]["value"], got))
        else:
            skipped.add(f)
    print("verify_sources: %d transcribed values re-checked against their source text, %s%s"
          % (seen, "all found verbatim" if not bad else "%d PROBLEM(S)" % len(bad),
             "" if not skipped else "; source file(s) not on this machine, skipped: %s" % ", ".join(sorted(skipped))))
    for b in bad:
        print("      - " + b)
    if bad:
        raise SystemExit("transcribed values disagree with their sources")


# ============================================================================================================ Fig. 1
def worked_example():
    """A two-hop retrieval with real HRR vectors, with the operations of the arena code (v13.py, v37b.py): random unit
    codebook rows, binding by circular convolution, a bundle per hop, unbinding by the approximate inverse, the unbound
    vector normalized and cleaned up by cosine against the candidate rows."""
    cfg = J("worked_example.json")
    d = cfg["d"]
    names = cfg["objects"] + cfg["persons"] + cfg["locations"]
    cb = np.random.RandomState(cfg["seed"]).standard_normal((len(names), d))
    cb /= np.linalg.norm(cb, axis=1, keepdims=True)
    code = dict(zip(names, cb))
    bind = lambda a, b: np.fft.irfft(np.fft.rfft(a) * np.fft.rfft(b), n=d)
    unbind = lambda ab, a: np.fft.irfft(np.fft.rfft(ab) * np.conj(np.fft.rfft(a)), n=d)
    unit = lambda v: v / np.linalg.norm(v)
    story = cfg["story"]                                               # (person, object, location)
    mem1 = sum(bind(code[p], code[o]) for p, o, l in story)            # holder bundle: object -> holder
    mem2 = sum(bind(code[l], code[p]) for p, o, l in story)            # location bundle: person -> location
    h = unit(unbind(mem1, code[cfg["query_object"]]))
    cos1 = [float(h @ code[p]) for p in cfg["persons"]]
    holder = cfg["persons"][int(np.argmax(cos1))]                      # clean-up between the hops
    g = unit(unbind(mem2, code[holder]))
    cos2 = [float(g @ code[l]) for l in cfg["locations"]]
    place = cfg["locations"][int(np.argmax(cos2))]
    truth = {o: (p, l) for p, o, l in story}[cfg["query_object"]]
    assert (holder, place) == truth, "the worked example must retrieve the right chain"
    off = (cb @ cb.T)[np.triu_indices(len(names), 1)]
    return dict(cfg=cfg, cos1=cos1, cos2=cos2, holder=holder, place=place, pairs=len(story),
                rows_cos_mean=float(off.mean()), rows_cos_absmax=float(np.abs(off).max()))


def fig1():
    # r50: the lower band (the BABILong delta-rule reader) was dropped -- those results are in the SI only, and the
    # band read as part of the coupling studied here (referee 10, Fig. 1a).
    fig = new_fig(180, 125)
    ax = axmm(fig, 5, 55, 172, 68, grid=False)
    schematic.draw_overview(ax, 172, 68)
    letter(fig, "a", 1, 124.5)

    ex = worked_example()
    cfg = ex["cfg"]
    hops = [(cfg["persons"], ex["cos1"], ex["holder"],
             "step 1: who holds the “%s”?\nsimilarity of the result to each person’s code" % cfg["query_object"]),
            (cfg["locations"], ex["cos2"], ex["place"],
             "step 2: where is “%s”?\nsimilarity of the result to each place’s code" % ex["holder"])]
    top = max(max(ex["cos1"]), max(ex["cos2"]))
    for k, (cand, cos, win, xl) in enumerate(hops):
        ax = axmm(fig, 16 + 90 * k, 19, 70, 27)
        iw = cand.index(win)
        assert iw == 0, "layout assumes that the queried chain is the first statement of the story"
        ax.bar(range(len(cand)), cos, 0.6, color=[C["handover"] if n == win else NEUTRAL for n in cand], linewidth=0, zorder=2)
        ax.axhline(0, color=INK, lw=0.6, zorder=3)
        ax.spines["bottom"].set_visible(False)
        ax.set_xticks(range(len(cand))); ax.set_xticklabels(cand); ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.6, len(cand) - 0.4)
        ymax = np.ceil(top * 10 + 1) / 10
        ax.set_ylim(-0.06, ymax); ax.spines["left"].set_bounds(0, ymax)
        ax.set_yticks(np.arange(0, ymax + 1e-9, 0.2))
        ax.set_ylabel("similarity (cosine)")
        ax.set_xlabel(xl, linespacing=1.3)
        # the winning row, and the residue the other pairs of the bundle leave on every other row
        ax.annotate("clean-up picks the closest code", xy=(iw + 0.33, cos[iw] * 0.93), xytext=(iw + 0.85, cos[iw] * 0.93),
                    fontsize=SMALL, color=INK, va="center", ha="left",
                    arrowprops=dict(arrowstyle="-", color=INK, lw=0.6, shrinkA=1.5, shrinkB=0))
        others = [c for i, c in enumerate(cos) if i != iw]
        yb = max(others) + 0.035
        ax.plot([0.68, 0.68, len(cand) - 0.68, len(cand) - 0.68], [yb - 0.018, yb, yb, yb - 0.018], color=MUTED, lw=0.6, zorder=3,
                solid_joinstyle="miter")
        ax.text((len(cand)) / 2, yb + 0.012, "cross-talk", fontsize=SMALL, color=INK, ha="center", va="bottom")
        rec("fig1_overview", "b", "hop %d" % (k + 1), candidates=cand, cosine=cos, selected=win,
            colours={"selected row": C["handover"], "other rows": NEUTRAL},
            source="computed by make_figures.py worked_example() from figdata/worked_example.json (seed %d, d = %d, %d pairs per bundle)"
                   % (cfg["seed"], cfg["d"], ex["pairs"]))
    rec("fig1_overview", "b", "set-up", seed=cfg["seed"], d=cfg["d"], pairs_per_bundle=ex["pairs"], objects=cfg["objects"],
        persons=cfg["persons"], locations=cfg["locations"], story=cfg["story"], question=cfg["question"],
        answer_chain=[cfg["query_object"], ex["holder"], ex["place"]],
        mean_cosine_between_different_codebook_rows=ex["rows_cos_mean"], largest_abs_cosine_between_different_codebook_rows=ex["rows_cos_absmax"],
        numpy=np.__version__, source=cfg["source"])
    letter(fig, "b", 1, 51.5)
    save(fig, "fig1_overview", schematic.BOXES["overview"])


def babi_table():
    """bAbI tasks 1 and 2 (Phi-2). Not drawn in any figure (author's decision): recorded for the Supplementary Table."""
    v49, v131 = J("v49_logs/v49.json"), J("v131_logs/v131b.json")
    best = max(v131["runs"], key=lambda r: mean(r["accs"]))
    assert best["tag"] == "r16_e6_lr0.0001" and len(v131["runs"]) == 12
    T = [("prompting", "v49_logs/v49.json cells['task1/zeroshot/all'].acc and cells['task2/zeroshot/all'].acc (n = 500 each; one deterministic decoding stored three times)",
          v49["cells"]["task1/zeroshot/all"]["acc"], v49["cells"]["task2/zeroshot/all"]["acc"]),
         ("LoRA fine-tune", "task 1: v50_logs/v50_task1_VOID_locoffset.json arms.lora_direct; task 2: v131_logs/v131b.json run r16_e6_lr0.0001 accs "
          "(highest test mean of 12 configurations)", J("v50_logs/v50_task1_VOID_locoffset.json")["arms"]["lora_direct"], best["accs"]),
         ("LoRA fine-tune, first configuration (not the best of twelve)", "v50_logs/v50_task2_BANKED.json arms.lora_direct", None,
          J("v50_logs/v50_task2_BANKED.json")["arms"]["lora_direct"]),
         ("projection trained through the binding loss", "v37_logs/v37b.json arms.spans (task 2 only; task 1 not run)", None,
          J("v37_logs/v37b.json")["arms"]["spans"]),
         ("handover", "task 1: v35_logs/v35.json cells[family none].chain; task 2: v37_logs/v37b.json arms.probe_argmax",
          rows(J("v35_logs/v35.json")["cells"], family="none")[0]["chain"], J("v37_logs/v37b.json")["arms"]["probe_argmax"]),
         ("frozen-state probe", "task 1: v30_logs/v30.json arms.floor; task 2: v48_logs/v48.json floor_last_token",
          J("v30_logs/v30.json")["arms"]["floor"], J("v48_logs/v48.json")["floor_last_token"])]
    for name, src, t1, t2 in T:
        rec("not_drawn_bAbI", "table", name, task1=None if t1 is None else {"mean": mean(t1), "per_seed": t1},
            task2=None if t2 is None else {"mean": mean(t2), "per_seed": t2}, source=src)


# ============================================================================================================ Fig. 5
def fig4_mechanism():
    fig = new_fig(180, 150)
    Cc = J("v13_logs/v13_fixed.json")            # arena C as published: panel b's floor and naming-head point
    # --- a: B31 (registered): every arm on frozen Phi-2 and Qwen3-8B, equalised baselines, order-keeping binding ---------
    B31 = J("b31.json")
    AK = {"probe": "equalised probe", "loss": "binding-loss map (perm, named)", "lf": "label-free InfoNCE map (perm, named)",
          "lora": "equalised LoRA", "handover": "naming head + VSA (perm, named)"}
    vals = lambda enc, g, k: [sd["value"] for sd in B31["arms"]["%s|%s|%s" % (enc, g, k)]["seeds"]]
    order = ["probe", "loss", "lf", "lora", "handover", "rows"]            # failing designs on top, the two that work below
    wrap = {"loss": "map trained only\non the answer", "probe": "probe of the hidden\nstates (equalised)",
            "lf": "same, codes kept\napart (no labels)", "lora": "LoRA fine-tune\n(equalised, tuned)",
            "handover": "naming head", "rows": "exact symbols"}
    heads = {"B": "one step (arena B)\n64 bindings, pairs held out", "C": "two steps (arena C)\n32 bindings per level",
             "A": "three steps (arena A)\n16 pairs per level"}
    ENC = [("Phi-2", -0.2, 1.0), ("Qwen3-8B", 0.2, 0.42)]
    for j, g in enumerate(["B", "C", "A"]):                        # ordered by the number of steps to compose
        ax = axmm(fig, 40 + 46 * j, 85, 42, 52, grid=False)
        ax.xaxis.grid(True, linewidth=0.5, color=GRID, zorder=0); ax.set_axisbelow(True)
        for i, k in enumerate(order):
            if k == "rows":
                ys = vals("no encoder", g, "exact symbols + VSA (perm)")
                ax.barh(i, mean(ys), 0.5, color=C[k], linewidth=0, zorder=2)
                ax.text(max(ys) + 0.025, i, "%.3f" % mean(ys), ha="left", va="center", fontsize=MIN_PT, color=INK)
                continue
            for enc, off, alpha in ENC:
                ys = vals(enc, g, AK[k])
                nofit = (k == "lora" and enc == "Phi-2" and g in "AC")     # r55: did not fit its own training split
                ax.barh(i + off, mean(ys), 0.36, color=C[k], alpha=alpha, linewidth=0, zorder=2,
                        hatch="////" if nofit else None, edgecolor="white" if nofit else None)
                ax.scatter(ys, [i + off] * len(ys), s=4, color=INK, edgecolors="white", linewidths=0.3, zorder=4)
                if k == "handover" and enc == "Qwen3-8B":               # value inside the bar: the open diamond sits beyond its end
                    ax.text(mean(ys) - 0.03, i + off, "%.3f" % mean(ys), ha="right", va="center", fontsize=MIN_PT, color=INK, zorder=5)
                else:
                    ax.text(max(ys) + 0.025, i + off, "%.3f" % mean(ys), ha="left", va="center", fontsize=MIN_PT, color=INK)
            if k == "handover":                                     # exploratory: Qwen3-8B naming head read from a dev-chosen layer
                ex = B31["explore1"][g]
                ax.scatter([ex["mean"]["value"]], [i + 0.2], s=16, marker="D", facecolors="none", edgecolors=INK, linewidths=0.8, zorder=6)
        ax.axvline(0.5, color=MUTED, lw=0.7, ls=DASH, zorder=3)
        ax.set_ylim(len(order) - 0.45, -0.55); ax.set_xlim(0, 1.25); ax.set_xticks([0, 0.5, 1.0]); ax.set_xticklabels(["0", "0.5", "1"])
        ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([wrap[k] for k in order] if j == 0 else [], linespacing=1.05, fontsize=MIN_PT)
        ax.text(0, -0.75, heads[g], fontsize=SMALL, color=MUTED, fontstyle="italic", ha="left", va="bottom", linespacing=1.15)
        if j == 1:
            ax.set_xlabel("accuracy (chance 0.5, dashed)")
    hb = [Patch(color=MUTED, alpha=1.0, label="Phi-2 (upper bar)"),
          Patch(color=MUTED, alpha=0.42, label="Qwen3-8B (lower bar)"),
          Line2D([], [], marker="D", ms=4, mfc="none", mec=INK, mew=0.8, ls="none", label="Qwen3-8B head, middle layer (exploratory)"),
          Patch(facecolor=C["lora"], hatch="////", edgecolor="white", label="fine-tune not fitted to its training split")]
    leg(fig, hb, 4, 149.6, ncol=4, columnspacing=0.6, handlelength=1.0)
    letter(fig, "a", 1, 149.5)
    for k in order:
        if k == "rows":
            rec("fig4_mechanism", "a", NAME[k], colour=C[k], source="b31.json arms 'no encoder|<arena>|exact symbols + VSA (perm)'",
                values={g: {"mean": mean(vals("no encoder", g, "exact symbols + VSA (perm)")),
                            "per_seed": vals("no encoder", g, "exact symbols + VSA (perm)")} for g in "ABC"})
            continue
        for enc, _, _ in ENC:
            rec("fig4_mechanism", "a", "%s, %s" % (NAME[k], enc), colour=C[k], source="b31.json arms '%s|<arena>|%s'" % (enc, AK[k]),
                values={g: {"mean": mean(vals(enc, g, AK[k])), "per_seed": vals(enc, g, AK[k])} for g in "ABC"})
    rec("fig4_mechanism", "a", "Qwen3-8B naming head from a dev-chosen layer (exploratory)", colour=INK, marker="D open",
        source="b31.json explore1", values={g: {"layer": B31["explore1"][g]["layer"], "mean": B31["explore1"][g]["mean"]["value"]} for g in "ABC"})

    # --- b: consistency sweep (B47, registered): exact orthogonal codes, zero coherence, each mention's code at a set
    #        cosine to its symbol's code; the label-free maps at their measured geometry ---------------------------------
    ax = axmm(fig, 14, 13, 96, 56)
    F = J("b44_fig2bc.json")
    VT = open("/path/to/vsassm/results/b44_verdict.txt").read() if os.path.exists("/path/to/vsassm/results/b44_verdict.txt") else None
    G = J("b47_fig2bc.json")
    V47 = open("/path/to/vsassm/results/b47_verdict.txt").read() if os.path.exists("/path/to/vsassm/results/b47_verdict.txt") else None
    men, form = G["b"]["consistency/mention/perm"], G["b"]["consistency/form/perm"]
    if V47 is not None:                                   # every plotted mean equals the verdict's table cell
        for k, rho in enumerate(men["rho"]):
            sec = V47[V47.index("## (3) Consistency sweep"):]
            row = [l for l in sec.split("\n") if l.startswith("  %.2f  " % rho) and "|" in l][0]
            cells = row.split("|")
            assert abs(float(cells[1].split()[0]) - men["acc_mean"][k]) < 6e-4, (rho, cells[1], men["acc_mean"][k])
            assert abs(float(cells[-1].split()[0]) - form["acc_mean"][k]) < 6e-4, (rho, cells[-1], form["acc_mean"][k])
    floor = G["b"]["floor_phi2"]["value"]
    for curve, col, ls_, lab in ((men, INK, "-", "every mention perturbed"), (form, MUTED, DASH, "only between a symbol's two written forms")):
        ax.plot(curve["same_mean"], curve["acc_mean"], color=col, lw=1.3, ls=ls_, marker="o", ms=3.0, zorder=5)
        rec("fig4_mechanism", "b", lab, colour=col, x_same_symbol_cosine=curve["same_mean"], y_mean=curve["acc_mean"],
            y_per_seed=curve["acc_per_seed"], source="figdata/b47_fig2bc.json (vsassm b47_out/b47_fig2bc.json; checked against b47_verdict.txt section 3)")
    P = G["label_free_points"]
    for key, lab, mk, col in (("infonce_phi2", "label-free map, Phi-2", "D", C["lf"]), ("infonce_qwen8", "label-free map, Qwen3-8B", "s", C["lf"])):
        q = P[key]
        ax.scatter([q["same"]], [q["acc"]], s=48, marker=mk, zorder=6, facecolors=col, edgecolors="white", linewidths=0.7)
        rec("fig4_mechanism", "b", lab, colour=col, marker=mk, x_same_symbol_cosine=q["same"], y=q["acc"], source=q["line"].strip())
    ax.axhline(floor, color=C["probe"], lw=0.8, ls=DOT, zorder=1)
    ax.text(0.99, floor - 0.012, "probe (floor)", fontsize=MIN_PT, color=MUTED, va="top", ha="right")
    ax.text(0.37, 0.56, "label-free maps (Phi-2 diamond,\nQwen3-8B square): consistent across\nmentions only to cosine 0.30", fontsize=SMALL,
            color=INK, ha="left", va="center", linespacing=1.15)
    ax.text(0.04, 0.93, "every mention's code\nperturbed (solid)", fontsize=SMALL, color=INK, ha="left", va="center", linespacing=1.1)
    ax.text(0.72, 0.70, "only across a symbol's\ntwo written forms (dashed)", fontsize=SMALL, color=MUTED, ha="left", va="center", linespacing=1.1)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0.4, 1.03)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xlabel("consistency (mean cosine between one symbol’s codes across mentions)")
    ax.set_ylabel("accuracy on arena C")
    letter(fig, "b", 1, 73)

    # --- c: bindings per level in arena C (B44, registered pipeline) -------------------------------------------
    ax = axmm(fig, 128, 13, 49, 56)
    cc = F["c"]
    ks = [r["k"] for r in cc["phi2"]]
    if VT is not None:
        for enc in ("phi2", "qwen8"):
            for r in cc[enc]:
                ln = [l for l in VT.split("\n") if l.startswith("  2c %s k=%d " % (enc, r["k"])) or l.startswith("  2c %s  k=%d " % (enc, r["k"]))][0]
                assert "naming head " + "/".join("%.3f" % x for x in r["name"]) in ln and "probe " + "/".join("%.3f" % x for x in r["probe"]) in ln
    series = [("exact codes", [r["exact"] for r in cc["phi2"]], INK, "o", "-"),
              ("naming head, Phi-2", [r["name"] for r in cc["phi2"]], C["handover"], "s", "-"),
              ("naming head, Qwen3-8B", [r["name"] for r in cc["qwen8"]], C["handover"], "s", DASH),
              ("probe, Phi-2", [r["probe"] for r in cc["phi2"]], C["probe"], "^", "-"),
              ("probe, Qwen3-8B", [r["probe"] for r in cc["qwen8"]], C["probe"], "^", DASH)]
    for lab, per, col, mk, ls_ in series:
        ax.plot(ks, [mean(p) for p in per], color=col, marker=mk, ms=3.2, lw=1.1, ls=ls_, zorder=3)
        rec("fig4_mechanism", "c", lab, colour=col, x=ks, y_mean=[mean(p) for p in per], y_per_seed=per,
            source="figdata/b44_fig2bc.json c (checked against b44_verdict.txt '2c' lines)")
    ax.text(ks[3], 0.93, "top line: exact codes\nand Phi-2 naming head", fontsize=SMALL, color=INK, ha="center", va="top", linespacing=1.1)
    ax.text(ks[0] * 1.03, 0.63, "Qwen3-8B\nnaming head\n(dashed)", fontsize=SMALL, color=C["handover"], ha="left", va="center", linespacing=1.1)
    ax.text(ks[4], 0.55, "probe", fontsize=SMALL, color=MUTED, ha="center", va="bottom")
    ax.set_xscale("log", base=2); ax.set_xticks(ks); ax.set_xticklabels([str(int(k)) for k in ks]); ax.minorticks_off()
    ax.set_xlabel("bindings per level"); ax.set_ylabel("accuracy on arena C"); ax.set_ylim(0.4, 1.05)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    letter(fig, "c", 115, 73)
    save(fig, "fig4_mechanism")


def figS5_coherence():
    """Supplementary: imposed coherence along one shared direction (B44, registered), raw, commutative and centred."""
    fig = new_fig(125, 72)
    # --- b: imposed coherence on exact codes (B44, registered pipeline: permuted binding, threshold refitted per level),
    #        raw and centred; the learned maps at their own coherence (B31 trained; B44 at initialisation) -----------------
    ax = axmm(fig, 14, 13, 92, 56)
    F = J("b44_fig2bc.json")
    VT = open("/path/to/vsassm/results/b44_verdict.txt").read() if os.path.exists("/path/to/vsassm/results/b44_verdict.txt") else None
    sw = F["b"]
    raw, comm, cen = sw["sweep_raw/perm"], sw["sweep_raw/comm"], sw["sweep_centred/perm"]
    if VT is not None:                                   # every plotted sweep mean equals the verdict's table cell
        for k, c in enumerate(raw["c"]):
            row = [l for l in VT.split("\n") if l.startswith("  %.2f  " % c)][0].split()
            assert abs(float(row[1].split("/")[0]) - raw["acc_mean"][k]) < 6e-4, (c, row[1], raw["acc_mean"][k])
            assert abs(float(row[3].split("/")[0]) - cen["acc_mean"][k]) < 6e-4
            assert abs(float(row[9].split("/")[0]) - comm["acc_mean"][k]) < 6e-4
        assert "largest measured cross-talk clearing it %.3f" % sw["tolerance_perm_phi2_floor"] in VT
    floor = sw["floor_phi2"]["value"]; tol = sw["tolerance_perm_phi2_floor"]; tolc = sw["tolerance_comm_phi2_floor"]
    ax.axvspan(tol, 1.2, color="#F4F4F4", zorder=0, lw=0)
    for curve, col, ls_, lab in ((raw, INK, "-", "exact codes, coherence added"), (comm, MUTED, DASH, "same, commutative binding"),
                                 (cen, C["handover"], "-", "same, codes centred")):
        ax.plot(curve["crosstalk_mean"] if curve is not cen else curve["c"], curve["acc_mean"], color=col, lw=1.3, ls=ls_, marker="o",
                ms=3.0, zorder=5)
        rec("figS5_coherence", "a", lab, colour=col, x=curve["c"], measured_coherence=curve["crosstalk_mean"], y_mean=curve["acc_mean"],
            y_per_seed=curve["acc_per_seed"], source="figdata/b44_fig2bc.json (vsassm results/b44_fig2bc.json; checked against b44_verdict.txt Bar 5)")
    pts = sw["points_phi2"]
    for key, lab, mk, filled, col in (("naming_head", NAME["handover"], "o", True, C["handover"]),
                                      ("binding_loss_init_raw", "map before training", "D", False, C["loss"]),
                                      ("binding_loss_trained", "map trained only on the answer", "D", True, C["loss"]),
                                      ("binding_loss_init_centred", "map before training, centred", "s", False, C["loss"])):
        q = pts[key]; xv, yv = q["different_per_seed"], q["acc_per_seed"]
        ax.scatter([mean(xv)], [mean(yv)], s=56, marker=mk, zorder=6, facecolors=col if filled else "none",
                   edgecolors="white" if filled else col, linewidths=0.7 if filled else 1.4)
        rec("figS5_coherence", "a", lab, colour=col, marker=mk + (" filled" if filled else " open"), x_mean=mean(xv), y_mean=mean(yv),
            x_per_seed=xv, y_per_seed=yv, source="figdata/b44_fig2bc.json points_phi2.%s (%s)" % (key, q["source"]))
    ax.axhline(floor, color=C["probe"], lw=0.8, ls=DOT, zorder=1)
    ax.text(0.005, floor - 0.012, "probe (floor)", fontsize=MIN_PT, color=MUTED, va="top", ha="left")
    ax.plot([tol, tol], [0.4, 1.0], color=INK, lw=0.7, ls=DASH, zorder=1)
    ax.text(tol + 0.03, 0.93, "coherence added along one shared\ndirection: harmful above %.2f,\nremoved entirely by centring" % tol,
            fontsize=SMALL, color=INK, va="top", ha="left", linespacing=1.15)
    ax.text(0.56, 0.72, "learned map: before training (open),\nafter (filled), centred before\ntraining (square): near the floor",
            fontsize=SMALL, color=INK, ha="left", va="center", linespacing=1.15)
    ax.set_xlim(-0.04, 1.04); ax.set_ylim(0.4, 1.02)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_xlabel("code coherence (mean cosine between different symbols’ codes)")
    ax.set_ylabel("accuracy on arena C (Phi-2)")
    

    save(fig, "figS5_coherence")


# ============================================================================================================ Fig. 4
def pool_cap():
    """The pool of the capacity, sharding and scaling runs holds 800 facts; a scale requested as 1024 is 800.
    Checked here against the prompt lengths of V-106, which grow linearly in the number of facts in the prompt."""
    pool = J("constants.json")["pool"]
    v = J("v106_logs/v106.json")
    assert v["config"]["n_pool"] == pool["instances"] and pool["instances"] * pool["facts_per_instance"] == pool["facts"]
    b73 = J("v73_logs/v73_BANKED.json")["Ks"]                   # bindings grow by two per pooled instance until the pool runs out
    k73 = sorted(b73, key=int)
    mb = lambda k: b73[k]["compositional"]["mean_bindings"]
    assert (mb(k73[-2]) - mb(k73[0])) / (int(k73[-2]) - int(k73[0])) == pool["facts_per_instance"]
    pr = v["prompt"]
    under = [p for p in pr if p["n_facts"] <= pool["facts"]]
    slope, icpt = np.polyfit([p["n_facts"] for p in under], [p["tokens"] for p in under], 1)
    implied = (pr[-1]["tokens"] - icpt) / slope
    assert pr[-1]["n_facts"] == pool["requested_last_scale"] and abs(implied - pool["facts"]) < 0.02 * pool["facts"], implied
    return pool, float(implied), float(slope)


def fact_axis_(ax, labelled, CAP, label_all, minor=()):
    ax.set_xscale("log", base=2)
    ax.set_xticks(labelled + [CAP])
    ax.set_xticklabels([str(t) for t in labelled] + [""])
    ax.minorticks_off()
    ax.annotate(label_all, (CAP, 0), xycoords=("data", "axes fraction"), xytext=(0, -15.5), textcoords="offset points",
                fontsize=SMALL, ha="center", va="top", color=INK)
    ax.xaxis.labelpad = 12.5
    if minor:
        ax.set_xticks(list(minor), minor=True); ax.set_xticklabels([], minor=True)
    ax.set_xlim(labelled[0] / 1.25, CAP * 1.3)


def fig3_store():
    fig = new_fig(180, 64)
    Y0, HH = 15, 44
    pool, implied, slope = pool_cap()
    CAP = pool["facts"]
    crit = J("constants.json")["capacity_criterion"]["threshold"]
    print("  pool: %d facts; V-106 prompt lengths imply %.0f facts at the scale requested as %d (%.2f tokens per fact)"
          % (CAP, implied, pool["requested_last_scale"], slope))
    v = J("v106_logs/v106.json")
    # --- a: one store, readout against the number of superposed facts (V-151, HRR) ---------------------------
    ax = axmm(fig, 14, Y0, 70, HH)
    v151 = J("v151_logs/v151.json")
    dims, kgrid = v151["config"]["dims"], v151["config"]["ks"]
    kx = [min(k, CAP) for k in kgrid]
    shades = ramp(C["store"], len(dims))
    assert v["config"]["dim"] == dims[-2]                    # the dimension of panel b carries the arm's own colour
    dims_f, models, fit, thr = capacity_fits()
    assert dims_f == dims and abs(thr - crit) < 1e-12
    h = []
    for i, (d_, col) in enumerate(zip(dims, shades)):
        y = [v151["capacity"]["HRR"][str(d_)][str(k)] for k in kgrid]
        ax.plot(kx, y, color=col, marker="o", ms=2.8, lw=1.3, zorder=3)
        ks_ = fit["HRR"]["kstar"][i]
        ax.plot([ks_], [crit], marker="o", ms=4.4, mfc="white", mec=col, mew=1.2, ls="none", zorder=6)
        ax.annotate("%d" % round(ks_), (ks_, crit), xytext=(2.5, 2.5), textcoords="offset points", ha="left", va="bottom",
                    fontsize=MIN_PT, color=INK, zorder=7)
        h.append(Line2D([], [], color=col, marker="o", ms=2.8, lw=1.3, label="$d$ = %d" % d_))
        rec("fig3_store", "a", "HRR, d = %d" % d_, colour=col, x_facts_drawn=kx, x_grid_as_banked=kgrid, y=y, kstar_marked=ks_,
            source="v151_logs/v151.json capacity.HRR['%d'] (mean of %d instances x %d capacity seeds; per-seed values not banked, no dots); "
                   "open marker: the %.2f crossing, interpolated in log k (capacity_fits)" % (d_, v151["config"]["n"], len(v151["config"]["cap_seeds"]), crit))
    ax.axhline(crit, color=INK, lw=0.7, ls=DASH, zorder=1)
    ax.text(kx[0] / 1.15, crit + 0.02, "%.2f criterion" % crit, fontsize=MIN_PT, color=MUTED, ha="left", va="bottom")
    fact_axis_(ax, [8, 32, 128, 512], CAP, pool["label_all"], minor=[16, 64, 256])
    ax.set_ylim(0, 1.0); ax.set_ylabel("readout exact match"); ax.set_xlabel("facts superposed in one store")
    inleg(ax, h, "lower left", (0.03, 0.04), handlelength=1.6, labelspacing=0.3)
    letter(fig, "a", 1, 63.5)

    # --- b: sharding at d = 2048 (V-102) ---------------------------------------------------------------------
    ax = axmm(fig, 106, Y0, 70, HH)
    grid = J("v102_logs/v102_scored.json")["grid"]
    dd = v["config"]["dim"]
    Ks = sorted({int(k.split("_")[1][1:]) for k in grid if k.startswith("d%d_" % dd)})
    Ss = sorted({int(k.split("_")[2][1:]) for k in grid if k.startswith("d%d_" % dd)})
    shades = ramp(C["store"], len(Ks))
    h = []
    for K_, col in zip(Ks, shades):
        per = [grid["d%d_K%d_S%d" % (dd, K_, s)] for s in Ss]
        ax.plot(Ss, [mean(p) for p in per], color=col, marker="o", ms=3.2, lw=1.3, zorder=3)
        for s, p in zip(Ss, per):
            line_dots(ax, s, p, col, alpha=0.6)
        lab = "$K$ = %d facts" % K_ if K_ <= CAP else "$K$ = %s" % pool["label_all"]
        h.append(Line2D([], [], color=col, marker="o", ms=3.2, lw=1.3, label=lab))
        rec("fig3_store", "b", lab.replace("$", ""), colour=col, facts_in_pool=min(K_, CAP), facts_requested=K_, x_stores=Ss,
            y_mean=[mean(p) for p in per], y_per_seed=per, source="v102_logs/v102_scored.json grid['d%d_K%d_S*']" % (dd, K_))
    ax.axhline(crit, color=INK, lw=0.7, ls=DASH, zorder=1)
    ax.set_xscale("log", base=2); ax.set_xticks(Ss); ax.set_xticklabels([str(s) for s in Ss]); ax.minorticks_off()
    ax.set_xlim(Ss[0] / 1.2, Ss[-1] * 1.2)
    ax.set_ylim(0, 1.0); ax.set_ylabel("readout exact match, $d$ = %d" % dd); ax.set_xlabel("stores the facts are spread over")
    inleg(ax, h, "lower right", (1.0, 0.04), handlelength=1.6, labelspacing=0.3)
    letter(fig, "b", 93, 63.5)
    save(fig, "fig3_store")


def figS2_poolsweep():
    """Supplementary: both routes given the same pooled facts (V-106); was panel a of Fig. 3 before r8."""
    fig = new_fig(110, 62)
    pool, implied, slope = pool_cap()
    CAP = pool["facts"]
    ax = axmm(fig, 14, 15, 90, 42)
    v = J("v106_logs/v106.json")
    assert [s["n_facts"] for s in v["store"]] == [p["n_facts"] for p in v["prompt"]] == v["scales"]
    xs = [min(n, CAP) for n in v["scales"]]
    st, pr = v["store"], v["prompt"]
    assert len({p["n"] for p in pr}) == 1 and pr[0]["n"] == v["config"]["n_eval"]
    per_store = max(x / s["shards"] for s, x in zip(st, xs))
    assert per_store == J("v151_logs/v151.json")["config"]["shard_at"] and all(s["shards"] == max(1, -(-s["n_facts"] // int(per_store))) for s in st)
    for s, x in zip(st, xs):
        line_dots(ax, x, s["acc_seeds"], C["store"], alpha=0.55)
    ax.plot(xs, [p["acc"] for p in pr], color=C["prompt"], marker="^", ms=4.4, lw=1.3, zorder=3)
    ax.plot(xs, [s["acc"] for s in st], color=C["store"], marker="o", ms=4.2, lw=1.6, zorder=4)
    tok = lambda p: "{:,.0f} tokens".format(p["tokens"])
    ax.annotate(tok(pr[0]), (xs[0], pr[0]["acc"]), xytext=(1, -17), textcoords="offset points", fontsize=MIN_PT, color=MUTED, ha="left", va="top")
    ax.annotate(tok(pr[-1]), (xs[-1], pr[-1]["acc"]), xytext=(3, -7), textcoords="offset points", fontsize=MIN_PT, color=MUTED, ha="right", va="top")
    ax.text(xs[2] * 1.1, [p["acc"] for p in pr][2] - 0.07, "%s (%s)" % (NAME["prompt"], v["config"]["model"].split("/")[-1]), fontsize=SMALL,
            color=INK, ha="left", va="top")
    ax.text(xs[0], 0.995, "HRR store, $d$ = %d, at most %d facts per store" % (v["config"]["dim"], per_store),
            fontsize=SMALL, color=INK, ha="left", va="bottom")
    fact_axis_(ax, [8, 64, 128, 512], CAP, pool["label_all"])
    ax.set_ylim(0, 1.08); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_ylabel("exact match"); ax.set_xlabel("pooled facts given to both routes")
    rec("figS2_poolsweep", "a", NAME["prompt"] + " (Qwen3-8B)", colour=C["prompt"], marker="^", x_facts_drawn=xs, x_facts_requested=v["scales"],
        y=[p["acc"] for p in pr], n=[p["n"] for p in pr], prompt_tokens=[p["tokens"] for p in pr],
        source="v106_logs/v106.json prompt[].acc, prompt[].tokens, prompt[].n (single runs, no dots)")
    rec("figS2_poolsweep", "a", "HRR store, d = 2048, at most 64 facts per store", colour=C["store"], marker="o", x_facts_drawn=xs,
        x_facts_requested=v["scales"], y_mean=[s["acc"] for s in st], y_per_seed=[s["acc_seeds"] for s in st], stores=[s["shards"] for s in st],
        facts_per_store=[x / s["shards"] for s, x in zip(st, xs)], codebook_seeds=v["config"]["cb_seeds"],
        n_eval=v["config"]["n_eval"], source="v106_logs/v106.json store[].acc, store[].acc_seeds, store[].shards")
    rec("figS2_poolsweep", "a", "not drawn: LoRA arm", y=[l["acc"] for l in v["lora"]], source="v106_logs/v106.json lora[].acc (not plotted, as briefed)")
    save(fig, "figS2_poolsweep")

# ============================================================================================================ Fig. 2
BENCH_STYLE = {      # colour, marker, line width of the systems of the BABILong panel (styles, not data)
    "ours": (C["ours"], "o", 1.9), "armt": (C["rows"], "s", 1.1), "gpt61sol": (C["gpt"], "^", 1.1),
    "opus55": (C["loss"], "D", 1.1), "gpt4": (C["probe"], "v", 1.5), "llama": (C["open"], "P", 1.5)}
END_LABELS = ("gpt4", "llama")     # leaderboard curves that end inside the panel and are named in the text: labelled at their last point


def fig2_benchmarks():
    """Four panels: BABILong accuracy against length (a); then one panel per public benchmark in one visual language,
    ours against the others at the hardest common setting: BABILong at 128k (b), RULER at 128k (c), CLUTRR (d). Ours in
    vermilion, everything else in greys, frontier models in their Fig. 6 colours."""
    fig = new_fig(180, 122)
    Bj = J("benchmarks/babilong.json")
    series = {s["id"]: s for s in Bj["series"]}
    val = {sid: {p["length"]: p["value"] for p in s["points"]} for sid, s in series.items()}
    LB = {r["model_name"]: r for r in csv.DictReader(open(os.path.join(D, "benchmarks", "babilong_leaderboard_v0_results.csv")))
          if r["task"] == "avg"}
    Sz = J("benchmarks/babilong_size.json")
    OPUS = "#7B4F9E"
    STY = {"ours": (C["ours"], 2.4, 5.0, "o"), "armt": ("#333333", 1.4, 3.6, "s"), "gpt61sol": (C["gpt"], 1.1, 3.2, "^"),
           "opus55": (OPUS, 1.1, 3.2, "D"), "gpt4": ("#A8A8A8", 1.1, 3.0, "v"), "llama": ("#A8A8A8", 1.1, 3.0, "P")}
    # --- a: accuracy against the length of the text -------------------------------------------------------------
    L = Bj["lengths"]
    pos = {l: i for i, l in enumerate(L)}
    ax = axmm(fig, 14, 74, 150, 42)
    for sid in ("gpt4", "llama", "opus55", "gpt61sol", "armt", "ours"):
        s = series[sid]; col, lw, ms, mk = STY[sid]
        xs = [pos[p["length"]] for p in s["points"]]; ys = [p["value"] for p in s["points"]]
        ax.plot(xs, ys, color=col, lw=lw, marker=mk, ms=ms, mec="white", mew=0.5,
                zorder={"ours": 6, "gpt61sol": 7, "opus55": 7}.get(sid, 3))   # frontier lines lie on ours: draw them on top
        rec("fig2_benchmarks", "a", s["label"], colour=col, values={p["length"]: p["value"] for p in s["points"]},
            n={p["length"]: p.get("n") for p in s["points"]}, source={p["length"]: p["source"] for p in s["points"]})
    B23 = J("benchmarks/b23.json")
    nm_pts = [p for p in B23["runs"]["Y_s0"]["points"] if p["length"] in pos]       # 32k is not on this axis
    NOMEM = tint(C["ours"], 0.55)
    ax.plot([pos[p["length"]] for p in nm_pts], [p["value"] for p in nm_pts], color=NOMEM, lw=1.3, ls=(0, (3, 1.6)),
            marker="o", ms=3.4, mfc="white", mec=NOMEM, mew=1.0, zorder=5)
    rec("fig2_benchmarks", "a", "the same 1.7B fine-tuned without the memory (B23, primary arm)", colour=NOMEM,
        values={p["length"]: p["value"] for p in nm_pts}, n={p["length"]: p["n"] for p in nm_pts},
        source={p["length"]: "b23_verdict.txt Y_s0: " + p["line"].strip()[:40] for p in nm_pts})
    p128 = [p for p in nm_pts if p["length"] == "128k"][0]
    tx_, ty_ = pos["64k"] - 0.2, 85.5          # label and a separate leader line (an arrow would enlarge the text's box)
    ax.text(tx_, ty_, "the same 1.7B, fine-tuned\nwithout the memory: %g%%" % p128["value"], fontsize=SMALL, color=INK,
            ha="right", va="center", linespacing=1.05)
    ax.plot([tx_ + 0.04, pos["128k"] - 0.06], [ty_ + 1.5, p128["value"] - 0.7], color=MUTED, lw=0.5, zorder=2)
    wx = pos["128k"] + 0.5
    ax.plot([wx, wx], [57, 91], color=NOMEM, lw=0.7, ls=(0, (1, 1.4)), zorder=2)
    ax.text(wx - 0.08, 61.5, "window of the 1.7B\nwithout the memory\n(131k tokens)", fontsize=MIN_PT, color=MUTED,
            ha="right", va="center", linespacing=1.1, style="italic")
    rec("fig2_benchmarks", "a", "window of the same 1.7B without the memory", tokens=B23["window_tokens"],
        source="b23_verdict.txt 'Beyond the window: 512k, 1M and 10M inputs exceed the 131k-token YaRN window'")
    lab = lambda sid, txt, dx, dy, **kw: ax.annotate(txt, (pos[series[sid]["points"][-1]["length"]], series[sid]["points"][-1]["value"]),
                                                    xytext=(dx, dy), textcoords="offset points", fontsize=SMALL, color=INK,
                                                    ha="left", va="center", **kw)
    lab("ours", "ours, 1.7B reader with the memory: %g%%" % val["ours"]["10M"], 6, 0, fontweight="bold")
    lab("armt", "ARMT (memory transformer, 145M): %g%%" % val["armt"]["10M"], 6, 0)
    lab("gpt4", "GPT-4: %g%%" % val["gpt4"]["128k"], 5, 0); lab("llama", "Llama-3.1-70B: %g%%" % val["llama"]["128k"], 5, 0)
    for sid, nm, ty in (("gpt61sol", "GPT-6.1 Sol*", 87.0), ("opus55", "Claude Opus 5.5*", 75.5)):
        p = series[sid]["points"][-1]
        tx = pos["512k"] - 0.25
        ax.text(tx, ty, nm, fontsize=SMALL, color=INK, ha="right", va="center")
        ax.plot([tx + 0.04, pos[p["length"]] - 0.04], [ty, p["value"] - 0.8], color=MUTED, lw=0.5, zorder=2)
    for j, r in enumerate(Bj["rejected"]):
        x = pos[r["length"]] + (j - (len(Bj["rejected"]) - 1) / 2) * 0.12
        rec("fig2_benchmarks", "a", "rejected input: " + series[r["series"]]["label"], length=r["length"], source=r["source"])
    ax.text(pos["1M"], 64, "1M tokens: input\ntoo long for both\nfrontier models", fontsize=MIN_PT, color=MUTED,
            ha="center", va="center", linespacing=1.1, style="italic")
    ax.set_xlim(-0.25, len(L) + 1.9); ax.set_ylim(30, 104); ax.set_yticks([40, 60, 80, 100])
    ax.set_xticks(range(len(L))); ax.set_xticklabels(L); ax.tick_params(axis="x", length=0)
    ax.spines["bottom"].set_bounds(-0.25, len(L) - 0.75)
    ax.set_ylabel("BABILong accuracy (%)"); ax.set_xlabel("length of the text (tokens; positions not to scale)")
    letter(fig, "a", 1, 121.5)

    # --- b, c: one ranking per benchmark at its hardest common length, the same visual language -----------------
    def ranking(ax, ent, xlabel, rest=None, rest_x=56):
        ax.xaxis.grid(True, linewidth=0.5, color=GRID, zorder=0); ax.set_axisbelow(True)
        for i, (nm, v, col, src, hatch) in enumerate(ent):
            ax.barh(-i, v, 0.7, color=col, linewidth=0, zorder=2, hatch=hatch, edgecolor="white" if hatch else None)
            bold = nm.startswith("ours")
            ax.text(v + 1.5, -i, "%g" % round(v, 1), fontsize=MIN_PT, color=INK, va="center", ha="left",
                    fontweight="bold" if bold else "normal")
        ax.set_yticks([-i for i in range(len(ent))]); ax.set_yticklabels([e[0] for e in ent], fontsize=MIN_PT)
        for t, e in zip(ax.get_yticklabels(), ent):
            if e[0].startswith("ours"):
                t.set_fontweight("bold")
        ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
        ax.set_ylim(-len(ent) + 0.4, 0.6); ax.set_xlim(0, 116); ax.set_xticks([0, 50, 100])
        ax.set_xlabel(xlabel)
        if rest:
            ax.text(rest_x, -len(ent) + 1, rest, fontsize=MIN_PT, color=MUTED, ha="left", va="center", linespacing=1.1)

    ent = [("ours (1.7B)", val["ours"]["128k"], C["ours"], "babilong.json ours 128k", None)]
    for m in Sz["models"]:
        size = ("%gB" % m["params_b"]) if m["params_b"] >= 1 else ("%gM" % round(m["params_b"] * 1000))
        ent.append(("%s (%s%s)" % (m["label"].replace("ChatQA-1.5-8B + RAG", "ChatQA-1.5 + RAG"), size,
                                   ", FT" if m["fine_tuned_on_babilong"] else ""),
                    float(LB[m["row"]]["128k"]), "#5A5A5A" if m["fine_tuned_on_babilong"] else "#B5B5B5",
                    "leaderboard row %r, 128k; size: %s" % (m["row"], m["params_source"]), None))
    for u in Sz["undisclosed"]:
        v = float(LB[u["row"]]["128k"]) if "row" in u else val[u["series"]]["128k"]
        col = {"GPT-4": "#B5B5B5", "GPT-6.1 Sol": C["gpt"], "Claude Opus 5.5": OPUS}[u["label"]]
        ent.append(("%s%s" % (u["label"], "*" if "series" in u else ""), v, col, u["source"], None))
    ent.append(("Qwen3-1.7B alone (FT)", p128["value"], "#5A5A5A", "b23_verdict.txt Y_s0 128k: " + p128["line"].strip()[:40], "////"))
    ent.sort(key=lambda e: -e[1])
    assert ent[0][0].startswith("ours")
    top, rest = ent[:10], ent[10:]
    ax = axmm(fig, 31, 12, 27, 48, grid=False)
    ranking(ax, top, "BABILong, 128k tokens (%)",
            "%d further:\n%g–%g%%" % (len(rest), min(e[1] for e in rest), max(e[1] for e in rest)), 62)
    for i, e in enumerate(top):
        rec("fig2_benchmarks", "b", e[0], acc_128k=e[1], rank=i + 1, source=e[3])
    for e in rest:
        rec("fig2_benchmarks", "b", e[0] + " (not drawn)", acc_128k=e[1], source=e[3])
    letter(fig, "b", 1, 63.5)

    Ru = J("benchmarks/ruler.json")
    # r40 (referee): ours is ranked by its average over the 11 tasks that the reader reads. Two of the 13 tasks count
    # words, which our system does with a word counter, so its 13-task average (94.9) is not set against models that
    # read them; the same Qwen3-8B alone is shown over the same 11 tasks (our run). Leaderboard rows report only the
    # 13-task average (500 samples per task).
    # r44 (referee 6): like for like on the task set -- every row is a 13-task average; ours does the two counting
    # tasks with a word counter (labelled). Leaderboard rows use 500 samples per task, ours 100.
    ours11 = {p["length"]: p["value"] for p in Ru["ours13"]["points"]}
    direct11 = {p["length"]: p["value"] for p in Ru["direct13"]["points"]}
    lbr = sorted(Ru["leaderboard"], key=lambda r: -r["points"][-1]["value"])
    size = lambda r: (" (%gB)" % r["params_b"]) if r["params_b"] else ""
    rent = [("ours (8B; word counter)", ours11["128k"], C["ours"], "b21_verdict.txt VSA store (ours), 13 tasks, 128k", None)]
    keep = [r for r in lbr if r["label"] not in ("GPT-4", "Qwen3-8B", "Llama-3.1-70B")][:7]
    for r in keep + [r for r in lbr if r["label"] in ("GPT-4", "Llama-3.1-70B")]:
        rent.append((r["label"] + size(r), r["points"][-1]["value"], "#B5B5B5", r["points"][-1]["line"][:60], None))
    rent.append(("Qwen3-8B alone (8B)", direct11["128k"], "#B5B5B5",
                 "b21_verdict.txt Qwen3-8B direct (B19), 13 tasks, 128k", "////"))
    rent.sort(key=lambda e: -e[1])
    ax = axmm(fig, 91, 12, 27, 48, grid=False)
    ranking(ax, rent, "RULER, 128k tokens (%)")
    for i, e in enumerate(rent):
        rec("fig2_benchmarks", "c", e[0], avg_128k=e[1], tasks=13, rank=i + 1, source=e[3])
    letter(fig, "c", 61, 63.5)

    # --- d: CLUTRR, all chains and long chains -------------------------------------------------------------------
    Cl = J("benchmarks/clutrr.json")
    by = J(Cl["ours"]["file"])["by_k"]
    for k, v_ in Cl["ours"]["check_even_k"].items():
        assert abs(round(by[k]["correct"] / by[k]["n"], 3) - v_) < 1e-9
    G = J("gains.json")
    ours_all = sum(c["correct"] for c in by.values()) / sum(c["n"] for c in by.values())
    long_ = [k for k in by if int(k) >= 7]
    ours_long = sum(by[k]["correct"] for k in long_) / sum(by[k]["n"] for k in long_)
    dlong = G["clutrr_direct_long"]["value"]
    refs = {r["id"]: r for r in Cl["references"]}
    rows = [("ours: frozen\nQwen3-8B + rules", 100 * ours_all, 100 * ours_long, C["ours"]),
            ("the same model\nalone", 100 * G["clutrr_direct_overall"]["value"], 100 * dlong, "#B5B5B5"),
            ("SatLM (rules given;\nanother test set)", refs["satlm"]["value"], None, "#7A7A7A"),
            ("DSR-LM", refs["dsrlm"]["value"], None, "#7A7A7A")]
    ax = axmm(fig, 149, 12, 25, 48, grid=False)
    ax.xaxis.grid(True, linewidth=0.5, color=GRID, zorder=0); ax.set_axisbelow(True)
    for i, (nm, a_, l_, col) in enumerate(rows):
        y = -i
        # r30: SatLM was tested on a different set of 1,042 stories, so its bar is dotted (referee item 9)
        h_ = "...." if nm.startswith("SatLM") else None
        ax.barh(y + (0.18 if l_ is not None else 0), a_, 0.34 if l_ is not None else 0.5, color=col, linewidth=0, zorder=2,
                hatch=h_, edgecolor="white" if h_ else None)
        ax.text(a_ + 1.5, y + (0.18 if l_ is not None else 0), "%.1f" % a_, fontsize=MIN_PT, color=INK, va="center",
                fontweight="bold" if i == 0 else "normal")
        if l_ is not None:
            ax.barh(y - 0.18, l_, 0.34, color=col, alpha=0.55, linewidth=0, zorder=2, hatch="////", edgecolor="white")
            ax.text(l_ + 1.5, y - 0.18, "%.1f" % l_, fontsize=MIN_PT, color=INK, va="center")
        rec("fig2_benchmarks", "d", nm.replace("\n", " "), all_chains=a_, chains_7_to_10=l_, colour=col)
    ax.set_yticks([-i for i in range(len(rows))]); ax.set_yticklabels([r[0] for r in rows], fontsize=MIN_PT, linespacing=1.05)
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_ylim(-len(rows) + 0.45, 0.55); ax.set_xlim(0, 116); ax.set_xticks([0, 50, 100]); ax.set_xlabel("CLUTRR (%)")
    leg(fig, [Patch(facecolor="#9A9A9A", label="all chains"),
              Patch(facecolor="#9A9A9A", alpha=0.55, hatch="////", edgecolor="white", label="7 to 10 relations")], 124, 63.6,
        ncol=2, columnspacing=0.8, handlelength=1.2)
    letter(fig, "d", 121, 63.5)
    save(fig, "fig2_benchmarks")


def figS4_clutrr_k():
    """Supplementary: CLUTRR accuracy against the number of relations to be chained (was Fig. 2d)."""
    fig = new_fig(124, 64)
    Cl = J("benchmarks/clutrr.json")
    ax = axmm(fig, 16, 13, 70, 46)
    by = J(Cl["ours"]["file"])["by_k"]
    ks = sorted(int(k) for k in by)
    ours = [by[str(k)]["correct"] / by[str(k)]["n"] for k in ks]
    ax.plot(ks, [100 * a for a in ours], color=C["ours"], marker="o", lw=1.9, ms=4.2, zorder=5, mec="white", mew=0.5)
    rec("figS4_clutrr_k", "a", Cl["ours"]["label"], k=ks, accuracy=ours, n=[by[str(k)]["n"] for k in ks],
        source="r1_logs/ours_by_k.json by_k")
    Dj = Cl["direct"]
    direct = {p["k"]: (p["value"], p["source"]) for p in Dj["points"]}
    dk = sorted(direct)
    ax.plot(dk, [100 * direct[k][0] for k in dk], color=C["prompt"], marker="^", lw=0, ms=4.0, zorder=4, mec="white", mew=0.5)
    for k0, k1 in zip(dk[:-1], dk[1:]):
        ax.plot([k0, k1], [100 * direct[k0][0], 100 * direct[k1][0]], color=C["prompt"], lw=1.1, zorder=3,
                ls="-" if k1 - k0 == 1 else (0, (2.5, 1.8)))
    rec("figS4_clutrr_k", "a", Dj["label"], k=dk, accuracy=[direct[k][0] for k in dk], source={str(k): direct[k][1] for k in dk})
    ax.text(5.6, 86, "ours: frozen Qwen3-8B\nwith the rule memory", fontsize=MIN_PT, color=INK, ha="center", va="bottom", linespacing=1.1)
    ax.text(8.0, 46.5, "the same model\nanswering directly", fontsize=MIN_PT, color=INK, ha="center", va="top", linespacing=1.1)
    LS = {"dotted": DOT, "dashdot": DASHDOT, "dashed": DASH}
    LC = {"rule_memory_gold": C["rows"], "satlm": "#555555", "dsrlm": C["probe"]}
    SHORT = {"rule_memory_gold": "gold relations, 97.5", "satlm": "SatLM, 68.3", "dsrlm": "DSR-LM, 61.0"}
    for r in Cl["references"]:
        ax.axhline(r["value"], color=LC[r["id"]], lw=0.9, ls=LS[r["linestyle"]], zorder=2)
        ax.text(10.55, r["value"], SHORT[r["id"]], fontsize=MIN_PT, color=INK, ha="left", va="center", clip_on=False)
    ax.set_xticks([2, 4, 6, 8, 10]); ax.set_xlabel("relations in the chain"); ax.set_ylabel("CLUTRR accuracy (%)")
    ax.set_ylim(35, 102.5); ax.set_xlim(1.6, 10.4); ax.set_yticks([40, 60, 80, 100])
    save(fig, "figS4_clutrr_k")


def figS3_per_task():
    """Supplementary: BABILong per task at one and ten million tokens, ours against ARMT's published numbers (was Fig. 2b)."""
    fig = new_fig(124, 62)
    T = J("benchmarks/babilong_per_task.json")
    ax = axmm(fig, 16, 12, 66, 44, grid=False)
    ax.xaxis.grid(True, linewidth=0.5, color=GRID, zorder=0); ax.set_axisbelow(True)
    tasks = [t["id"] for t in T["tasks"]]
    assert T["lengths"] == ["1M", "10M"]
    off = {"1M": -0.17, "10M": 0.17}
    MK = {"ours": ("o", 4.8, C["ours"], 5), "armt": ("s", 4.2, C["rows"], 4)}
    for i, t in enumerate(tasks):
        for Lk in T["lengths"]:
            y = i + off[Lk]
            vo, va = T["ours"]["values"][Lk][t]["value"], T["armt"]["values"][Lk][t]["value"]
            ax.plot([va, vo], [y, y], color="#BBBBBB", lw=1.0, zorder=1, solid_capstyle="butt")
            for who, v in (("armt", va), ("ours", vo)):
                mk, ms, col, z = MK[who]
                ax.plot([v], [y], marker=mk, ms=ms, ls="none", zorder=z, mfc=col if Lk == "1M" else "white", mec=col, mew=1.1)
    for who in ("ours", "armt"):
        for Lk in T["lengths"]:
            name = "ours" if who == "ours" else "ARMT"
            rec("figS3_per_task", "a", "%s, %s tokens" % (name, Lk), values={t: T[who]["values"][Lk][t]["value"] for t in tasks},
                n={t: T[who]["values"][Lk][t].get("n") for t in tasks}, source={t: T[who]["values"][Lk][t]["source"] for t in tasks})
    lb = {(r["model_name"], r["task"]): r for r in csv.DictReader(open(os.path.join(D, "benchmarks", "babilong_leaderboard_v0_results.csv")))}
    row = T["cross_checks"]["leaderboard"]["row"]
    for Lk in T["lengths"]:
        pv = [T["armt"]["values"][Lk][t]["value"] for t in tasks]
        assert abs(mean(pv) - float(lb[(row, "avg")][Lk])) < 0.051, (Lk, mean(pv))
    ax.set_yticks(range(len(tasks))); ax.set_yticklabels([t["label"] for t in T["tasks"]])
    ax.set_ylim(len(tasks) - 0.45, -0.55); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlim(30, 103); ax.set_xticks([40, 60, 80, 100]); ax.set_xlabel("BABILong accuracy (%)")
    hb = [Line2D([], [], ls="none", marker="o", ms=4.8, mfc=C["ours"], mec=C["ours"], mew=1.1, label="ours, 1M tokens"),
          Line2D([], [], ls="none", marker="o", ms=4.8, mfc="white", mec=C["ours"], mew=1.1, label="ours, 10M tokens"),
          Line2D([], [], ls="none", marker="s", ms=4.2, mfc=C["rows"], mec=C["rows"], mew=1.1, label="ARMT, 1M tokens"),
          Line2D([], [], ls="none", marker="s", ms=4.2, mfc="white", mec=C["rows"], mew=1.1, label="ARMT, 10M tokens")]
    leg(fig, hb, 85, 56, handlelength=1.0, labelspacing=0.4)
    save(fig, "figS3_per_task")


# ============================================================================================================ Fig. 3
def fig6_oplogs():
    fig = new_fig(180, 128)
    O = J("oplogs.json")
    S_ = O["systems"]
    order = ["sor", "qwen", "gpt", "ours"]
    col = {k: C[S_[k]["colour_key"]] for k in order}
    OPUS = "#7B4F9E"
    # one key for every panel, across the top
    hs = [Patch(facecolor=col[k], label=S_[k]["label"]) for k in reversed(order)]
    hs.append(Line2D([], [], color=OPUS, marker="s", ms=3.8, lw=1.2, label="Claude Opus 5.5 (d)"))
    leg(fig, hs, 14, 127, ncol=3, columnspacing=1.8, labelspacing=0.4, handlelength=1.2)
    # --- a: three question sets, four systems ----------------------------------------------------------------
    ax = axmm(fig, 14, 72, 74, 36)
    sets = O["sets"]
    G = [s["id"] for s in sets]
    arms = [(k, S_[k]["label"], col[k], {s["id"]: (s["values"][k]["value"] if s["values"][k] else None) for s in sets}, {}) for k in order]
    pos, w = grouped(ax, G, arms, gap=0.2, dots=False)
    for s in sets:
        for k, v_ in s.get("repaired", {}).items():           # the repaired parse: an open marker above the bar
            ax.plot([pos[(k, s["id"])]], [v_["value"]], marker="o", ms=4.6, mfc="white", mec=col[k], mew=1.2, ls="none", zorder=5)
            rec("fig6_oplogs", "a", "%s, repaired parse (open marker)" % S_[k]["label"], group=s["label"], colour=col[k],
                value=v_["value"], source=v_["source"])
        for k, why in s.get("not_run", {}).items():
            ax.text(pos[(k, s["id"])], 0.02, "not run", rotation=90, ha="center", va="bottom", fontsize=MIN_PT, color=MUTED)
            rec("fig6_oplogs", "a", "%s: not run" % S_[k]["label"], group=s["label"], note=why)
        for k in order:
            if s["values"][k]:
                rec("fig6_oplogs", "a", S_[k]["label"], group=s["label"], n=s["n"]["value"], colour=col[k], value=s["values"][k]["value"],
                    source=s["values"][k]["source"])
        v_o = s["values"]["ours"]["value"]
        ax.text(pos[("ours", s["id"])], v_o + 0.015, "%.3f" % v_o, ha="center", va="bottom", fontsize=MIN_PT, color=INK)
    ax.set_xticklabels(["%s\n($n$ = %d)" % (s["label"], s["n"]["value"]) for s in sets], linespacing=1.3)
    ax.set_xlim(-0.52, len(G) - 0.48)
    ax.set_ylim(0, 1.06); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_ylabel("questions answered exactly")
    letter(fig, "a", 1, 113)

    # --- b: paired outcomes against GPT-6.1 Sol (OpenAI API), as two 2 x 2 tables ----------------------------
    ax = axmm(fig, 104, 70, 73, 40, grid=False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    Pd = O["paired"]
    keys = [s["key"] for s in Pd["segments"]]
    assert keys == ["both_right", "only_ours", "only_gpt", "both_wrong"], keys
    fill = {"both_right": "#E3E3E3", "only_ours": tint(C["ours"], 0.42), "only_gpt": tint(C["gpt"], 0.42), "both_wrong": "#4D4D4D"}
    cw, chh, y0 = 0.19, 0.25, 0.20
    for j, row in enumerate(Pd["rows"]):
        n = row["n"]
        assert sum(row[k] for k in keys) == n
        either = (n - row["both_wrong"]) / n
        assert "%.3f" % either == row["either_right"]["text"], (either, row["either_right"])
        x0 = 0.18 + 0.44 * j
        cells = {("r", "r"): "both_right", ("r", "w"): "only_ours", ("w", "r"): "only_gpt", ("w", "w"): "both_wrong"}
        for (ro, cg), k in cells.items():
            cx = x0 + (0 if cg == "r" else cw); cy = y0 + (chh if ro == "r" else 0)
            ax.add_patch(matplotlib.patches.Rectangle((cx, cy), cw - 0.008, chh - 0.012, facecolor=fill[k], edgecolor="none"))
            ax.text(cx + (cw - 0.008) / 2, cy + (chh - 0.012) / 2, str(row[k]), ha="center", va="center", fontsize=SMALL + 0.5,
                    color="white" if k == "both_wrong" else INK)
        ax.text(x0 + cw - 0.004, y0 + 2 * chh + 0.20, "%s ($n$ = %d)" % (row["label"], n), ha="center", va="bottom", fontsize=SMALL, color=INK)
        for cg, lab in (("r", "right"), ("w", "wrong")):
            ax.text(x0 + (0 if cg == "r" else cw) + (cw - 0.008) / 2, y0 + 2 * chh + 0.02, lab, ha="center", va="bottom", fontsize=MIN_PT, color=MUTED)
        ax.text(x0 + cw - 0.004, y0 + 2 * chh + 0.11, "GPT-6.1 Sol", ha="center", va="bottom", fontsize=MIN_PT, color=MUTED)
        ax.text(x0 + cw - 0.004, y0 - 0.05, "either right: %s" % row["either_right"]["text"], ha="center", va="top", fontsize=SMALL, color=INK)
        rec("fig6_oplogs", "b", row["label"], n=n, **{k: row[k] for k in keys},
            either_right=row["either_right"]["value"], either_right_recomputed=either, source=row["source"],
            route="OpenAI API")
    for ro, lab in (("r", "ours right"), ("w", "ours wrong")):
        ax.text(0.165, y0 + (chh if ro == "r" else 0) + (chh - 0.012) / 2, lab, ha="right", va="center", fontsize=MIN_PT, color=MUTED)
    for k in keys:
        rec("fig6_oplogs", "b", "cell colour: " + k, colour=fill[k])
    letter(fig, "b", 94, 113)

    # --- c: past-window questions by site ----------------------------------------------------------------------
    ax = axmm(fig, 14, 14, 74, 36)
    PS = O["per_site"]["sites"]
    pale = tint(col["qwen"], 0.45)
    arms = [("qwen", S_["qwen"]["label"], pale, {s["id"]: s["values"]["qwen"]["value"] for s in PS}, {}),
            ("gpt", S_["gpt"]["label"], col["gpt"], {s["id"]: s["values"]["gpt"]["value"] for s in PS}, {}),
            ("ours", S_["ours"]["label"], col["ours"], {s["id"]: s["values"]["ours"]["value"] for s in PS}, {})]
    grouped(ax, [s["id"] for s in PS], arms, gap=0.26, dots=False)
    ax.set_xticklabels(["%s\n($n$ = %d)" % (s["label"], s["n"]["value"]) for s in PS], linespacing=1.3)
    ax.set_xlim(-0.52, len(PS) - 0.48)
    ax.set_ylim(0, 1.0); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_ylabel("questions answered exactly")
    assert sum(s["n"]["value"] for s in PS) == [s for s in sets if s["id"] == "window"][0]["n"]["value"]
    for k, lab, c_, d_, _ in arms:
        rec("fig6_oplogs", "c", lab, colour=c_, values=d_, n={s["id"]: s["n"]["value"] for s in PS},
            correct_implied={s["id"]: s["values"][k]["correct_implied"] for s in PS},
            site_labels={s["id"]: s["label"] for s in PS}, source={s["id"]: s["values"][k]["source"] for s in PS})
    letter(fig, "c", 1, 55)

    # --- d: impact questions over records of one, four and eight days -------------------------------------------
    M = J("multiday.json")
    ax = axmm(fig, 114, 14, 63, 36)
    dd = M["days"]
    ax.plot(dd, M["pipeline"]["set_exact"], color=col["ours"], marker="o", ms=4.2, lw=1.4, zorder=4)
    if os.path.exists(M["gpt_direct"]["file"]):         # the drawn GPT-6.1 Sol line is the verdict's row, verbatim
        assert M["gpt_direct"]["line"] in open(M["gpt_direct"]["file"]).read().splitlines(), "multiday gpt_direct: source line not found"
        assert M["gpt_direct"]["line"].split("|")[2:5] and all(("%.3f" % v) in c for v, c in zip(M["gpt_direct"]["set_exact"], M["gpt_direct"]["line"].split("|")[2:5]))
    ax.plot(dd, M["gpt_direct"]["set_exact"], color=col["gpt"], marker="o", ms=4.2, lw=1.2, zorder=3)
    ax.plot(dd, M["opus_normalised"]["set_exact"], color=OPUS, marker="s", ms=3.8, lw=1.2, zorder=3)
    i8 = dd.index(8)                                     # direct labels at the line ends (the lines cross no label)
    for k_, nm_ in (("gpt_direct", "GPT-6.1 Sol"), ("opus_normalised", "Claude Opus 5.5"), ("pipeline", "ours")):
        ax.annotate(nm_, (8, M[k_]["set_exact"][i8]), xytext=(5, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=MIN_PT, color=INK)
    for k_, lab_ in (("pipeline", "ours"), ("gpt_direct", "GPT-6.1 Sol, OpenAI API"),
                     ("gpt_registered", "GPT-6.1 Sol, reseller gateway, registered parse (not drawn)"),
                     ("gpt_repaired", "GPT-6.1 Sol, reseller gateway, repaired parse (not drawn)"), ("opus_normalised", "Claude Opus 5.5, item names normalized"),
                     ("opus_registered", "Claude Opus 5.5, registered parse (not drawn)")):
        rec("fig6_oplogs", "d", lab_, days=dd, n=M["n"], set_exact=M[k_]["set_exact"], micro_f1=M[k_]["micro_f1"], source=M[k_]["source"])
    ax.set_xticks(dd); ax.set_xticklabels(["%d\n(%.0fk)" % (d_, tk) for d_, tk in zip(dd, M["tokens_k"])], linespacing=1.2)
    ax.set_xlim(0.3, 12.6); ax.set_ylim(0.5, 1.0); ax.set_yticks([0.5, 0.625, 0.75, 0.875, 1.0])
    ax.spines["bottom"].set_bounds(0.3, 8.7)
    ax.set_xlabel("days of record (tokens per prompt)"); ax.set_ylabel("impact questions\nanswered exactly", linespacing=1.2)
    ax.text(12.5, 0.52, "$n$ = %d, %d, %d" % tuple(M["n"]), ha="right", va="bottom", fontsize=MIN_PT, color=MUTED)
    letter(fig, "d", 101, 55)
    save(fig, "fig6_oplogs")

# ============================================================================================================ Fig. 5
def fig5_conditions():
    """What the memory adds where a language model must read and combine the evidence itself (a), and, against a
    fine-tune of the same model on short tasks, when a fine-tune is enough (b)."""
    fig = new_fig(130, 100)
    tr = matplotlib.transforms.blended_transform_factory
    ALONE = "#6E6E6E"
    # --- b: against a fine-tune of the same model on short tasks ----------------------------------------------------
    Cd = J("conditions.json")
    rows = {r["label"]: r for g in Cd["groups"] for r in g["rows"]}
    B31 = J("b31.json")                     # arenas: naming head (order-keeping binding, every symbol named) vs the equalised LoRA
    for lab in [l for l in rows if l.startswith("arena")]:
        del rows[lab]
    for g, enc in [(g, e) for g in "CAB" for e in ("Phi-2", "Qwen3-8B")]:
        v = lambda k: [sd["value"] for sd in B31["arms"]["%s|%s|%s" % (enc, g, k)]["seeds"]]
        rows["arena %s, %s" % (g, enc)] = {"memory": v("naming head + VSA (perm, named)"), "lora": v("equalised LoRA"),
                                          "source": "b31.json arms '%s|%s|naming head + VSA (perm, named)' and '|equalised LoRA'" % (enc, g)}
    B44 = J("b44.json")                     # arena C, Qwen3-8B, n = 800: answer-only LoRA and the scratchpad LoRA (told the bridge symbol)
    for var, lab in (("std", "arena C, Qwen3-8B"), ("ho", "arena C, Qwen3-8B, pairs held out")):
        rows[lab] = {"memory": [x["value"] for x in B44[var]["memory"]["seeds"]], "lora": [x["value"] for x in B44[var]["lora"]["seeds"]],
                     "scratch": [x["value"] for x in B44[var]["scratch"]["seeds"]],
                     "source": "b44.json %s (n = 800): naming head, equalised LoRA, scratchpad LoRA" % var}
    rows["arena C, Phi-2"]["scratch"] = [x["value"] for x in J("b47_fig2bc.json")["phi2_scratch_n800"]["seeds"]]
    rows["arena C, Phi-2"]["source"] += "; scratchpad LoRA from b47_fig2bc.json phi2_scratch_n800 (b47_verdict.txt)"
    B28 = J("b28.json")                     # order-keeping binding (bAbI 4, 15) and bAbI task 2 on all 500 questions
    for lab, task in (("bAbI task 4", "bAbI task 4"), ("bAbI task 15", "bAbI task 15")):
        r = dict(rows[lab])
        r["memory_first"] = r["memory"]                    # the commutative binding that we used first
        r["memory"] = [s["value"] for s in B28["enc"][task + "|naming head|perm"]["seeds"]]
        r["source"] = r["source"] + "; memory now b28.json naming head, permuted binding (memory_first = commutative)"
        rows[lab] = r
    B33 = J("b33.json")                     # bAbI task 2, corrected parse, 1,000 fresh questions (registered primary set)
    r = dict(rows["bAbI task 2 (two hops)"])
    r["memory"] = [s["value"] for s in B33["fresh"]["memory"]["seeds"]]
    r["lora"] = [s["value"] for s in B33["fresh"]["ft_best"]["seeds"]]
    r.pop("lora_first", None)                # r44: first-configuration markers dropped (process history; Methods, SI)
    r["source"] = "b33.json fresh (1,000 questions no part of the paper had scored): memory, ft_best"
    rows["bAbI task 2 (two hops)"] = r
    PLAIN = {"arena C, Phi-2": "arena C (two steps), Phi-2", "arena C, Qwen3-8B": "arena C (two steps), Qwen3-8B",
             "arena C, Qwen3-8B, pairs held out": "arena C, pairs held out, Qwen3-8B",
             "arena A, Phi-2": "arena A (three steps), Phi-2", "arena A, Qwen3-8B": "arena A (three steps), Qwen3-8B",
             "arena B, Phi-2": "arena B (one step), Phi-2", "arena B, Qwen3-8B": "arena B (one step), Qwen3-8B",
             "bAbI task 1 (one hop)": "bAbI 1: where is a person?",
             "bAbI task 2 (two hops)": "bAbI 2: where is a carried object?",
             "bAbI task 4": "bAbI 4: relations with a direction",
             "bAbI task 15": "bAbI 15: deduction from a rule"}
    GROUPS = [("answer-only fine-tune fails; diamond: told the intermediate symbol", ["arena C, Phi-2", "arena C, Qwen3-8B",
               "arena C, Qwen3-8B, pairs held out", "arena A, Phi-2"]),
              ("the fine-tune learns the task: the memory adds little or trails",
               ["arena A, Qwen3-8B", "arena B, Phi-2", "arena B, Qwen3-8B",
                "bAbI task 1 (one hop)", "bAbI task 2 (two hops)", "bAbI task 4", "bAbI task 15"])]
    assert sorted(l for _, ls in GROUPS for l in ls) == sorted(rows) == sorted(PLAIN)
    ax = axmm(fig, 85, 12, 32, 78, grid=False)
    ax.xaxis.grid(True, linewidth=0.5, color=GRID, zorder=0); ax.set_axisbelow(True)
    y, yt, ylab, gtops = 0.0, [], [], []
    for glab, labs in GROUPS:
        gtops.append((y, glab))
        y -= 0.75
        for lab in labs:
            r = rows[lab]
            m, l_ = r["memory"], r["lora"]
            mm, ml = mean(m), mean(l_)
            nofit = lab in ("arena A, Phi-2", "arena C, Phi-2")      # r55: answer-only fine-tune did not fit its training split
            ax.plot([ml, mm], [y, y], color="#BBBBBB", lw=1.4, zorder=1, solid_capstyle="butt")
            if nofit:
                ax.plot([ml], [y], marker="s", ms=4.6, mfc="white", mec=C["lora"], mew=1.2, ls="none", zorder=4)
            else:
                ax.plot([ml], [y], marker="o", ms=5.0, color=C["lora"], ls="none", zorder=4)
            ax.plot([mm], [y], marker="o", ms=5.0, color=C["handover"], ls="none", zorder=5)
            if len(l_) > 1:
                ax.scatter(l_, [y - 0.24] * len(l_), s=4, color=C["lora"], alpha=0.85, linewidths=0, zorder=3)
            if len(m) > 1:
                ax.scatter(m, [y + 0.24] * len(m), s=4, color=C["handover"], alpha=0.85, linewidths=0, zorder=3)
            if r.get("memory_first"):                # bAbI 4, 15: the commutative binding that we used first
                ax.plot([mean(r["memory_first"])], [y], marker="o", ms=5.0, mfc="white", mec=C["handover"], mew=1.2, ls="none", zorder=4)
                ax.plot([mean(r["memory_first"]), mm], [y, y], color="#BBBBBB", lw=1.0, ls=(0, (2, 1.5)), zorder=1)
            if r.get("lora_first"):                  # first configuration (bAbI 2: before the sweep selected on the official test set)
                ax.plot([mean(r["lora_first"])], [y], marker="o", ms=5.0, mfc="white", mec=C["lora"], mew=1.2, ls="none", zorder=4)
                ax.plot([mean(r["lora_first"]), ml], [y, y], color="#BBBBBB", lw=1.0, ls=(0, (2, 1.5)), zorder=1)
            if r.get("scratch"):                     # B44: the fine-tune told the bridge symbol (diamond)
                ms_ = mean(r["scratch"])
                ax.plot([ms_], [y], marker="D", ms=4.6, mfc="white", mec=C["lora"], mew=1.2, ls="none", zorder=6)
                ml = max(ml, ms_)
            d = mm - ml
            nocomp = nofit and not r.get("scratch")          # no fitted or step-supervised fine-tune: no margin is shown
            ax.text(1.06, y, "n.c." if nocomp else "%+.2f" % d, transform=tr(ax.transAxes, ax.transData), ha="left", va="center",
                    fontsize=SMALL, color=MUTED if nocomp else INK, fontweight="normal")
            if nocomp:
                d = None
            yt.append(y); ylab.append(PLAIN[lab])
            rec("fig5_conditions", "b", lab, memory=m, lora=l_, scratchpad=r.get("scratch"), difference_vs_best_finetune=d, group=glab, plain_label=PLAIN[lab],
                lora_first_open_marker=r.get("lora_first"), memory_first_open_marker=r.get("memory_first"), not_drawn={"prompt": r.get("prompt")}, source=r["source"])
            y -= 1.0
        y -= 0.2
    ax.set_yticks(yt); ax.set_yticklabels(ylab, fontsize=MIN_PT)
    ax.tick_params(axis="y", length=0)
    for (yg, lab) in gtops:
        ax.text(-1.36, yg - 0.2, lab, transform=tr(ax.transAxes, ax.transData), ha="left", va="center", fontsize=MIN_PT,
                fontweight="bold", color=INK)
    ax.set_ylim(y + 0.75, 0.3)
    ax.set_xlim(0.45, 1.03); ax.set_xticks([0.5, 0.75, 1.0]); ax.set_xlabel("accuracy")
    ax.spines["left"].set_visible(False)
    hb = [Line2D([], [], marker="o", ms=5.0, color=C["handover"], ls="none", label="memory"),
          Line2D([], [], marker="o", ms=5.0, color=C["lora"], ls="none", label="LoRA fine-tune, answer only (tuned)"),
          Line2D([], [], marker="D", ms=4.6, mfc="white", mec=C["lora"], mew=1.2, ls="none", label="LoRA told the intermediate symbol"),
          Line2D([], [], marker="o", ms=5.0, mfc="white", mec=C["handover"], mew=1.2, ls="none", label="memory, commutative"),
          Line2D([], [], marker="s", ms=4.6, mfc="white", mec=C["lora"], mew=1.2, ls="none", label="answer-only fine-tune that did not fit its training split")]
    leg(fig, hb, 4, 99.6, ncol=2, columnspacing=0.7, handlelength=0.8, labelspacing=0.25)
    save(fig, "fig5_conditions")


def figS6_gains():
    """Supplementary: the same language model without and with an exact store on the public benchmarks."""
    fig = new_fig(120, 100)
    tr = matplotlib.transforms.blended_transform_factory
    ALONE = "#6E6E6E"
    # --- a: the language model alone against the same task with the memory ----------------------------------------
    Bj = {s["id"]: {p["length"]: p["value"] for p in s["points"]} for s in J("benchmarks/babilong.json")["series"]}
    by = J("r1_logs/ours_by_k.json")
    G = J("gains.json")
    Ru = J("benchmarks/ruler.json")
    ru = lambda key: {p["length"]: p["value"] for p in Ru[key]["points"]}["128k"] / 100
    A_ROWS = [
        ("BABILong, 128k, same model", "1.7B fine-tuned without vs with memory",
         [p for p in J("benchmarks/b23.json")["runs"]["Y_s0"]["points"] if p["length"] == "128k"][0]["value"] / 100,
         Bj["ours"]["128k"] / 100, "benchmarks/b23.json runs Y_s0 128k (b23_verdict.txt; n = 500) and babilong.json ours 128k"),
        ("RULER, 128k, memory tasks", "Qwen3-8B vs the same + vector store", ru("directNV"), ru("oursNV"),
         "benchmarks/ruler.json directNV and oursNV at 128k: the 9 tasks that use the memory (b21_verdict.txt; n = 900)"),
        ("RULER, 128k: facts in the prompt", "8B + facts in prompt vs + VSA store",
         J("benchmarks/b24.json")["at_128k"]["prompt"]["value"] / 100, J("benchmarks/b24.json")["at_128k"]["store"]["value"] / 100,
         "benchmarks/b24.json at_128k prompt and store (b24_verdict.txt; n = 900)"),
        ("RULER, 128k: retrieval (BM25)", "8B + top-20 pieces vs + VSA store",
         J("benchmarks/b24.json")["at_128k"]["bm25"]["value"] / 100, J("benchmarks/b24.json")["at_128k"]["store"]["value"] / 100,
         "benchmarks/b24.json at_128k bm25 and store (b24_verdict.txt; n = 900)"),
        ("CLUTRR, all chain lengths", "Qwen3-8B vs the same + rule memory", G["clutrr_direct_overall"]["value"],
         by["correct"] / by["n_rows"], "gains.json clutrr_direct_overall (r1_verdict.txt); r1_logs/ours_by_k.json correct / n_rows")]
    assert by["n_rows"] == 1146
    ax = axmm(fig, 48, 12, 33, 78, grid=False)
    ax.xaxis.grid(True, linewidth=0.5, color=GRID, zorder=0); ax.set_axisbelow(True)
    ys = []
    for i, (lab, comp, lo, hi, src) in enumerate(A_ROWS):
        y = -1.25 * i
        ys.append(y)
        ax.plot([lo, hi], [y, y], color="#BBBBBB", lw=1.6, zorder=1, solid_capstyle="butt")
        ax.plot([lo], [y], marker="o", ms=5.6, mfc="white", mec=ALONE, mew=1.3, ls="none", zorder=4)
        ax.plot([hi], [y], marker="o", ms=5.8, color=C["handover"], ls="none", zorder=5)
        ax.text(1.06, y, "%+.2f" % (hi - lo), transform=tr(ax.transAxes, ax.transData), ha="left", va="center", fontsize=SMALL,
                fontweight="bold", color=INK)
        ax.text(-0.04, y + 0.17, lab, transform=tr(ax.transAxes, ax.transData), ha="right", va="center", fontsize=SMALL, color=INK)
        ax.text(-0.04, y - 0.25, comp, transform=tr(ax.transAxes, ax.transData), ha="right", va="center", fontsize=MIN_PT, color=MUTED)
        rec("figS6_gains", "a", lab, comparator=comp, alone=lo, with_memory=hi, gain=hi - lo, source=src)
    ax.text(1.06, 0.75, "gain", transform=tr(ax.transAxes, ax.transData), ha="left", va="center", fontsize=MIN_PT, color=MUTED)
    ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_ylim(ys[-1] - 0.8, 1.05)
    ax.set_xlim(0.15, 1.03); ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_xlabel("accuracy")
    ha = [Line2D([], [], marker="o", ms=5.6, mfc="white", mec=ALONE, mew=1.3, ls="none", label="without the memory (per row)"),
          Line2D([], [], marker="o", ms=5.8, color=C["handover"], ls="none", label="with an exact store")]
    leg(fig, ha, 8, 99.2, ncol=2, columnspacing=1.4, handlelength=0.8)
    

    save(fig, "figS6_gains")


def fig_compression():
    """Main figure (r79l, user 11-25 'not intuitive'): per encoder, the accuracies themselves (a) -- Hebbian VSA store
    (final design), corrected least squares, best of seven classical memories, exact index -- with an arrow from the
    store to least squares, and least squares' gain with its 95% interval (b). Same sources as before (B49/B60/B64)."""
    fig = new_fig(180, 74)
    E = J("b49_compression.json")["encoders"]
    EF = J("final_compression.json")["encoders"]
    E60 = J("b60_compression.json")["encoders"]
    assert [r["name"] for r in E] == [r["name"] for r in EF] == [r["name"] for r in E60]
    rows = []
    for r, rf, r6 in zip(E, EF, E60):
        st = rf["acc_final"]["value"]
        # least squares and best of seven as absolute means: store minus the paired region-mean difference
        # (same 12 conditions); cross-check against the first design's differences in B60 / B64.
        ls = st - rf["final_lamc"]["value"]
        b7 = st - rf["final_best7"]["value"]
        assert abs((r6["acc_vsa_b60"]["value"] - r6["lamc"]["value"]) - ls) < 0.06, r["name"]
        assert abs((r["acc_vsa"]["value"] - J("b64_compression.json")["encoders"][E.index(r)]["best7"]["value"]) - b7) < 0.1, r["name"]
        g = rf["final_lamc"]
        rows.append(dict(name=r["name"], kind=r["kind"], store=st, ls=ls, best7=b7, exact=r["acc_exact"]["value"],
                         gain=-g["value"], glo=-g["hi"], ghi=-g["lo"]))
    ys = [-(k + (0.6 if q["kind"] == "sent" else 0.0)) for k, q in enumerate(rows)]
    STORE, LS, B7, EX = C["handover"], INK, C["sor"], INK

    # --- a: accuracies -------------------------------------------------------------------------------------------
    ax = axmm(fig, 30, 12, 98, 50, grid=False)
    ax.xaxis.grid(True, linewidth=0.5, color=GRID, zorder=0); ax.set_axisbelow(True)
    for y, q in zip(ys, rows):
        ax.annotate("", xy=(q["ls"] - 0.45, y), xytext=(q["store"] + 0.45, y),
                    arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.9, mutation_scale=7, shrinkA=0, shrinkB=0), zorder=3)
        ax.plot([q["best7"]], [y + 0.28], ls="none", marker="o", ms=3.2, color=B7, zorder=4)
        ax.plot([q["store"]], [y], ls="none", marker="o", ms=4.6, color=STORE, zorder=5)
        ax.plot([q["ls"]], [y], ls="none", marker="D", ms=3.9, color=LS, zorder=5)
        ax.plot([q["exact"]], [y], ls="none", marker="^", ms=4.2, mfc="white", mec=EX, mew=0.8, zorder=5)
    for key, lab, col in (("store", "Hebbian VSA store (final design)", STORE), ("ls", "least-squares memory (corrected)", LS),
                          ("best7", "best of seven classical memories", B7), ("exact", "exact index, unlimited bits", EX)):
        rec("fig_compression", "a", lab, colour=col, values={q["name"]: round(q[key], 2) for q in rows},
            source={"store": "figdata/final_compression.json acc_final (vsassm results/b60_part_<enc>.json cells, STORE, mean of 12)",
                    "ls": "acc_final minus final_lamc (vsassm results/b60_verdict.txt 'PRIMARY (ideal): STORE - LAMc', R1); checked against acc_vsa_b60 minus lamc",
                    "best7": "acc_final minus final_best7 (vsassm results/b64_verdict.txt 'VSA - BEST7 (ideal)', R1); checked against acc_vsa minus best7",
                    "exact": "figdata/b49_compression.json (b49_summary_tables.txt 'R1 mean accuracy')"}[key])
    ax.set_yticks(ys); ax.set_yticklabels([q["name"] for q in rows], fontsize=MIN_PT); ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(40, 90); ax.set_xticks([40, 50, 60, 70, 80, 90])
    ax.set_xlabel("mean accuracy at 4 to 16 bits per fact (%)")
    ax.set_ylim(ys[-1] - 0.7, ys[0] + 0.75)
    # group headers sit above each group's row names (user 11-28: on the right they read as labels of the triangles)
    for yy, lab in ((ys[0] + 0.62, "language models"), (ys[4] + 0.62, "sentence encoders")):
        ax.text(-0.012, yy, lab, transform=ax.get_yaxis_transform(), fontsize=SMALL, color=MUTED, fontweight="bold",
                ha="right", va="center")
    hb = [Line2D([], [], color=STORE, marker="o", ms=4.6, lw=0, label="Hebbian VSA store"),
          Line2D([], [], color=LS, marker="D", ms=3.9, lw=0, label="least squares"),
          Line2D([], [], color=B7, marker="o", ms=3.2, lw=0, label="best of seven classical memories"),
          Line2D([], [], color=EX, marker="^", ms=4.2, mfc="white", mew=0.8, lw=0, label="exact index (unlimited bits)")]
    leg(fig, hb, 30, 72.5, ncol=4, columnspacing=1.4, handlelength=0.9, handletextpad=0.4)
    letter(fig, "a", 1, 66.5)

    # --- b: least squares' gain -----------------------------------------------------------------------------------
    ax = axmm(fig, 138, 12, 38, 50, grid=False)
    ax.xaxis.grid(True, linewidth=0.5, color=GRID, zorder=0); ax.set_axisbelow(True)
    for y, q in zip(ys, rows):
        ax.barh([y], [q["gain"]], height=0.5, color=NEUTRAL, zorder=3, linewidth=0)
        ax.plot([q["glo"], q["ghi"]], [y, y], color=INK, lw=1.1, zorder=4, solid_capstyle="butt")
        ax.text(q["ghi"] + 0.25, y, "+%.1f" % q["gain"], fontsize=MIN_PT, color=INK, va="center", ha="left")
    rec("fig_compression", "b", "least squares minus Hebbian VSA store (final design), points, 95% interval",
        values={q["name"]: [q["gain"], q["glo"], q["ghi"]] for q in rows},
        source="figdata/final_compression.json final_lamc, sign flipped (vsassm results/b60_verdict.txt 'PRIMARY (ideal): STORE - LAMc', R1)")
    ax.set_yticks(ys); ax.set_yticklabels([]); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.axvline(0, color=MUTED, lw=0.8, zorder=2)
    ax.set_xlim(0, 9.5); ax.set_xticks([0, 2, 4, 6, 8]); ax.set_ylim(ys[-1] - 0.7, ys[0] + 0.75)
    ax.set_xlabel("gain from least squares (points)")
    letter(fig, "b", 132, 66.5)
    save(fig, "fig_compression")


# ============================================================================================================ Fig. 7
def fig7_reasoning():
    """Reasoning over the stored state: one traced answer (a), read against true state (b), the language model given
    the read state (c), and selective accuracy from the learned reliability (d)."""
    from matplotlib.patches import FancyBboxPatch
    fig = new_fig(180, 128)
    Rz = J("reasoning.json")
    W_, H_ = fig._mm
    T = Rz["trace"]
    boxes = []
    def node(x, y, s, fc="#F4F4F4", ec="#999999", w=None):
        w = w or (3.0 + 1.55 * len(s))
        p = FancyBboxPatch(((x - w / 2) / W_, (y - 2.6) / H_), w / W_, 5.2 / H_, boxstyle="round,pad=0,rounding_size=0.006",
                           transform=fig.transFigure, fc=fc, ec=ec, lw=0.7)
        fig.patches.append(p)
        t = fig.text(x / W_, y / H_, s, ha="center", va="center", fontsize=SMALL, color=INK)
        boxes.append((p, t)); return w
    def edge(x0, x1, y0, y1, col=INK):
        fig.patches.append(matplotlib.patches.FancyArrowPatch((x0 / W_, y0 / H_), (x1 / W_, y1 / H_), transform=fig.transFigure,
                           arrowstyle="-|>", mutation_scale=6.5, lw=0.8, color=col, shrinkA=0, shrinkB=0))
    def cite_(x, y, s, col=MUTED, style="normal", ha="center"):
        fig.text(x / W_, y / H_, s, ha=ha, va="center", fontsize=MIN_PT, color=col, fontstyle=style)
    # question and answer
    fig.text(4 / W_, 121 / H_, "Question:", fontsize=SMALL, fontweight="bold", color=INK, va="center")
    fig.text(19 / W_, 121 / H_, T["question"], fontsize=SMALL, color=INK, va="center")
    items = sorted(T["hops"])
    hop = T["hops"]
    ya, yb = 108.0, 90.0                       # the two specimen chains
    xs = [26, 76, 118, 160]
    for it, yy in zip(items, (ya, yb)):
        node(xs[0], yy, it, fc="#FBE3D5", ec=C["handover"])
    yc = (ya + yb) / 2
    node(xs[1], yc, hop[items[0]][1]["x"]); node(xs[2], yc, hop[items[0]][2]["x"]); node(xs[3], yc, hop[items[0]][2]["y"])
    assert hop[items[0]][0]["y"] == hop[items[1]][0]["y"] == hop[items[0]][1]["x"]
    for it, yy in zip(items, (ya, yb)):
        h0 = hop[it][0]
        edge(xs[0] + 9.5, xs[1] - 7.0, yy, yc + (1.4 if yy > yc else -1.4))
        if h0["kind"] == "human":
            cite_(48, yy - (4.6 if yy < yc else -4.6) + (0 if yy > yc else 0), "line %d, %s chat, %s: “%s”" % (
                h0["line"], h0["src"], h0["who"], h0["text"]), col=C["handover"], style="italic")
        else:
            cite_(48, yy + 4.4, "line %d, %s %s" % (h0["line"], h0["src"].upper(), h0["text"].split("|")[2].strip()), )
        rec("fig7_reasoning", "a", it, hops=[(h["x"], h["y"], h["line"], h["kind"], h["src"]) for h in hop[it]],
            source=T["source"])
    h1, h2 = hop[items[0]][1], hop[items[0]][2]
    edge(xs[1] + 7.5, xs[2] - 5.5, yc, yc); cite_((xs[1] + xs[2]) / 2, yc + 4.4, "line %d, %s %s" % (h1["line"], h1["src"].upper(), h1["text"].split("|")[2].strip()))
    edge(xs[2] + 5.5, xs[3] - 7.5, yc, yc); cite_((xs[2] + xs[3]) / 2, yc + 4.4, "line %d, %s %s" % (h2["line"], h2["src"].upper(), h2["text"].split("|")[2].strip()))
    ok = sorted(T["pred"]) == sorted(T["gold"])
    extra = sorted(set(T["system_of_record_answer"]) - set(T["gold"])); miss = sorted(set(T["gold"]) - set(T["system_of_record_answer"]))
    fig.text(4 / W_, 79.0 / H_, "Answer of the memory: %s and %s, %s; every hop cites the log line from which it was read." % (
        items[0], items[1], "exactly the true set" if ok else "not exact"), fontsize=SMALL, color=INK, va="center")
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}
    fig.text(4 / W_, 74.0 / H_, ("System of record: %s is still logged at the %s, so it misses that specimen and lists %s others"
             " that are no longer in %s.") % (miss[0], T["sor_place"][miss[0]], words.get(len(extra), str(len(extra))), T["target"]),
             fontsize=SMALL, color=MUTED, va="center")
    rec("fig7_reasoning", "a", "system of record", answer=T["system_of_record_answer"], missing=miss, extra=extra, source=T["source"])
    letter(fig, "a", 1, 127)

    # --- b: read state against true state ---------------------------------------------------------------------
    ax = axmm(fig, 13, 13, 46, 44)
    RT = Rz["read_true"]
    xs_ = np.arange(len(RT)); w = 0.36
    for i, r in enumerate(RT):
        ax.bar(i - w / 2, r["read"]["value"], w * 0.92, color=C["handover"], zorder=2, linewidth=0)
        ax.bar(i + w / 2, r["true"]["value"], w * 0.92, color=C["rows"], zorder=2, linewidth=0)
        rec("fig7_reasoning", "b", r["label"], read=r["read"]["value"], true=r["true"]["value"],
            source=[r["read"]["source"], r["true"]["source"]])
    ax.set_xticks(xs_); ax.set_xticklabels(["CLUTRR", "impact", "past\nwindow"], linespacing=1.2); ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 1.0); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_ylabel("accuracy")
    leg(fig, [Patch(facecolor=C["handover"], label="reader's state"), Patch(facecolor=C["rows"], label="true state")], 14, 70.5,
        ncol=1, handlelength=1.0)
    letter(fig, "b", 1, 70.5)

    # --- c: the language model given the state that the memory read ------------------------------------------------
    ax = axmm(fig, 75, 13, 46, 44)
    Ls = Rz["llm_state"]
    vals = [("Qwen3-32B\nover the log", Ls["log"]["value"], C["open"]), ("Qwen3-32B\nover the\nread state", Ls["state"]["value"], C["open"]),
            ("ours", Ls["ours"]["value"], C["ours"])]
    for i, (lab, v, c_) in enumerate(vals):
        ax.bar(i, v, 0.62, color=c_, zorder=2, linewidth=0, hatch="////" if i == 1 else None, edgecolor="white" if i == 1 else None)
        ax.text(i, v + 0.02, "%.3f" % v, ha="center", va="bottom", fontsize=SMALL, color=INK)
        rec("fig7_reasoning", "c", lab.replace("\n", " "), value=v, colour=c_)
    ax.set_xticks(range(3)); ax.set_xticklabels([v[0] for v in vals], linespacing=1.2); ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 1.0); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_ylabel("impact questions answered exactly")
    ax.set_xlim(-0.55, 2.55)
    letter(fig, "c", 62, 70.5)

    # --- d: selective accuracy from the learned reliability ------------------------------------------------------
    ax = axmm(fig, 137, 13, 40, 44)
    Sv = Rz["selective"]
    cov = np.array(Sv["coverage"]) * 100; acc = np.array(Sv["accuracy"])
    keep = cov >= 30 - 1e-9
    ax.plot(cov[keep], acc[keep], color=C["ours"], lw=1.3, marker="o", ms=2.6, zorder=3)
    ax.axhline(Sv["all_answers"], color=MUTED, lw=0.8, ls=DASH, zorder=2)
    ax.text(31, Sv["all_answers"] - 0.004, "all answers, %.3f" % Sv["all_answers"], fontsize=MIN_PT, color=MUTED, va="top")
    i70 = int(np.argmin(abs(cov - 70)))
    ax.annotate("%.3f at 70%%" % acc[i70], (cov[i70], acc[i70]), xytext=(cov[i70] - 22, acc[i70] + 0.017), fontsize=MIN_PT,
                color=INK, arrowprops=dict(arrowstyle="-", lw=0.6, color=INK))
    ax.set_xlim(28, 102); ax.set_xticks([40, 60, 80, 100])
    ax.set_ylim(0.90, 1.0); ax.set_yticks([0.90, 0.925, 0.95, 0.975, 1.0])
    ax.set_xlabel("answers kept, most reliable\nfirst (%)", linespacing=1.2); ax.set_ylabel("accuracy of the kept answers")
    rec("fig7_reasoning", "d", "selective accuracy, min reliability", coverage=Sv["coverage"], accuracy=Sv["accuracy"],
        n=Sv["n"], wrong=Sv["wrong"], auroc=Sv["auroc"], source=Sv["source"])
    letter(fig, "d", 124, 70.5)
    save(fig, "fig7_reasoning", boxes)


# ================================================================================================ Supplementary Fig. 1
def kstar(ks, accs_, thr):
    """The crossing of the criterion, by linear interpolation in log k between the two grid points that bracket it."""
    if accs_[0] < thr or accs_[-1] >= thr:
        return None
    for i in range(len(ks) - 1):
        a0, a1 = accs_[i], accs_[i + 1]
        if a0 >= thr > a1:
            f = (a0 - thr) / (a0 - a1)
            return float(np.exp(np.log(ks[i]) + f * (np.log(ks[i + 1]) - np.log(ks[i]))))
    return None


def capacity_fits():
    """k* per model and dimension, the exponent of k* ~ d^alpha and its bootstrap interval, as the frozen scorer of
    V-151 computes them (same seed, same order of models, so the intervals reproduce the ledger's)."""
    K = J("constants.json")["capacity_criterion"]
    v = J("v151_logs/v151.json")
    dims, ks, models = v["config"]["dims"], v["config"]["ks"], v["config"]["models"]
    rng = np.random.default_rng(K["bootstrap_seed"])
    out = {}
    for m in models:
        kst = [kstar(ks, [v["capacity"][m][str(d)][str(k)] for k in ks], K["threshold"]) for d in dims]
        assert all(k is not None for k in kst), "censored cell in %s" % m
        x, y = np.log(dims), np.log(kst)
        a, c = np.polyfit(x, y, 1)
        draws = []
        for _ in range(K["bootstrap_draws"]):
            idx = rng.integers(0, len(x), len(x))
            if len(set(x[idx])) < 2:
                continue
            draws.append(float(np.polyfit(x[idx], y[idx], 1)[0]))
        out[m] = {"kstar": kst, "alpha": float(a), "c": float(c), "ci": (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))}
    led = K["ledger"]
    assert [round(k, 1) for k in out["HRR"]["kstar"]] == led["HRR_kstar"] and round(out["HRR"]["alpha"], 3) == led["HRR_exponent"]
    assert [round(c, 3) for c in out["HRR"]["ci"]] == led["HRR_ci95"]
    return dims, models, out, K["threshold"]


def figS1():
    fig = new_fig(180, 64)
    Y0, HH = 15, 44
    # --- a: chain depth, each arm trained at that depth (V-57) -----------------------------------------------
    ax = axmm(fig, 14, Y0, 44, HH)
    v57 = J("v57_logs/v57.json")
    dep = v57["depths"]
    P = "F3e_depth"
    ha_ = []
    for key, jk, mk, lw, ms, z in [("rows", "parser", "o", 2.2, 4.2, 2), ("handover", "probe_argmax", "s", 1.1, 3.2, 3),
                                   ("lora", "lora_direct", "v", 1.1, 3.6, 3), ("probe", "floor", "^", 1.1, 3.6, 3)]:
        per = same(P, "%s per seed per depth" % jk, [v57["cells"][str(k)][jk] for k in dep])
        ax.plot(dep, [mean(p) for p in per], color=C[key], marker=mk, ms=ms, lw=lw, zorder=z)
        ha_.append(Line2D([], [], color=C[key], marker=mk, ms=ms, lw=lw, label=NAME[key]))
        for d_, p in zip(dep, per):
            line_dots(ax, d_, p, C[key])
        rec("figS1_sweeps", "a", NAME[key], colour=C[key], marker=mk, x=dep, y_mean=[mean(p) for p in per], y_per_seed=per,
            source="v57_logs/v57.json cells[depth].%s" % jk)
    ax.set_xticks(dep); ax.set_xlim(dep[0] - 0.25, dep[-1] + 0.25)
    ax.set_xlabel("chain depth\n(each arm trained at that depth)", linespacing=1.3); ax.set_ylabel("accuracy")
    ax.set_ylim(0.4, 1.02); ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    inleg(ax, ha_, "center left", (0.03, 0.555), labelspacing=0.3, handlelength=2.0)
    letter(fig, "a", 1, 63.5)

    # --- b: one fitted projection, read continuously or snapped to the nearest codebook row (V-21) -----------
    ax = axmm(fig, 76, Y0, 42, HH)
    v21 = J("v21_logs/v21.json")
    P = "F3d_read_by_budget"
    ep = [c["fit_epochs"] for c in v21["cells"]]
    xs = np.arange(len(ep))
    means = {}
    for lab, jk, ilab, col, mk in [("read continuously", "frozen_cont", "continuous read per seed", C["cont"], "o"),
                                   ("snapped to the nearest\ncodebook row", "proj_argmax", "snapped read per seed", C["handover"], "s")]:
        per = same(P, ilab, [c[jk] for c in v21["cells"]])
        means[jk] = [mean(p) for p in per]
        ax.plot(xs, means[jk], color=col, marker=mk, ms=3.8, lw=1.3, zorder=3)
        for x, p in zip(xs, per):
            line_dots(ax, x, p, col)
        rec("figS1_sweeps", "b", lab.replace("\n", " "), colour=col, marker=mk, x_epochs=ep, y_mean=[mean(p) for p in per], y_per_seed=per,
            source="v21_logs/v21.json cells[].%s (hops %d, n_ctx %d, n_sym %d)" % (jk, v21["hops"], v21["n_ctx"], v21["n_sym"]))
    ax.text(1.15, means["proj_argmax"][1] - 0.03, "snapped to the\nnearest code", fontsize=MIN_PT, color=INK, ha="left", va="top",
            linespacing=1.15)
    ax.text(xs[3] + 0.1, means["frozen_cont"][3] - 0.045, "not snapped", fontsize=MIN_PT, color=INK, ha="center", va="top")
    for key, jk, va, dy in [("rows", "parser", "bottom", 0.008), ("probe", "floor", "bottom", 0.008)]:
        m = mean(v21[jk])
        ax.axhline(m, color=C[key], lw=0.8, ls=DASH, zorder=1)
        ax.text(xs[-1] + 0.3, m + dy, NAME[key], fontsize=MIN_PT, color=INK if key == "rows" else MUTED, va=va, ha="right")
        rec("figS1_sweeps", "b", NAME[key] + " (dashed line)", colour=C[key], y_mean=m, y_per_seed=v21[jk], source="v21_logs/v21.json %s" % jk)
    ax.set_xticks(xs); ax.set_xticklabels([str(int(e)) for e in ep]); ax.set_xlim(-0.35, len(ep) - 0.65)
    ax.set_xlabel("epochs fitting the projection"); ax.set_ylabel("accuracy"); ax.set_ylim(0.4, 1.06)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    letter(fig, "b", 63.5, 63.5)

    # --- c: facts held at the criterion against dimension (V-151) --------------------------------------------
    dims, models, fit, thr = capacity_fits()
    print("  V-151: " + "; ".join("%s k* %s, exponent %.3f [%.3f, %.3f]" % (m, "/".join("%.1f" % k for k in fit[m]["kstar"]), fit[m]["alpha"],
                                                                         fit[m]["ci"][0], fit[m]["ci"][1]) for m in models))
    ax = axmm(fig, 136, Y0, 41, HH)
    style = {"HRR": (C["store"], "o", 1.9, 4.6, 4), "MAP": ("#5A5A5A", "s", 0.7, 3.0, 3), "BSC": ("#8A8A8A", "^", 0.7, 3.2, 3),
             "FHRR": ("#B0B0B0", "D", 0.7, 2.8, 3)}
    assert models[0] == "HRR"
    nudge = {"HRR": 0.86, "MAP": 1.16, "BSC": 1.0, "FHRR": 1.0}      # end labels: HRR and MAP end 4% apart, so they are moved apart
    for m in models:
        col, mk, lw, ms, z = style[m]
        ax.plot(dims, fit[m]["kstar"], color=col, marker=mk, ms=ms, lw=lw, zorder=z)
        ax.text(dims[-1] * 1.22, fit[m]["kstar"][-1] * nudge[m], m, fontsize=MIN_PT, color=INK, ha="left", va="center")
        rec("figS1_sweeps", "c", m, colour=col, marker=mk, x_dimension=dims, kstar=fit[m]["kstar"], exponent=fit[m]["alpha"],
            exponent_ci95=list(fit[m]["ci"]), intercept_log=fit[m]["c"],
            source="v151_logs/v151.json capacity.%s; %.2f crossing interpolated in log k; least-squares fit of log k* on log d; bootstrap of the "
                   "four points as constants.json capacity_criterion" % (m, thr))
    xx = np.array([dims[0] / 1.2, dims[-1] * 1.2])
    ax.plot(xx, np.exp(fit["HRR"]["c"]) * xx ** fit["HRR"]["alpha"], color=INK, lw=0.8, ls=DASH, zorder=5)
    ax.text(dims[-1] * 1.9, 30, "dashed: power law\nfitted to HRR,\nexponent %.3f" % fit["HRR"]["alpha"], fontsize=MIN_PT, color=MUTED,
            ha="right", va="bottom", linespacing=1.15)
    ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
    ax.set_xticks(dims); ax.set_xticklabels([str(d_) for d_ in dims]); ax.minorticks_off()
    yt = [32, 64, 128, 256, 512]
    ax.set_yticks(yt); ax.set_yticklabels([str(t) for t in yt])
    ax.set_xlim(dims[0] / 1.4, dims[-1] * 2.0); ax.set_ylim(24, 560)
    ax.xaxis.grid(True, linewidth=0.5, color=GRID, zorder=0)
    ax.set_xlabel("dimension $d$"); ax.set_ylabel("facts held at %.2f readout\n(interpolated)" % thr, linespacing=1.3)
    letter(fig, "c", 123.5, 63.5)
    save(fig, "figS1_sweeps")

def fig8_hardware():
    """The RULER vector stores under real load (a) and under hardware-like precision (b), read noise (c) and failed
    components (d) (B22), with what more or longer vectors buy back. Open markers: levels added after the registered
    ones were scored (amendment 1; descriptive)."""
    fig = new_fig(180, 124)
    H = J("ruler_loads.json")
    R = {r["cond"]: dict(r, added=False) for r in J("b22_robust.json")}
    if os.path.exists(os.path.join(D, "b22_robust_ext.json")):
        R.update({r["cond"]: dict(r, added=True) for r in J("b22_robust_ext.json")})
    if os.path.exists(os.path.join(D, "b22_robust_bip.json")):
        R.update({r["cond"]: dict(r, added=True) for r in J("b22_robust_bip.json")})
    if os.path.exists(os.path.join(D, "b22_robust_ext_trades.json")):
        R.update({r["cond"]: dict(r, added=True) for r in J("b22_robust_ext_trades.json")})
    # --- a: load per vector ----------------------------------------------------------------------------------------
    ax = axmm(fig, 13, 82, 34, 36)
    ks = sorted(int(k) for k in H["hist"])
    ax.bar(ks, [H["hist"][str(k)] / 1000 for k in ks], 0.8, color=C["ours"], linewidth=0, zorder=2)
    mean_load = sum(k * H["hist"][str(k)] for k in ks) / H["n_vectors"]
    ax.axvline(mean_load, color=INK, lw=0.7, ls=DASH, zorder=3)
    ax.text(mean_load + 1.2, ax.get_ylim()[1] * 0.97, "mean %.1f" % mean_load, fontsize=MIN_PT, color=INK, va="top", ha="left")
    ax.set_xlabel("pairs in one vector"); ax.set_ylabel("vectors (thousands)")
    ax.set_xticks([0, 10, 20, 30, 40]); ax.set_xlim(-1, 41)
    rec("fig8_hardware", "a", "pairs per vector", hist=H["hist"], n_vectors=H["n_vectors"], n_samples=H["n_samples"],
        mean=mean_load, source="figdata/ruler_loads.json (recomputed from the B21 pairs)")
    letter(fig, "a", 1, 122)
    # --- b-d: one degradation each, default store and the trades ---------------------------------------------------
    TR = [("8 per vector", "v", "#555555", "8 pairs per vector"), ("4 per vector", "^", "#555555", "4 pairs per vector"),
          ("d=16384", "s", INK, "vectors twice as long")]
    def panel(x0, levels, labels, key, xlabel, lett, lx):
        ax = axmm(fig, x0, 12, 33, 40)
        lv = [l for l in levels if l is None or key(l) in R]
        xs = list(range(len(lv)))
        cond = lambda l: R["float"] if l is None else R[key(l)]
        ys = [100 * cond(l)["frac"] for l in lv]
        ax.plot(xs, ys, color=C["ours"], lw=1.6, zorder=3)
        for x, l, y in zip(xs, lv, ys):
            added = l is not None and cond(l)["added"]
            ax.plot([x], [y], marker="o", ms=3.8, color=C["ours"], mfc="white" if added else C["ours"],
                    mec=C["ours"] if added else "white", mew=1.0 if added else 0.5, ls="none", zorder=4)
            rec("fig8_hardware", lett, "float" if l is None else key(l), frac=y / 100, added_after_scoring=added,
                source="b22_robust%s.json" % ("_ext" if added else ""))
        ax.axhline(99.5, color=MUTED, lw=0.7, ls=DOT, zorder=1)
        worst = next((l for l in lv if l is not None and cond(l)["frac"] < 0.995), None)
        if worst is not None:
            xw = lv.index(worst)
            for j, (suf, mk, col, _) in enumerate(TR):
                c = R.get(key(worst) + " | " + suf)
                if c:
                    ax.plot([xw + 0.22 * (j + 1)], [100 * c["frac"]], marker=mk, ms=4.2, color=col, ls="none", zorder=5,
                            mec="white", mew=0.4)
                    rec("fig8_hardware", lett, key(worst) + " | " + suf, frac=c["frac"], keys=c["keys"], source="b22_robust json")
        ax.set_xticks(xs); ax.set_xticklabels([labels[levels.index(l)] for l in lv])
        ax.set_xlim(-0.4, len(lv) - 0.4 + 0.5)
        ax.set_xlabel(xlabel)
        letter(fig, lett, lx, 58)
        return ax, ys
    q = [None, 8, 6, 4, 3, 2, 1]
    ax, yq = panel(13, q, ["float", "8", "6", "4", "3", "2", "1"], lambda b: "Q b=%d" % b, "bits per number", "b", 1)
    ax.set_ylabel("facts read back exactly (%)")
    if "Q b=1 bipolar" in R:                 # amendment 2: the usual bipolar 1-bit store (real part, random ties)
        xb = len([l for l in q if l is None or "Q b=%d" % l in R]) - 1
        vb = 100 * R["Q b=1 bipolar"]["frac"]
        ax.plot([xb - 0.25], [vb], marker="D", ms=3.8, mfc="white", mec=C["ours"], mew=1.0, ls="none", zorder=5)
        rec("fig8_hardware", "b", "1 bit, bipolar store (amendment 2, post hoc)", frac=vb / 100, source="b22_robust_bip.json")
    n = [None, 0.25, 0.5, 1, 2, 3, 4, 6, 8]
    _, yn = panel(74, n, ["0", ".25", ".5", "1", "2", "3", "4", "6", "8"], lambda r: "N r=%g" % r,
                  "read noise (× signal size)", "c", 62)
    f = [None, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    _, yf = panel(135, f, ["0", "5", "10", "20", "30", "40", "50", "60", "70"], lambda p: "F p=%g" % p,
                  "failed components (%)", "d", 123)
    fig.axes[1].set_ylim(94, 100.4); fig.axes[1].set_yticks([94, 96, 98, 100])
    for a in fig.axes[2:]:
        a.set_ylim(0, 103); a.set_yticks([0, 25, 50, 75, 100])
    fig.axes[3].tick_params(axis="x", labelsize=MIN_PT)
    hs = [Line2D([], [], color=C["ours"], lw=1.6, marker="o", ms=3.8, mec="white", label="16 pairs per vector, d = 8,192"),
          Line2D([], [], color=C["ours"], lw=0, marker="o", ms=3.8, mfc="white", label="level added after scoring"),
          Line2D([], [], color=C["ours"], lw=0, marker="D", ms=3.8, mfc="white", label="bipolar 1-bit store (added)")] + \
         [Line2D([], [], marker=mk, color=col, ls="none", ms=4.2, label=lab) for _, mk, col, lab in TR]
    leg(fig, hs, 13, 68.5, ncol=3, columnspacing=1.0, handlelength=1.4)
    # --- e: a measured phase-change-memory device model (B25): store and clean-up on simulated crossbars --------------
    P = {t["cond"]: t for t in J("b25_pcm.json")["totals"]}
    times = ["60 s", "1 h", "1 d", "30 d", "1 y"]
    ax = axmm(fig, 72, 82, 62, 36)
    SER = [("B1-SC gdc", "16/vec", C["ours"], "-", "o", "drift compensated, 16 pairs per vector"),
           ("B1-SC gdc", "64/vec", C["ours"], (0, (3, 1.6)), "s", "drift compensated, 64 pairs per vector"),
           ("B1-SC none", "16/vec", "#8A8A8A", "-", "^", "no drift compensation")]
    for arm, load, col, ls, mk, lab in SER:
        ys = [100 * P["%s | %s | %s" % (arm, t, load)]["frac"] for t in times]
        ax.plot(range(len(times)), ys, color=col, lw=1.5, ls=ls, marker=mk, ms=3.6, mec="white", mew=0.5, zorder=4)
        rec("fig8_hardware", "e", lab, times=times, frac=[y / 100 for y in ys], source="figdata/b25_pcm.json (B25)")
    ax.text(4.25, 100, "compensated,\n16 or 64 pairs\nper vector", fontsize=MIN_PT, color=INK, va="center", ha="left", linespacing=1.1)
    ax.text(4.25, 6, "no drift\ncompensation", fontsize=MIN_PT, color=MUTED, va="center", ha="left", linespacing=1.1)
    ax.set_xticks(range(len(times))); ax.set_xticklabels(times); ax.set_xlim(-0.3, 4.2)
    ax.set_ylim(-3, 104); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("time after writing"); ax.set_ylabel("facts read back exactly (%)")
    letter(fig, "e", 60, 122)
    save(fig, "fig8_hardware")


# ============================================================================================================ Fig. 6
def fig6_ecc():
    """Equal cells on simulated phase-change devices (B38): cells per stored pair that each memory needs for >= 99.5%
    exact reads under scattered errors (a) and under failed lines without an interleaver (b), and accuracy of designs
    fixed for one noise level as the error rate rises beyond it (c). Every stored bit is counted."""
    B = J("b38.json")
    fig = new_fig(180, 80)
    TOP = 5800.0                                           # where censored budgets ('>4096') are drawn
    S = [("V-last", C["ours"], "-", "o", "full", "VSA store"),
         ("V-cb", C["ours"], (0, (3, 1.6)), "o", "none", "VSA store, codebook on the cells"),
         ("E-LDPC", "#0072B2", "-", "s", "full", "table with LDPC code"),
         ("E-BCH", "#56B4E9", "-", "D", "full", "table with BCH code"),
         ("T-vhdd", "#009E73", "-", "^", "full", "random-code table, no superposition"),
         ("T-rep", "#A08C78", "-", "v", "full", "table with repeated bits")]
    def val(c):
        return TOP if c.get("censored") else c["value"]
    def series(ax, xs, cells, col, ls, mk, fill, lw=1.3):
        ys = [val(c) for c in cells]
        ax.plot(xs, ys, color=col, lw=lw, ls=ls, zorder=3)
        for x, c, y in zip(xs, cells, ys):
            cen = c.get("censored")
            ax.plot([x], [y], marker="^" if cen else mk, ms=3.6, color=col,
                    mfc="white" if (fill == "none" or cen) else col, mec=col, mew=0.9, ls="none", zorder=4)
        return ys
    def logy(ax, note=True):
        ax.set_yscale("log", base=2)
        ax.set_ylim(10, 7600)
        ax.set_yticks([16, 64, 256, 1024, 4096]); ax.set_yticklabels(["16", "64", "256", "1,024", "4,096"])
        ax.axhspan(4700, 7600, color="#F2F2F2", zorder=0, lw=0)
        if note:
            ax.text(0.02, 0.985, "not reached\n(> 4,096)", transform=ax.transAxes, fontsize=MIN_PT, color=MUTED, va="top",
                ha="left", linespacing=1.0)
    # --- a: scattered (i.i.d.) errors --------------------------------------------------------------------------------
    ax = axmm(fig, 13, 11, 46, 44)
    conds = B["iid"]["conds"]
    eps = [c["value"] for c in B["iid"]["eps"]]
    xs = list(range(len(conds)))
    for key, col, ls, mk, fill, lab in S:
        ys = series(ax, xs, B["iid"][key], col, ls, mk, fill)
        rec("fig6_ecc", "a", lab, conds=conds, eps=eps, cells_per_pair=ys, censored_drawn_at=TOP, source="figdata/b38.json iid " + key)
    cap = [c["value"] for c in B["iid"]["capacity"]]
    ax.plot(xs, cap, color=MUTED, lw=0.9, ls=DOT, zorder=2)
    rec("fig6_ecc", "a", "capacity bound", cells_per_pair=cap, source="figdata/b38.json iid capacity")
    ax.text(xs[-1] + 0.15, cap[-1], "bound", fontsize=MIN_PT, color=MUTED, va="center", ha="left")
    logy(ax)
    ax.set_xticks(xs); ax.set_xticklabels(["0", ".01", "2.1", "5", "12", "28", "37"])
    ax.set_xlim(-0.4, len(xs) - 0.1)
    ax.set_xlabel("cell-error rate (%; conditions evenly spaced)"); ax.set_ylabel("cells per stored pair, 99.5% exact")
    letter(fig, "a", 1, 60)
    # --- b: failed lines, natural layout (no interleaver) ---------------------------------------------------------------
    ax = axmm(fig, 72, 11, 46, 44)
    dc = ["k4", "e05", "k8", "k16"]
    f = [4.2, 10, 24.5, 55.2]                              # f = 2 eps (prereg: 0.0416, 0.10, 0.2453, 0.5520)
    xs = list(range(len(dc)))
    for key, col, ls, mk, fill, lab in S:
        if key in ("V-cb", "T-rep"):
            continue
        cells = [B["damage"]["lines-nat-%s" % c][key] for c in dc]
        src = "figdata/b38.json damage lines-nat"
        if key == "E-LDPC":                                # B58 part (c): B38's package decoder overflowed (NaN read as
            L = J("b58_ldpc.json")                         # "no error"); re-scored with the guarded decoder
            cells = [L["lines-nat"][c] for c in dc]
            src = "figdata/b58_ldpc.json lines-nat E-LDPC* (B58 re-score)"
        ys = series(ax, xs, cells, col, ls, mk, fill)
        rec("fig6_ecc", "b", lab + ", no interleaver", damaged_share_pct=f, cells_per_pair=ys, source=src)
    cells = [B["damage"]["lines-il-%s" % c]["E-LDPC"] for c in dc]
    ys = series(ax, xs, cells, "#0072B2", (0, (3, 1.6)), "s", "none", lw=1.0)
    rec("fig6_ecc", "b", "table with LDPC code, interleaved", damaged_share_pct=f, cells_per_pair=ys, source="figdata/b38.json damage lines-il")
    ax.text(xs[-1] + 0.12, ys[-1], "LDPC,\ninterleaved", fontsize=MIN_PT, color="#0072B2", va="center", ha="left", linespacing=1.0)
    logy(ax, note=False)
    ax.set_xticks(xs); ax.set_xticklabels(["4", "10", "25", "55"])
    ax.set_xlim(-0.4, len(xs) + 0.2)
    ax.set_xlabel("cells on failed lines (%; evenly spaced)")
    letter(fig, "b", 63, 60)
    # --- c: designs fixed for 12% errors, beyond the design point -------------------------------------------------------
    ax = axmm(fig, 131, 11, 46, 44)
    gc = ["k8", "e16", "e20", "e24", "k16", "e32", "k32"]
    ge = [100 * eps[conds.index("k8")], 16, 20, 24, 100 * eps[conds.index("k16")], 32, 100 * eps[conds.index("k32")]]
    for key, col, ls, mk, fill, lab in S:
        if key in ("V-cb", "T-rep"):
            continue
        ys = [B["graceful"][c][key]["value"] for c in gc]
        ax.plot(ge, ys, color=col, lw=1.3, zorder=3, marker=mk, ms=3.4, mec="white", mew=0.4)
        rec("fig6_ecc", "c", lab, eps_pct=ge, exact_pct=ys, cells_per_pair=128, source="figdata/b38.json graceful")
    ax.axvline(ge[0], color=MUTED, lw=0.7, ls=DOT, zorder=1)
    ax.text(ge[0] + 0.6, 6, "design\npoint", fontsize=MIN_PT, color=MUTED, va="bottom", ha="left", linespacing=1.0)
    ax.set_xlim(10, 39); ax.set_xticks([12, 20, 28, 36])
    ax.set_ylim(-3, 104); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("cell-error rate (%)"); ax.set_ylabel("facts read exactly (%)")
    letter(fig, "c", 121, 60)
    hs = [Line2D([], [], color=col, ls=ls, lw=1.3, marker=mk, ms=3.6, mfc="white" if fill == "none" else col, mec=col,
                 label=lab) for _, col, ls, mk, fill, lab in S] + \
         [Line2D([], [], color=MUTED, ls=DOT, lw=0.9, label="capacity bound")]
    leg(fig, hs, 13, 79, ncol=3, columnspacing=1.4, handlelength=2.0)
    save(fig, "fig6_ecc")

FIGS = {"1": fig1, "2": fig2_benchmarks, "3": fig3_store, "4": fig4_mechanism, "5": fig5_conditions, "6": fig6_oplogs,
        "7": fig7_reasoning, "8": fig8_hardware, "E": fig6_ecc, "K": fig_compression, "S5": figS5_coherence, "S6": figS6_gains, "S1": figS1, "S2": figS2_poolsweep, "S3": figS3_per_task, "S4": figS4_clutrr_k}

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--no-verify" not in sys.argv:
        verify_sources()
    for k in args or list(FIGS):
        FIGS[k]()
    babi_table()
    old = json.load(open(VALUES_JSON)) if (args and os.path.exists(VALUES_JSON)) else {}
    old.update(REC)
    json.dump(old, open(VALUES_JSON, "w"), indent=1, ensure_ascii=False, sort_keys=True)
    print("plotted values -> %s" % VALUES_JSON)
