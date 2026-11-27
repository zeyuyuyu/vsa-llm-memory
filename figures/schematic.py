"""Fig. 1a, drawn with matplotlib primitives only: three couplings of one frozen language model.

    prompting                 the evidence goes into the prompt, the model generates the answer
    LoRA fine-tune            a low-rank update of the weights, the model generates the answer
    vector symbolic memory    identity head -> codebook rows -> bind and bundle (one superposed vector per hop)
                              -> unbind and clean up -> answer (always a codebook row)

Coordinates are millimetres: the caller gives draw_overview an axes that is exactly W x H mm, so a font size in points
and a box size in data units can be compared. BOXES["overview"] lists every (box, label) pair for the layout check.
"""
from matplotlib.patches import FancyBboxPatch

from figstyle import C, INK, MUTED, SMALL, MIN_PT

BOXES = {}
BAND = "#FBEDE5"          # pale tint of the memory's vermilion
LM_FILL = "#EFEFEF"


def _box(ax, x, y, w, h, fc="#FFFFFF", ec=INK, lw=0.7, r=1.2, z=2):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=%f" % r, fc=fc, ec=ec, lw=lw, zorder=z, clip_on=False)
    ax.add_patch(p)
    return p


def _arrow(ax, p0, p1, color=INK, lw=0.9, scale=8):
    ax.annotate("", xy=p1, xytext=p0, zorder=3,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=scale))


def _right_of(ax, t, gap):
    """x (data units) just right of a text object"""
    bb = t.get_window_extent(ax.figure.canvas.get_renderer())
    return ax.transData.inverted().transform((bb.x1, bb.y0))[0] + gap


def draw_overview(ax, W=172.0, H=68.0):
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    pairs = BOXES["overview"] = []
    V_, P_, L_ = C["handover"], C["prompt"], C["lora"]

    def labelled(x, y, w, h, text, fontsize=8.5, weight="bold", **kw):
        p = _box(ax, x, y, w, h, **kw)
        t = ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, fontweight=weight, color=INK,
                    linespacing=1.25, zorder=4)
        pairs.append((p, t))
        return p

    # ---- the input: two facts and a two-hop question -----------------------------------------------------------
    card = _box(ax, 0, 55.5, 66, 12, lw=0.6)
    t1 = ax.text(3, 63.9, "Mary took the football. Mary went to the hallway.", fontsize=8, style="italic", va="center", color=INK, zorder=4)
    t2 = ax.text(3, 59.2, "Where is the football?", fontsize=8, fontweight="bold", va="center", color=INK, zorder=4)
    pairs += [(card, t1), (card, t2)]

    # ---- the frozen language model -----------------------------------------------------------------------------
    labelled(0, 34, 38, 16, "frozen\nlanguage model", fc=LM_FILL, lw=0.7)
    _arrow(ax, (19, 55.5), (19, 50))

    # ---- two couplings that end in generated text: thin lanes --------------------------------------------------
    x_end, x_ans = 150.5, 152.0
    for y, col, name, how in ((46.5, P_, "prompting", "evidence in the prompt"),
                              (38.5, L_, "LoRA fine-tune", "low-rank update of the weights")):
        _arrow(ax, (38, y), (x_end, y), color=col, lw=1.1)
        t = ax.text(44, y + 1.1, name, fontsize=8.5, fontweight="bold", color=INK, va="bottom", ha="left", zorder=4)
        ax.text(_right_of(ax, t, 2.2), y + 1.1, how, fontsize=SMALL, color=MUTED, va="bottom", ha="left", zorder=4)
        ax.text(x_ans, y, "answer", fontsize=8.5, color=INK, va="center", ha="left", zorder=4)

    # ---- the vector symbolic memory: the primary lane ----------------------------------------------------------
    ax.add_patch(FancyBboxPatch((0, 0.4), W, 28.6, boxstyle="round,pad=0,rounding_size=1.6", fc=BAND, ec="none", zorder=1,
                                clip_on=False))
    _arrow(ax, (19, 34), (19, 21), color=V_, lw=1.1)
    ax.text(21.5, 31.6, "hidden states", fontsize=SMALL, color=MUTED, va="center", ha="left", zorder=4)
    ax.text(25.5, 25.2, "vector symbolic memory", fontsize=9.5, fontweight="bold", color=INK, va="center", ha="left", zorder=4)
    yb, hb = 9.0, 12.0
    ym = yb + hb / 2
    stages = [(5, 28, "naming head", "the only trained part"),
              (39, 28, "exact codes", "one fixed random vector each"),
              (73, 30, "bind and bundle", "tie pairs, add them up"),
              (109, 36, "unbind and clean up", "look up, snap to a symbol")]
    for i, (x, w, name, note) in enumerate(stages):
        labelled(x, yb, w, hb, name, ec=V_, lw=1.0)
        ax.text(x + w / 2, yb - 1.7, note, fontsize=MIN_PT, color=MUTED, ha="center", va="top", zorder=4)
        nxt = stages[i + 1][0] if i + 1 < len(stages) else x_end
        _arrow(ax, (x + w, ym), (nxt, ym), color=V_, lw=1.1)
    ax.text(x_ans, ym, "answer", fontsize=8.5, fontweight="bold", color=INK, va="center", ha="left", zorder=4)
    ax.text(x_ans, yb - 1.7, "a known symbol", fontsize=MIN_PT, color=MUTED, ha="left", va="top", zorder=4)

    ax.text(W, H - 0.5, "controlled experiments (Fig. 2)", fontsize=SMALL, color=MUTED, ha="right", va="top", zorder=4)


def draw_reader(ax, W=172.0, H=25.0):
    """Fig. 1a, lower band: the system behind the benchmark and log results (Table 1): a fine-tuned reader writes events
    into a delta-rule memory and event logs, and rules read the answer from them."""
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    pairs = BOXES.setdefault("overview", [])
    V_ = C["handover"]

    def labelled(x, y, w, h, text, fontsize=8.5, weight="bold", **kw):
        p = _box(ax, x, y, w, h, **kw)
        t = ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, fontweight=weight, color=INK,
                    linespacing=1.2, zorder=4)
        pairs.append((p, t))
        return p

    ax.add_patch(FancyBboxPatch((0, 0.4), W, H - 0.8, boxstyle="round,pad=0,rounding_size=1.6", fc=BAND, ec="none", zorder=1,
                                clip_on=False))
    ax.text(5, H - 4.0, "reader with an exact store, used for BABILong", fontsize=9.5, fontweight="bold", color=INK,
            va="center", ha="left", zorder=4)
    ax.text(W - 2, H - 4.0, "Figs 3a and 4a,b", fontsize=SMALL, color=MUTED, va="center", ha="right", zorder=4)
    yb, hb = 6.0, 9.5
    ym = yb + hb / 2
    x_ans = 152.0
    stages = [(5, 28, "small reader", "Qwen3-1.7B, fine-tuned"),
              (39, 28, "events", "take(Mary, apple), ..."),
              (73, 30, "symbolic memory\nand event logs", "a move overwrites; logs grow"),
              (109, 36, "rules", "read the answer from the memory")]
    for i, (x, w, name, note) in enumerate(stages):
        labelled(x, yb, w, hb, name, ec=V_, lw=1.0)
        ax.text(x + w / 2, yb - 1.5, note, fontsize=MIN_PT, color=MUTED, ha="center", va="top", zorder=4)
        nxt = stages[i + 1][0] if i + 1 < len(stages) else 150.5
        _arrow(ax, (x + w, ym), (nxt, ym), color=V_, lw=1.1)
    ax.text(x_ans, ym, "answer", fontsize=8.5, fontweight="bold", color=INK, va="center", ha="left", zorder=4)
    ax.text(x_ans, yb - 1.5, "cites its log lines", fontsize=MIN_PT, color=MUTED, ha="left", va="top", zorder=4)
