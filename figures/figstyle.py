"""Shared figure style for the rewritten npj manuscript (five main figures, one supplementary figure).

Nature-family conventions: sans-serif (Liberation Sans, metric-compatible with Arial), bold lower-case panel letters,
the colour-blind-safe Okabe-Ito palette, 180 mm design width, vector PDF with embedded TrueType.

Sizes at the 180 mm design width: axis labels 8.5 pt, tick labels and legends 7.5 pt, nothing under 7 pt.

One colour per arm in every figure (C) and one name per arm in every legend (NAME).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MM = 1 / 25.4
DOUBLE = 180 * MM

LABEL, SMALL, MIN_PT, LETTER = 8.5, 7.5, 7.0, 10.0      # axis labels; ticks, legends, notes; floor; panel letters

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": LABEL, "axes.labelsize": LABEL, "xtick.labelsize": SMALL, "ytick.labelsize": SMALL,
    "legend.fontsize": SMALL, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 3.0, "ytick.major.size": 3.0, "xtick.minor.size": 1.8, "ytick.minor.size": 1.8,
    "xtick.minor.width": 0.5, "ytick.minor.width": 0.5, "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False, "axes.labelpad": 3.0,
    "lines.linewidth": 1.2, "lines.markersize": 4.0, "legend.frameon": False,
    "legend.handlelength": 1.4, "legend.handletextpad": 0.6, "legend.labelspacing": 0.45, "legend.columnspacing": 1.6,
    "hatch.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "custom",
    "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic", "mathtext.bf": "Liberation Sans:bold",
    "mathtext.cal": "Liberation Sans:italic", "mathtext.sf": "Liberation Sans", "mathtext.tt": "Liberation Mono",
    "figure.dpi": 150, "savefig.dpi": 300,
})

# Okabe-Ito, colour-blind safe. A colour names one arm in every figure.
C = {
    "prompt":   "#0072B2",   # blue           prompting (the frozen model, evidence in its prompt)
    "lora":     "#E69F00",   # orange         LoRA fine-tune
    "handover": "#D55E00",   # vermilion      handover / the vector symbolic memory / the HRR store / ours
    "loss":     "#CC79A7",   # pink           projection trained through the binding loss
    "rows":     "#000000",   # black          exact codebook rows
    "probe":    "#7F7F7F",   # grey           frozen-state probe
    "cont":     "#56B4E9",   # sky blue       continuous handover (Fig. 2b, Supplementary Fig. 1b only)
    "gpt":      "#009E73",   # bluish green   GPT-6.1 Sol
    "open":     "#56B4E9",   # sky blue       Qwen3-32B and other open models read as external baselines (Figs 4, 5)
    "sor":      "#A08C78",   # grey-brown     system of record
    "lf":       "#882255",   # wine           map trained on the answer with a term that keeps codes apart (B27, no labels)
}
C["store"] = C["ours"] = C["handover"]

NAME = {
    "prompt": "prompting", "lora": "LoRA fine-tune", "loss": "map trained only on the answer",
    "handover": "naming head", "rows": "exact codes (parser)", "probe": "probe of the hidden states",
    "cont": "naming head, not snapped",
    "lf": "map that keeps codes apart (no labels)",
}

INK = "#222222"        # text, axes
MUTED = "#666666"      # secondary text
GRID = "#E6E6E6"       # hairline grid
NEUTRAL = "#BDBDBD"    # marks that are not a named arm
