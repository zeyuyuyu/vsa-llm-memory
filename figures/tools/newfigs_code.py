# ============================================================================================================ Fig. 5
def fig5_conditions():
    """When the memory helps: the memory against a LoRA fine-tune in six settings (a), and the check (b)."""
    from matplotlib.patches import FancyBboxPatch
    fig = new_fig(180, 76)
    Cd = J("conditions.json")
    # --- a: dot plot, one row per setting, grouped by the condition that holds or fails -------------------------
    ax = axmm(fig, 50, 12, 62, 60, grid=False)
    ax.xaxis.grid(True, linewidth=0.5, color=GRID, zorder=0); ax.set_axisbelow(True)
    y, yt, ylab, gtops = 0.0, [], [], []
    for g in Cd["groups"]:
        gtops.append((y, g["label"]))
        y -= 0.62
        for r in g["rows"]:
            m, l_ = r["memory"], r["lora"]
            ax.plot([mean(l_), mean(m)], [y, y], color="#BBBBBB", lw=1.0, zorder=1, solid_capstyle="butt")
            if "prompt" in r:
                ax.plot([r["prompt"]], [y], marker="s", ms=3.6, color=C["prompt"], ls="none", zorder=3)
            if "lora_first" in r:
                ax.plot([mean(r["lora_first"])], [y], marker="o", ms=4.4, mfc="white", mec=C["lora"], mew=1.1, ls="none", zorder=3)
            ax.plot([mean(l_)], [y], marker="o", ms=5.0, color=C["lora"], ls="none", zorder=4)
            ax.plot([mean(m)], [y], marker="o", ms=5.0, color=C["handover"], ls="none", zorder=5)
            if len(l_) > 1:
                ax.scatter(l_, [y - 0.2] * len(l_), s=5, color=C["lora"], alpha=0.8, linewidths=0, zorder=3)
            if len(m) > 1:
                ax.scatter(m, [y + 0.2] * len(m), s=5, color=C["handover"], alpha=0.8, linewidths=0, zorder=3)
            yt.append(y); ylab.append(r["label"])
            rec("fig5_conditions", "a", r["label"], memory=m, lora=l_, lora_first=r.get("lora_first"), prompt=r.get("prompt"),
                group=g["label"], source=r["source"])
            y -= 1.0
        y -= 0.15
    ax.set_yticks(yt); ax.set_yticklabels(ylab)
    ax.tick_params(axis="y", length=0)
    for (yg, lab) in gtops:
        ax.text(-0.03, yg - 0.12, lab, transform=matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData),
                ha="right", va="center", fontsize=SMALL, fontstyle="italic", color=MUTED)
    ax.set_ylim(y + 0.55, 0.25)
    ax.set_xlim(0.33, 1.02); ax.set_xticks([0.4, 0.6, 0.8, 1.0]); ax.set_xlabel("accuracy")
    ax.spines["left"].set_visible(False)
    hs = [Line2D([], [], marker="o", ms=5.0, color=C["handover"], ls="none", label="vector symbolic memory (handover)"),
          Line2D([], [], marker="o", ms=5.0, color=C["lora"], ls="none", label="LoRA fine-tune (best configuration)"),
          Line2D([], [], marker="o", ms=4.4, mfc="white", mec=C["lora"], mew=1.1, ls="none", label="LoRA fine-tune, first configuration"),
          Line2D([], [], marker="s", ms=3.6, color=C["prompt"], ls="none", label="prompting")]
    leg(fig, hs, 50, 75.5, ncol=2, columnspacing=1.2, handlelength=0.8)
    letter(fig, "a", 1, 75.5)
    # --- b: the check, as a decision diagram -------------------------------------------------------------------
    W_, H_ = fig._mm
    boxes = []
    def box(x, y, w, h, s, fc, ec, bold=False):
        p = FancyBboxPatch((x / W_, y / H_), w / W_, h / H_, boxstyle="round,pad=0,rounding_size=0.008",
                           transform=fig.transFigure, fc=fc, ec=ec, lw=0.8)
        fig.patches.append(p)
        t = fig.text((x + w / 2) / W_, (y + h / 2) / H_, s, ha="center", va="center", fontsize=SMALL, color=INK,
                     fontweight="bold" if bold else "normal", linespacing=1.25)
        boxes.append((p, t))
    def arrow(x0, y0, x1, y1, lab=None, lx=0, ly=0):
        fig.patches.append(matplotlib.patches.FancyArrowPatch((x0 / W_, y0 / H_), (x1 / W_, y1 / H_), transform=fig.transFigure,
                           arrowstyle="-|>", mutation_scale=7, lw=0.8, color=INK))
        if lab:
            fig.text((x0 + x1) / 2 / W_ + lx / W_, (y0 + y1) / 2 / H_ + ly / H_, lab, fontsize=SMALL, color=INK,
                     ha="center", va="center")
    q1 = (128, 56, 46, 13); q2 = (128, 30, 46, 13)
    box(*q1, "Does the model's own\nsingle pass fail?", "#F2F2F2", "#888888")
    box(*q2, "Is the algebra exact on\nthe task's role structure?", "#F2F2F2", "#888888")
    box(128, 4, 46, 12, "add the vector\nsymbolic memory", "#FBE3D5", C["handover"], bold=True)
    box(178 - 28, 69.5 - 7.5 - 46.5 + 8, 0, 0, "", "white", "white")
    box(119, 70, 1, 1, "", "white", "white")
    arrow(151, 56, 151, 43.2, "yes", lx=3.2)
    arrow(151, 30, 151, 16.2, "yes", lx=3.2)
    fig.text(176 / W_, 62.5 / H_, "no:\nprompt or\nfine-tune", fontsize=SMALL, color=MUTED, ha="left", va="center", linespacing=1.2)
    fig.text(176 / W_, 36.5 / H_, "no:\nfine-tune", fontsize=SMALL, color=MUTED, ha="left", va="center", linespacing=1.2)
    fig.patches[:] = [p for p in fig.patches if not (p.get_width() == 0 or p.get_width() < 2 / W_)]
    boxes[:] = [b for b in boxes if b[1].get_text()]
    letter(fig, "b", 120, 75.5)
    rec("fig5_conditions", "b", "decision diagram", note="drawn from the two conditions stated in the Results; no data")
    save(fig, "fig5_conditions", boxes)


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
    fig.text(4 / W_, 74.0 / H_, ("System of record: %s is still logged at the %s, so it misses that specimen and lists %d others"
             " that are no longer in %s.") % (miss[0], T["sor_place"][miss[0]], len(extra), T["target"]),
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
    ax = axmm(fig, 75, 13, 40, 44)
    Ls = Rz["llm_state"]
    vals = [("Qwen3-32B,\nwhole log", Ls["log"]["value"], C["open"]), ("Qwen3-32B,\nread state", Ls["state"]["value"], C["open"]),
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
    ax = axmm(fig, 132, 13, 45, 44)
    Sv = Rz["selective"]
    cov = np.array(Sv["coverage"]) * 100; acc = np.array(Sv["accuracy"])
    keep = cov >= 30 - 1e-9
    ax.plot(cov[keep], acc[keep], color=C["ours"], lw=1.3, marker="o", ms=2.6, zorder=3)
    ax.axhline(Sv["all_answers"], color=MUTED, lw=0.8, ls=DASH, zorder=2)
    ax.text(31, Sv["all_answers"] - 0.004, "all answers, %.3f" % Sv["all_answers"], fontsize=MIN_PT, color=MUTED, va="top")
    i70 = int(np.argmin(abs(cov - 70)))
    ax.annotate("%.3f at 70%%" % acc[i70], (cov[i70], acc[i70]), xytext=(cov[i70] - 22, acc[i70] + 0.017), fontsize=MIN_PT,
                color=INK, arrowprops=dict(arrowstyle="-", lw=0.6, color=INK))
    ax.set_xlim(28, 102); ax.set_xticks([30, 50, 70, 90, 100][:-1] + [100])
    ax.set_ylim(0.90, 1.0); ax.set_yticks([0.90, 0.925, 0.95, 0.975, 1.0])
    ax.set_xlabel("answers kept, most reliable first (%)"); ax.set_ylabel("accuracy of the kept answers")
    rec("fig7_reasoning", "d", "selective accuracy, min reliability", coverage=Sv["coverage"], accuracy=Sv["accuracy"],
        n=Sv["n"], wrong=Sv["wrong"], auroc=Sv["auroc"], source=Sv["source"])
    letter(fig, "d", 119, 70.5)
    save(fig, "fig7_reasoning", boxes)


