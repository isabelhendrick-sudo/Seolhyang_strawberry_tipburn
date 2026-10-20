"""Stage 1: compare tipburn symptoms across nutrient treatment groups (Seolhyang strawberry, AI Hub).

Run:  python analyse_tipburn.py --labels data/normal_labels data/tipburn_labels
"""
import argparse
import json
import logging
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy import stats

OUT = Path("outputs")
SOURCE = "Source: AI Hub facility-crop dataset, Seolhyang strawberry, farm AI22_003."
MIN_ANN = 20
TITLE_PT, SUBTITLE_PT, AXIS_PT, TICK_PT, NOTE_PT = 16, 10, 13, 11, 9

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
    "font.size": TICK_PT, "axes.labelsize": AXIS_PT,
    "xtick.labelsize": TICK_PT, "ytick.labelsize": TICK_PT,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#444444", "xtick.color": "#333333", "ytick.color": "#333333",
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.facecolor": "white",
})

GROUP_COLOURS = {
    "Normal": "#666666", "Ca deficiency": "#1b9e77", "EC 0.75": "#d95f02",
    "EC 3.0": "#7570b3", "pH 5.0": "#e7298a", "pH 7.5": "#66a61e",
}
ORGANS = {"잎": "leaf", "줄기": "stem", "과실": "fruit", "화방": "flower cluster", "꽃": "flower"}
SENSORS = {"ti_value": "Air temp.", "hi_value": "Humidity", "ci_value": "CO2", "ir_value": "Light",
           "tl_value": "Solution temp.", "ei_value": "Solution EC", "pl_value": "Solution pH",
           "el_value": "Substrate EC", "pi_value": "Substrate pH"}


def colour_of(group):
    return GROUP_COLOURS.get(group, "#a6761d")


def add_header(fig, title, subtitle=None, x=0.03, top=0.975):
    """Draw a bold title and plain subtitle; return where the plot area can start."""
    h = fig.get_size_inches()[1]
    fig.text(x, top, title, fontsize=TITLE_PT, fontweight="bold", ha="left", va="top")
    y = top - TITLE_PT * 1.45 / 72 / h
    if subtitle:
        fig.text(x, y, subtitle, fontsize=SUBTITLE_PT, ha="left", va="top", color="#333333")
        y -= SUBTITLE_PT * 1.6 * (subtitle.count("\n") + 1) / 72 / h
    return y - 0.25 / h


def add_note(fig, text, x=0.03, y=0.012):
    fig.text(x, y, text, fontsize=NOTE_PT, color="#666666", ha="left", va="bottom")


def nice_step(maxval, target=6):
    """Round block size (1, 2 or 5 x 10^k) that splits maxval into about `target` blocks."""
    raw = maxval / target
    mag = 10 ** np.floor(np.log10(raw))
    for m in (1, 2, 5, 10):
        if raw <= m * mag:
            return m * mag
    return 10 * mag


def squarify(sizes, x, y, dx, dy):
    """Squarified treemap layout; sizes must be sorted largest first and sum to dx * dy."""
    def worst(row, w):
        s = sum(row)
        return max(max(w * w * r / (s * s), s * s / (w * w * r)) for r in row)

    rects, sizes = [], list(sizes)
    while sizes:
        w = min(dx, dy)
        row, i = [sizes[0]], 1
        while i < len(sizes) and worst(row + [sizes[i]], w) <= worst(row, w):
            row.append(sizes[i])
            i += 1
        s = sum(row)
        if dx >= dy:
            rw, yy = s / dy, y
            for r in row:
                rects.append((x, yy, rw, r / rw))
                yy += r / rw
            x, dx = x + rw, dx - rw
        else:
            rh, xx = s / dx, x
            for r in row:
                rects.append((xx, y, r / rh, rh))
                xx += r / rh
            y, dy = y + rh, dy - rh
        sizes = sizes[i:]
    return rects


def fix_name(info):
    """Decode Korean zip entry names stored without the UTF-8 flag."""
    if info.flag_bits & 0x800:
        return info.filename
    try:
        return info.filename.encode("cp437").decode("cp949")
    except UnicodeError:
        return info.filename


def group_label(method, cause):
    """Turn the Korean treatment description into a short English group name."""
    s = f"{method or ''} {cause or ''}"
    if "정상" in s and not re.search(r"(?i)ph|ec|칼슘|calcium|ca", s):
        return "Normal"
    m = re.search(r"(?i)ph\s*([0-9.]+)", s)
    if m:
        return f"pH {float(m.group(1)):.1f}"
    m = re.search(r"(?i)ec\s*([0-9.]+)", s)
    if m:
        v = f"{float(m.group(1)):.2f}".rstrip("0")
        return f"EC {v}0" if v.endswith(".") else f"EC {v}"
    if re.search(r"(?i)칼슘|calcium|\bca\b", s):
        return "Ca deficiency"
    return (method or cause or "Unknown").strip()


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def iter_json(path):
    """Yield (name, json) from a label zip, a folder of JSON files, or a folder of zips."""
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"Path not found: {path.resolve()}")
    zips = [path] if path.is_file() else sorted(p for p in path.rglob("*.zip"))
    if path.is_dir():
        for f in sorted(path.rglob("*.json")):
            yield f.name, json.loads(f.read_bytes().decode("utf-8-sig"))
    for zp in zips:
        with zipfile.ZipFile(zp) as z:
            for info in z.infolist():
                name = fix_name(info)
                if name.lower().endswith(".json"):
                    yield name, json.loads(z.read(info).decode("utf-8-sig"))


def read_labels(path):
    rows, env_rows = [], []
    for name, d in iter_json(path):
        im, gi = d["images"], d.get("growth_index") or {}
        group = group_label(im.get("disease_cause_method"), im.get("disease_cause"))
        captured = pd.to_datetime(im.get("date_captured"), errors="coerce")
        base = dict(
            image_file=im.get("fname"), farm_id=im.get("farm_id"), crops_id=im.get("crops_id"),
            group=group, captured=captured,
            stem_length=to_float(gi.get("stem_length")), leaf_cnt=to_float(gi.get("leaf_cnt")),
            leaf_length=to_float(gi.get("leaf_length")), stem_thick=to_float(gi.get("stem_thick")),
        )
        cats = {c.get("id"): c.get("name") for c in d.get("categories") or []}
        for a in d.get("annotations", []):
            x, y, w, h = (a.get("bbox") or [np.nan] * 4)[:4]
            organ = cats.get(a.get("category_id"), "unknown")
            rows.append({**base, "leaf_idx": a.get("category_id"), "organ": ORGANS.get(organ, organ),
                         "status": a.get("disease_status"), "bx": x, "by": y, "bw": w, "bh": h})
        for e in d.get("environments") or []:
            env_rows.append({"group": group, "farm_id": im.get("farm_id"), "captured": captured,
                             **{k: to_float(v) for k, v in e.items() if k.endswith("_value")}})
    if not rows:
        raise SystemExit(f"No annotations found in {Path(path).resolve()}")
    return rows, env_rows


def boot_ci(values, n=2000, seed=0):
    values = np.asarray(values, float)
    if len(values) < 2:
        return np.nan, np.nan
    means = np.random.default_rng(seed).choice(values, (n, len(values))).mean(1)
    return np.percentile(means, [2.5, 97.5])


def order_groups(groups):
    rest = sorted(g for g in groups if g != "Normal")
    return (["Normal"] if "Normal" in groups else []) + rest


def weekly_plant_props(leaves):
    """Symptomatic proportion per plant per week; sparse plant-weeks become missing."""
    pw = (leaves.groupby(["group", "plant", "wk"])
          .agg(n=("symptom", "size"), y=("symptom", "sum")).reset_index())
    pw["prop"] = np.where(pw.n >= MIN_ANN, pw.y / pw.n, np.nan)
    return pw


def fig1_overview(leaves, groups, period_text, path):
    ov = (leaves.groupby("group")
          .agg(plants=("plant", "nunique"), images=("image_file", "nunique"), leaves=("symptom", "size"))
          .reindex(groups).dropna())
    fig = plt.figure(figsize=(12, 9.6))
    top = add_header(fig, "Overview of Seolhyang Strawberry Leaf Dataset Composition per Experimental Condition",
                     f"Number of plants, images and leaf annotations in each group, {period_text}")
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1, 1.1], left=0.03, right=0.97,
                  top=top - 0.045, bottom=0.085, wspace=0.28, hspace=0.42)

    ax = fig.add_subplot(gs[:, 0])
    order = ov.plants.sort_values(ascending=False)
    W, H = 1.0, 1.25
    sizes = list(order.values / order.values.sum() * W * H)
    for (g, n), (x, y, w, h) in zip(order.items(), squarify(sizes, 0, 0, W, H)):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=colour_of(g), edgecolor="white", lw=3))
        ax.text(x + 0.025, y + 0.035, g, color="white", fontsize=AXIS_PT, fontweight="bold", va="top")
        ax.text(x + 0.025, y + 0.11, f"{int(n)} plants", color="white", fontsize=TICK_PT, va="top")
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")
    ax.set_title("A   Plants per group", loc="left", fontsize=AXIS_PT, fontweight="bold", pad=24)
    ax.text(0, 1.01, "Tile area is proportional to the number of plants", transform=ax.transAxes,
            fontsize=SUBTITLE_PT, color="#555555", va="bottom")

    panels = [(gs[0, 1], "images", "B   Images per group", "images"),
              (gs[1, 1], "leaves", "C   Leaf annotations per group", "annotations")]
    for slot, col, label, unit in panels:
        ax = fig.add_subplot(slot)
        vals = ov[col].sort_values(ascending=False)
        step = nice_step(vals.max())
        ys = np.arange(len(vals))
        ax.barh(ys, vals, height=0.68, color=[colour_of(g) for g in vals.index])
        for yv, v in zip(ys, vals):
            ax.vlines(np.arange(step, v, step), yv - 0.34, yv + 0.34, color="white", lw=2)
            ax.text(v + vals.max() * 0.015, yv, f"{int(v):,}", va="center", fontsize=TICK_PT)
        ax.set_yticks(ys, vals.index)
        ax.invert_yaxis()
        ax.set_xlim(0, vals.max() * 1.16)
        ax.set_xticks([])
        for sp in ("top", "right", "bottom"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(label, loc="left", fontsize=AXIS_PT, fontweight="bold", pad=24)
        ax.text(0, 1.02, f"Each block = {int(step):,} {unit}", transform=ax.transAxes,
                fontsize=SUBTITLE_PT, color="#555555", va="bottom")
    add_note(fig, f"{SOURCE} Leaf annotations are leaves labelled symptomatic (Y) or healthy (N); each leaf is "
                  "photographed\nseveral times a day, so one leaf can contribute several annotations.")
    fig.savefig(path)
    plt.close(fig)


def fig2_symptoms_by_group(plants, groups, period_text, path):
    fig, ax = plt.subplots(figsize=(11, 7.2))
    top = add_header(fig, "Symptom Proportion Across Experimental Treatment Groups",
                     "Individual plant responses and group means with 95% bootstrap confidence intervals")
    rng = np.random.default_rng(1)
    xmax = plants.prop.max() * 100
    for i, g in enumerate(groups):
        v = plants.loc[plants.group == g, "prop"].values * 100
        c = colour_of(g)
        if len(v) >= 3 and v.std() > 0:
            grid = np.linspace(v.min(), v.max(), 300)
            d = stats.gaussian_kde(v)(grid)
            d = d / d.max() * 0.38
            ax.fill_between(grid, i - d, i + d, color=c, alpha=0.18, lw=0)
            for sign in (-1, 1):
                ax.plot(np.r_[grid[0], grid, grid[-1]], np.r_[i, i + sign * d, i], color=c, lw=1)
        ax.scatter(v, i + rng.uniform(-0.14, 0.14, len(v)), s=55, color=c, edgecolor="white",
                   linewidth=0.8, zorder=3)
        lo, hi = boot_ci(v)
        ax.plot([lo, hi], [i, i], color="black", lw=2.2, zorder=4, solid_capstyle="butt")
        for end in (lo, hi):
            ax.plot([end, end], [i - 0.1, i + 0.1], color="black", lw=2, zorder=4)
        ax.scatter([v.mean()], [i], marker="D", s=70, color="black", edgecolor="white", zorder=5)
    ax.set_yticks(range(len(groups)), [f"{g}\n(n = {int((plants.group == g).sum())} plants)" for g in groups])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-1, xmax * 1.06)
    ax.set_ylim(len(groups) - 0.45, -0.55)
    ax.set_xlabel("Symptomatic leaves per plant (%)")
    ax.grid(axis="x", color="#e5e5e5", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    handles = [Line2D([], [], marker="o", ls="", color="#888888", markeredgecolor="white", ms=8,
                      label="Individual plant"),
               Line2D([], [], marker="D", ls="-", color="black", ms=7, label="Group mean with 95% CI")]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=TICK_PT)
    H, p = kruskal(plants, groups)
    fig.subplots_adjust(left=0.2, right=0.97, top=top - 0.01, bottom=0.17)
    add_note(fig, f"{SOURCE} {period_text}.\nEach dot is one plant's mean daily proportion of symptomatic "
                  f"leaves; shaded shapes are smoothed distributions. Kruskal-Wallis across groups: "
                  f"H = {H:.2f}, p = {p:.3f}.")
    fig.savefig(path)
    plt.close(fig)


def fig3_over_time(leaves, groups, week_names, path):
    wk = weekly_plant_props(leaves).groupby(["group", "wk"]).prop.mean().mul(100).reset_index()
    weeks = sorted(week_names)
    fig, ax = plt.subplots(figsize=(11, 6.6))
    top = add_header(fig, "Symptom Progression Over Time by Treatment Group",
                     "Weekly mean proportion of symptomatic leaves across the plants in each group; "
                     f"W1 = {week_names[weeks[0]]} \u2026 W{weeks[-1]} = {week_names[weeks[-1]]}")
    ends = []
    for g in groups:
        s = wk[wk.group == g].dropna().sort_values("wk")
        if len(s):
            ax.plot(s.wk, s.prop, "-o", lw=2.8, ms=6, color=colour_of(g), clip_on=False)
            ends.append((g, s.wk.iloc[-1], s.prop.iloc[-1]))
    ymax = wk.prop.max() * 1.1
    ax.set_ylim(0, ymax)

    # Spread the end-of-line labels so they never overlap
    ends.sort(key=lambda e: e[2])
    gap, ypos = ymax * 0.055, []
    for k, (_, _, yv) in enumerate(ends):
        ypos.append(max(yv, gap * 0.6) if k == 0 else max(yv, ypos[-1] + gap))
    shift = max(0, ypos[-1] - ymax * 0.98) if ypos else 0
    for (g, xw, yv), yl in zip(ends, ypos):
        yl -= shift
        moved = abs(yl - yv) > gap * 0.3 or xw != weeks[-1]
        ax.annotate(g, xy=(xw, yv), xytext=(weeks[-1] + 0.18, yl), color=colour_of(g),
                    fontsize=TICK_PT, fontweight="bold", va="center",
                    arrowprops=dict(arrowstyle="-", color=colour_of(g), lw=0.8, alpha=0.6) if moved else None)

    ax.set_xticks(weeks, [f"W{w}" for w in weeks])
    ax.set_xlim(weeks[0] - 0.3, weeks[-1] + 1.25)
    ax.set_xlabel("Week")
    ax.set_ylabel("Symptomatic leaves (%)")
    ax.grid(axis="y", color="#e5e5e5", lw=0.8)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.09, right=0.97, top=top - 0.02, bottom=0.15)
    add_note(fig, f"{SOURCE} Plant-weeks with fewer than {MIN_ANN} leaf annotations are excluded.")
    fig.savefig(path)
    plt.close(fig)


def fig4_heatmap(leaves, groups, week_names, path):
    hm = weekly_plant_props(leaves).set_index(["group", "plant", "wk"]).prop.unstack("wk")
    hm = hm.reindex(columns=sorted(week_names))
    hm = hm.reindex(sorted(hm.index, key=lambda t: (groups.index(t[0]), t[1])))
    fig = plt.figure(figsize=(11, max(7, 0.2 * len(hm) + 3.2)))
    top = add_header(fig, "Individual Plant Symptom Trajectories Across Treatment Groups",
                     "Heatmap of symptom proportion (%) per individual plant over time "
                     "(white = missing/not photographed)")
    ax = fig.add_axes([0.2, 0.12, 0.62, top - 0.15])
    cax = fig.add_axes([0.85, 0.12, 0.022, top - 0.15])
    im = ax.imshow(hm.values * 100, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100, interpolation="nearest")
    ax.set_xticks(range(hm.shape[1]), [f"W{w}" for w in hm.columns])
    ax.set_yticks([])
    ax.set_xlabel("Week")
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-0.5, hm.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, len(hm)), minor=True)
    ax.grid(which="minor", color="white", lw=0.6)
    ax.tick_params(which="minor", length=0)
    pos = 0
    counts = hm.index.get_level_values(0).value_counts()
    for g in groups:
        n = int(counts.get(g, 0))
        if pos and n:
            ax.axhline(pos - 0.5, color="black", lw=1.4)
        if n:
            ax.text(-0.65, pos + n / 2 - 0.5, f"{g}\n({n} plants)", ha="right", va="center",
                    fontsize=TICK_PT, fontweight="bold")
        pos += n
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Symptomatic leaves (%)", fontsize=AXIS_PT)
    cb.outline.set_visible(False)
    add_note(fig, f"{SOURCE} Each row is one plant. Plant-weeks with fewer than {MIN_ANN} "
                  "leaf annotations are shown as missing.")
    fig.savefig(path)
    plt.close(fig)


def fig5_growth(plants, groups, period_text, path):
    measures = {"leaf_cnt": "Leaf count", "stem_length": "Plant height (cm)",
                "leaf_length": "Leaf length (cm)", "stem_thick": "Crown diameter (mm)"}
    measures = {k: v for k, v in measures.items() if plants[k].notna().sum() > 3}
    if not measures:
        return
    fig, axs = plt.subplots(1, len(measures), figsize=(3.3 * len(measures) + 1.6, 5.4), sharey=True)
    axs = np.atleast_1d(axs)
    top = add_header(fig, "Plant Growth Measurements and Symptom Proportion",
                     "Each dot is one plant; Spearman correlation between each measurement and the "
                     "proportion of symptomatic leaves")
    for ax, (col, label) in zip(axs, measures.items()):
        for g in groups:
            p = plants[plants.group == g]
            ax.scatter(p[col], p.prop * 100, s=45, color=colour_of(g), edgecolor="white", lw=0.6, label=g)
        ok = plants[[col, "prop"]].dropna()
        rho, pv = stats.spearmanr(ok[col], ok.prop)
        ax.set_title(f"\u03c1 = {rho:.2f}, p = {pv:.3f}", fontsize=TICK_PT, loc="left")
        ax.set_xlabel(label)
        ax.grid(color="#eeeeee", lw=0.8)
        ax.set_axisbelow(True)
    axs[0].set_ylabel("Symptomatic leaves per plant (%)")
    axs[-1].legend(frameon=False, fontsize=TICK_PT, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig.subplots_adjust(left=0.07, right=0.86, top=top - 0.06, bottom=0.18, wspace=0.12)
    add_note(fig, f"{SOURCE} {period_text}. Growth values are each plant's mean measurement. "
                  "Correlation does not imply causation.")
    fig.savefig(path)
    plt.close(fig)


def fig6_sensors(env, groups, period_text, path):
    cols = [c for c in SENSORS if c in env and env[c].notna().any()]
    if env.empty or not cols:
        return
    uniq = env.groupby("group")[cols].nunique().reindex(groups)
    fig, ax = plt.subplots(figsize=(10, 6))
    top = add_header(fig, "Sensor Readings Stored in the Label Files",
                     "Number of distinct values per sensor field; 1\u20132 values means the reading barely "
                     "changes and cannot explain differences between groups")
    ax.imshow(uniq.values, cmap="Blues", aspect="auto", vmin=0, vmax=max(5, np.nanmax(uniq.values)))
    for (i, j), v in np.ndenumerate(uniq.values):
        ax.text(j, i, "-" if pd.isna(v) else int(v), ha="center", va="center", fontsize=AXIS_PT,
                color="#1a1a1a")
    ax.set_xticks(range(len(cols)), [SENSORS[c] for c in cols])
    ax.set_yticks(range(len(groups)), groups)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.subplots_adjust(left=0.17, right=0.97, top=top - 0.03, bottom=0.12)
    add_note(fig, f"{SOURCE} {period_text}.\nAir temperature, humidity, CO2 and light fields are empty "
                  "and are not shown.")
    fig.savefig(path)
    plt.close(fig)


def kruskal(plants, groups):
    samples = [plants.loc[plants.group == g, "prop"].values for g in groups]
    samples = [s for s in samples if len(s) > 1]
    return stats.kruskal(*samples) if len(samples) > 1 else (np.nan, np.nan)


def holm(pvals):
    order, m, adj, run = np.argsort(pvals), len(pvals), np.empty(len(pvals)), 0
    for k, idx in enumerate(order):
        run = max(run, min(1, (m - k) * pvals[idx]))
        adj[idx] = run
    return adj


def write_stats(leaves, plants, groups, organ_table, period_text, path):
    ov = (leaves.groupby("group")
          .agg(plants=("plant", "nunique"), images=("image_file", "nunique"),
               leaf_annotations=("symptom", "size"), symptomatic=("symptom", "sum"))
          .reindex(groups))
    summ = plants.groupby("group").prop.agg(["count", "mean", "median", "std", "min", "max"]).reindex(groups)
    H, p = kruskal(plants, groups)
    lines = ["TIPBURN ANALYSIS - SUMMARY STATISTICS", f"Period: {period_text}", "",
             "Annotations by object type and label (only leaves are analysed):", organ_table.to_string(), "",
             "Data per group:", ov.to_string(), "",
             "Proportion of symptomatic leaves per plant (unit of analysis = plant):", summ.round(3).to_string(),
             "", f"Kruskal-Wallis across groups: H = {H:.2f}, p = {p:.4f}"]
    if "Normal" in groups:
        ref = plants.loc[plants.group == "Normal", "prop"]
        res = []
        for g in groups[1:]:
            a = plants.loc[plants.group == g, "prop"]
            if len(a) > 1 and len(ref) > 1:
                U, pv = stats.mannwhitneyu(a, ref, alternative="two-sided")
                res.append([g, len(a), round(a.mean(), 3), U, pv, round(2 * U / (len(a) * len(ref)) - 1, 3)])
        if res:
            r = pd.DataFrame(res, columns=["group", "n", "mean_prop", "U", "p", "rank_biserial"])
            r["p_holm"] = holm(r.p.values)
            lines += ["", "Each group vs Normal (Mann-Whitney U, Holm-corrected):",
                      r.to_string(index=False, formatters={"p": "{:.4f}".format, "p_holm": "{:.4f}".format})]
    lines += ["", "These are associations with recorded treatment groups; small numbers of plants per group "
                  "limit statistical power."]
    path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="Compare tipburn symptoms across treatment groups.")
    ap.add_argument("--labels", nargs="+", required=True, help="label zips or folders")
    ap.add_argument("--start", default="2022-11-01", help="first date analysed")
    ap.add_argument("--end", default="2022-12-07", help="last date analysed")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    rows, env = [], []
    for p in args.labels:
        r, e = read_labels(p)
        print(f"{Path(p).name}: {len(r):,} annotations")
        rows += r
        env += e
    df = pd.DataFrame(rows).drop_duplicates(["image_file", "leaf_idx", "bx", "by"])
    env = pd.DataFrame(env)

    # Keep only normal plants from the farm where the treatments were applied
    treat_farms = set(df.loc[df.group != "Normal", "farm_id"])
    df = df[(df.group != "Normal") | df.farm_id.isin(treat_farms)].copy()
    env = env[(env.group != "Normal") | env.farm_id.isin(treat_farms)]

    lo, hi = pd.Timestamp(args.start), pd.Timestamp(args.end)
    df["date"] = df.captured.dt.normalize()
    df = df[(df.date >= lo) & (df.date <= hi)].copy()
    env = env[(env.captured.dt.normalize() >= lo) & (env.captured.dt.normalize() <= hi)]
    if df.empty:
        raise SystemExit(f"No annotations between {args.start} and {args.end}.")
    df["plant"] = df.farm_id.astype(str) + "|" + df.crops_id.astype(str) + "|" + df.group
    organ_table = pd.crosstab([df.organ, df.status], df.group)

    # Leaves labelled Y (symptomatic) or N (healthy); stems, fruit, flowers and E labels are dropped
    leaves = df[(df.organ == "leaf") & df.status.isin(["Y", "N"])].copy()
    leaves["symptom"] = (leaves.status == "Y").astype(int)
    leaves["wk"] = (leaves.date - lo).dt.days // 7 + 1
    leaves.to_csv(OUT / "leaves.csv", index=False, encoding="utf-8-sig")

    groups = order_groups(leaves.group.unique())
    d0, d1 = leaves.date.min(), leaves.date.max()
    period_text = f"{d0.day} {d0:%B} \u2013 {d1.day} {d1:%B %Y}"
    week_names = {}
    for w, dates in leaves.groupby("wk").date:
        a = lo + pd.Timedelta(days=7 * (w - 1))
        b = min(a + pd.Timedelta(days=6), dates.max())
        name = f"{a.day}\u2013{b.day} {b:%b}" if a.month == b.month else f"{a.day} {a:%b} \u2013 {b.day} {b:%b}"
        week_names[int(w)] = name + (" (partial week)" if (b - a).days < 6 else "")

    pdays = (leaves.groupby(["group", "plant", "date"])
             .agg(leaves=("symptom", "size"), y=("symptom", "sum")).reset_index())
    pdays["prop"] = pdays.y / pdays.leaves
    pdays.to_csv(OUT / "plant_day.csv", index=False)
    plants = pdays.groupby(["group", "plant"]).prop.mean().reset_index()
    growth = leaves.groupby("plant")[["leaf_cnt", "stem_length", "leaf_length", "stem_thick"]].mean()
    plants = plants.join(growth, on="plant")
    plants.to_csv(OUT / "plant_summary.csv", index=False)

    fig1_overview(leaves, groups, period_text, OUT / "fig01_dataset_overview.png")
    fig2_symptoms_by_group(plants, groups, period_text, OUT / "fig02_symptoms_by_group.png")
    fig3_over_time(leaves, groups, week_names, OUT / "fig03_symptoms_over_time.png")
    fig4_heatmap(leaves, groups, week_names, OUT / "fig04_plant_heatmap.png")
    fig5_growth(plants, groups, period_text, OUT / "fig05_growth_vs_symptoms.png")
    fig6_sensors(env, groups, period_text, OUT / "fig06_sensor_check.png")
    write_stats(leaves, plants, groups, organ_table, period_text, OUT / "stats.txt")
    print(f"\nOutputs saved to {OUT.resolve()}")


if __name__ == "__main__":
    main()
