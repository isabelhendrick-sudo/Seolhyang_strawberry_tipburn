"""Stage 2: train a CPU-friendly SVM to recognise symptomatic leaves in the photos.

Run after analyse_tipburn.py:  python train_classifier.py --images data/normal_images data/tipburn_images
"""
import argparse
import io
import zipfile
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report, confusion_matrix,
                             precision_recall_curve, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from analyse_tipburn import (AXIS_PT, SOURCE, SUBTITLE_PT, TICK_PT, TITLE_PT, add_header, add_note,
                             colour_of, fix_name, order_groups)

OUT = Path("outputs")
SIZE = 128
SVM_COLOUR, LOGREG_COLOUR = "#1a1a1a", "#e6ab02"
HEALTHY_COLOUR, SYMPTOM_COLOUR = "#bdbdbd", "#1a1a1a"


class ImageStore:
    """Find a photo by file name inside the given zips or folders."""

    def __init__(self, sources):
        self.index, self.zips = {}, {}
        for src in map(Path, sources):
            if not src.exists():
                raise SystemExit(f"Path not found: {src.resolve()}")
            zips = [src] if src.is_file() else list(src.rglob("*.zip"))
            if src.is_dir():
                for f in src.rglob("*"):
                    if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        self.index[f.name] = ("dir", f)
            for zp in zips:
                z = zipfile.ZipFile(zp)
                self.zips[str(zp)] = z
                for info in z.infolist():
                    name = Path(fix_name(info)).name
                    if name.lower().endswith((".jpg", ".jpeg", ".png")):
                        self.index[name] = ("zip", (str(zp), info))
        print(f"Indexed {len(self.index):,} images")

    def load(self, name):
        kind, ref = self.index[name]
        data = ref.read_bytes() if kind == "dir" else self.zips[ref[0]].read(ref[1])
        return np.array(Image.open(io.BytesIO(data)).convert("RGB"))


def crop_leaf(img, row, pad=0.05):
    H, W = img.shape[:2]
    x0, y0 = int(max(0, row.bx - pad * row.bw)), int(max(0, row.by - pad * row.bh))
    x1, y1 = int(min(W, row.bx + row.bw * (1 + pad))), int(min(H, row.by + row.bh * (1 + pad)))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return cv2.resize(img[y0:y1, x0:x1], (SIZE, SIZE), interpolation=cv2.INTER_AREA)


def features(crop):
    """Colour histograms, colour statistics, brown/green pixel shares and HOG texture."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
    f = []
    for ch, bins, rng in [(0, 18, (0, 180)), (1, 8, (0, 256)), (2, 8, (0, 256))]:
        hist = cv2.calcHist([hsv], [ch], None, [bins], rng).ravel()
        f += list(hist / hist.sum())
    for arr in (hsv, lab):
        for i in range(3):
            f += [arr[..., i].mean(), arr[..., i].std()]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    brown = (h >= 5) & (h <= 25) & (s > 60) & (v > 30) & (v < 200)
    green = (h > 30) & (h < 90) & (s > 40)
    edge = np.ones_like(brown)
    m = SIZE // 6
    edge[m:-m, m:-m] = False
    f += [brown.mean(), green.mean(), brown[edge].mean(), brown[~edge].mean(),
          brown.sum() / max((brown | green).sum(), 1)]
    grey = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), (64, 64))
    f += list(hog(grey, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2)))
    return np.asarray(f, np.float32)


# Column positions of the colour features plotted in fig08 (after 34 histogram bins and 12 statistics)
GREEN_ALL, BROWN_EDGE, BROWN_ALL = 47, 48, 50


def fig_example_leaves(data, crops, y, groups, path, rng):
    fig, axs = plt.subplots(len(groups), 8, figsize=(12, 1.55 * len(groups) + 1.6), squeeze=False)
    top = add_header(fig, "Examples of Expert-Labelled Leaves by Treatment Group",
                     "Four randomly chosen symptomatic and healthy leaf crops per group, used to check "
                     "that the labels match tipburn")
    for r, g in enumerate(groups):
        for c, lab in enumerate([1] * 4 + [0] * 4):
            ax = axs[r, c]
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            idx = np.where((data.group == g) & (y == lab))[0]
            if len(idx):
                ax.imshow(crops[rng.choice(idx)])
            if r == 0:
                ax.set_title("Symptomatic" if lab else "Healthy", fontsize=TICK_PT, fontweight="bold")
        axs[r, 0].set_ylabel(g, rotation=0, ha="right", va="center", fontsize=TICK_PT,
                             fontweight="bold", color=colour_of(g))
    fig.subplots_adjust(left=0.13, right=0.98, top=top - 0.02, bottom=0.06, wspace=0.08, hspace=0.12)
    add_note(fig, f"{SOURCE} Crops are taken from each leaf's bounding box and resized to 64 x 64 pixels.")
    fig.savefig(path)
    plt.close(fig)


def fig_colour_features(X, y, path):
    panels = [(BROWN_EDGE, "Brown pixels near the leaf edge"), (BROWN_ALL, "Brown share of plant pixels"),
              (GREEN_ALL, "Green pixels")]
    fig, axs = plt.subplots(1, 3, figsize=(12, 5.4))
    top = add_header(fig, "Colour Features of Symptomatic and Healthy Leaves",
                     "Distribution of three simple colour measurements per leaf crop; boxes show the middle 50% "
                     "of leaves and the line the median")
    for ax, (col, label) in zip(axs, panels):
        bp = ax.boxplot([X[y == 0, col] * 100, X[y == 1, col] * 100], showfliers=False, widths=0.55,
                        patch_artist=True)
        for patch, med, colr, mcol in zip(bp["boxes"], bp["medians"], (HEALTHY_COLOUR, SYMPTOM_COLOUR),
                                          ("#1a1a1a", "white")):
            patch.set(facecolor=colr, edgecolor=colr)
            med.set(color=mcol, lw=2)
        ax.set_xticks([1, 2], ["Healthy", "Symptomatic"])
        ax.set_ylabel("Share of pixels (%)")
        ax.set_title(label, loc="left", fontsize=AXIS_PT, fontweight="bold")
        ax.grid(axis="y", color="#eeeeee", lw=0.8)
        ax.set_axisbelow(True)
    fig.subplots_adjust(left=0.07, right=0.98, top=top - 0.06, bottom=0.12, wspace=0.35)
    add_note(fig, f"{SOURCE} Pixel shares are measured on each 128 x 128 leaf crop.")
    fig.savefig(path)
    plt.close(fig)


def fig_performance(y, svm_pred, scores, n_plants, date_text, path):
    cm = confusion_matrix(y, svm_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    prec, rec, prev = tp / max(tp + fp, 1), tp / max(tp + fn, 1), y.mean()
    fig = plt.figure(figsize=(13, 6.6))
    top = add_header(fig, "Tipburn Detection Performance of the Leaf Image Classifier",
                     f"Leaves the SVM flags as symptomatic are correct {prec:.0%} of the time (precision), "
                     f"but it finds only {rec:.0%} of symptomatic leaves (recall).\n"
                     "Lowering the decision threshold finds more symptomatic leaves at the cost of more false alarms.")
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.25], left=0.12, right=0.97,
                  top=top - 0.06, bottom=0.17, wspace=0.42)

    ax = fig.add_subplot(gs[0, 0])
    share = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    names = [["Correctly cleared", "False alarm"], ["Missed", "Correctly detected"]]
    for i in range(2):
        for j in range(2):
            col = "white" if share[i, j] > 0.55 else "#1a1a1a"
            ax.text(j, i - 0.17, f"{cm[i, j]:,}", ha="center", va="center", fontsize=TITLE_PT,
                    fontweight="bold", color=col)
            ax.text(j, i + 0.08, f"{share[i, j]:.0%} of {'healthy' if i == 0 else 'symptomatic'}",
                    ha="center", va="center", fontsize=TICK_PT, color=col)
            ax.text(j, i + 0.27, names[i][j], ha="center", va="center", fontsize=SUBTITLE_PT,
                    color=col, style="italic")
    ax.set_xticks([0, 1], ["Healthy", "Symptomatic"])
    ax.set_yticks([0, 1], ["Healthy", "Symptomatic"])
    ax.set_xlabel("Predicted by the SVM")
    ax.set_ylabel("Expert label")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("A   Predictions at the default threshold", loc="left", fontsize=AXIS_PT,
                 fontweight="bold", pad=12)

    ax = fig.add_subplot(gs[0, 1])
    styles = {"SVM": (SVM_COLOUR, 2.6), "Logistic regression": (LOGREG_COLOUR, 2.0)}
    for name, sc in scores.items():
        colr, lw = styles[name]
        pr, rc, _ = precision_recall_curve(y, sc)
        ax.plot(rc, pr, color=colr, lw=lw, label=f"{name} (AP = {average_precision_score(y, sc):.2f})")
    ax.axhline(prev, ls="--", color="#999999", lw=1.2, label=f"Chance ({prev:.0%} symptomatic)")

    # Panel A is the filled point on the SVM curve; the open point shows a lower threshold
    pr, rc, _ = precision_recall_curve(y, scores["SVM"])
    ax.scatter([rec], [prec], s=120, color=SVM_COLOUR, edgecolor="white", lw=1.5, zorder=5)
    ax.annotate(f"Default threshold (panel A)\nprecision {prec:.0%}, recall {rec:.0%}", xy=(rec, prec),
                xytext=(rec, max(prec - 0.26, prev + 0.06)), fontsize=TICK_PT, ha="center",
                arrowprops=dict(arrowstyle="-", color="#444444", lw=1))
    k = int(np.argmin(np.abs(rc - min(0.8, max(rec + 0.3, 0.6)))))
    ax.scatter([rc[k]], [pr[k]], s=90, facecolor="white", edgecolor=SVM_COLOUR, lw=2, zorder=5)
    ax.annotate(f"Lower threshold\nrecall {rc[k]:.0%}, precision {pr[k]:.0%}", xy=(rc[k], pr[k]),
                xytext=(min(rc[k], 0.85), min(pr[k] + 0.2, 0.95)), fontsize=TICK_PT, ha="center",
                arrowprops=dict(arrowstyle="-", color="#444444", lw=1))
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Recall (share of symptomatic leaves found)")
    ax.set_ylabel("Precision (share of flagged leaves\nthat are truly symptomatic)")
    ax.grid(color="#e5e5e5", lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", frameon=False, fontsize=TICK_PT)
    ax.set_title("B   Precision\u2013recall trade-off", loc="left", fontsize=AXIS_PT, fontweight="bold", pad=12)
    add_note(fig, f"{SOURCE} {len(y):,} leaves from {n_plants} plants, {date_text}. 5-fold cross-validation with "
                  "each plant kept entirely in either training or test data.\nSymptomatic leaves make up "
                  f"{prev:.0%} of the sample, so guessing at random would give {prev:.0%} precision. "
                  "AP = average precision.")
    fig.savefig(path)
    plt.close(fig)


def fig_by_group(per, path):
    fig, ax = plt.subplots(figsize=(11, 6.4))
    top = add_header(fig, "Classifier Performance Within Each Treatment Group",
                     "Precision, recall and F1 score of the SVM for the leaves of each group; groups with few "
                     "symptomatic leaves give unreliable scores")
    marks = {"precision": ("o", "#1a1a1a"), "recall": ("s", "#8c8c8c"), "f1": ("D", "#e6ab02")}
    ys = np.arange(len(per))
    for metric, (marker, colr) in marks.items():
        ax.scatter(per[metric], ys, marker=marker, s=90, color=colr, edgecolor="white", zorder=3,
                   label="F1 score" if metric == "f1" else metric.capitalize())
    for yv, row in zip(ys, per.itertuples()):
        vals = [row.precision, row.recall, row.f1]
        ax.plot([min(vals), max(vals)], [yv, yv], color="#dddddd", lw=3, zorder=1)
    ax.set_yticks(ys, [f"{r.group}\n({r.symptomatic} symptomatic leaves)" for r in per.itertuples()])
    for lbl, r in zip(ax.get_yticklabels(), per.itertuples()):
        lbl.set_color(colour_of(r.group) if r.symptomatic >= 20 else "#aaaaaa")
    ax.set_ylim(len(per) - 0.5, -0.5)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score (0 = worst, 1 = best)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color="#e5e5e5", lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, fontsize=TICK_PT, ncol=3)
    fig.subplots_adjust(left=0.24, right=0.97, top=top - 0.02, bottom=0.14)
    add_note(fig, f"{SOURCE} Groups in grey have fewer than 20 symptomatic leaves in the sample.")
    fig.savefig(path)
    plt.close(fig)


def fig_errors(crops, y, pred, path, rng):
    fp = np.where((pred == 1) & (y == 0))[0]
    fn = np.where((pred == 0) & (y == 1))[0]
    fig, axs = plt.subplots(2, 8, figsize=(12, 4.6))
    top = add_header(fig, "Examples of Classification Errors",
                     "Leaves the SVM got wrong, chosen at random")
    for r, (idx, label) in enumerate([(fp, "False alarm: healthy leaf flagged as symptomatic"),
                                      (fn, "Missed: symptomatic leaf classed as healthy")]):
        pick = rng.choice(idx, min(8, len(idx)), replace=False) if len(idx) else []
        for c in range(8):
            axs[r, c].axis("off")
            if c < len(pick):
                axs[r, c].imshow(crops[pick[c]])
        axs[r, 0].set_title(label, loc="left", fontsize=TICK_PT, fontweight="bold")
    fig.subplots_adjust(left=0.03, right=0.97, top=top - 0.06, bottom=0.08, hspace=0.35, wspace=0.08)
    add_note(fig, SOURCE)
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Train an SVM to recognise symptomatic leaves.")
    ap.add_argument("--images", nargs="+", required=True, help="image zips or folders")
    ap.add_argument("--neg-per-pos", type=float, default=3.0, help="healthy leaves per symptomatic leaf")
    ap.add_argument("--max-leaves", type=int, default=6000, help="maximum leaves used")
    ap.add_argument("--min-side", type=float, default=30, help="smallest leaf box in pixels")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    leaves = pd.read_csv(OUT / "leaves.csv", encoding="utf-8-sig", parse_dates=["date"])
    leaves = leaves[(leaves.bw >= args.min_side) & (leaves.bh >= args.min_side)]
    store = ImageStore(args.images)
    leaves = leaves[leaves.image_file.isin(store.index.keys())]
    if leaves.symptom.sum() == 0:
        raise SystemExit("No symptomatic leaves have a matching photo; check the --images paths.")

    # Balance the sample: every symptomatic leaf (up to the cap) plus a fixed number of healthy ones
    pos, neg = leaves[leaves.symptom == 1], leaves[leaves.symptom == 0]
    n_pos = min(len(pos), int(args.max_leaves / (1 + args.neg_per_pos)))
    sample = pd.concat([pos.sample(n_pos, random_state=args.seed),
                        neg.sample(min(len(neg), int(n_pos * args.neg_per_pos)), random_state=args.seed)])
    sample = sample.reset_index(drop=True)
    print(f"Using {n_pos:,} symptomatic and {len(sample) - n_pos:,} healthy leaves")

    cache = OUT / "features_cache.npz"
    key = np.array(sample.image_file + "#" + sample.leaf_idx.astype(str))
    if cache.exists() and np.array_equal(np.load(cache, allow_pickle=True)["key"], key):
        c = np.load(cache, allow_pickle=True)
        X, keep, crops = c["X"], c["keep"], c["crops"]
        print("Loaded cached features")
    else:
        X, keep, crops = [], [], []
        for i, (fname, rows) in enumerate(sample.groupby("image_file")):
            if i % 200 == 0:
                print(f"  {i:,}/{sample.image_file.nunique():,} images processed")
            img = store.load(fname)
            for idx, row in rows.iterrows():
                crop = crop_leaf(img, row)
                if crop is not None:
                    X.append(features(crop))
                    keep.append(idx)
                    crops.append(cv2.resize(crop, (64, 64)))
        X, keep, crops = np.vstack(X), np.array(keep), np.stack(crops)
        np.savez_compressed(cache, X=X, keep=keep, crops=crops, key=key)
    data = sample.loc[keep].reset_index(drop=True)
    y, plants = data.symptom.values, data.plant.values
    groups = order_groups(data.group.unique())

    fig_example_leaves(data, crops, y, groups, OUT / "fig07_example_leaves.png", rng)
    fig_colour_features(X, y, OUT / "fig08_colour_features.png")

    # Each plant stays entirely in training or test data in every fold
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    models = {"Baseline (always healthy)": DummyClassifier(strategy="most_frequent"),
              "Logistic regression": make_pipeline(StandardScaler(),
                                                   LogisticRegression(max_iter=2000, class_weight="balanced")),
              "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale",
                                                         class_weight="balanced"))}
    results, scores, preds = [], {}, {}
    for name, model in models.items():
        print(f"Cross-validating {name} ...")
        preds[name] = cross_val_predict(model, X, y, groups=plants, cv=cv, n_jobs=-1)
        if name != "Baseline (always healthy)":
            scores[name] = cross_val_predict(model, X, y, groups=plants, cv=cv, n_jobs=-1,
                                             method="decision_function")
        p, r, f, _ = precision_recall_fscore_support(y, preds[name], average="binary", zero_division=0)
        results.append(dict(model=name, accuracy=(preds[name] == y).mean(), precision=p, recall=r, f1=f))
    svm_pred = preds["SVM"]

    d0, d1 = data.date.min(), data.date.max()
    date_text = f"{d0.day} {d0:%B} \u2013 {d1.day} {d1:%B %Y}"
    fig_performance(y, svm_pred, scores, data.plant.nunique(), date_text, OUT / "fig09_classifier_performance.png")

    per = []
    for g in groups:
        m = (data.group == g).values
        p, r, f, _ = precision_recall_fscore_support(y[m], svm_pred[m], average="binary", zero_division=0)
        per.append(dict(group=g, leaves=int(m.sum()), symptomatic=int(y[m].sum()),
                        accuracy=(svm_pred[m] == y[m]).mean(), precision=p, recall=r, f1=f))
    per = pd.DataFrame(per)
    fig_by_group(per, OUT / "fig10_performance_by_group.png")
    fig_errors(crops, y, svm_pred, OUT / "fig11_error_examples.png", rng)

    final = models["SVM"].fit(X, y)
    joblib.dump({"model": final, "crop_size": SIZE}, OUT / "tipburn_svm.joblib")
    report = classification_report(y, svm_pred, target_names=["Healthy", "Symptomatic"], digits=3)
    text = ["CLASSIFIER RESULTS (5-fold cross-validation, grouped by plant)", f"Photos: {date_text}",
            f"{len(data):,} leaves from {data.plant.nunique()} plants, {y.mean():.1%} symptomatic", "",
            pd.DataFrame(results).round(3).to_string(index=False), "", "SVM classification report:", report,
            "SVM performance per group:", per.round(3).to_string(index=False)]
    (OUT / "classifier_results.txt").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(text))
    print(f"\nOutputs saved to {OUT.resolve()}")


if __name__ == "__main__":
    main()
