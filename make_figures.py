"""Generate figures + summary tables from pilot result JSONs (final version).
Run locally: python make_figures.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R1 = os.path.join(HERE, "..", "results", "pilot_v1")
R2 = os.path.join(HERE, "..", "results", "pilot_v2")
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.dpi": 150})


def j(*p):
    return os.path.join(*p)


def load(d, name):
    with open(j(d, name), encoding="utf-8") as fh:
        return json.load(fh)


# ---------- Fig 1: RGB baseline vs v1 pooled vs v2 injected (5-shot) ----------
fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
panels = [
    (axes[0], "Pavia University", 3.38, 33.52, 50.82),
    (axes[1], "Indian Pines", 2.59, None, 50.05),
]
for ax, title, rgb, v1, v2 in panels:
    vals, labels, colors = [rgb], ["RGB-direct\nCLIP"], ["#999999"]
    if v1 is not None:
        vals.append(v1); labels.append("v1 pooled\nadapter"); colors.append("#5b8db8")
    vals.append(v2); labels.append("v2 injected\nadapter"); colors.append("#c44e52")
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}", ha="center", fontsize=10)
    ax.set_ylabel("mIoU (%)")
    ax.set_title(f"{title} (5 labelled px/class)")
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(j(FIG, "fig1_fewshot_vs_baselines.png"), bbox_inches="tight")
plt.close()

# ---------- Fig 2: label scaling ----------
fig, ax = plt.subplots(figsize=(5.6, 3.8))
shots = [5, 15, 50, 42776]
oa_v1 = [48.9, 68.6, 76.4, 78.6]
miou_v1 = [33.5, None, None, 60.6]
ax.semilogx(shots, oa_v1, "o-", color="#5b8db8", label="v1 pooled adapter (OA)")
ax.plot([5], [69.3], "s", color="#c44e52", label="v2 injected, 5-shot (OA)")
ax.plot([5], [50.8], "D", color="#c44e52", alpha=0.55, label="v2 injected, 5-shot (mIoU)")
ax.plot([42776], [60.6], "D", color="#5b8db8", alpha=0.55, label="v1 full supervision (mIoU)")
ax.axhline(3.4, ls=":", color="#777", label="RGB-direct CLIP zero-shot (OA 9.3)")
ax.set_xlabel("labelled pixels (total, log scale)")
ax.set_ylabel("%")
ax.legend(fontsize=8)
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("Label efficiency (Pavia University)")
plt.tight_layout()
plt.savefig(j(FIG, "fig2_label_scaling.png"), bbox_inches="tight")
plt.close()

# ---------- Fig 3: open-vocabulary base/novel analysis ----------
fig, ax = plt.subplots(figsize=(7.4, 3.9))
groups = ["RGB zero-shot\n+balancing", "adapter 5-shot\nprior-fuse +bal", "adapter 5-shot\nprior-KL +bal", "adapter 50-shot\nprior-fuse +bal"]
base_v = [7.2, 42.8, 18.7, 59.2]
novel_v = [3.4, 2.4, 3.1, 7.7]
x = np.arange(len(groups))
ax.bar(x - 0.19, base_v, 0.38, label="base classes", color="#5b8db8")
ax.bar(x + 0.19, novel_v, 0.38, label="novel classes", color="#c44e52")
for xi, v in zip(x - 0.19, base_v):
    ax.text(xi, v + 0.6, f"{v:.1f}", ha="center", fontsize=9)
for xi, v in zip(x + 0.19, novel_v):
    ax.text(xi, v + 0.6, f"{v:.1f}", ha="center", fontsize=9)
ax.set_xticks(x, groups, fontsize=8.5)
ax.set_ylabel("mIoU (%)")
ax.set_title("Open-vocabulary base/novel split (Pavia University)")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(j(FIG, "fig3_base_novel.png"), bbox_inches="tight")
plt.close()

# ---------- Fig 4: prediction maps ----------
gt = None
import scipy.io as sio
gt = sio.loadmat(j(HERE, "..", "data", "PaviaU_gt.mat"))["paviaU_gt"]
pred_v2 = np.load(j(R2, "pred_map_v2_paviau_s5_seed0.npy"))
pred_rgb = np.load(j(R2, "pred_map_rgb_paviau.npy"))
palette = plt.get_cmap("tab10")
fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
for ax, m, t in [(axes[0], gt - 1, "Ground truth"),
                 (axes[1], pred_rgb, "RGB-direct CLIP (mIoU 3.4%)"),
                 (axes[2], pred_v2, "v2 injected adapter (mIoU 50.8%, 45 labels)")]:
    mm = np.where(gt > 0, m.astype(np.int16), np.int16(-1))
    ax.imshow(np.ma.masked_less(mm, 0), cmap=palette, vmin=0, vmax=9, interpolation="nearest")
    ax.set_title(t, fontsize=9.5)
    ax.axis("off")
plt.tight_layout()
plt.savefig(j(FIG, "fig4_prediction_maps.png"), bbox_inches="tight")
plt.close()


# ---------- Fig 5: M4 tokenize mechanism ablation ----------
RI = os.path.join(HERE, "..", "results", "innovation")
tok_rows = []
for tok in ["wl100", "wl50", "wl25", "bandeq9", "bandindex"]:
    f_in = j(RI, f"train_metrics_v2_paviau_s5_inj{'_' + tok if tok != 'wl50' else ''}_seed0.json")
    f_zs = j(RI, f"train_metrics_zs_v2_paviau_s5_inj{'_' + tok if tok != 'wl50' else ''}_seed0_to_indianpines.json")
    if not os.path.exists(f_in):
        continue
    d_in = json.load(open(f_in, encoding="utf-8"))
    d_zs = json.load(open(f_zs, encoding="utf-8")) if os.path.exists(f_zs) else {}
    rgb_ip = json.load(open(j(R1, "rgb_baseline_indianpines.json"), encoding="utf-8"))
    tok_rows.append((tok, 100 * d_in["mIoU"],
                     100 * d_zs.get("mRecall", 0) / max(100 * rgb_ip["mRecall"], 1e-9)))
if tok_rows:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
    names = [r[0] for r in tok_rows]
    axes[0].bar(names, [r[1] for r in tok_rows], color="#5b8db8")
    axes[0].set_ylabel("PaviaU 5-shot mIoU (%)")
    axes[0].set_title("In-domain")
    axes[1].bar(names, [r[2] for r in tok_rows], color="#c44e52")
    axes[1].axhline(1.0, ls=":", color="#777")
    axes[1].set_ylabel("IP zero-shot mRecall / RGB baseline")
    axes[1].set_title("Cross-sensor transfer (ratio to RGB-direct)")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("M4: spectral-axis tokenization mechanism", y=1.02)
    plt.tight_layout()
    plt.savefig(j(FIG, "fig5_tokenize_ablation.png"), bbox_inches="tight")
    plt.close()

# ---------- Fig 3b: TSC points into base/novel chart ----------
# ---------- summary tables ----------
rows = []
for dname, d in [("pilot_v1", R1), ("pilot_v2", R2)]:
    for f in sorted(os.listdir(d)):
        if f.endswith(".json"):
            rows.append((dname, f, load(d, f)))
lines = ["# Results summary (auto-generated by make_figures.py)\n"]
for dname, f, d in rows:
    keys = [k for k in ("tag", "scene", "mIoU", "OA", "mRecall", "base_mIoU", "novel_mIoU", "hMIoU",
                        "s_residual", "shots", "trainable_params_M", "train_time_s", "peak_mem_GB",
                        "ratio_mRecall", "eval_time_s") if k in d]
    lines.append(f"## [{dname}] {f}\n")
    lines.append("| field | value |")
    lines.append("|---|---|")
    for k in keys:
        v = d[k]
        lines.append(f"| {k} | {v if not isinstance(v, float) else round(v, 4)} |")
    lines.append("")
with open(j(HERE, "..", "results", "summary_tables.md"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
print("OK: figures ->", FIG)
