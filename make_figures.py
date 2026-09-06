"""Generate figures + summary tables from pilot result JSONs (paper-final version).

Numbering matches the manuscript (论文_终稿.md):
  fig1_fewshot_vs_baselines.png   -> 图1  封闭集few-shot主结果
  fig2_prediction_maps.png        -> 图2  Pavia U 5-shot定性预测
  fig3_tokenize_ablation.png      -> 图3  tokenize消融
  fig4_transfer_mechanism.png     -> 图4  2x2因子分解
  fig5_base_novel.png             -> 图5  base-novel坍缩与TSC
  fig6_label_scaling.png          -> 图6  标注预算-精度前沿

Run locally: python make_figures.py
No internal milestone codenames in figure titles; Chinese captions.
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
RI = os.path.join(HERE, "..", "results", "innovation")
FIG = os.path.join(HERE, "..", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({
    "font.size": 11, "figure.dpi": 150,
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
})


def j(*p):
    return os.path.join(*p)


def load(d, name):
    with open(j(d, name), encoding="utf-8") as fh:
        return json.load(fh)


def pct(x):
    return 100.0 * x


# ---------- 图1: 封闭集few-shot主结果（RGB基线 / 光谱1-NN / 池化式 / 注入式） ----------
one_nn = load(os.path.join(HERE, "..", "results"), "spectral_1nn_baseline_seed0.json")
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
panels = [
    (axes[0], "Pavia University", 3.38, one_nn["paviau"]["mIoU"], 33.52, 50.82),
    (axes[1], "Indian Pines", 2.59, one_nn["indianpines"]["mIoU"], None, 50.05),
]
for ax, title, rgb, knn_v, v1, v2 in panels:
    vals, labels, colors = [rgb], ["RGB直用\nCLIP"], ["#999999"]
    vals.append(knn_v); labels.append("光谱1-NN\n(零训练)"); colors.append("#8c8c3f")
    if v1 is not None:
        vals.append(v1); labels.append("池化式\n适配器"); colors.append("#5b8db8")
    vals.append(v2); labels.append("注入式适配器\n(本文)"); colors.append("#c44e52")
    bars = ax.bar(labels, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.8, f"{v:.1f}", ha="center", fontsize=10)
    ax.set_ylabel("mIoU (%)")
    ax.set_title(f"{title}（每类5个标注像素，无转导）")
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(j(FIG, "fig1_fewshot_vs_baselines.png"), bbox_inches="tight")
plt.close()

# ---------- 图2: 定性预测图 ----------
import scipy.io as sio
gt = sio.loadmat(j(HERE, "..", "data", "PaviaU_gt.mat"))["paviaU_gt"]
pred_v2 = np.load(j(R2, "pred_map_v2_paviau_s5_seed0.npy"))
pred_rgb = np.load(j(R2, "pred_map_rgb_paviau.npy"))
palette = plt.get_cmap("tab10")
fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
for ax, m, t in [(axes[0], gt - 1, "真值"),
                 (axes[1], pred_rgb, "RGB直用CLIP（mIoU 3.4%）"),
                 (axes[2], pred_v2, "注入式适配器（mIoU 50.8%，45个标注像素）")]:
    mm = np.where(gt > 0, m.astype(np.int16), np.int16(-1))
    ax.imshow(np.ma.masked_less(mm, 0), cmap=palette, vmin=0, vmax=9, interpolation="nearest")
    ax.set_title(t, fontsize=9.5)
    ax.axis("off")
plt.tight_layout()
plt.savefig(j(FIG, "fig2_prediction_maps.png"), bbox_inches="tight")
plt.close()

# ---------- 图3: tokenize消融（域内mIoU vs 跨传感器mRecall比） ----------
tok_rows = []
rgb_ip = load(R1, "rgb_baseline_indianpines.json")
rgb_ip_mr = pct(rgb_ip["mRecall"])
for tok, label in [("wl100", "wl100\n(5 token)"), ("wl50", "wl50\n(9 token, 默认)"),
                   ("wl25", "wl25\n(18 token)"), ("bandeq9", "bandeq9\n(9 token, 无对齐)"),
                   ("bandindex", "bandindex\n(16 token, 无对齐)")]:
    f_in = j(RI, f"train_metrics_v2_paviau_s5_inj{'_' + tok if tok != 'wl50' else ''}_seed0.json")
    f_zs = j(RI, f"train_metrics_zs_v2_paviau_s5_inj{'_' + tok if tok != 'wl50' else ''}_seed0_to_indianpines.json")
    if not os.path.exists(f_in):
        continue
    d_in = json.load(open(f_in, encoding="utf-8"))
    d_zs = json.load(open(f_zs, encoding="utf-8")) if os.path.exists(f_zs) else {}
    tok_rows.append((label, pct(d_in["mIoU"]), pct(d_zs.get("mRecall", 0)) / rgb_ip_mr))
fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.8))
names = [r[0] for r in tok_rows]
axes[0].bar(names, [r[1] for r in tok_rows], color="#5b8db8")
axes[0].set_ylabel("域内 mIoU (%)（Pavia U 5-shot）")
axes[0].set_title("域内判别力")
axes[1].bar(names, [r[2] for r in tok_rows], color="#c44e52")
axes[1].axhline(1.0, ls=":", color="#777")
axes[1].set_ylabel("跨传感器零样本 mRecall比\n（相对IP的RGB基线7.57%）")
axes[1].set_title("跨传感器迁移")
for ax in axes:
    ax.tick_params(axis="x", labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(j(FIG, "fig3_tokenize_ablation.png"), bbox_inches="tight")
plt.close()

# ---------- 图4: 2x2因子分解（tokenize x 内核，跨传感器mRecall比） ----------
cells = {
    ("物理波长对齐\n(wl50)", "Transformer"): 1.75,
    ("物理波长对齐\n(wl50)", "MLP"): 1.21,
    ("序号patch\n(bandindex)", "Transformer"): 1.27,
    ("序号patch\n(bandindex)", "MLP"): None,          # 未测
}
fig, ax = plt.subplots(figsize=(5.8, 3.9))
xs = np.arange(2)
w = 0.36
for xi, (xtok, color) in enumerate(zip(["物理波长对齐\n(wl50)", "序号patch\n(bandindex)"], ["#c44e52", "#5b8db8"])):
    ts_v = cells[(xtok, "Transformer")]
    mlp_v = cells[(xtok, "MLP")]
    ax.bar(xi - w / 2, ts_v, w, color=color, label="Transformer内核" if xi == 0 else None)
    ax.text(xi - w / 2, ts_v + 0.03, f"{ts_v:.2f}", ha="center", fontsize=10)
    if mlp_v is not None:
        ax.bar(xi + w / 2, mlp_v, w, color=color, alpha=0.55, label="MLP内核" if xi == 0 else None)
        ax.text(xi + w / 2, mlp_v + 0.03, f"{mlp_v:.2f}", ha="center", fontsize=10)
    else:
        ax.bar(xi + w / 2, 0.02, w, color="#bbbbbb", hatch="//", fill=False)
        ax.text(xi + w / 2, 0.10, "未测", ha="center", fontsize=10)
ax.axhline(1.0, ls=":", color="#777")
ax.set_xticks(xs, ["物理波长对齐 (wl50)", "序号patch (bandindex)"])
ax.set_ylabel("跨传感器零样本 mRecall比")
ax.set_title("tokenize × 内核 2×2 因子分解（单种子）")
ax.set_ylim(0, 2.0)
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(j(FIG, "fig4_transfer_mechanism.png"), bbox_inches="tight")
plt.close()

# ---------- 图5: base-novel坍缩、常规手段与TSC（全部含/不含转导逐行标注见论文表6） ----------
d_tsc05 = load(RI, "train_metrics_v2_paviau_s5_prior_base0-1-2-3-5_tsc0.5_seed0.json")
d_tsc20 = load(RI, "train_metrics_v2_paviau_s5_prior_base0-1-2-3-5_tsc2.0_seed0.json")
d_s50 = load(R2, "train_metrics_v2_paviau_s50_prior_base0-1-2-3-5_seed0.json")
d_s50_tsc = load(RI, "train_metrics_v2_paviau_s50_prior_base0-1-2-3-5_tsc1.0_seed0.json")
groups = ["RGB直用\n+均衡", "纯注入\n(无缓解, 无转导)", "先验融合\n+均衡", "先验KL\n+均衡",
          "TSC β=0.5\n+均衡", "TSC β=2.0\n+均衡", "50-shot先验融合\n+均衡", "50-shot TSC β=1\n+均衡"]
base_v = [7.2, 50.0, 42.8, 18.7, pct(d_tsc05["base_mIoU"]), pct(d_tsc20["base_mIoU"]),
          pct(d_s50["base_mIoU"]), pct(d_s50_tsc["base_mIoU"])]
novel_v = [3.4, 0.0, 2.4, 3.1, pct(d_tsc05["novel_mIoU"]), pct(d_tsc20["novel_mIoU"]),
           pct(d_s50["novel_mIoU"]), pct(d_s50_tsc["novel_mIoU"])]
x = np.arange(len(groups))
fig, ax = plt.subplots(figsize=(8.6, 4.1))
ax.bar(x - 0.19, base_v, 0.38, label="base类 mIoU", color="#5b8db8")
ax.bar(x + 0.19, novel_v, 0.38, label="novel类 mIoU", color="#c44e52")
for xi, v in zip(x - 0.19, base_v):
    ax.text(xi, v + 0.6, f"{v:.1f}", ha="center", fontsize=8.5)
for xi, v in zip(x + 0.19, novel_v):
    ax.text(xi, v + 0.6, f"{v:.2f}".rstrip("0").rstrip("."), ha="center", fontsize=8.5)
ax.set_xticks(x, groups, fontsize=8)
ax.set_ylabel("mIoU (%)")
ax.set_title("同场景base-novel：坍缩、常规先验保持手段与TSC（Pavia U，单种子）")
ax.legend()
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(j(FIG, "fig5_base_novel.png"), bbox_inches="tight")
plt.close()

# ---------- 图6: 标注预算-精度前沿（协议口径逐点标注） ----------
fig, ax = plt.subplots(figsize=(6.4, 4.2))
shots = [5, 15, 50, 42776]
oa_v1 = [48.9, 68.6, 76.4, 78.6]
ax.semilogx(shots, oa_v1, "o-", color="#5b8db8",
            label="池化式适配器（OA；全监督点78.6为train-inclusive口径）")
ax.plot([5], [69.3], "s", color="#c44e52", label="注入式适配器 5-shot（OA 69.3，无转导）")
ax.plot([5], [50.8], "D", color="#c44e52", alpha=0.55, label="注入式适配器 5-shot（mIoU 50.8）")
ax.plot([21388], [99.98], "^", color="#c44e52",
        label="注入式适配器 全监督（OA 99.98，50/50 held-out，seed 123）")
ax.axhline(3.4, ls=":", color="#777", label="RGB直用CLIP零样本（mIoU 3.4，无转导）")
ax.set_xlabel("标注像素总数（对数刻度）")
ax.set_ylabel("%")
ax.legend(fontsize=7.5, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
ax.set_title("标注预算-精度前沿（Pavia University，单种子）")
plt.tight_layout()
plt.savefig(j(FIG, "fig6_label_scaling.png"), bbox_inches="tight")
plt.close()

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
