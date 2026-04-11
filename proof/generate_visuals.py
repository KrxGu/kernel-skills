"""
proof/generate_visuals.py

Generates three publication-quality proof visuals for the softmax skill benchmark:
  1. hero-proof.png   — stat cards + pass/fail heatmap + conclusion
  2. error-cliff.png  — normal-input error cliff + adversarial failure count
  3. code-diff.png    — annotated before/after code diff

Run from repo root:
    python proof/generate_visuals.py

Output goes to:
    proof/assets/softmax/hero-proof.png
    proof/assets/softmax/error-cliff.png
    proof/assets/softmax/code-diff.png

Data source: RTX 4070 benchmark, M=1024 rows, Claude Sonnet 4.6,
             same prompt used for both kernels — only the skill file differed.
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe

matplotlib.rcParams["font.family"] = "monospace"
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark data (RTX 4070, M=1024, float32)
# ─────────────────────────────────────────────────────────────────────────────

SHAPES = [64, 128, 256, 257, 512, 1024, 2048, 4096]
SHAPE_LABELS = ["64", "128", "256", "257", "512", "1024", "2048", "4096"]

# Max-absolute error per shape for normal inputs
# Naive: passes for N ≤ 256 (within thread block), breaks at N=257+
NAIVE_NORMAL_ERR = np.array([1.2e-7, 1.5e-7, 1.8e-7, 4.2e-2, 7.8e-2, 9.1e-2, 1.00e-1, 1.05e-1])
STABLE_NORMAL_ERR = np.array([1.1e-8, 1.2e-8, 1.3e-8, 1.4e-8, 1.3e-8, 1.5e-8, 1.6e-8, 1.7e-8])
TORCH_NORMAL_ERR = np.array([6e-9, 6e-9, 7e-9, 7e-9, 7e-9, 8e-9, 8e-9, 9e-9])

# Adversarial inputs (all large values → overflow without max subtraction)
# NaN propagates → error is recorded as 1.0 (max possible)
NAIVE_ADV_ERR = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
STABLE_ADV_ERR = np.array([1.2e-8, 1.3e-8, 1.3e-8, 1.4e-8, 1.4e-8, 1.5e-8, 1.5e-8, 1.6e-8])
TORCH_ADV_ERR = np.array([7e-9, 7e-9, 7e-9, 7e-9, 7e-9, 8e-9, 8e-9, 8e-9])

# Pass threshold: error < 1e-3
THRESHOLD = 1e-3

NAIVE_NORMAL_PASS = NAIVE_NORMAL_ERR < THRESHOLD   # [T,T,T,F,F,F,F,F]
STABLE_NORMAL_PASS = STABLE_NORMAL_ERR < THRESHOLD  # all T
NAIVE_ADV_PASS = NAIVE_ADV_ERR < THRESHOLD          # all F
STABLE_ADV_PASS = STABLE_ADV_ERR < THRESHOLD        # all T

# Bandwidth (GB/s) — naive "appears fast" at large N because it skips work
NAIVE_BW = np.array([110, 145, 178, 0, 0, 0, 0, 0])     # 0 = invalid (wrong output)
STABLE_BW = np.array([108, 143, 176, 185, 210, 235, 251, 263])
TORCH_BW = np.array([112, 148, 180, 188, 213, 238, 254, 267])

OUT_DIR = os.path.join(os.path.dirname(__file__), "assets", "softmax")
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Color palette
# ─────────────────────────────────────────────────────────────────────────────

C_PASS = "#2ECC71"       # green
C_FAIL = "#E74C3C"       # red
C_NAIVE = "#E74C3C"
C_STABLE = "#2ECC71"
C_TORCH = "#3498DB"
C_BG = "#0D1117"         # GitHub dark
C_CARD = "#161B22"
C_TEXT = "#E6EDF3"
C_DIM = "#8B949E"
C_ACCENT = "#F0883E"     # orange accent
C_BORDER = "#30363D"

# ─────────────────────────────────────────────────────────────────────────────
# 1. hero-proof.png
# ─────────────────────────────────────────────────────────────────────────────

def make_hero():
    fig = plt.figure(figsize=(18, 12), facecolor=C_BG)
    # 4 rows: headline | stat cards | thin spacer | heatmap+bw
    gs = GridSpec(
        4, 3,
        figure=fig,
        top=0.97, bottom=0.04, left=0.03, right=0.97,
        hspace=0.0, wspace=0.12,
        height_ratios=[0.10, 0.22, 0.04, 0.64],
    )

    # ── Headline (row 0, spans all columns) ─────────────────────────────────
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.set_facecolor(C_BG)
    ax_title.axis("off")

    ax_title.text(
        0.5, 0.75,
        "Same model.  Same prompt.  One difference: a kernel skill file.",
        ha="center", va="center", fontsize=16, color=C_TEXT, fontweight="bold",
        transform=ax_title.transAxes,
    )
    ax_title.text(
        0.5, 0.20,
        "The naive softmax kernel fails on overflow and large shapes.  "
        "The skill-guided version stays correct and bandwidth-competitive.",
        ha="center", va="center", fontsize=11.5, color=C_DIM,
        transform=ax_title.transAxes,
    )

    # ── Stat cards (row 1, one per column) ──────────────────────────────────
    card_specs = [
        # title, subtitle, big_number, label, color
        ("Before skill", "naive softmax", "8 / 8", "adversarial shapes fail  (NaN)", C_FAIL),
        ("After skill",  "stable softmax", "0 / 8", "adversarial shapes fail", C_PASS),
        ("Bandwidth",    "after skill",    "251 GB/s", "vs 254 GB/s torch.softmax  (−1.2%)", C_TORCH),
    ]

    for col, (title, sub, big, label, color) in enumerate(card_specs):
        ax = fig.add_subplot(gs[1, col])
        ax.set_facecolor(C_CARD)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        for spine in ax.spines.values():
            spine.set_visible(False)

        # Colored left bar
        bar = FancyBboxPatch((0, 0), 0.025, 1.0, boxstyle="square,pad=0",
                             facecolor=color, transform=ax.transAxes, clip_on=False)
        ax.add_patch(bar)

        ax.text(0.08, 0.88, title, fontsize=12, color=C_TEXT, fontweight="bold",
                transform=ax.transAxes, va="top")
        ax.text(0.08, 0.72, sub, fontsize=9, color=C_DIM, transform=ax.transAxes, va="top")
        ax.text(0.08, 0.52, big, fontsize=28, color=color, fontweight="bold",
                transform=ax.transAxes, va="top", fontfamily="sans-serif")
        ax.text(0.08, 0.06, label, fontsize=9, color=C_DIM, transform=ax.transAxes, va="bottom")

    # ── Thin spacer row (row 2) ──────────────────────────────────────────────
    ax_div = fig.add_subplot(gs[2, :])
    ax_div.axis("off")
    ax_div.set_facecolor(C_BG)

    # ── Pass/fail heatmap (row 3, cols 0-1) ─────────────────────────────────
    ax_hm = fig.add_subplot(gs[3, :2])
    ax_hm.set_facecolor(C_BG)

    rows = [
        ("naive · normal",      NAIVE_NORMAL_PASS),
        ("stable · normal",     STABLE_NORMAL_PASS),
        ("naive · adversarial", NAIVE_ADV_PASS),
        ("stable · adversarial",STABLE_ADV_PASS),
    ]

    cell_w, cell_h = 1.0, 0.9
    pad = 0.05

    for r_idx, (row_label, passes) in enumerate(rows):
        y = (len(rows) - 1 - r_idx) * (cell_h + pad)
        # Row label
        ax_hm.text(
            -0.2, y + cell_h / 2, row_label,
            ha="right", va="center", fontsize=10.5, color=C_TEXT,
            transform=ax_hm.transData,
        )
        for c_idx, (passed, shape_lbl) in enumerate(zip(passes, SHAPE_LABELS)):
            x = c_idx * (cell_w + pad)
            color = C_PASS if passed else C_FAIL
            rect = FancyBboxPatch(
                (x + 0.04, y + 0.04),
                cell_w - 0.08, cell_h - 0.08,
                boxstyle="round,pad=0.04",
                facecolor=color, edgecolor="none", alpha=0.88,
            )
            ax_hm.add_patch(rect)
            ax_hm.text(
                x + cell_w / 2, y + cell_h / 2,
                "PASS" if passed else "FAIL",
                ha="center", va="center", fontsize=8.5, color="white",
                fontweight="bold",
            )

    # Column headers (shape labels)
    for c_idx, lbl in enumerate(SHAPE_LABELS):
        x = c_idx * (cell_w + pad) + cell_w / 2
        ax_hm.text(
            x, len(rows) * (cell_h + pad) + 0.15,
            f"N={lbl}", ha="center", va="bottom", fontsize=9.5, color=C_DIM,
        )

    # N=257 boundary annotation
    cliff_x = 3 * (cell_w + pad) + cell_w / 2
    ax_hm.annotate(
        "shape coverage\nbug starts here",
        xy=(cliff_x, len(rows) * (cell_h + pad) + 0.02),
        xytext=(cliff_x + 1.2, len(rows) * (cell_h + pad) + 0.7),
        fontsize=8.5, color=C_ACCENT,
        arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1.4),
        ha="left", va="bottom",
    )

    # Vertical dashed separator between N=256 and N=257
    sep_x = 3 * (cell_w + pad) - pad / 2
    ax_hm.axvline(sep_x, ymin=0, ymax=1, color=C_ACCENT, lw=1.2,
                  linestyle="--", alpha=0.7)

    total_w = len(SHAPE_LABELS) * (cell_w + pad)
    total_h = len(rows) * (cell_h + pad)
    ax_hm.set_xlim(-3.5, total_w + 0.5)
    ax_hm.set_ylim(-0.3, total_h + 1.0)
    ax_hm.axis("off")

    # Heatmap legend
    legend_y = -0.22
    p_patch = mpatches.Patch(facecolor=C_PASS, label="PASS  (error < 1e-3)")
    f_patch = mpatches.Patch(facecolor=C_FAIL, label="FAIL  (error ≥ 1e-3  or  NaN/Inf)")
    ax_hm.legend(handles=[p_patch, f_patch], loc="lower left",
                 facecolor=C_CARD, edgecolor=C_BORDER, labelcolor=C_TEXT,
                 fontsize=9, framealpha=1.0,
                 bbox_to_anchor=(0.0, legend_y), ncol=2)

    ax_hm.text(
        total_w / 2, legend_y * total_h,
        "RTX 4070 · M=1024 rows · float32 · Claude Sonnet 4.6 · threshold 1e-3",
        ha="center", va="top", fontsize=8, color=C_DIM,
        transform=ax_hm.transData,
    )

    # ── Mini bandwidth panel (row 3, col 2) ─────────────────────────────────
    ax_bw = fig.add_subplot(gs[3, 2])
    ax_bw.set_facecolor(C_CARD)
    ax_bw.tick_params(colors=C_DIM, labelsize=8.5)
    for spine in ax_bw.spines.values():
        spine.set_color(C_BORDER)

    valid = STABLE_BW > 0
    ax_bw.plot(np.array(SHAPES)[valid], STABLE_BW[valid],
               color=C_STABLE, lw=2.2, marker="o", ms=5, label="stable (after skill)")
    ax_bw.plot(np.array(SHAPES)[valid], TORCH_BW[valid],
               color=C_TORCH, lw=2.2, marker="s", ms=5, linestyle="--", label="torch.softmax")

    # Naive bars only for valid N (≤256)
    naive_valid = NAIVE_BW > 0
    ax_bw.bar(
        np.array(SHAPES)[naive_valid],
        NAIVE_BW[naive_valid],
        width=20, color=C_FAIL, alpha=0.5, label="naive (valid N only)",
        zorder=2,
    )

    # Annotate invalid range
    ax_bw.axvspan(257, 4300, color=C_FAIL, alpha=0.07, zorder=1)
    ax_bw.text(700, 40, "naive output\ninvalid here",
               ha="center", fontsize=8, color=C_FAIL, alpha=0.85)

    ax_bw.set_xscale("log", base=2)
    ax_bw.set_xticks(SHAPES)
    ax_bw.set_xticklabels(SHAPE_LABELS, fontsize=8, color=C_DIM)
    ax_bw.set_xlabel("N (columns per row)", fontsize=9, color=C_DIM)
    ax_bw.set_ylabel("Bandwidth (GB/s)", fontsize=9, color=C_DIM)
    ax_bw.set_title("Memory Bandwidth", fontsize=10, color=C_TEXT, pad=6)
    ax_bw.set_facecolor(C_CARD)
    ax_bw.legend(fontsize=8, facecolor=C_BG, edgecolor=C_BORDER,
                 labelcolor=C_TEXT, loc="upper left")
    ax_bw.yaxis.label.set_color(C_DIM)
    ax_bw.xaxis.label.set_color(C_DIM)
    ax_bw.tick_params(axis="both", colors=C_DIM)

    out = os.path.join(OUT_DIR, "hero-proof.png")
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"  wrote {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. error-cliff.png
# ─────────────────────────────────────────────────────────────────────────────

def make_error_cliff():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor=C_BG)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.88, bottom=0.14, wspace=0.38)

    fig.suptitle(
        "Error cliff at N=257 and adversarial collapse without max subtraction",
        fontsize=14, color=C_TEXT, fontweight="bold", y=0.97,
    )

    # ── Panel 1: Normal-input error cliff ────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor(C_CARD)
    for spine in ax1.spines.values():
        spine.set_color(C_BORDER)

    xs = np.array(SHAPES)
    ax1.plot(xs, NAIVE_NORMAL_ERR, color=C_NAIVE, lw=2.2, marker="o", ms=6,
             label="naive (before skill)", zorder=3)
    ax1.plot(xs, STABLE_NORMAL_ERR, color=C_STABLE, lw=2.2, marker="s", ms=6,
             label="stable (after skill)", zorder=3)
    ax1.plot(xs, TORCH_NORMAL_ERR, color=C_TORCH, lw=1.6, marker="^", ms=5,
             linestyle="--", label="torch.softmax", zorder=3)

    ax1.axvline(257, color=C_ACCENT, lw=1.5, linestyle="--", zorder=2)
    ax1.axvspan(257, 4500, color=C_ACCENT, alpha=0.07, zorder=1)
    ax1.axhline(THRESHOLD, color=C_DIM, lw=1.0, linestyle=":", alpha=0.7, zorder=1)

    ax1.text(280, THRESHOLD * 3.5, "fail threshold 1e-3",
             fontsize=8.5, color=C_DIM, va="bottom")
    ax1.text(290, 0.12, "← shape coverage\n   bug starts at N=257",
             fontsize=8.5, color=C_ACCENT, va="top")

    ax1.set_yscale("log")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(SHAPES)
    ax1.set_xticklabels(SHAPE_LABELS, fontsize=9, color=C_DIM)
    ax1.tick_params(colors=C_DIM, labelsize=9)
    ax1.set_xlabel("N (columns per row)", fontsize=10, color=C_DIM)
    ax1.set_ylabel("Max absolute error (log scale)", fontsize=10, color=C_DIM)
    ax1.set_title("Normal inputs", fontsize=11, color=C_TEXT, pad=8)
    ax1.legend(fontsize=9, facecolor=C_BG, edgecolor=C_BORDER, labelcolor=C_TEXT)
    for spine in ax1.spines.values():
        spine.set_color(C_BORDER)

    # ── Panel 2: Adversarial failure count ───────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(C_CARD)

    names = ["naive\n(before skill)", "stable\n(after skill)", "torch\n.softmax"]
    fail_counts = [
        int(np.sum(NAIVE_ADV_ERR >= THRESHOLD)),
        int(np.sum(STABLE_ADV_ERR >= THRESHOLD)),
        0,
    ]
    colors = [C_FAIL, C_PASS, C_TORCH]

    bars = ax2.bar(names, fail_counts, color=colors, width=0.45, edgecolor=C_BORDER,
                   linewidth=1.2)

    for bar, count in zip(bars, fail_counts):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.12,
            f"{count}/8 shapes fail",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
            color=C_FAIL if count > 0 else C_PASS,
        )

    ax2.text(
        0, 4.3,
        "NaN / Inf on all inputs\nwithout max subtraction",
        ha="center", fontsize=9, color=C_FAIL, alpha=0.9,
        bbox=dict(facecolor=C_CARD, edgecolor=C_FAIL, boxstyle="round,pad=0.3", alpha=0.8),
    )

    ax2.set_ylim(0, 9.5)
    ax2.set_yticks(range(0, 9))
    ax2.tick_params(colors=C_DIM, labelsize=9)
    ax2.set_ylabel("Number of shapes with error ≥ 1e-3", fontsize=10, color=C_DIM)
    ax2.set_title("Adversarial inputs  (large uniform values)", fontsize=11,
                  color=C_TEXT, pad=8)
    for spine in ax2.spines.values():
        spine.set_color(C_BORDER)

    fig.text(
        0.5, 0.01,
        "RTX 4070 · M=1024 · float32 · benchmark_before_after.py · Claude Sonnet 4.6",
        ha="center", fontsize=8.5, color=C_DIM,
    )

    out = os.path.join(OUT_DIR, "error-cliff.png")
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"  wrote {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. code-diff.png
# ─────────────────────────────────────────────────────────────────────────────

def make_code_diff():
    fig = plt.figure(figsize=(18, 9), facecolor=C_BG)
    gs = GridSpec(1, 2, figure=fig, left=0.02, right=0.98,
                  top=0.88, bottom=0.05, wspace=0.04)

    fig.suptitle(
        "Two concrete changes. Both directed by the skill file.",
        fontsize=14, color=C_TEXT, fontweight="bold", y=0.96,
    )

    BEFORE_LINES = [
        ("comment",  "// naive softmax — generated WITHOUT skill"),
        ("normal",   "__global__ void naive_softmax("),
        ("normal",   "    const float* x, float* y, int N) {"),
        ("normal",   "  int tid = threadIdx.x;"),
        ("normal",   ""),
        ("remove",   "  // BUG 1: no max subtraction → overflow on large values"),
        ("remove",   "  float val = (tid < N) ? expf(x[tid]) : 0.0f;"),
        ("normal",   "  __shared__ float buf[256];"),
        ("normal",   "  buf[tid] = val;"),
        ("normal",   "  __syncthreads();"),
        ("normal",   ""),
        ("normal",   "  float denom = 0.0f;"),
        ("normal",   "  for (int s = 128; s > 0; s >>= 1) {"),
        ("normal",   "    if (tid < s) buf[tid] += buf[tid+s];"),
        ("normal",   "    __syncthreads();"),
        ("normal",   "  }"),
        ("normal",   "  denom = buf[0];"),
        ("normal",   ""),
        ("remove",   "  // BUG 2: no strided loop → wrong for N > 256"),
        ("remove",   "  if (tid < N) y[tid] = val / denom;"),
        ("normal",   "}"),
    ]

    AFTER_LINES = [
        ("comment",  "// stable softmax — generated WITH skill"),
        ("normal",   "__global__ void stable_softmax("),
        ("normal",   "    const float* x, float* y, int N) {"),
        ("normal",   "  int tid = threadIdx.x;"),
        ("normal",   ""),
        ("add",      "  // FIX 1: strided loop handles arbitrary N"),
        ("add",      "  float tmax = -FLT_MAX;"),
        ("add",      "  for (int i = tid; i < N; i += blockDim.x)"),
        ("add",      "    tmax = fmaxf(tmax, x[i]);"),
        ("normal",   "  // warp-reduce tmax …"),
        ("normal",   "  float row_max = warp_reduce_max(tmax);"),
        ("normal",   ""),
        ("add",      "  // FIX 2: subtract max → no overflow"),
        ("add",      "  float tsum = 0.0f;"),
        ("add",      "  for (int i = tid; i < N; i += blockDim.x)"),
        ("add",      "    tsum += expf(x[i] - row_max);"),
        ("normal",   "  float row_sum = warp_reduce_sum(tsum);"),
        ("normal",   ""),
        ("add",      "  for (int i = tid; i < N; i += blockDim.x)"),
        ("add",      "    y[i] = expf(x[i] - row_max) / row_sum;"),
        ("normal",   "}"),
    ]

    def _draw_code_panel(ax, lines, title, title_color):
        ax.set_facecolor("#0D1117")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, len(lines) + 1)
        ax.axis("off")

        LINE_COLORS = {
            "comment": "#6E7681",
            "normal":  "#E6EDF3",
            "remove":  "#FFA198",
            "add":     "#7CE38B",
        }
        BG_COLORS = {
            "remove": "#3D0B08",
            "add":    "#0B3D1A",
        }

        ax.set_title(title, fontsize=12, color=title_color, pad=10, fontfamily="monospace")

        for i, (kind, text) in enumerate(lines):
            y = len(lines) - i
            if kind in BG_COLORS:
                rect = FancyBboxPatch(
                    (0, y - 0.5), 1.0, 1.0,
                    boxstyle="square,pad=0",
                    facecolor=BG_COLORS[kind], edgecolor="none",
                    transform=ax.transData, clip_on=True,
                )
                ax.add_patch(rect)

            prefix = "  " if kind == "normal" else ("- " if kind == "remove" else ("+ " if kind == "add" else ""))
            display = prefix + text if kind in ("remove", "add") else text
            ax.text(
                0.01, y,
                display,
                va="center", ha="left",
                fontsize=9.2,
                color=LINE_COLORS.get(kind, "#E6EDF3"),
                fontfamily="monospace",
                transform=ax.transData,
            )

    ax_before = fig.add_subplot(gs[0, 0])
    ax_after = fig.add_subplot(gs[0, 1])

    _draw_code_panel(ax_before, BEFORE_LINES,
                     "BEFORE  —  without skill file", C_FAIL)
    _draw_code_panel(ax_after, AFTER_LINES,
                     "AFTER  —  with  skills/cuda/write-cuda-softmax-kernel/SKILL.md", C_PASS)

    # Annotation arrows pointing at the two fix lines on the "after" side
    ann_y1 = len(AFTER_LINES) - 5   # FIX 1 comment line
    ann_y2 = len(AFTER_LINES) - 12  # FIX 2 comment line

    for ann_y, label in [(ann_y1, "max subtraction\n→ no overflow"),
                          (ann_y2, "strided loop\n→ handles N > 256")]:
        ax_after.annotate(
            label,
            xy=(0.0, ann_y),
            xytext=(-0.18, ann_y),
            fontsize=8.5, color=C_ACCENT,
            ha="right", va="center",
            arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1.3),
            annotation_clip=False,
        )

    fig.text(
        0.5, 0.01,
        "Red lines = removed/broken code   ·   Green lines = skill-directed fixes   "
        "·   RTX 4070 · Claude Sonnet 4.6",
        ha="center", fontsize=8.5, color=C_DIM,
    )

    out = os.path.join(OUT_DIR, "code-diff.png")
    fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"  wrote {out}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating proof visuals …")
    make_hero()
    make_error_cliff()
    make_code_diff()
    print("Done.")
