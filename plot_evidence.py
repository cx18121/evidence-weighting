"""Plot how reported probabilities respond to the specified evidence."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
METRICS = ROOT / "experiments/evidence_count/results/metrics.json"
MODELS = [
    ("jev", "Jev (Noul)"),
    ("gemma", "Gemma-3-27B"),
    ("deepseek", "DeepSeek-V3.2"),
    ("mistral", "Mistral-Small-3.2"),
    ("llama", "Llama-3.3-70B"),
    ("haiku", "Haiku 4.5"),
]


def series(data):
    rows = []
    for model, label in MODELS:
        slope = data["e1"][f"{model}|listed|primary"]["slope|0.52"]
        assert 0 <= slope["lo"] <= slope["est"] <= slope["hi"]
        rows.append((label, slope))
    return rows


def draw(out):
    data = json.loads(METRICS.read_text())
    rows = series(data)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, (ax, amount) = plt.subplots(2, 1, figsize=(7.0, 3.8), gridspec_kw={"height_ratios": [1.35, 1]})
    ax.axvline(1, color="#555555", linewidth=1.3, linestyle="--", zorder=1)
    for y, (label, slope) in enumerate(rows):
        color = "#4b5563" if slope["lo"] <= 1 <= slope["hi"] else "#1f4f70" if slope["est"] < 1 else "#ad4935"
        ax.errorbar(
            slope["est"], y,
            xerr=[[slope["est"] - slope["lo"]], [slope["hi"] - slope["est"]]],
            fmt="o", color=color, markersize=5, capsize=2.5, linewidth=1.5, zorder=2,
        )
    ax.set_yticks(range(len(rows)), [label for label, _ in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 3.4)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xlabel("A. Slope of reported versus specified log odds")
    ax.grid(axis="x", color="#e7e7e7", linewidth=0.8)
    ax.set_axisbelow(True)

    counts = [4, 8, 16, 32, 64]
    reference = [1 / (1 + (0.48 / 0.52) ** (n // 2)) for n in counts]
    amount.plot(counts, reference, "k--", marker="o", linewidth=1.5, label="Specified", zorder=4)
    for key, label, color in (
        ("jev", "Jev (Noul)", "#1f4f70"),
        ("llama", "Llama-3.3-70B", "#ad4935"),
        ("haiku", "Haiku 4.5", "#d17822"),
    ):
        values = [data["e1"][f"{key}|listed|primary"][f"p75|0.52|{n}"] for n in counts]
        amount.errorbar(
            counts, [v["est"] for v in values],
            yerr=[[v["est"] - v["lo"] for v in values], [v["hi"] - v["est"] for v in values]],
            fmt="o-", color=color, markersize=3, capsize=2, linewidth=1.1, label=label,
        )
    amount.set(xscale="log", xlim=(3.5, 73), ylim=(0.48, 1.04),
               xticks=counts, ylabel="P(YES)", xlabel="B. Reports shown (75% favor YES; reliability 52%)")
    amount.set_xticklabels([str(n) for n in counts])
    amount.grid(axis="y", color="#e7e7e7", linewidth=0.8)
    amount.legend(ncol=2, loc="lower right", frameon=True, fontsize=7.5, edgecolor="#dddddd")
    fig.tight_layout(h_pad=1.25)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "evidence.pdf")
    fig.savefig(out / "evidence.png", dpi=220)
    plt.close(fig)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    for label, slope in draw(args.out):
        print(f"{label}: {slope['est']:.2f} [{slope['lo']:.2f}, {slope['hi']:.2f}]")
