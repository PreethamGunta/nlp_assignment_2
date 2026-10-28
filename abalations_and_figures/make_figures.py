"""
generate_analysis.py
Generates all analysis outputs and visualizations for the PA2 report:
  1. Per-tag precision, recall, F1 (BiLSTM vs HMM)
  2. Confusion matrix
  3. Training curves (from ablation_results.json)
  4. Seen/unseen accuracy breakdown
  5. Error examples (gold vs HMM vs BiLSTM)
  6. Sentence-level accuracy
  7. Per-tag F1 bar chart comparison

Usage: python generate_analysis.py dev_current.out dev.tags [hmm.out]

Outputs saved to figures/
"""

import sys
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import defaultdict

# Create output directory
os.makedirs("figures", exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_tagged_file(path):
    """Return list-of-sentences, each sentence is list of (word, tag)."""
    sentences = []
    current = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if current:
                    sentences.append(current)
                    current = []
                continue
            sent_tokens = []
            for tok in line.split():
                parts = tok.rsplit("/", 1)
                if len(parts) == 2:
                    sent_tokens.append((parts[0], parts[1]))
            if sent_tokens:
                sentences.append(sent_tokens)
    if current:
        sentences.append(current)
    return sentences


def flatten(sentences):
    """Flatten list-of-sentences to flat list of (word, tag)."""
    return [(w, t) for sent in sentences for (w, t) in sent]


# ─────────────────────────────────────────────────────────────────────────────
#  Load word vocabulary (for IV/OOV)
# ─────────────────────────────────────────────────────────────────────────────

def load_vocab():
    try:
        with open("data/word2idx.json", encoding="utf-8") as f:
            w2i = json.load(f)
        return set(w2i.keys()) - {"<PAD>", "<UNK>"}
    except FileNotFoundError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  Per-tag precision / recall / F1
# ─────────────────────────────────────────────────────────────────────────────

def compute_per_tag_f1(pred_flat, gold_flat):
    """Returns dict tag -> {precision, recall, f1, support}."""
    from collections import Counter
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    for (pw, pt), (gw, gt) in zip(pred_flat, gold_flat):
        if pt == gt:
            tp[gt] += 1
        else:
            fp[pt] += 1
            fn[gt]  += 1
    support = defaultdict(int)
    for _, gt in gold_flat:
        support[gt] += 1
    tags = sorted(support.keys())
    result = {}
    for tag in tags:
        prec = tp[tag] / (tp[tag] + fp[tag]) if (tp[tag] + fp[tag]) > 0 else 0.0
        rec  = tp[tag] / (tp[tag] + fn[tag]) if (tp[tag] + fn[tag]) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        result[tag] = {"precision": prec, "recall": rec, "f1": f1,
                       "support": support[tag], "correct": tp[tag]}
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Confusion matrix
# ─────────────────────────────────────────────────────────────────────────────

def build_confusion_matrix(pred_flat, gold_flat, tags):
    tag2i = {t: i for i, t in enumerate(tags)}
    n = len(tags)
    cm = np.zeros((n, n), dtype=int)
    for (_, pt), (_, gt) in zip(pred_flat, gold_flat):
        if gt in tag2i and pt in tag2i:
            cm[tag2i[gt]][tag2i[pt]] += 1
    return cm


def plot_confusion_matrix(cm, tags, title, out_path, topk=20):
    """Plot normalized confusion matrix for the topk most-frequent tags."""
    support = cm.sum(axis=1)
    topk_idx = np.argsort(support)[::-1][:topk]
    topk_tags = [tags[i] for i in topk_idx]
    sub_cm = cm[np.ix_(topk_idx, topk_idx)]
    row_sums = sub_cm.sum(axis=1, keepdims=True)
    norm_cm = np.divide(sub_cm, row_sums, where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(norm_cm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(topk))
    ax.set_yticks(range(topk))
    ax.set_xticklabels(topk_tags, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(topk_tags, fontsize=9)
    ax.set_ylabel("Gold tag", fontsize=11)
    ax.set_xlabel("Predicted tag", fontsize=11)
    ax.set_title(title, fontsize=12, pad=10)

    # Annotate non-diagonal cells with rate if > 0.02
    for i in range(topk):
        for j in range(topk):
            val = norm_cm[i, j]
            if val > 0.02:
                color = "white" if val > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved confusion matrix: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Per-tag F1 comparison bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_tag_f1(neural_f1, hmm_f1, out_path, min_support=50):
    """Grouped bar chart: HMM vs BiLSTM F1 per tag (tags with support >= min_support)."""
    all_tags = sorted(
        [t for t, v in neural_f1.items() if v["support"] >= min_support],
        key=lambda t: -neural_f1[t]["support"]
    )
    if not all_tags:
        print("  No tags with sufficient support for F1 chart")
        return

    n = len(all_tags)
    x = np.arange(n)
    w = 0.35

    neural_vals = [neural_f1[t]["f1"] * 100 for t in all_tags]
    hmm_vals    = [hmm_f1.get(t, {}).get("f1", 0) * 100 for t in all_tags]

    fig, ax = plt.subplots(figsize=(max(10, n * 0.55), 5))
    bars1 = ax.bar(x - w/2, hmm_vals,    w, label="HMM",         color="#7bafd4", alpha=0.85)
    bars2 = ax.bar(x + w/2, neural_vals, w, label="BiLSTM+CharCNN", color="#e07b54", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(all_tags, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("F1 score (%)", fontsize=11)
    ax.set_title("Per-tag F1: HMM vs BiLSTM+CharCNN", fontsize=12)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    ax.axhline(y=90, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved per-tag F1 chart: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Training curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(ablation_json, out_prefix):
    """Plot training loss and val accuracy curves for ablation experiments."""
    if not os.path.exists(ablation_json):
        print(f"  {ablation_json} not found — skipping training curves")
        return

    with open(ablation_json) as f:
        results = json.load(f)

    # Also add the full model curves if they exist
    full_model_curves_path = "data/full_model_curves.json"
    full_curves = None
    if os.path.exists(full_model_curves_path):
        with open(full_model_curves_path) as f:
            full_curves = json.load(f)

    # Plot val accuracy over epochs for all ablation experiments
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results) + 1))

    if full_curves:
        ax1.plot(range(1, len(full_curves["train_losses"]) + 1),
                 full_curves["train_losses"], color=colors[0], linewidth=2,
                 label="Full model (BiLSTM+CharCNN+Drop)")
        ax2.plot(range(1, len(full_curves["val_accs"]) + 1),
                 full_curves["val_accs"], color=colors[0], linewidth=2,
                 label="Full model (BiLSTM+CharCNN+Drop)")

    for i, r in enumerate(results):
        c = colors[i + 1]
        name = r["name"]
        short = (name[:35] + "...") if len(name) > 38 else name
        if "train_losses" in r and r["train_losses"]:
            epochs = range(1, len(r["train_losses"]) + 1)
            ax1.plot(epochs, r["train_losses"], color=c, linewidth=1.5, label=short)
            ax2.plot(epochs, r["val_accs"],     color=c, linewidth=1.5, label=short)

    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Training Loss", fontsize=11)
    ax1.set_title("Training Loss", fontsize=12)
    ax1.legend(fontsize=7, loc="upper right")
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Validation Accuracy (%)", fontsize=11)
    ax2.set_title("Validation Accuracy", fontsize=12)
    ax2.legend(fontsize=7, loc="lower right")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{out_prefix}_curves.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{out_prefix}_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved training curves: {out_prefix}_curves.{{pdf,png}}")


def plot_full_model_curves(curves_path, out_prefix):
    """Plot only the full model's training loss + val acc over epochs."""
    if not os.path.exists(curves_path):
        print(f"  {curves_path} not found — skipping full model curves")
        return
    with open(curves_path) as f:
        data = json.load(f)

    train_losses = data.get("train_losses", [])
    val_accs     = data.get("val_accs", [])
    if not train_losses:
        return

    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(epochs, train_losses, color="#2196F3", linewidth=2, marker="o", markersize=3)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax1.set_title("Training Loss (Full Model)", fontsize=12)
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, val_accs, color="#E91E63", linewidth=2, marker="o", markersize=3)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Validation Accuracy (%)", fontsize=11)
    ax2.set_title("Validation Accuracy (Full Model)", fontsize=12)
    # mark best epoch
    best_epoch = int(np.argmax(val_accs)) + 1
    best_acc   = max(val_accs)
    ax2.axvline(x=best_epoch, color="green", linestyle="--", linewidth=1.5,
                label=f"Best epoch {best_epoch} ({best_acc:.2f}%)")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{out_prefix}_full_curves.pdf", dpi=150, bbox_inches="tight")
    plt.savefig(f"{out_prefix}_full_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved full model curves: {out_prefix}_full_curves.{{pdf,png}}")


# ─────────────────────────────────────────────────────────────────────────────
#  Seen / Unseen accuracy by sentence length
# ─────────────────────────────────────────────────────────────────────────────

def plot_sent_length_accuracy(pred_sents, gold_sents, out_path):
    """Token accuracy grouped by sentence length."""
    bins = [(1, 10), (11, 20), (21, 30), (31, 50), (51, 200)]
    bin_correct = defaultdict(int)
    bin_total   = defaultdict(int)

    for pred, gold in zip(pred_sents, gold_sents):
        L = len(gold)
        for (pw, pt), (gw, gt) in zip(pred, gold):
            for lo, hi in bins:
                if lo <= L <= hi:
                    label = f"{lo}–{hi}"
                    bin_total[label]   += 1
                    if pt == gt:
                        bin_correct[label] += 1
                    break

    labels = [f"{lo}–{hi}" for lo, hi in bins]
    accs   = [bin_correct[lb] / bin_total[lb] * 100 if bin_total[lb] > 0 else 0
              for lb in labels]
    counts = [bin_total[lb] for lb in labels]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, accs, color="#4CAF50", alpha=0.8, edgecolor="black", linewidth=0.5)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                f"n={cnt}", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Sentence length (tokens)", fontsize=11)
    ax.set_ylabel("Token accuracy (%)", fontsize=11)
    ax.set_title("Token Accuracy by Sentence Length (BiLSTM+CharCNN)", fontsize=12)
    ax.set_ylim(90, 100)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved sentence-length accuracy: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Ablation bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_ablation_bars(ablation_json, full_model_acc, out_path):
    """Horizontal bar chart for ablation results."""
    if not os.path.exists(ablation_json):
        print(f"  {ablation_json} not found — skipping ablation chart")
        return

    with open(ablation_json) as f:
        results = json.load(f)

    names = ["Full model\n(BiLSTM+CharCNN+Drop)"] + [r["name"] for r in results]
    accs  = [full_model_acc]                         + [r["test_acc"] for r in results]

    # Shorten names for display
    short_names = []
    for n in names:
        n2 = n.replace("BiLSTM+CharCNN, ", "– ").replace("BiLSTM+CharCNN+WordDrop (FULL)", "Full model")
        short_names.append(n2)

    colors = ["#E91E63" if i == 0 else "#7bafd4" for i in range(len(accs))]
    y = np.arange(len(accs))

    fig, ax = plt.subplots(figsize=(10, max(5, len(accs) * 0.7)))
    bars = ax.barh(y, accs, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f"{acc:.2f}%", va="center", ha="left", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(short_names, fontsize=9)
    ax.set_xlabel("Dev Set Token Accuracy (%)", fontsize=11)
    ax.set_title("Ablation Study Results", fontsize=12)
    ax.set_xlim(min(accs) - 3, 100)
    ax.axvline(x=full_model_acc, color="red", linestyle="--", linewidth=1, alpha=0.5)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved ablation chart: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Error examples (qualitative)
# ─────────────────────────────────────────────────────────────────────────────

def find_error_examples(neural_sents, gold_sents, hmm_sents=None, n=10):
    """Find interesting error examples for the report."""
    examples = []
    for i, (pred, gold) in enumerate(zip(neural_sents, gold_sents)):
        hmm = hmm_sents[i] if hmm_sents else None
        errors = [(j, pw, pt, gw, gt) for j, ((pw, pt), (gw, gt))
                  in enumerate(zip(pred, gold)) if pt != gt]
        if not errors:
            continue
        for j, pw, pt, gw, gt in errors:
            ctx_start = max(0, j - 2)
            ctx_end   = min(len(gold), j + 3)
            context   = [gw for (gw, _) in gold[ctx_start:ctx_end]]
            hmm_tag   = hmm[j][1] if hmm and j < len(hmm) else "—"
            examples.append({
                "sent_idx": i, "tok_idx": j,
                "word": gw, "gold": gt,
                "neural": pt, "hmm": hmm_tag,
                "context": " ".join(context)
            })
    return examples[:n]


# ─────────────────────────────────────────────────────────────────────────────
#  Print error table to stdout
# ─────────────────────────────────────────────────────────────────────────────

def print_error_table(examples):
    print("\nError Examples (word, context, gold, HMM, BiLSTM+CharCNN):")
    print(f"  {'Word':<18} {'Context':<35} {'Gold':>6} {'HMM':>6} {'Neural':>10}")
    print(f"  {'-'*80}")
    for ex in examples:
        print(f"  {ex['word']:<18} {ex['context']:<35} {ex['gold']:>6} {ex['hmm']:>6} {ex['neural']:>10}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    pred_path = sys.argv[1] if len(sys.argv) > 1 else "data/dev_current.out"
    gold_path = sys.argv[2] if len(sys.argv) > 2 else "data/dev.tags"
    hmm_path  = sys.argv[3] if len(sys.argv) > 3 else "data/hmm.out"

    print(f"Loading predictions: {pred_path}")
    print(f"Loading gold tags:   {gold_path}")

    pred_sents  = parse_tagged_file(pred_path)
    gold_sents  = parse_tagged_file(gold_path)
    hmm_sents   = parse_tagged_file(hmm_path) if hmm_path and os.path.exists(hmm_path) else None

    pred_flat = flatten(pred_sents)
    gold_flat = flatten(gold_sents)
    hmm_flat  = flatten(hmm_sents) if hmm_sents else None

    # Align lengths
    n = min(len(pred_flat), len(gold_flat))
    pred_flat = pred_flat[:n]
    gold_flat = gold_flat[:n]

    correct = sum(p[1] == g[1] for p, g in zip(pred_flat, gold_flat))
    acc = correct / n * 100
    print(f"\nOverall accuracy: {acc:.2f}%")

    all_tags = sorted(set(gt for _, gt in gold_flat))

    # ── 1. Per-tag F1 ─────────────────────────────────────────────────────
    neural_f1 = compute_per_tag_f1(pred_flat, gold_flat)
    hmm_f1    = compute_per_tag_f1(hmm_flat, gold_flat) if hmm_flat else {}

    print("\nPer-tag F1 (BiLSTM+CharCNN):")
    print(f"  {'Tag':<8} {'Prec':>7} {'Rec':>7} {'F1':>7} {'Supp':>7}")
    for tag in all_tags:
        v = neural_f1[tag]
        print(f"  {tag:<8} {v['precision']*100:>6.2f}% {v['recall']*100:>6.2f}% "
              f"{v['f1']*100:>6.2f}% {v['support']:>7}")

    # Save F1 to JSON for report
    with open("data/neural_f1_results.json", "w") as f:
        json.dump({t: {k: float(v) if isinstance(v, (float, np.floating)) else int(v)
                      for k, v in d.items()}
                  for t, d in neural_f1.items()}, f, indent=2)

    if hmm_f1:
        with open("data/hmm_f1_results.json", "w") as f:
            json.dump({t: {k: float(v) if isinstance(v, (float, np.floating)) else int(v)
                          for k, v in d.items()}
                      for t, d in hmm_f1.items()}, f, indent=2)

    # ── 2. Confusion matrix ────────────────────────────────────────────────
    cm = build_confusion_matrix(pred_flat, gold_flat, all_tags)
    plot_confusion_matrix(
        cm, all_tags,
        "BiLSTM+CharCNN Confusion Matrix (top-20 tags by frequency)",
        "figures/confusion_matrix.pdf"
    )
    plot_confusion_matrix(
        cm, all_tags,
        "BiLSTM+CharCNN Confusion Matrix (top-20 tags by frequency)",
        "figures/confusion_matrix.png"
    )

    # ── 3. Per-tag F1 chart ────────────────────────────────────────────────
    plot_per_tag_f1(neural_f1, hmm_f1, "figures/per_tag_f1.pdf")
    plot_per_tag_f1(neural_f1, hmm_f1, "figures/per_tag_f1.png")

    # ── 4. Training curves ─────────────────────────────────────────────────
    plot_training_curves("data/ablation_results.json", "figures/ablation")
    plot_full_model_curves("data/full_model_curves.json", "figures")

    # ── 5. Sentence-length accuracy ────────────────────────────────────────
    # Re-align sentences
    min_sents = min(len(pred_sents), len(gold_sents))
    plot_sent_length_accuracy(
        pred_sents[:min_sents], gold_sents[:min_sents],
        "figures/sent_length_acc.pdf"
    )
    plot_sent_length_accuracy(
        pred_sents[:min_sents], gold_sents[:min_sents],
        "figures/sent_length_acc.png"
    )

    # ── 6. Ablation bar chart ──────────────────────────────────────────────
    plot_ablation_bars("data/ablation_results.json", acc, "figures/ablation_bars.pdf")
    plot_ablation_bars("data/ablation_results.json", acc, "figures/ablation_bars.png")

    # ── 7. Error examples ──────────────────────────────────────────────────
    examples = find_error_examples(
        pred_sents[:min_sents], gold_sents[:min_sents],
        hmm_sents[:min_sents] if hmm_sents else None, n=20)
    print_error_table(examples)

    # Save errors to JSON
    with open("data/error_examples.json", "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)
    print(f"\nSaved error examples: error_examples.json")

    print("\nAll figures saved to figures/")


if __name__ == "__main__":
    main()

