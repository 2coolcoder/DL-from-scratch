"""Train on every dataset over several seeds, write results/ and the README table.

    python experiments/run_all.py                 # all datasets, 5 seeds
    python experiments/run_all.py --datasets iris --seeds 0 1

Reported accuracy is the *final*-epoch test accuracy, not the best epoch: the
best epoch is chosen by looking at the test set and would be optimistic.
"""

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dlscratch as dl
from experiments.config import CONFIGS, SEEDS
from experiments.run import build_and_train

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def sklearn_baselines(name, cfg, seeds):
    """Reference numbers from scikit-learn on the identical splits, so the
    scratch engine is measured against something rather than self-graded."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    scores = {"logreg": [], "sklearn_mlp": []}
    for seed in seeds:
        d = dl.load(name, seed=seed)
        for key, clf in (
            ("logreg", LogisticRegression(max_iter=5000, random_state=seed)),
            ("sklearn_mlp", MLPClassifier(hidden_layer_sizes=tuple(cfg["hidden"]),
                                          activation="tanh", max_iter=2000,
                                          random_state=seed)),
        ):
            clf.fit(d.X_train, d.y_train)
            scores[key].append(float(clf.score(d.X_test, d.y_test)))
    return scores


def plot_curves(name, runs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    for i, h in enumerate(runs):
        alpha = 1.0 if i == 0 else 0.35
        label = "seed 0" if i == 0 else None
        ax_loss.plot(h["epoch"], h["train_loss"], color="C0", alpha=alpha,
                     label="train" if i == 0 else None)
        ax_loss.plot(h["epoch"], h["test_loss"], color="C1", alpha=alpha,
                     label="test" if i == 0 else None)
        ax_acc.plot(h["epoch"], h["train_acc"], color="C0", alpha=alpha,
                    label="train" if i == 0 else None)
        ax_acc.plot(h["epoch"], h["test_acc"], color="C1", alpha=alpha,
                    label="test" if i == 0 else None)

    ax_loss.set(xlabel="epoch", ylabel="cross-entropy", title=f"{name} — loss")
    ax_acc.set(xlabel="epoch", ylabel="accuracy", title=f"{name} — accuracy")
    for ax in (ax_loss, ax_acc):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"{name}  ({len(runs)} seeds, faint lines = seeds 1+)", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def mean_std(xs):
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=list(CONFIGS))
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary = {}

    for name in args.datasets:
        cfg = CONFIGS[name]
        print(f"\n{'=' * 70}\n{name}  |  {cfg}\n{'=' * 70}")
        runs, started = [], time.perf_counter()

        for seed in args.seeds:
            _, data, h = build_and_train(name, cfg, seed=seed, verbose=False)
            runs.append(h)
            print(f"  seed {seed}: final {h['final_test_acc']:.4f}  "
                  f"best {h['best_test_acc']:.4f} @ epoch {h['best_epoch']}  "
                  f"({h['total_seconds']:.1f}s)")

        finals = [h["final_test_acc"] for h in runs]
        trains = [h["train_acc"][-1] for h in runs]
        f_mean, f_std = mean_std(finals)
        t_mean, _ = mean_std(trains)
        print(f"  -> test {f_mean:.4f} +/- {f_std:.4f}   (train {t_mean:.4f})")

        record = {
            "dataset": name,
            "config": cfg,
            "seeds": args.seeds,
            "dims": runs[0]["dims"],
            "n_params": runs[0]["n_params"],
            "n_train": len(data.X_train),
            "n_test": len(data.X_test),
            "n_features": data.n_features,
            "n_classes": data.n_classes,
            "class_names": data.class_names,
            "final_test_acc": finals,
            "final_train_acc": trains,
            "test_acc_mean": f_mean,
            "test_acc_std": f_std,
            "train_acc_mean": t_mean,
            "sec_per_epoch": statistics.mean(h["sec_per_epoch"] for h in runs),
            "total_seconds": time.perf_counter() - started,
            "confusion_matrix_seed0": runs[0]["confusion_matrix"],
            "history_seed0": {k: runs[0][k] for k in
                              ("epoch", "train_loss", "test_loss", "train_acc", "test_acc")},
        }

        if not args.no_baselines:
            base = sklearn_baselines(name, cfg, args.seeds)
            for key, vals in base.items():
                m, s = mean_std(vals)
                record[f"{key}_mean"], record[f"{key}_std"] = m, s
                print(f"     baseline {key:<12}: {m:.4f} +/- {s:.4f}")

        with open(os.path.join(RESULTS_DIR, f"{name}.json"), "w") as fh:
            json.dump(record, fh, indent=2)
        if not args.no_plots:
            plot_curves(name, runs, os.path.join(RESULTS_DIR, f"{name}_curves.png"))
        summary[name] = record

    write_table(summary, args.datasets)


def write_table(summary, order):
    """Emit results/table.md from *every* result on disk, not just this run's
    datasets, so `--datasets digits` does not clobber the other rows."""
    for name in CONFIGS:
        if name in summary:
            continue
        path = os.path.join(RESULTS_DIR, f"{name}.json")
        if os.path.exists(path):
            with open(path) as fh:
                summary[name] = json.load(fh)
    order = [n for n in CONFIGS if n in summary]

    header = (
        "| Dataset | Samples | Features | Classes | Architecture | Params | "
        "Test accuracy | sklearn MLP | Logistic reg. | s/epoch |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for name in order:
        r = summary[name]
        arch = "-".join(str(d) for d in r["dims"])
        def fmt(key):
            if f"{key}_mean" not in r:
                return "—"
            return f"{r[f'{key}_mean'] * 100:.1f} ± {r[f'{key}_std'] * 100:.1f}"
        rows.append(
            f"| `{name}` | {r['n_train'] + r['n_test']} | {r['n_features']} | "
            f"{r['n_classes']} | {arch} | {r['n_params']} | "
            f"**{r['test_acc_mean'] * 100:.1f} ± {r['test_acc_std'] * 100:.1f}** | "
            f"{fmt('sklearn_mlp')} | {fmt('logreg')} | {r['sec_per_epoch']:.2f} |"
        )
    table = header + "\n".join(rows) + "\n"
    path = os.path.join(RESULTS_DIR, "table.md")
    with open(path, "w") as fh:
        fh.write(table)
    print(f"\n{'=' * 70}\nResults table (also written to {path}):\n")
    print(table)


if __name__ == "__main__":
    main()
