"""Train the scratch MLP on one dataset.

    python experiments/run.py --dataset digits
    python experiments/run.py --dataset iris --hidden 16 8 --epochs 100 --lr 0.05
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dlscratch as dl
from experiments.config import CONFIGS


def build_and_train(name, cfg, seed=0, verbose=True, log_every=1):
    data = dl.load(name, seed=seed)
    dims = [data.n_features] + list(cfg["hidden"]) + [data.n_classes]
    model = dl.MLP(dims, hidden_act=cfg["act"], seed=seed)
    if verbose:
        print(data)
        print(model)
        print(f"  {len(model.parameters())} parameters, seed {seed}")
    history = dl.train(
        model, data,
        epochs=cfg["epochs"], lr=cfg["lr"], batch_size=cfg["batch_size"],
        momentum=cfg["momentum"], weight_decay=cfg["weight_decay"],
        lr_decay=cfg["lr_decay"], seed=seed, verbose=verbose, log_every=log_every,
    )
    history["dims"] = dims
    history["n_params"] = len(model.parameters())
    return model, data, history


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="iris", choices=sorted(dl.DATASETS))
    ap.add_argument("--hidden", type=int, nargs="*", default=None)
    ap.add_argument("--act", default=None, choices=["tanh", "relu"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--momentum", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--lr-decay", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = dict(CONFIGS[args.dataset])
    for key in ("hidden", "act", "epochs", "lr", "batch_size",
                "momentum", "weight_decay", "lr_decay"):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val

    _, data, h = build_and_train(args.dataset, cfg, seed=args.seed)
    print(f"\nfinal test accuracy : {h['final_test_acc']:.4f}")
    print(f"best  test accuracy : {h['best_test_acc']:.4f} (epoch {h['best_epoch']})")
    print(f"training time       : {h['total_seconds']:.1f}s "
          f"({h['sec_per_epoch']:.2f}s/epoch)")
    print("\nconfusion matrix (rows = true, cols = predicted):")
    print(dl.format_confusion_matrix(h["confusion_matrix"], data.class_names))


if __name__ == "__main__":
    main()
