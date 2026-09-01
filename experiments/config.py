"""Per-dataset architecture and optimizer settings used for the reported results."""

CONFIGS = {
    "iris": dict(hidden=[8], act="tanh", epochs=60, lr=0.10, batch_size=16,
                 momentum=0.9, weight_decay=1e-4, lr_decay=0.98),
    "wine": dict(hidden=[16], act="tanh", epochs=60, lr=0.05, batch_size=16,
                 momentum=0.9, weight_decay=1e-4, lr_decay=0.98),
    "breast_cancer": dict(hidden=[16], act="tanh", epochs=40, lr=0.05, batch_size=32,
                          momentum=0.9, weight_decay=1e-4, lr_decay=0.98),
    "digits": dict(hidden=[64], act="tanh", epochs=40, lr=0.10, batch_size=32,
                   momentum=0.9, weight_decay=1e-4, lr_decay=0.97),
}

SEEDS = [0, 1, 2, 3, 4]
