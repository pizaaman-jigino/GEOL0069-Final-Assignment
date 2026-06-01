"""Supervised classification: Random Forest and a compact CNN.

Two complementary supervised approaches, both covered in the AI4EO class:

* **Random Forest** trains on the engineered feature vector
  (:mod:`src.features`). It is fast, needs no GPU, and exposes feature
  importances -- a transparent baseline.
* **SimpleCNN** is a small convolutional network (PyTorch) that learns
  directly from the raw 64x64x3 patches. It is deliberately lightweight so
  that it trains in a few minutes on a CPU while still demonstrating the
  end-to-end deep-learning workflow (data -> conv blocks -> pooling -> dense
  head -> softmax, optimised with Adam and cross-entropy).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier


# --------------------------------------------------------------------------- #
# Random Forest                                                               #
# --------------------------------------------------------------------------- #
def train_random_forest(X_train, y_train, n_estimators: int = 300,
                        max_depth: int | None = None, seed: int = 42):
    """Train a Random Forest on tabular features and return the fitted model."""
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        n_jobs=-1,
        random_state=seed,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)
    return clf


# --------------------------------------------------------------------------- #
# Compact CNN (PyTorch, CPU-friendly)                                         #
# --------------------------------------------------------------------------- #
def _import_torch():
    import torch
    import torch.nn as nn
    return torch, nn


def build_cnn(n_classes: int = 10):
    """Construct the SimpleCNN module."""
    torch, nn = _import_torch()

    class SimpleCNN(nn.Module):
        """3 conv blocks (16->32->64 channels) + global pool + linear head."""

        def __init__(self, n_classes: int):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
                nn.MaxPool2d(2),                                    # 64 -> 32
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d(2),                                    # 32 -> 16
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),                            # -> 64 x 1 x 1
            )
            self.head = nn.Sequential(
                nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, n_classes)
            )

        def forward(self, x):
            return self.head(self.features(x))

    return SimpleCNN(n_classes)


def _to_tensor_dataset(X, y):
    """numpy (N,H,W,C) uint8 -> torch float tensor (N,C,H,W) in [0,1]."""
    torch, _ = _import_torch()
    Xt = torch.from_numpy(X.astype(np.float32) / 255.0).permute(0, 3, 1, 2).contiguous()
    yt = torch.from_numpy(np.asarray(y, dtype=np.int64))
    return torch.utils.data.TensorDataset(Xt, yt)


def train_cnn(
    X_train, y_train, X_val, y_val,
    n_classes: int = 10, epochs: int = 20, batch_size: int = 64,
    lr: float = 1e-3, weight_decay: float = 1e-4, seed: int = 42, verbose: bool = True,
):
    """Train SimpleCNN and return ``(model, history)``.

    Training uses Adam with weight decay and a **cosine learning-rate
    schedule** (which decays the step size smoothly to ~0, stabilising the
    final epochs). The weights from the **best validation epoch** are restored
    before returning, so the reported model is the best one seen, not whatever
    the last (possibly noisy) epoch produced.

    ``X_val`` / ``y_val`` should be a held-out *validation* split carved from
    the training data -- NOT the test set -- so test accuracy stays unbiased.

    ``history`` is a dict of per-epoch train loss and validation accuracy.
    """
    import copy
    torch, nn = _import_torch()
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, (torch.get_num_threads() or 1)))

    model = build_cnn(n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    train_ds = _to_tensor_dataset(X_train, y_train)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    history = {"train_loss": [], "val_acc": [], "lr": []}
    best_acc, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
        sched.step()
        train_loss = running / len(train_ds)

        val_acc = evaluate_cnn(model, X_val, y_val, batch_size=256)["accuracy"]
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(float(opt.param_groups[0]["lr"]))
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
        if verbose:
            print(f"  epoch {ep:2d}/{epochs}  train_loss={train_loss:.3f}  val_acc={val_acc:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val_acc"] = best_acc
    if verbose:
        print(f"  -> restored best-validation weights (val_acc={best_acc:.3f})")
    return model, history


def evaluate_cnn(model, X, y, batch_size: int = 256):
    """Run the CNN on ``X`` and return ``{'accuracy', 'y_true', 'y_pred'}``."""
    torch, _ = _import_torch()
    device = next(model.parameters()).device
    model.eval()
    ds = _to_tensor_dataset(X, y)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds = []
    with torch.no_grad():
        for xb, _yb in dl:
            logits = model(xb.to(device))
            preds.append(logits.argmax(1).cpu().numpy())
    y_pred = np.concatenate(preds)
    y_true = np.asarray(y)
    acc = float((y_pred == y_true).mean())
    return {"accuracy": acc, "y_true": y_true, "y_pred": y_pred}


def count_parameters(model) -> int:
    """Total number of trainable parameters (for the report / FLOP estimate)."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
