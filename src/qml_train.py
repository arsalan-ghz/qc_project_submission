from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 50
    batch_size: int = 16
    lr: float = 1e-2
    seed: int = 42


def _as_tensor(x: np.ndarray, *, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def train_regression(
    model: torch.nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    config: TrainConfig = TrainConfig(),
    device: Optional[torch.device] = None,
) -> Dict[str, list]:
    """Minimal torch training loop for residual regression.

    - model: torch module returning a scalar per sample (shape (batch,) or (batch,1)).
    - X_train: (N, n_features)
    - y_train: (N,) targets

    Returns a history dict with train/val loss curves.
    """

    if device is None:
        device = torch.device("cpu")

    torch.manual_seed(int(config.seed))
    np.random.seed(int(config.seed))

    model = model.to(device)
    model.train()

    Xtr = _as_tensor(X_train, device=device)
    ytr = _as_tensor(y_train, device=device).view(-1)

    if X_val is not None and y_val is not None:
        Xva = _as_tensor(X_val, device=device)
        yva = _as_tensor(y_val, device=device).view(-1)
    else:
        Xva = yva = None

    dataset = torch.utils.data.TensorDataset(Xtr, ytr)
    loader = torch.utils.data.DataLoader(dataset, batch_size=int(config.batch_size), shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=float(config.lr))
    loss_fn = torch.nn.MSELoss()

    history = {"train_mse": [], "val_mse": []}

    for _epoch in range(int(config.epochs)):
        model.train()
        train_losses = []
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            pred = pred.view(-1)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            train_losses.append(loss.detach().item())

        history["train_mse"].append(float(np.mean(train_losses)) if train_losses else float("nan"))

        if Xva is not None:
            model.eval()
            with torch.no_grad():
                pred = model(Xva).view(-1)
                val_loss = loss_fn(pred, yva).item()
            history["val_mse"].append(float(val_loss))

    return history


def predict(model: torch.nn.Module, X: np.ndarray, *, device: Optional[torch.device] = None) -> np.ndarray:
    if device is None:
        device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        xt = _as_tensor(X, device=device)
        yt = model(xt).view(-1).detach().cpu().numpy()
    return yt


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    # R2 (simple, avoids sklearn dependency here)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}
