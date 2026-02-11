#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional
import time

import numpy as np
import pandas as pd
import sys

from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import torch
from tqdm import tqdm

from src.qc_quantum_corr import AngleMap, angles_from_z
from src.qml_interpretation import from_string
from src.qml_model import QNNSettings, build_qnn_base_model


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train QCardEst-style quantum correction models (Torch + Qiskit Machine Learning).\n"
            "Pipeline: Ridge baseline -> residual Δ -> QNN base model -> ReshapeSumLayer(nOutputs) -> NormLayer -> Interpretation."
        )
    )

    p.add_argument("--data", type=str, default="data/model_dataset_object_level.csv")
    p.add_argument(
        "--features",
        nargs="+",
        default=["tau_cent_median", "sigma_line_median", "fwhm_median"],
        help="Feature columns used for baseline and quantum correction.",
    )
    p.add_argument("--target", type=str, default="target_hbeta_log10")

    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)

    # Evaluation strategies
    p.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated list of seeds for repeated holdout (overrides --seed). Example: 0,1,2,3,4",
    )
    p.add_argument(
        "--kfolds",
        type=int,
        default=0,
        help="If >1, run K-fold CV instead of a single train/test split.",
    )
    p.add_argument("--kfold-seed", type=int, default=42, help="Random seed for KFold shuffle.")

    # QNN model settings
    p.add_argument("--n-outputs", type=int, default=2)
    p.add_argument("--norm", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--shots", type=int, default=200)
    p.add_argument(
        "--readout",
        choices=["sum", "bitstring"],
        default="sum",
        help="Choose how to reduce the QNN outputs before interpretation. 'bitstring' selects explicit basis states (QCardEst-style), while 'sum' retains the original reshape+sum behavior.",
    )
    p.add_argument(
        "--bitstrings",
        nargs="+",
        default=None,
        help="Optional explicit bitstrings (e.g. 000 111) used when --readout bitstring is active. Defaults to ['0..0','1..1'] for 2 outputs.",
    )

    # Optional circuit transpilation/compilation
    p.add_argument(
        "--transpile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If enabled, transpile the QNN circuit before execution.",
    )
    p.add_argument("--optimization-level", type=int, default=1, help="Qiskit transpile optimization level (0-3).")
    p.add_argument(
        "--seed-transpiler",
        type=int,
        default=None,
        help="Optional seed for Qiskit transpiler (controls stochastic passes).",
    )

    # Angle encoding
    p.add_argument("--clip", type=float, default=2.0)
    p.add_argument("--angle-scale", type=float, default=np.pi / 2)

    # Training
    p.add_argument("--archs", nargs="+", default=["linear", "rational", "rationalLog"])
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-2)

    # Outputs
    p.add_argument("--out-metrics", type=str, default="results/qcardest_style_metrics.csv")
    p.add_argument(
        "--out-metrics-agg",
        type=str,
        default=None,
        help="Optional aggregated metrics output (mean/std over seeds or folds).",
    )
    p.add_argument("--out-preds", type=str, default="results/qcardest_style_predictions.csv")

    return p.parse_args()


def _parse_seed_list(seeds: Optional[str]) -> Optional[List[int]]:
    if seeds is None:
        return None
    parts = [p.strip() for p in str(seeds).split(",") if p.strip()]
    if not parts:
        raise ValueError("--seeds was provided but no seeds were parsed")
    return [int(p) for p in parts]


def _default_agg_path(out_metrics: Path) -> Path:
    if out_metrics.suffix.lower() == ".csv":
        return out_metrics.with_name(out_metrics.stem + "_agg.csv")
    return out_metrics.with_name(out_metrics.name + "_agg.csv")


def _aggregate_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate MAE/RMSE/R2 (and baseline metrics) as mean/std over seed/fold."""

    key_cols = [
        "arch",
        "n_outputs",
        "reps",
        "norm",
        "shots",
        "readout",
        "bitstrings",
        "epochs",
        "batch_size",
        "lr",
    ]
    existing_keys = [c for c in key_cols if c in metrics_df.columns]

    metric_cols = [
        "baseline_MAE",
        "baseline_RMSE",
        "baseline_R2",
        "MAE",
        "RMSE",
        "R2",
        "train_mse_final",
        "test_mse_final",
    ]
    existing_metrics = [c for c in metric_cols if c in metrics_df.columns]

    grouped = metrics_df.groupby(existing_keys, dropna=False)

    agg_parts = []
    for col in existing_metrics:
        agg_parts.append(grouped[col].mean().rename(f"{col}_mean"))
        agg_parts.append(grouped[col].std(ddof=1).rename(f"{col}_std"))

    agg_df = pd.concat(agg_parts, axis=1).reset_index()
    return agg_df


def _run_one_experiment(
    *,
    df: pd.DataFrame,
    features: List[str],
    target: str,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed_split: int,
    seed_train: int,
    seed_qnn: int,
    fold: Optional[int],
    args: argparse.Namespace,
) -> tuple[list[dict], pd.DataFrame]:
    """Run baseline + QNN residual learning on a fixed train/test index split.

    Conceptually this is a two-stage model:
    1) Classical baseline (Ridge) predicts the target.
    2) Quantum+Torch model learns the residual Δ = y - ŷ_base.

    Final prediction: ŷ_quant = ŷ_base + Δ̂.
    """

    X = df[features].copy()
    y = df[target].astype(float).values

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    # ----- Baseline Ridge -----
    # A simple, stable baseline is valuable here because the QNN is only asked
    # to learn the remaining error structure (residual learning), not the full y.
    base_model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0, random_state=int(seed_split))),
        ]
    )
    base_model.fit(X_train, y_train)

    y_base_train = base_model.predict(X_train)
    y_base_test = base_model.predict(X_test)

    # Residual targets for the quantum model.
    delta_train = y_train - y_base_train
    delta_test = y_test - y_base_test

    # ----- Quantum preprocessing: z-score -> angles -----
    # We standardize features, then map z-scores into rotation angles.
    # The clip/scale mapping controls how large rotations can get.
    scaler_q = Pipeline(steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    Z_train = scaler_q.fit_transform(X_train)
    Z_test = scaler_q.transform(X_test)

    angle_map = AngleMap(clip=float(args.clip), scale=float(args.angle_scale))
    Xang_train = np.array([angles_from_z(z, angle_map) for z in Z_train], dtype=float)
    Xang_test = np.array([angles_from_z(z, angle_map) for z in Z_test], dtype=float)

    # ----- QNN base settings -----
    # If readout=bitstring, these select explicit computational-basis components
    # from the 2**n_qubits probability vector before the interpretation head.
    bitstrings = tuple(args.bitstrings) if args.bitstrings is not None else None

    qnn_settings = QNNSettings(
        n_outputs=int(args.n_outputs),
        reps=int(args.reps),
        norm=bool(args.norm),
        shots=int(args.shots),
        seed=int(seed_qnn),
        readout=str(args.readout),
        bitstrings=bitstrings,
        transpile=bool(args.transpile),
        optimization_level=int(args.optimization_level),
        seed_transpiler=int(args.seed_transpiler) if args.seed_transpiler is not None else None,
    )

    metrics_rows: list[dict] = []
    preds_rows: list[pd.DataFrame] = []

    for arch in args.archs:
        # For each architecture we rebuild the QNN module so the initial quantum
        # weights are seeded consistently and results are attributable per-arch.
        t0 = time.perf_counter()
        base_qnn = build_qnn_base_model(n_inputs=Xang_train.shape[1], settings=qnn_settings)
        # `from_string` wraps the base QNN with a QCardEst-style interpretation head:
        # - linear: learned scale on the first reduced component
        # - rational: (x0+eps)/(x1+eps)
        # - rationalLog: log((x0+eps)/(x1+eps))
        model = from_string(str(arch), base_qnn)

        result = _train_one(
            model=model,
            X_train=Xang_train,
            y_train=delta_train,
            X_test=Xang_test,
            y_test=delta_test,
            seed=int(seed_train),
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
        )

        delta_hat_test = result["pred_test"]
        # Compose baseline + correction.
        y_pred_quant = y_base_test + delta_hat_test

        mae = mean_absolute_error(y_test, y_pred_quant)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_quant)))
        r2 = r2_score(y_test, y_pred_quant)

        metrics_rows.append(
            {
                "arch": arch,
                **asdict(qnn_settings),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "seed_split": int(seed_split),
                "seed_train": int(seed_train),
                "seed_qnn": int(seed_qnn),
                "fold": int(fold) if fold is not None else None,
                "wall_time_s": float(time.perf_counter() - t0),
                "baseline_MAE": float(mean_absolute_error(y_test, y_base_test)),
                "baseline_RMSE": float(np.sqrt(mean_squared_error(y_test, y_base_test))),
                "baseline_R2": float(r2_score(y_test, y_base_test)),
                "MAE": float(mae),
                "RMSE": float(rmse),
                "R2": float(r2),
                "train_mse_final": float(result["history"]["train_mse"][-1]),
                "test_mse_final": float(result["history"]["test_mse"][-1]),
            }
        )

        pred_df = df.iloc[test_idx][["varname", "target_hbeta_source"]].copy()
        pred_df["arch"] = arch
        pred_df["seed_split"] = int(seed_split)
        pred_df["seed_train"] = int(seed_train)
        pred_df["fold"] = int(fold) if fold is not None else None
        pred_df["y_true"] = y_test
        pred_df["y_pred_base"] = y_base_test
        pred_df["delta_hat"] = delta_hat_test
        pred_df["y_pred_quant"] = y_pred_quant
        preds_rows.append(pred_df)

    preds_df = pd.concat(preds_rows, ignore_index=True) if preds_rows else pd.DataFrame()
    return metrics_rows, preds_df


def _train_one(
    *,
    model: torch.nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
) -> dict:
    device = torch.device("cpu")

    # Training determinism: this seed controls Torch parameter init and any
    # stochasticity in the torch dataloader shuffle.
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    model = model.to(device)
    model.train()

    Xtr = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    ytr = torch.as_tensor(y_train, dtype=torch.float32, device=device).view(-1)

    Xte = torch.as_tensor(X_test, dtype=torch.float32, device=device)
    yte = torch.as_tensor(y_test, dtype=torch.float32, device=device).view(-1)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtr, ytr),
        batch_size=int(batch_size),
        shuffle=True,
    )

    opt = torch.optim.Adam(model.parameters(), lr=float(lr))
    loss_fn = torch.nn.MSELoss()

    history = {"train_mse": [], "test_mse": []}

    for epoch in range(1, int(epochs) + 1):
        batch_losses = []
        # tqdm defaults to stderr; write to stdout so captured logs preserve ordering.
        pbar = tqdm(loader, desc=f"epoch {epoch:03d}/{epochs}", leave=False, file=sys.stdout)
        for xb, yb in pbar:
            opt.zero_grad(set_to_none=True)
            pred = model(xb).view(-1)
            loss = loss_fn(pred, yb)
            # TorchConnector enables backprop through the QNN primitive.
            loss.backward()
            opt.step()

            loss_val = float(loss.detach().cpu().item())
            batch_losses.append(loss_val)
            pbar.set_postfix(train_mse=f"{loss_val:.4f}")

        train_mse = float(np.mean(batch_losses)) if batch_losses else float("nan")
        history["train_mse"].append(train_mse)

        model.eval()
        with torch.no_grad():
            test_pred = model(Xte).view(-1)
            test_mse = float(loss_fn(test_pred, yte).cpu().item())
        history["test_mse"].append(test_mse)
        model.train()

        print(f"  epoch {epoch:03d}: train_mse={train_mse:.6f} test_mse={test_mse:.6f}", flush=True)

    model.eval()
    with torch.no_grad():
        pred_test = model(Xte).view(-1).detach().cpu().numpy()

    return {"history": history, "pred_test": pred_test, "train_seed": int(seed)}


def main() -> None:
    args = _parse_args()

    seeds = _parse_seed_list(args.seeds)
    kfolds = int(args.kfolds)
    if seeds is not None and kfolds > 1:
        raise ValueError("Use either --seeds (repeated holdout) or --kfolds (CV), not both.")

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found: {data_path}")

    df = pd.read_csv(data_path)

    features: List[str] = list(args.features)
    target: str = str(args.target)

    print(f"Loaded {data_path} shape={df.shape}", flush=True)
    print(f"Features: {features}", flush=True)

    metrics_rows: list[dict] = []
    preds_rows: list[pd.DataFrame] = []

    if kfolds and kfolds > 1:
        print(f"Mode: {kfolds}-fold CV (shuffle=True, seed={args.kfold_seed})", flush=True)
        kf = KFold(n_splits=int(kfolds), shuffle=True, random_state=int(args.kfold_seed))
        indices = np.arange(len(df))

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(indices), start=1):
            seed_split = int(args.kfold_seed)
            seed_train = int(args.seed) + int(fold_idx)
            print(f"\n--- Fold {fold_idx}/{kfolds} train={len(train_idx)} test={len(test_idx)} seed_train={seed_train}")

            fold_metrics, fold_preds = _run_one_experiment(
                df=df,
                features=features,
                target=target,
                train_idx=train_idx,
                test_idx=test_idx,
                seed_split=seed_split,
                seed_train=seed_train,
                seed_qnn=seed_train,
                fold=int(fold_idx),
                args=args,
            )
            metrics_rows.extend(fold_metrics)
            preds_rows.append(fold_preds)

    else:
        run_seeds = seeds if seeds is not None else [int(args.seed)]
        print(f"Mode: train/test split repeated over {len(run_seeds)} seed(s)", flush=True)

        indices = np.arange(len(df))

        for seed_run in run_seeds:
            train_idx, test_idx = train_test_split(
                indices, test_size=float(args.test_size), random_state=int(seed_run)
            )
            train_idx = np.asarray(train_idx, dtype=int)
            test_idx = np.asarray(test_idx, dtype=int)

            print(f"\n--- Seed {seed_run} train={len(train_idx)} test={len(test_idx)}")

            split_metrics, split_preds = _run_one_experiment(
                df=df,
                features=features,
                target=target,
                train_idx=train_idx,
                test_idx=test_idx,
                seed_split=int(seed_run),
                seed_train=int(seed_run),
                seed_qnn=int(seed_run),
                fold=None,
                args=args,
            )
            metrics_rows.extend(split_metrics)
            preds_rows.append(split_preds)

    metrics_df = pd.DataFrame(metrics_rows).sort_values("MAE")
    preds_df = pd.concat(preds_rows, ignore_index=True) if preds_rows else pd.DataFrame()

    out_metrics = Path(args.out_metrics)
    out_metrics_agg = Path(args.out_metrics_agg) if args.out_metrics_agg else _default_agg_path(out_metrics)
    out_preds = Path(args.out_preds)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(out_metrics, index=False)
    preds_df.to_csv(out_preds, index=False)

    if len(metrics_df) > 1:
        agg_df = _aggregate_metrics(metrics_df)
        agg_df.to_csv(out_metrics_agg, index=False)

    print("\n=== Summary (sorted by MAE) ===")
    summary_cols = [
        "arch",
        "MAE",
        "RMSE",
        "R2",
        "train_mse_final",
        "test_mse_final",
        "n_outputs",
        "norm",
        "reps",
        "shots",
        "readout",
    ]
    available_cols = [c for c in summary_cols if c in metrics_df.columns]
    print(metrics_df[available_cols].to_string(index=False))
    print("\nSaved:")
    print(" -", out_metrics)
    if len(metrics_df) > 1:
        print(" -", out_metrics_agg)
    print(" -", out_preds)


if __name__ == "__main__":
    main()
