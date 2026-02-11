# src/qc_data.py
from __future__ import annotations
from pathlib import Path
import pandas as pd


def load_csv_safely(path: str | Path) -> pd.DataFrame:
    """
    Loads CSV whether delimiter is ',' or ';' (sometimes Excel/Germany flips this).
    """
    path = Path(path)
    try:
        return pd.read_csv(path, sep=",", encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(path, sep=";", encoding="utf-8-sig")


def build_object_level_dataset(
    df: pd.DataFrame,
    agg: str = "median",
    use_hbeta_rows_for_features: bool = True,
    add_optional_features: bool = True,
) -> pd.DataFrame:
    """
    Returns one row per object (varname) with aggregated features + target.
    Target used: target_hbeta_log10 (object-level).
    Features from numeric parsed columns.
    """
    ID_COL = "varname"
    LINE_COL = "Line"

    # numeric feature columns available in your dataset
    TAU = "τcent (days)_val"
    SIGMA = "σline (km s-1)_val"
    FWHM = "FWHM (km s-1)_val"
    LOGL = "log LAGN,5100 (ergs s-1)_val"

    # target + provenance
    Y = "target_hbeta_log10"
    SRC = "target_hbeta_source"

    # ---- decide which rows contribute to FEATURES ----
    feat_df = df.copy()
    if use_hbeta_rows_for_features:
        feat_df = feat_df[feat_df[LINE_COL].astype(str).str.contains("Hβ", na=False)].copy()

    # ---- aggregation function ----
    if agg not in {"median", "mean"}:
        raise ValueError("agg must be 'median' or 'mean'")

    aggfunc = "median" if agg == "median" else "mean"

    # ---- core features (keep minimal) ----
    feature_cols = [TAU, SIGMA]
    if add_optional_features:
        feature_cols += [FWHM, LOGL]

    # Aggregate numeric features per object (ignore missing values automatically)
    X = getattr(feat_df.groupby(ID_COL)[feature_cols], aggfunc)()
    X = X.rename(columns={
        TAU: f"tau_cent_{aggfunc}",
        SIGMA: f"sigma_line_{aggfunc}",
        FWHM: f"fwhm_{aggfunc}",
        LOGL: f"logL_{aggfunc}",
    })

    # counts as extra stable features
    n_total = df.groupby(ID_COL).size().rename("n_measurements_total")
    n_hbeta = feat_df.groupby(ID_COL).size().rename("n_hbeta_rows")

    X = X.join(n_total, how="left").join(n_hbeta, how="left")

    # ---- target per object (use first non-null; target is repeated anyway) ----
    y = (
        df.groupby(ID_COL)[Y]
        .apply(lambda s: s.dropna().iloc[0] if s.dropna().size else pd.NA)
        .rename("target_hbeta_log10")
    )

    src = (
        df.groupby(ID_COL)[SRC]
        .apply(lambda s: s.dropna().iloc[0] if s.dropna().size else pd.NA)
        .rename("target_hbeta_source")
    )

    out = X.join(y, how="inner").join(src, how="left")

    # drop objects with missing required core features or missing target
    out = out.dropna(subset=["tau_cent_" + aggfunc, "sigma_line_" + aggfunc, "target_hbeta_log10"])

    # keep a predictable column order
    cols_order = [
        f"tau_cent_{aggfunc}",
        f"sigma_line_{aggfunc}",
        f"fwhm_{aggfunc}",
        f"logL_{aggfunc}",
        "n_measurements_total",
        "n_hbeta_rows",
        "target_hbeta_log10",
        "target_hbeta_source",
    ]
    cols_order = [c for c in cols_order if c in out.columns]
    out = out[cols_order].reset_index()  # keep varname as a column

    return out


def phaseB_report(df_raw: pd.DataFrame, df_obj: pd.DataFrame) -> str:
    """
    Generates a small text report you can save under results/.
    """
    lines = []
    lines.append(f"Raw rows: {df_raw.shape[0]}")
    lines.append(f"Raw unique objects (varname): {df_raw['varname'].nunique()}")
    lines.append("")
    lines.append(f"Object-level rows: {df_obj.shape[0]}")
    lines.append(f"Object-level unique objects: {df_obj['varname'].nunique()}")
    lines.append("")
    if "target_hbeta_source" in df_obj.columns:
        lines.append("Target source counts (object-level):")
        lines.append(str(df_obj["target_hbeta_source"].value_counts(dropna=False)))
    return "\n".join(lines)
