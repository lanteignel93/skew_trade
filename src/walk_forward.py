"""
Walk-Forward PCA Analysis for Skew Strategy

This module implements walk-forward validation of the PCA-based skew signal:
- Expanding or rolling window PCA (minimum 3 years, configurable max lookback)
- Monthly rebalancing (refit PCA at each month-end)
- Decile classification (no look-ahead bias in bucket design)
- Non-overlapping monthly forward returns
- Window length calibration: structural/predictive metrics across a grid

Usage:
    python walk_forward.py [--rebalance-week 0]                              # month-end, rolling 756d
    python walk_forward.py --max-train-days 1260 --rebalance-week 0          # 5-year rolling
    python walk_forward.py --rebalance-freq biweekly --output-path wf.parquet  # bi-weekly (~26/year)
    python walk_forward.py --calibrate                                       # calibration study
    python walk_forward.py --calibrate --window-grid "756,1260,None"         # custom grid

Output:
    Parquet file with walk-forward results for notebook analysis
"""

import polars as pl
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from datetime import date
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Optional, Union
import json
import argparse
import time


SCRIPT_DIR = Path(__file__).parent


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward PCA analysis."""

    # Input data
    surface_data_path: str = str(SCRIPT_DIR / "data" / "surface_data.parquet")

    # PCA settings
    n_components: int = 6
    skew_pc_index: int = 1  # 0-based, PC2

    # Walk-forward settings
    min_train_days: int = 756  # 3 years minimum
    max_train_days: Optional[int] = 756  # 3-year rolling window (None = expanding)

    # Rebalance timing within month:
    #   0 or -1 = last trading day of month (month-end)
    #   1 = end of week 1 (last trading day of days 1-7)
    #   2 = end of week 2 (last trading day of days 8-14)
    #   3 = end of week 3 (last trading day of days 15-21)
    #   4 = end of week 4 (last trading day of days 22-28)
    rebalance_week: int = 0  # 0 = month-end; ignored when rebalance_freq="biweekly"
    rebalance_freq: str = "monthly"  # "monthly" or "biweekly" (twice per month, ~26/year)

    # Bucket settings (deciles = 10 equal buckets)
    n_buckets: int = 10

    # Column names
    date_col: str = "date"
    surface_prefix: str = "int_surface_vol_"

    # Output
    output_path: str = str(SCRIPT_DIR / "data" / "walk_forward_results.parquet")
    log_path: str = str(SCRIPT_DIR / "walk_forward_log.json")

    # Sign convention: ensure positive PC2 = steep skew
    sign_fix_pc2: bool = True

    @classmethod
    def default(cls) -> "WalkForwardConfig":
        return cls()

    def get_output_suffix(self) -> str:
        """Get suffix for output files based on rebalance timing."""
        if self.rebalance_freq == "biweekly":
            return "biweekly"
        if self.rebalance_week <= 0:
            return "month_end"
        return f"week{self.rebalance_week}"


def get_surface_columns(df: pl.DataFrame, prefix: str) -> tuple[list[str], np.ndarray]:
    """Get surface columns and their moneyness values, sorted by moneyness."""
    cols = []
    xs = []
    for c in df.columns:
        if c.startswith(prefix):
            try:
                x = float(c[len(prefix):])
                cols.append(c)
                xs.append(x)
            except ValueError:
                continue

    if not cols:
        raise ValueError(f"No surface columns found with prefix='{prefix}'")

    # Sort by moneyness
    order = np.argsort(xs)
    cols = [cols[i] for i in order]
    xs = np.array([xs[i] for i in order])

    return cols, xs


def get_rebalance_dates(
    dates: pl.Series,
    rebalance_week: int = 0,
    rebalance_freq: str = "monthly",
) -> pl.DataFrame:
    """
    Get rebalance dates based on timing selection.

    Args:
        dates: Series of trading dates
        rebalance_week: Which week's end to use (ignored when rebalance_freq="biweekly")
            0 or -1 = last trading day of month (month-end)
            1 = end of week 1 (last trading day of days 1-7)
            2 = end of week 2 (last trading day of days 8-14)
            3 = end of week 3 (last trading day of days 15-21)
            4 = end of week 4 (last trading day of days 22-28)
        rebalance_freq: "monthly" (default) or "biweekly" (twice per month, ~26/year).
            Biweekly splits each month into two halves: days 1-14 and days 15+.

    Returns:
        DataFrame with 'rebalance_date' column
    """
    df = pl.DataFrame({"date": dates})

    # Add month and day-of-month
    df = df.with_columns([
        pl.col("date").dt.truncate("1mo").alias("month"),
        pl.col("date").dt.day().alias("day_of_month"),
    ])

    if rebalance_freq == "biweekly":
        # Two rebalance dates per month: last day of days 1-14, last day of days 15+
        first_half = (
            df.filter(pl.col("day_of_month") <= 14)
            .group_by("month")
            .agg(pl.col("date").max().alias("rebalance_date"))
        )
        second_half = (
            df.filter(pl.col("day_of_month") >= 15)
            .group_by("month")
            .agg(pl.col("date").max().alias("rebalance_date"))
        )
        result = pl.concat([first_half, second_half]).sort("rebalance_date")
    elif rebalance_week <= 0:
        # Month-end: last trading day of each month
        result = (
            df.group_by("month")
            .agg(pl.col("date").max().alias("rebalance_date"))
            .sort("month")
        )
    else:
        # Specific week within month
        # Week 1: days 1-7, Week 2: days 8-14, etc.
        week_start = (rebalance_week - 1) * 7 + 1
        week_end = rebalance_week * 7

        result = (
            df.filter(
                (pl.col("day_of_month") >= week_start) &
                (pl.col("day_of_month") <= week_end)
            )
            .group_by("month")
            .agg(pl.col("date").max().alias("rebalance_date"))
            .sort("month")
        )

    return result.select("rebalance_date")


def fit_pca_on_window(
    X: np.ndarray,
    n_components: int,
    skew_pc_index: int,
    sign_fix: bool,
    deep_otm_idx: int,
) -> tuple[PCA, StandardScaler, np.ndarray, np.ndarray]:
    """
    Fit PCA on a data window.

    Returns:
        (pca, scaler, loadings, scores)
    """
    scaler = StandardScaler()
    Z = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(Z)
    loadings = pca.components_.T

    # Sign-fix PC2 so positive = steep skew (positive loading at deep OTM)
    if sign_fix and loadings[deep_otm_idx, skew_pc_index] < 0:
        loadings[:, skew_pc_index] *= -1
        scores[:, skew_pc_index] *= -1

    return pca, scaler, loadings, scores


def project_to_pc2(
    X_new: np.ndarray,
    scaler: StandardScaler,
    pca: PCA,
    skew_pc_index: int,
    sign_flip: bool,
) -> np.ndarray:
    """Project new observations onto fitted PCA, return PC2 scores."""
    Z_new = scaler.transform(X_new)
    scores = pca.transform(Z_new)
    pc2 = scores[:, skew_pc_index]
    if sign_flip:
        pc2 = -pc2
    return pc2


def compute_decile(value: float, historical_values: np.ndarray, n_buckets: int = 10) -> int:
    """
    Compute decile (1-10) based on historical distribution.

    D1 = lowest 10% (flattest skew)
    D10 = highest 10% (steepest skew)
    """
    percentile_rank = np.mean(historical_values <= value)
    decile = int(np.ceil(percentile_rank * n_buckets))
    return max(1, min(n_buckets, decile))  # Clamp to [1, n_buckets]


@dataclass
class RebalMetadata:
    """Per-refit metadata captured during walk-forward."""
    date: date
    n_train_days: int
    pc2_loading: np.ndarray  # PC2 loading vector for this refit
    pc2_evr: float           # PC2 explained variance ratio


def run_walk_forward(
    config: WalkForwardConfig,
    return_metadata: bool = False,
) -> Union[pl.DataFrame, tuple[pl.DataFrame, list[RebalMetadata]]]:
    """
    Run walk-forward PCA analysis.

    At each rebalance date:
    1. Fit PCA on training window (expanding or rolling, per max_train_days)
    2. Compute PC2 score for current observation
    3. Classify into decile based on historical PC2 distribution
    4. Record forward return (ΔPC2 to next rebalance date)

    Args:
        config: Walk-forward configuration
        return_metadata: If True, also return per-refit RebalMetadata list

    Returns:
        DataFrame with walk-forward results, or (DataFrame, metadata_list) if return_metadata
    """
    print("Loading surface data...")
    df = pl.read_parquet(config.surface_data_path)
    df = df.sort(config.date_col)

    # Get surface columns
    surface_cols, x_grid = get_surface_columns(df, config.surface_prefix)

    # Drop ATM column (all zeros after demeaning)
    keep_mask = ~np.isclose(x_grid, 0.0)
    surface_cols_use = [c for c, k in zip(surface_cols, keep_mask) if k]
    x_use = x_grid[keep_mask]
    deep_otm_idx = int(np.argmin(x_use))

    print(f"Using {len(surface_cols_use)} surface columns")
    print(f"Deep OTM index: {deep_otm_idx} (moneyness={x_use[deep_otm_idx]:.2f})")

    # Extract data
    dates = df[config.date_col].to_numpy()
    X = df.select(surface_cols_use).to_numpy().astype(np.float64)

    # Get rebalance dates
    rebal_df = get_rebalance_dates(df[config.date_col], config.rebalance_week, config.rebalance_freq)
    rebalance_dates = rebal_df["rebalance_date"].to_numpy()

    if config.rebalance_freq == "biweekly":
        week_desc = "biweekly"
    elif config.rebalance_week <= 0:
        week_desc = "month-end"
    else:
        week_desc = f"week {config.rebalance_week}"
    print(f"Total observations: {len(dates)}")
    print(f"Rebalance dates ({week_desc}): {len(rebalance_dates)}")
    print(f"Date range: {dates[0]} to {dates[-1]}")

    # Find first valid month-end (need min_train_days before it)
    date_to_idx = {d: i for i, d in enumerate(dates)}

    results = []
    log_entries = []
    rebal_metadata: list[RebalMetadata] = []

    window_desc = f"rolling {config.max_train_days}d" if config.max_train_days else "expanding"
    print(f"Training window: {window_desc}")

    for i, rebal_date in enumerate(rebalance_dates):
        if rebal_date not in date_to_idx:
            continue

        current_idx = date_to_idx[rebal_date]

        # Check if we have enough training data
        if current_idx < config.min_train_days:
            continue

        # Training data: rolling or expanding window up to current month-end
        if config.max_train_days is not None:
            start_idx = max(0, current_idx + 1 - config.max_train_days)
        else:
            start_idx = 0
        X_train = X[start_idx:current_idx + 1]
        dates_train = dates[start_idx:current_idx + 1]

        # Fit PCA on training data
        pca, scaler, loadings, scores_train = fit_pca_on_window(
            X_train,
            config.n_components,
            config.skew_pc_index,
            config.sign_fix_pc2,
            deep_otm_idx,
        )

        # Determine whether project_to_pc2 needs an additional sign flip.
        # fit_pca_on_window() already flips pca.components_ in-place via the
        # loadings view (loadings = pca.components_.T is a NumPy view, not a copy).
        # After that flip, loadings[deep_otm_idx, skew_pc_index] is guaranteed >= 0,
        # so this expression evaluates to False — meaning project_to_pc2 will use
        # pca.transform() directly, which already reflects the corrected sign.
        sign_flip = (config.sign_fix_pc2 and
                     loadings[deep_otm_idx, config.skew_pc_index] < 0)

        assert loadings[deep_otm_idx, config.skew_pc_index] >= 0, (
            f"PC2 sign convention violated at {rebal_date}: "
            f"deep OTM loading = {loadings[deep_otm_idx, config.skew_pc_index]:.4f}"
        )

        # Capture per-refit metadata
        date_as_py = np.datetime64(rebal_date, 'D').astype('datetime64[D]').astype(date)
        rebal_metadata.append(RebalMetadata(
            date=date_as_py,
            n_train_days=len(X_train),
            pc2_loading=loadings[:, config.skew_pc_index].copy(),
            pc2_evr=float(pca.explained_variance_ratio_[config.skew_pc_index]),
        ))

        # Get PC2 scores for training period
        pc2_train = scores_train[:, config.skew_pc_index]

        # Current PC2 score (last observation in training)
        pc2_current = pc2_train[-1]

        # Classify into decile based on historical distribution
        # Use all but current for threshold computation (avoid including current in its own percentile)
        pc2_historical = pc2_train[:-1] if len(pc2_train) > 1 else pc2_train
        decile = compute_decile(pc2_current, pc2_historical, config.n_buckets)

        # Forward return: ΔPC2 to next rebalance date
        fwd_dpc2 = np.nan  # Default
        if i + 1 < len(rebalance_dates):
            next_rebal_date = rebalance_dates[i + 1]
            if next_rebal_date in date_to_idx:
                next_idx = date_to_idx[next_rebal_date]

                # Project next rebalance date observation onto current PCA
                X_next = X[next_idx:next_idx + 1]

                # Check for NaN in input data
                if not np.any(np.isnan(X_next)):
                    pc2_next = project_to_pc2(
                        X_next, scaler, pca, config.skew_pc_index, sign_flip
                    )[0]

                    if np.isfinite(pc2_next) and np.isfinite(pc2_current):
                        fwd_dpc2 = pc2_next - pc2_current

        # Record result
        results.append({
            "date": date_as_py,
            "rebalance_week": config.rebalance_week,
            "pc2_score": pc2_current,
            "decile": decile,
            "fwd_1m_dpc2": fwd_dpc2,
            "n_train_days": len(X_train),
            "pc1_var_ratio": pca.explained_variance_ratio_[0],
            "pc2_var_ratio": pca.explained_variance_ratio_[config.skew_pc_index],
            "pc2_historical_mean": float(np.mean(pc2_historical)),
            "pc2_historical_std": float(np.std(pc2_historical)),
        })

        # Log entry
        log_entries.append({
            "date": str(rebal_date),
            "n_train": len(X_train),
            "pc2_score": float(pc2_current),
            "decile": decile,
            "evr": [float(x) for x in pca.explained_variance_ratio_],
        })

    print(f"\nGenerated {len(results)} walk-forward observations")

    # Create output DataFrame
    result_df = pl.DataFrame(results)

    # Save results
    result_df.write_parquet(config.output_path)
    print(f"Results saved to: {config.output_path}")

    # Save log
    log_data = {
        "config": {
            "surface_data_path": config.surface_data_path,
            "min_train_days": config.min_train_days,
            "max_train_days": config.max_train_days,
            "n_components": config.n_components,
            "n_buckets": config.n_buckets,
            "rebalance_week": config.rebalance_week,
        },
        "summary": {
            "n_observations": len(results),
            "date_range": [str(result_df["date"].min()), str(result_df["date"].max())],
            "decile_counts": result_df.group_by("decile").len().sort("decile").to_dicts(),
        },
        "entries": log_entries[:10],  # First 10 for inspection
    }

    with open(config.log_path, "w") as f:
        json.dump(log_data, f, indent=2, default=str)
    print(f"Log saved to: {config.log_path}")

    if return_metadata:
        return result_df, rebal_metadata
    return result_df


def calibrate_window_length(
    config: WalkForwardConfig,
    window_grid: Optional[list[Optional[int]]] = None,
) -> pl.DataFrame:
    """
    Run walk-forward for multiple window lengths and compare structural/predictive metrics.

    Args:
        config: Base config (rebalance_week, surface_data_path, etc.)
        window_grid: List of max_train_days values. None entries = expanding window.
                     Default: [756, 1008, 1260, 1512, 1764, None] (~3y–7y + expanding)

    Returns:
        DataFrame with one row per window length and all metrics.
        Also saves calibration_results.parquet.
    """
    from scipy.stats import spearmanr, chi2 as chi2_dist

    if window_grid is None:
        window_grid = [756, 1008, 1260, 1512, 1764, None]

    print("=" * 70)
    print("WINDOW LENGTH CALIBRATION")
    print("=" * 70)
    print(f"Grid: {[w if w is not None else 'expanding' for w in window_grid]}")
    if config.rebalance_freq == "biweekly":
        rebal_desc = "biweekly"
    elif config.rebalance_week <= 0:
        rebal_desc = "month-end"
    else:
        rebal_desc = f"week {config.rebalance_week}"
    print(f"Rebalance: {rebal_desc}")
    print()

    rows = []
    t0 = time.time()

    for w in window_grid:
        label = f"{w}d" if w is not None else "expanding"
        print(f"  Running {label}...", end=" ", flush=True)
        t_w = time.time()

        # Build config for this window
        cfg = WalkForwardConfig(
            surface_data_path=config.surface_data_path,
            n_components=config.n_components,
            skew_pc_index=config.skew_pc_index,
            min_train_days=config.min_train_days,
            max_train_days=w,
            rebalance_week=config.rebalance_week,
            rebalance_freq=config.rebalance_freq,
            n_buckets=config.n_buckets,
            date_col=config.date_col,
            surface_prefix=config.surface_prefix,
            sign_fix_pc2=config.sign_fix_pc2,
            # Suppress file output during calibration
            output_path=str(SCRIPT_DIR / "data" / f"_calibration_tmp_{label}.parquet"),
            log_path=str(SCRIPT_DIR / f"_calibration_tmp_{label}.json"),
        )

        result_df, metadata = run_walk_forward(cfg, return_metadata=True)

        # --- Metric 1: Loading stability (cosine similarity between consecutive refits) ---
        cos_sims = []
        for j in range(1, len(metadata)):
            v1 = metadata[j - 1].pc2_loading
            v2 = metadata[j].pc2_loading
            dot = np.dot(v1, v2)
            norm = np.linalg.norm(v1) * np.linalg.norm(v2)
            if norm > 0:
                cos_sims.append(dot / norm)
        loading_stability = float(np.mean(cos_sims)) if cos_sims else np.nan

        # --- Metric 2: Signal rank correlation (Spearman between decile and fwd ΔPC2) ---
        valid = result_df.filter(
            pl.col("fwd_1m_dpc2").is_not_null() & pl.col("fwd_1m_dpc2").is_not_nan()
        )
        if valid.height > 5:
            rho, rho_p = spearmanr(
                valid["decile"].to_numpy(), valid["fwd_1m_dpc2"].to_numpy()
            )
        else:
            rho, rho_p = np.nan, np.nan

        # --- Metric 3: Monotonicity violations ---
        # Count adjacent decile pairs where mean fwd ΔPC2 doesn't decrease
        decile_means = (
            valid.group_by("decile")
            .agg(pl.col("fwd_1m_dpc2").mean().alias("mean_dpc2"))
            .sort("decile")
        )
        dm_vals = decile_means["mean_dpc2"].to_numpy()
        mono_violations = int(np.sum(np.diff(dm_vals) > 0)) if len(dm_vals) > 1 else config.n_buckets

        # --- Metric 4: Decile uniformity (chi-squared + entropy) ---
        decile_counts = (
            result_df.group_by("decile").len().sort("decile")["len"].to_numpy()
        )
        n_total = decile_counts.sum()
        expected = n_total / config.n_buckets
        chi2_stat = float(np.sum((decile_counts - expected) ** 2 / expected))
        n_dec = len(decile_counts)
        chi2_p = 1.0 - float(chi2_dist.cdf(chi2_stat, df=n_dec - 1)) if n_dec > 1 else np.nan
        # Entropy (normalized to [0, 1] where 1 = perfectly uniform)
        probs = decile_counts / n_total
        probs = probs[probs > 0]
        entropy = float(-np.sum(probs * np.log(probs)) / np.log(config.n_buckets))

        # --- Metric 5: EVR stability (CV of PC2 explained variance ratio across refits) ---
        evrs = np.array([m.pc2_evr for m in metadata])
        evr_cv = float(np.std(evrs) / np.mean(evrs)) if np.mean(evrs) > 0 else np.nan

        # --- Supplementary: hit rates for extreme deciles ---
        d1_3 = valid.filter(pl.col("decile") <= 3)
        d9_10 = valid.filter(pl.col("decile") >= 9)
        hit_long = float((d1_3["fwd_1m_dpc2"] > 0).mean()) if d1_3.height > 0 else np.nan
        hit_short = float((d9_10["fwd_1m_dpc2"] < 0).mean()) if d9_10.height > 0 else np.nan

        elapsed_w = time.time() - t_w
        print(f"done ({elapsed_w:.1f}s, {len(metadata)} refits)")

        rows.append({
            "window": label,
            "max_train_days": w if w is not None else -1,
            "n_obs": result_df.height,
            "n_refits": len(metadata),
            "loading_stability": loading_stability,
            "spearman_rho": rho,
            "spearman_p": rho_p,
            "mono_violations": mono_violations,
            "chi2_stat": chi2_stat,
            "chi2_p": chi2_p,
            "entropy": entropy,
            "evr_cv": evr_cv,
            "evr_mean": float(np.mean(evrs)),
            "hit_d1_3_long": hit_long,
            "hit_d9_10_short": hit_short,
            "n_d1_3": d1_3.height,
            "n_d9_10": d9_10.height,
        })

    elapsed_total = time.time() - t0
    cal_df = pl.DataFrame(rows)

    # Clean up temp files
    for w in window_grid:
        label = f"{w}d" if w is not None else "expanding"
        for ext in [".parquet", ".json"]:
            tmp = SCRIPT_DIR / "data" / f"_calibration_tmp_{label}{ext}"
            if tmp.exists():
                tmp.unlink()

    # Save results
    out_path = SCRIPT_DIR / "calibration_results.parquet"
    cal_df.write_parquet(str(out_path))

    # --- Print formatted summary ---
    print(f"\nTotal calibration time: {elapsed_total:.1f}s")
    print()
    print("=" * 100)
    print("CALIBRATION RESULTS")
    print("=" * 100)
    print()

    # Header
    print(f"{'Window':<12} {'Load.Stab':>9} {'Spearman':>9} {'MonoViol':>9} "
          f"{'Chi2':>8} {'Entropy':>8} {'EVR_CV':>8} │ "
          f"{'D1-3 Hit':>8} {'D9-10 Hit':>9} {'n_obs':>6}")
    print("─" * 100)

    for r in rows:
        w_str = r["window"]
        print(
            f"{w_str:<12} "
            f"{r['loading_stability']:>9.4f} "
            f"{r['spearman_rho']:>9.4f} "
            f"{r['mono_violations']:>9d} "
            f"{r['chi2_stat']:>8.1f} "
            f"{r['entropy']:>8.4f} "
            f"{r['evr_cv']:>8.4f} │ "
            f"{r['hit_d1_3_long']:>8.1%} "
            f"{r['hit_d9_10_short']:>9.1%} "
            f"{r['n_obs']:>6d}"
        )

    print("─" * 100)
    print()
    print("Interpretation:")
    print("  Loading stability:  higher = more consistent PC2 structure across refits")
    print("  Spearman rho:       more negative = stronger monotonic signal (decile → fwd ΔPC2)")
    print("  Mono violations:    lower = cleaner decile ordering (max possible = n_buckets-1)")
    print("  Chi2:               lower = more uniform decile distribution")
    print("  Entropy:            higher = more uniform (1.0 = perfect)")
    print("  EVR CV:             lower = more stable PC2 variance explained")
    print()
    print(f"Results saved to: {out_path}")

    return cal_df


def print_summary(df: pl.DataFrame) -> None:
    """Print summary statistics of walk-forward results."""
    print("\n" + "=" * 60)
    print("WALK-FORWARD SUMMARY")
    print("=" * 60)

    print(f"\nObservations: {df.height}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Valid forward returns (excluding null AND NaN values)
    valid = df.filter(
        pl.col("fwd_1m_dpc2").is_not_null() &
        pl.col("fwd_1m_dpc2").is_not_nan()
    )
    print(f"Valid forward returns: {valid.height}")

    # Check for NaN values
    nan_count = df.filter(pl.col("fwd_1m_dpc2").is_nan()).height
    if nan_count > 0:
        print(f"WARNING: {nan_count} observations have NaN forward returns")

    print("\n--- Decile Distribution ---")
    print(df.group_by("decile").len().sort("decile"))

    print("\n--- Forward ΔPC2 by Decile ---")
    fwd_stats = (
        valid
        .group_by("decile")
        .agg([
            pl.len().alias("n"),
            pl.col("fwd_1m_dpc2").mean().alias("mean_dpc2"),
            pl.col("fwd_1m_dpc2").std().alias("std_dpc2"),
            (pl.col("fwd_1m_dpc2") > 0).mean().alias("pct_positive"),
        ])
        .sort("decile")
    )
    print(fwd_stats)

    print("\n--- PC Variance Ratios (mean across fits) ---")
    print(f"PC1: {df['pc1_var_ratio'].mean():.4f}")
    print(f"PC2: {df['pc2_var_ratio'].mean():.4f}")


def run_single_config(config: WalkForwardConfig) -> pl.DataFrame:
    """Run walk-forward for a single configuration."""
    if config.rebalance_freq == "biweekly":
        week_desc = "biweekly"
    elif config.rebalance_week <= 0:
        week_desc = "month-end"
    else:
        week_desc = f"week {config.rebalance_week}"
    print("\n" + "=" * 70)
    print(f"Running walk-forward: {week_desc}")
    print("=" * 70)
    print(f"  Surface data: {config.surface_data_path}")
    print(f"  Min train days: {config.min_train_days}")
    print(f"  N buckets: {config.n_buckets}")
    print(f"  Output: {config.output_path}")

    result_df = run_walk_forward(config)
    print_summary(result_df)

    return result_df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Walk-forward PCA analysis")
    parser.add_argument(
        "--surface-path",
        type=str,
        default=None,
        help="Path to surface data parquet"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path for output parquet (only used with --rebalance-week)"
    )
    parser.add_argument(
        "--min-train-days",
        type=int,
        default=None,
        help="Minimum training days (default: 756 = 3 years)"
    )
    parser.add_argument(
        "--n-buckets",
        type=int,
        default=None,
        help="Number of buckets/deciles (default: 10)"
    )
    parser.add_argument(
        "--max-train-days",
        type=int,
        default=None,
        help="Maximum training window in trading days. None = expanding (default)."
    )
    parser.add_argument(
        "--rebalance-week",
        type=int,
        default=None,
        help="Which week to rebalance: 0=month-end, 1-4=end of that week. "
             "If not specified, runs ALL 5 options (0-4). Ignored when --rebalance-freq=biweekly."
    )
    parser.add_argument(
        "--rebalance-freq",
        type=str,
        default="monthly",
        choices=["monthly", "biweekly"],
        help="Rebalance frequency: 'monthly' (default) or 'biweekly' (twice per month, ~26/year)."
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Run window length calibration study instead of normal walk-forward."
    )
    parser.add_argument(
        "--window-grid",
        type=str,
        default=None,
        help="Comma-separated window lengths for calibration (use 'None' for expanding). "
             "Example: '756,1260,None'"
    )

    args = parser.parse_args()

    # Build base config
    base_config = WalkForwardConfig.default()

    if args.surface_path:
        base_config.surface_data_path = args.surface_path
    if args.min_train_days:
        base_config.min_train_days = args.min_train_days
    if args.n_buckets:
        base_config.n_buckets = args.n_buckets
    if args.max_train_days is not None:
        base_config.max_train_days = args.max_train_days
    base_config.rebalance_freq = args.rebalance_freq

    # --- Calibration mode ---
    if args.calibrate:
        # Default rebalance_week=0 for calibration if not specified
        if args.rebalance_week is not None:
            base_config.rebalance_week = args.rebalance_week
        else:
            base_config.rebalance_week = 0

        # Parse window grid
        grid = None
        if args.window_grid:
            grid = []
            for tok in args.window_grid.split(","):
                tok = tok.strip()
                if tok.lower() == "none":
                    grid.append(None)
                else:
                    grid.append(int(tok))

        cal_df = calibrate_window_length(base_config, window_grid=grid)
        return {"calibration": cal_df}

    # --- Biweekly walk-forward mode ---
    if args.rebalance_freq == "biweekly":
        config = WalkForwardConfig(
            surface_data_path=base_config.surface_data_path,
            n_components=base_config.n_components,
            skew_pc_index=base_config.skew_pc_index,
            min_train_days=base_config.min_train_days,
            max_train_days=base_config.max_train_days,
            rebalance_week=0,
            rebalance_freq="biweekly",
            n_buckets=base_config.n_buckets,
            date_col=base_config.date_col,
            surface_prefix=base_config.surface_prefix,
            sign_fix_pc2=base_config.sign_fix_pc2,
        )
        if args.output_path:
            config.output_path = args.output_path
            config.log_path = args.output_path.replace(".parquet", "_log.json")
        else:
            config.output_path = str(SCRIPT_DIR / "data" / "walk_forward_results_biweekly.parquet")
            config.log_path = str(SCRIPT_DIR / "walk_forward_log_biweekly.json")
        result_df = run_single_config(config)
        return {"biweekly": result_df}

    # --- Normal monthly walk-forward mode ---
    # Determine which weeks to run
    if args.rebalance_week is not None:
        # Single week specified
        weeks_to_run = [args.rebalance_week]
    else:
        # Run all 5 options: month-end (0) and weeks 1-4
        weeks_to_run = [0, 1, 2, 3, 4]
        print("No --rebalance-week specified. Running all 5 rebalance timings...")

    results = {}

    for week in weeks_to_run:
        # Create config for this week
        config = WalkForwardConfig(
            surface_data_path=base_config.surface_data_path,
            n_components=base_config.n_components,
            skew_pc_index=base_config.skew_pc_index,
            min_train_days=base_config.min_train_days,
            max_train_days=base_config.max_train_days,
            rebalance_week=week,
            n_buckets=base_config.n_buckets,
            date_col=base_config.date_col,
            surface_prefix=base_config.surface_prefix,
            sign_fix_pc2=base_config.sign_fix_pc2,
        )

        # Set output paths
        if args.output_path and len(weeks_to_run) == 1:
            config.output_path = args.output_path
            config.log_path = args.output_path.replace(".parquet", "_log.json")
        else:
            suffix = config.get_output_suffix()
            config.output_path = str(SCRIPT_DIR / "data" / f"walk_forward_results_{suffix}.parquet")
            config.log_path = str(SCRIPT_DIR / f"walk_forward_log_{suffix}.json")

        # Run
        result_df = run_single_config(config)
        results[week] = result_df

    # Print combined summary if multiple weeks
    if len(weeks_to_run) > 1:
        print("\n" + "=" * 70)
        print("COMBINED SUMMARY ACROSS ALL REBALANCE TIMINGS")
        print("=" * 70)

        for week, df in results.items():
            week_desc = "month-end" if week <= 0 else f"week {week}"
            valid = df.filter(
                pl.col("fwd_1m_dpc2").is_not_null() &
                pl.col("fwd_1m_dpc2").is_not_nan()
            )

            # Quick hit rate for extreme deciles
            d1 = valid.filter(pl.col("decile") == 1)
            d10 = valid.filter(pl.col("decile") == 10)

            d1_hit = (d1["fwd_1m_dpc2"] > 0).mean() if d1.height > 0 else float("nan")
            d10_hit = (d10["fwd_1m_dpc2"] < 0).mean() if d10.height > 0 else float("nan")

            print(f"\n{week_desc}:")
            print(f"  Observations: {df.height}")
            print(f"  D1 (flat) hit rate (expect +): {d1_hit:.1%} (n={d1.height})")
            print(f"  D10 (steep) hit rate (expect -): {d10_hit:.1%} (n={d10.height})")

    return results


if __name__ == "__main__":
    main()
