import argparse
import polars as pl
import numpy as np
import datetime
from dataclasses import dataclass
from pathlib import Path
from scipy.special import erf

# Directory where this script lives
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"


@dataclass
class SurfaceConfig:
    """Configuration for vol surface construction."""
    interpolation_points: list[float]
    target_years: float
    start_date: datetime.date
    ticker: str
    iv_spread_threshold: float = 0.1
    # Data quality filters
    max_moneyness_extrap_gap: float = 0.10  # Drop if extrapolating more than this in moneyness
    min_expiries: int = 2  # Drop days with fewer expiries (can't interpolate term structure)
    # Output paths
    surface_output_path: str = str(DATA_DIR / "surface_data.parquet")
    log_dir: Path = SCRIPT_DIR

    @classmethod
    def default(cls) -> "SurfaceConfig":
        return cls(
            interpolation_points=[-0.3, -0.25, -0.2, -0.17, -0.14, -0.12, -0.1,
                                  -0.09, -0.08, -0.07, -0.06, -0.05, -0.04,
                                  -0.03, -0.02, -0.01, 0],
            target_years=3/12,
            start_date=datetime.date(2015, 1, 1),
            ticker="SPXW",
        )


def interp_by_moneyness(
    moneyness: np.ndarray,
    iv: np.ndarray,
    target_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Linear interpolation of IV by moneyness values (not row indices).

    Args:
        moneyness: Array of actual moneyness values (sorted)
        iv: Array of IV values corresponding to moneyness
        target_points: Target moneyness points to interpolate to

    Returns:
        (interpolated_iv, is_extrapolated): IV values and boolean mask for extrapolation
    """
    # Remove any NaN values
    valid_mask = ~np.isnan(iv) & ~np.isnan(moneyness)
    if valid_mask.sum() < 2:
        # Not enough points to interpolate
        return np.full_like(target_points, np.nan), np.ones_like(target_points, dtype=bool)

    m_valid = moneyness[valid_mask]
    iv_valid = iv[valid_mask]

    # Sort by moneyness
    sort_idx = np.argsort(m_valid)
    m_sorted = m_valid[sort_idx]
    iv_sorted = iv_valid[sort_idx]

    # Interpolate (numpy.interp does linear interpolation, extrapolates at boundaries)
    interp_iv = np.interp(target_points, m_sorted, iv_sorted)

    # Mark extrapolated points (outside the range of actual data)
    min_m, max_m = m_sorted.min(), m_sorted.max()
    is_extrapolated = (target_points < min_m) | (target_points > max_m)

    return interp_iv, is_extrapolated


def interp_by_variance(
    years: np.ndarray,
    iv: np.ndarray,
    target_years: float,
) -> tuple[float, bool]:
    """
    Interpolate IV in variance space (sqrt-of-time scaling).

    Total variance w = σ² * T is interpolated linearly, then σ is recovered.

    Args:
        years: Array of time-to-expiry values
        iv: Array of IV values
        target_years: Target time-to-expiry

    Returns:
        (interpolated_iv, is_extrapolated)
    """
    valid_mask = ~np.isnan(iv) & ~np.isnan(years) & (years > 0)
    if valid_mask.sum() < 2:
        return np.nan, True

    t_valid = years[valid_mask]
    iv_valid = iv[valid_mask]

    # Convert to total variance
    w_valid = iv_valid**2 * t_valid

    # Sort by time
    sort_idx = np.argsort(t_valid)
    t_sorted = t_valid[sort_idx]
    w_sorted = w_valid[sort_idx]

    # Interpolate total variance
    w_target = np.interp(target_years, t_sorted, w_sorted)

    # Convert back to IV
    iv_target = np.sqrt(w_target / target_years) if target_years > 0 else np.nan

    # Check if extrapolated
    is_extrapolated = (target_years < t_sorted.min()) | (target_years > t_sorted.max())

    return iv_target, is_extrapolated


def load_option_data(config: SurfaceConfig) -> pl.DataFrame:
    """Load and filter option data from local parquet."""
    df = pl.read_parquet(DATA_DIR / "options_raw.parquet")
    df = df.filter(pl.col('date') > config.start_date)
    df = df.filter(pl.col('ticker') == config.ticker)

    # Compute forward and moneyness (f_price already in parquet, but recompute for safety)
    if 'f_price' not in df.columns:
        df = df.with_columns([
            (pl.col('u_price') * (pl.col('rate') * pl.col('t_years')).exp()).alias('f_price'),
        ])
    df = df.with_columns([
        (pl.col('strike') / pl.col('f_price')).log().alias('moneyness'),
    ])

    df = df.filter(pl.col('cp') == 'Put')

    # Keep OTM puts and slightly ITM puts (up to moneyness ~ +0.05)
    # This allows proper interpolation to ATM (moneyness = 0)
    # We'll drop moneyness > 0 after interpolation
    df = df.filter(pl.col('moneyness') <= 0.05)

    # Data quality filters (applied before IV spread filter)
    df = df.filter(
        (pl.col('surface_vol') > 0) &               # Remove zero-vol 
        (pl.col('surface_vol') <= 3.0) &            # Remove extreme outliers (>300% vol)
        (pl.col('ask_vol') >= pl.col('bid_vol')) &  # Remove crossed IV markets
        (pl.col('delta') != 0) &                    # Remove worthless/expired (zero greeks)
        (pl.col('vega') != 0) &                     # Remove worthless/expired (zero greeks)
        (pl.col('theta') <= 0) &                    # Remove positive-theta puts (data anomaly)
        (pl.col('t_years') > 0)                     # Remove non-positive DTE
    )

    # IV spread filter
    df = df.with_columns([
        (pl.col('ask_vol') - pl.col('bid_vol')).abs().alias('spreadIV'),
    ])
    df = df.filter(
        ((pl.col('spreadIV') / pl.col('surface_vol')) < config.iv_spread_threshold) &
        (pl.col('bid_vol') > 0) &
        (pl.col('ask_vol') > 0)
    )

    # Keep lowest spread per strike
    lowest_spread = df.sort(['date', 'expiration', 'cp', 'spreadIV'])
    lowest_spread = lowest_spread.filter(
        pl.col('spreadIV') == pl.col('spreadIV').min().over("date", "expiration", "strike")
    )
    df = df.join(
        lowest_spread.select(["date", 'expiration', 'strike', 'cp']),
        on=['date', 'expiration', 'strike', 'cp'],
        how='inner'
    )

    return df


def detect_arbitrage_dates(
    df: pl.DataFrame,
    price_tol: float = 0.10,
    intrinsic_tol: float = 0.05,
) -> tuple[set, pl.DataFrame]:
    """Detect dates with no-arbitrage violations in option prices.

    Checks:
      1. Put spread monotonicity: P(K1) <= P(K2) for K1 < K2 (same expiry)
      2. Butterfly convexity: λP(K1) + (1-λ)P(K3) - P(K2) >= 0 (consecutive triplets)
      3. Intrinsic value floor: P >= max(0, (K-F)*exp(-rT))

    Args:
        df: Option data with columns [date, expiration, strike, cp,
            c_price, f_price, rate, t_years].
        price_tol: Dollar tolerance for put spread and butterfly checks.
        intrinsic_tol: Dollar tolerance for intrinsic value check.

    Returns:
        (bad_dates, diagnostics_df): Set of dates to exclude and violation details.
    """
    puts = (
        df.filter(pl.col('cp') == 'Put')
        .sort(['date', 'expiration', 'strike'])
    )

    diag_parts: list[pl.DataFrame] = []

    # --- Check 1: Put spread monotonicity ---
    # Within same (date, expiry), sorted by strike ascending:
    # c_price at lower strike should be <= c_price at higher strike + tol
    mono = puts.with_columns([
        pl.col('c_price').shift(-1).over('date', 'expiration').alias('_next_prc'),
        pl.col('strike').shift(-1).over('date', 'expiration').alias('_next_xx'),
    ]).filter(pl.col('_next_prc').is_not_null())

    mono_v = mono.filter(pl.col('c_price') > pl.col('_next_prc') + price_tol)
    if mono_v.height > 0:
        diag_parts.append(mono_v.select([
            'date', 'expiration',
            pl.lit('put_spread').alias('check'),
            pl.col('strike').cast(pl.Float64).alias('strike'),
            (pl.col('c_price') - pl.col('_next_prc')).cast(pl.Float64).alias('violation_amount'),
            pl.concat_str([
                pl.lit('K='), pl.col('strike').cast(pl.Utf8),
                pl.lit(' $'), pl.col('c_price').round(2).cast(pl.Utf8),
                pl.lit(' > K='), pl.col('_next_xx').cast(pl.Utf8),
                pl.lit(' $'), pl.col('_next_prc').round(2).cast(pl.Utf8),
            ]).alias('detail'),
        ]))

    # --- Check 2: Butterfly convexity ---
    # Consecutive triplet K1 < K2 < K3:
    # λ = (K3-K2)/(K3-K1), butterfly = λ*P1 + (1-λ)*P3 - P2 >= 0
    bfly = puts.with_columns([
        pl.col('strike').shift(-1).over('date', 'expiration').alias('_K2'),
        pl.col('strike').shift(-2).over('date', 'expiration').alias('_K3'),
        pl.col('c_price').shift(-1).over('date', 'expiration').alias('_P2'),
        pl.col('c_price').shift(-2).over('date', 'expiration').alias('_P3'),
    ]).filter(pl.col('_K3').is_not_null())

    bfly = bfly.with_columns(
        ((pl.col('_K3') - pl.col('_K2')) / (pl.col('_K3') - pl.col('strike'))).alias('_lam'),
    ).with_columns(
        (pl.col('_lam') * pl.col('c_price')
         + (1 - pl.col('_lam')) * pl.col('_P3')
         - pl.col('_P2')).alias('_bfly_val'),
    )

    bfly_v = bfly.filter(pl.col('_bfly_val') < -price_tol)
    if bfly_v.height > 0:
        diag_parts.append(bfly_v.select([
            'date', 'expiration',
            pl.lit('butterfly').alias('check'),
            pl.col('_K2').cast(pl.Float64).alias('strike'),
            (-pl.col('_bfly_val')).cast(pl.Float64).alias('violation_amount'),
            pl.concat_str([
                pl.lit('K1='), pl.col('strike').cast(pl.Utf8),
                pl.lit(' K2='), pl.col('_K2').cast(pl.Utf8),
                pl.lit(' K3='), pl.col('_K3').cast(pl.Utf8),
                pl.lit(' bfly=$'), pl.col('_bfly_val').round(4).cast(pl.Utf8),
            ]).alias('detail'),
        ]))

    # --- Check 3: Intrinsic value floor ---
    # European put: P >= max(0, (K - F) * exp(-rT))
    intrinsic = puts.with_columns(
        pl.max_horizontal(
            pl.lit(0.0),
            (pl.col('strike') - pl.col('f_price'))
            * (-pl.col('rate') * pl.col('t_years')).exp(),
        ).alias('_intrinsic'),
    )

    intr_v = intrinsic.filter(pl.col('c_price') < pl.col('_intrinsic') - intrinsic_tol)
    if intr_v.height > 0:
        diag_parts.append(intr_v.select([
            'date', 'expiration',
            pl.lit('intrinsic_floor').alias('check'),
            pl.col('strike').cast(pl.Float64).alias('strike'),
            (pl.col('_intrinsic') - pl.col('c_price')).cast(pl.Float64).alias('violation_amount'),
            pl.concat_str([
                pl.lit('K='), pl.col('strike').cast(pl.Utf8),
                pl.lit(' $'), pl.col('c_price').round(2).cast(pl.Utf8),
                pl.lit(' < intrinsic=$'), pl.col('_intrinsic').round(2).cast(pl.Utf8),
            ]).alias('detail'),
        ]))

    if diag_parts:
        diag_df = pl.concat(diag_parts).sort(['date', 'check', 'expiration'])
        bad_dates = set(diag_df['date'].unique().to_list())
    else:
        diag_df = pl.DataFrame(schema={
            'date': pl.Date, 'expiration': pl.Date, 'check': pl.Utf8,
            'strike': pl.Float64, 'violation_amount': pl.Float64, 'detail': pl.Utf8,
        })
        bad_dates = set()

    return bad_dates, diag_df


def diagnose_surface_arbitrage(
    fixedterm_df: pl.DataFrame,
    price_tol: float = 0.10,
) -> tuple[dict, pl.DataFrame]:
    """Diagnose no-arbitrage violations on the interpolated vol surface.

    Converts surface IVs to Black-Scholes put prices, then checks:
      1. Put spread monotonicity: P(K_i) <= P(K_{i+1}) for K_i < K_{i+1}
      2. Butterfly convexity: weighted butterfly >= 0 for consecutive triplets

    Runs on the fixed-term surface (absolute IVs, before ATM demeaning).
    This is diagnostic — it reports but does not filter.

    Args:
        fixedterm_df: Long-format surface from build_fixedterm_surface() with
            columns [date, u_price, f_price, t_years, moneyness, surface_vol].
        price_tol: Dollar tolerance for violations.

    Returns:
        (summary, violations_df)
    """
    df = fixedterm_df.filter(
        pl.col('surface_vol').is_not_null() & pl.col('surface_vol').is_finite() & (pl.col('surface_vol') > 0)
    ).sort(['date', 'moneyness'])

    # Compute BS put prices from surface IVs
    F = df['f_price'].to_numpy().astype(np.float64)
    x = df['moneyness'].to_numpy().astype(np.float64)
    vol = df['surface_vol'].to_numpy().astype(np.float64)
    T = df['t_years'].to_numpy().astype(np.float64)
    S = df['u_price'].to_numpy().astype(np.float64)

    rate = np.log(F / S) / T
    K = F * np.exp(x)
    sqrt_t = np.sqrt(T)
    d1 = (-x + 0.5 * vol**2 * T) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    _ncdf = lambda z: 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    discount = np.exp(-rate * T)
    bs_price = discount * (K * _ncdf(-d2) - F * _ncdf(-d1))

    df = df.with_columns([
        pl.Series('bs_put_price', bs_price),
        pl.Series('strike', K),
    ])

    diag_parts: list[pl.DataFrame] = []

    # --- Check 1: Put spread monotonicity ---
    mono = df.with_columns([
        pl.col('bs_put_price').shift(-1).over('date').alias('_next_prc'),
        pl.col('moneyness').shift(-1).over('date').alias('_next_x'),
    ]).filter(pl.col('_next_prc').is_not_null())

    mono_v = mono.filter(pl.col('bs_put_price') > pl.col('_next_prc') + price_tol)
    if mono_v.height > 0:
        diag_parts.append(mono_v.select([
            'date',
            pl.lit('put_spread').alias('check'),
            pl.col('moneyness').alias('moneyness_point'),
            (pl.col('bs_put_price') - pl.col('_next_prc')).alias('violation_amount'),
            pl.concat_str([
                pl.lit('x='), pl.col('moneyness').round(3).cast(pl.Utf8),
                pl.lit(' P=$'), pl.col('bs_put_price').round(4).cast(pl.Utf8),
                pl.lit(' > x='), pl.col('_next_x').round(3).cast(pl.Utf8),
                pl.lit(' P=$'), pl.col('_next_prc').round(4).cast(pl.Utf8),
            ]).alias('detail'),
        ]))

    # --- Check 2: Butterfly convexity ---
    bfly = df.with_columns([
        pl.col('strike').shift(-1).over('date').alias('_K2'),
        pl.col('strike').shift(-2).over('date').alias('_K3'),
        pl.col('bs_put_price').shift(-1).over('date').alias('_P2'),
        pl.col('bs_put_price').shift(-2).over('date').alias('_P3'),
        pl.col('moneyness').shift(-1).over('date').alias('_x2'),
    ]).filter(pl.col('_K3').is_not_null())

    bfly = bfly.with_columns(
        ((pl.col('_K3') - pl.col('_K2')) / (pl.col('_K3') - pl.col('strike'))).alias('_lam'),
    ).with_columns(
        (pl.col('_lam') * pl.col('bs_put_price')
         + (1.0 - pl.col('_lam')) * pl.col('_P3')
         - pl.col('_P2')).alias('_bfly_val'),
    )

    bfly_v = bfly.filter(pl.col('_bfly_val') < -price_tol)
    if bfly_v.height > 0:
        diag_parts.append(bfly_v.select([
            'date',
            pl.lit('butterfly').alias('check'),
            pl.col('_x2').alias('moneyness_point'),
            (-pl.col('_bfly_val')).alias('violation_amount'),
            pl.concat_str([
                pl.lit('x1='), pl.col('moneyness').round(3).cast(pl.Utf8),
                pl.lit(' x2='), pl.col('_x2').round(3).cast(pl.Utf8),
                pl.lit(' bfly=$'), pl.col('_bfly_val').round(4).cast(pl.Utf8),
            ]).alias('detail'),
        ]))

    n_total = df['date'].n_unique()

    if diag_parts:
        violations_df = pl.concat(diag_parts).sort(['date', 'check'])
        n_bad = violations_df['date'].n_unique()
        n_put_spread = violations_df.filter(pl.col('check') == 'put_spread').height
        n_butterfly = violations_df.filter(pl.col('check') == 'butterfly').height
    else:
        violations_df = pl.DataFrame(schema={
            'date': pl.Date, 'check': pl.Utf8,
            'moneyness_point': pl.Float64, 'violation_amount': pl.Float64,
            'detail': pl.Utf8,
        })
        n_bad = 0
        n_put_spread = 0
        n_butterfly = 0

    summary = {
        'n_dates_checked': n_total,
        'n_dates_with_violations': n_bad,
        'pct_dates_violated': n_bad / n_total * 100 if n_total > 0 else 0,
        'n_put_spread_violations': n_put_spread,
        'n_butterfly_violations': n_butterfly,
        'total_violations': n_put_spread + n_butterfly,
    }

    return summary, violations_df


def build_moneyness_surface(
    df: pl.DataFrame,
    config: SurfaceConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Build moneyness-interpolated surface using actual x-value interpolation.

    Returns:
        (surface_df, extrapolation_stats): Surface data and extrapolation diagnostics
    """
    group_keys = ["date", "ticker", "expiration"]
    target_points = np.array(config.interpolation_points, dtype=np.float64)

    # Get unique curves
    unique_curves = df.select(group_keys + ["t_years", "u_price", "f_price"]).unique()

    results = []
    extrap_records = []

    # Process each curve
    for curve_row in unique_curves.iter_rows(named=True):
        trading_date = curve_row["date"]
        ticker = curve_row["ticker"]
        expiration = curve_row["expiration"]
        t_years = curve_row["t_years"]
        uclose = curve_row["u_price"]
        uforward = curve_row["f_price"]

        # Get data for this curve
        curve_data = df.filter(
            (pl.col("date") == trading_date) &
            (pl.col("ticker") == ticker) &
            (pl.col("expiration") == expiration)
        )

        if curve_data.height < 2:
            continue

        moneyness = curve_data["moneyness"].to_numpy()
        iv = curve_data["surface_vol"].to_numpy()

        # Interpolate using actual moneyness values
        interp_iv, is_extrap = interp_by_moneyness(moneyness, iv, target_points)

        # Record extrapolation stats
        for i, (m, ext) in enumerate(zip(target_points, is_extrap)):
            extrap_records.append({
                "date": trading_date,
                "expiration": expiration,
                "moneyness": m,
                "is_extrapolated": ext,
                "min_actual_moneyness": float(moneyness.min()),
                "max_actual_moneyness": float(moneyness.max()),
                "n_actual_points": len(moneyness),
            })

        # Build result rows
        for m, iv_val in zip(target_points, interp_iv):
            results.append({
                "date": trading_date,
                "ticker": ticker,
                "expiration": expiration,
                "t_years": t_years,
                "u_price": uclose,
                "f_price": uforward,
                "moneyness": m,
                "surface_vol": iv_val,
            })

    surface_df = pl.DataFrame(results)
    extrap_df = pl.DataFrame(extrap_records)

    return surface_df, extrap_df


def build_fixedterm_surface(
    moneyness_df: pl.DataFrame,
    config: SurfaceConfig,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Build fixed-term surface using variance-space interpolation.

    Returns:
        (surface_df, term_extrap_stats): Surface data and term extrapolation diagnostics
    """
    target_years = config.target_years
    group_keys = ["date", "ticker", "moneyness"]

    # Get bracketing expirations for each date
    unique_exps = moneyness_df.select(["date", "expiration", "t_years"]).unique()

    floor_exps = (
        unique_exps
        .filter(pl.col("t_years") <= target_years)
        .sort("t_years", descending=True)
        .group_by("date").head(1)
    )

    ceil_exps = (
        unique_exps
        .filter(pl.col("t_years") > target_years)
        .sort("t_years", descending=False)
        .group_by("date").head(1)
    )

    target_keys = pl.concat([floor_exps, ceil_exps])
    filtered_df = moneyness_df.join(target_keys, on=["date", "expiration"], how="inner")

    # Get unique (date, moneyness) combinations
    unique_points = filtered_df.select(["date", "ticker", "moneyness", "u_price", "f_price"]).unique()

    results = []
    term_extrap_records = []

    for row in unique_points.iter_rows(named=True):
        trading_date = row["date"]
        ticker = row["ticker"]
        moneyness = row["moneyness"]
        uclose = row["u_price"]
        uforward = row["f_price"]

        # Get data for this point
        point_data = filtered_df.filter(
            (pl.col("date") == trading_date) &
            (pl.col("ticker") == ticker) &
            (pl.col("moneyness") == moneyness)
        )

        if point_data.height < 1:
            continue

        years_arr = point_data["t_years"].to_numpy()
        iv_arr = point_data["surface_vol"].to_numpy()

        if point_data.height == 1:
            # Only one expiry available - use it directly (extrapolation)
            iv_target = iv_arr[0]
            is_extrap = True
        else:
            # Interpolate in variance space
            iv_target, is_extrap = interp_by_variance(years_arr, iv_arr, target_years)

        term_extrap_records.append({
            "date": trading_date,
            "moneyness": moneyness,
            "is_extrapolated": is_extrap,
            "n_expiries": len(years_arr),
            "min_years": float(years_arr.min()),
            "max_years": float(years_arr.max()),
        })

        results.append({
            "date": trading_date,
            "ticker": ticker,
            "u_price": uclose,
            "f_price": uforward,
            "t_years": target_years,
            "moneyness": moneyness,
            "surface_vol": iv_target,
        })

    surface_df = pl.DataFrame(results)
    term_extrap_df = pl.DataFrame(term_extrap_records)

    # Average u_price/f_price per date (they should be same, but ensure consistency)
    surface_df = surface_df.with_columns([
        pl.col(c).mean().over('date').alias(c) for c in ['u_price', 'f_price']
    ])

    return surface_df, term_extrap_df


def print_extrapolation_diagnostics(
    moneyness_extrap_df: pl.DataFrame,
    term_extrap_df: pl.DataFrame,
    config: SurfaceConfig,
) -> None:
    """Print extrapolation diagnostics."""
    print("\n" + "="*60)
    print("EXTRAPOLATION DIAGNOSTICS")
    print("="*60)

    # Moneyness extrapolation by point
    print("\n--- Moneyness Extrapolation by Target Point ---")
    moneyness_stats = (
        moneyness_extrap_df
        .group_by("moneyness")
        .agg([
            pl.len().alias("n_total"),
            pl.col("is_extrapolated").sum().alias("n_extrapolated"),
        ])
        .with_columns([
            (pl.col("n_extrapolated") / pl.col("n_total") * 100).alias("pct_extrapolated"),
        ])
        .sort("moneyness")
    )
    print(moneyness_stats)

    # Focus on the deepest OTM point
    deep_otm = config.interpolation_points[0]
    deep_otm_stats = moneyness_extrap_df.filter(pl.col("moneyness") == deep_otm)

    print(f"\n--- Deep OTM ({deep_otm}) Extrapolation Details ---")
    extrap_pct = deep_otm_stats["is_extrapolated"].mean() * 100
    print(f"Extrapolation rate: {extrap_pct:.1f}%")

    if deep_otm_stats.filter(pl.col("is_extrapolated") == True).height > 0:
        extrap_cases = deep_otm_stats.filter(pl.col("is_extrapolated") == True)
        print(f"When extrapolated, max actual moneyness (closest to ATM we had):")
        print(f"  Mean: {extrap_cases['max_actual_moneyness'].mean():.4f}")
        print(f"  Min:  {extrap_cases['max_actual_moneyness'].min():.4f}")
        print(f"  Max:  {extrap_cases['max_actual_moneyness'].max():.4f}")

        # Identify worst offenders - days where we had to extrapolate far
        print(f"\n--- Worst Moneyness Extrapolation Days (max_actual_moneyness < -0.5) ---")
        worst_days = (
            extrap_cases
            .filter(pl.col("max_actual_moneyness") < -0.5)
            .select(["date", "expiration", "max_actual_moneyness", "min_actual_moneyness", "n_actual_points"])
            .unique()
            .sort("max_actual_moneyness")
        )
        if worst_days.height > 0:
            print(worst_days)
        else:
            print("No severely problematic days found.")

    # Term structure extrapolation
    print("\n--- Term Structure Extrapolation ---")
    term_stats = (
        term_extrap_df
        .group_by("moneyness")
        .agg([
            pl.len().alias("n_total"),
            pl.col("is_extrapolated").sum().alias("n_extrapolated"),
        ])
        .with_columns([
            (pl.col("n_extrapolated") / pl.col("n_total") * 100).alias("pct_extrapolated"),
        ])
        .sort("moneyness")
    )
    print(term_stats)

    overall_term_extrap = term_extrap_df["is_extrapolated"].mean() * 100
    print(f"\nOverall term extrapolation rate: {overall_term_extrap:.1f}%")

    # Breakdown by n_expiries
    print("\n--- Term Extrapolation by Number of Available Expiries ---")
    print("n_expiries = 1 means only one expiry available, can't interpolate (extrapolation)")
    print("n_expiries = 2+ means proper interpolation between floor/ceiling expiries")
    by_n_exp = (
        term_extrap_df
        .group_by("n_expiries")
        .agg([
            pl.len().alias("n_total"),
            pl.col("is_extrapolated").sum().alias("n_extrapolated"),
        ])
        .sort("n_expiries")
    )
    print(by_n_exp)

    # Identify days with only 1 expiry
    single_expiry_days = (
        term_extrap_df
        .filter(pl.col("n_expiries") == 1)
        .select(["date", "min_years", "max_years"])
        .unique()
    )
    if single_expiry_days.height > 0:
        print(f"\n--- Days with Only 1 Expiry Available ---")
        print(single_expiry_days)


def filter_bad_data(
    moneyness_df: pl.DataFrame,
    moneyness_extrap_df: pl.DataFrame,
    term_extrap_df: pl.DataFrame,
    config: SurfaceConfig,
) -> tuple[pl.DataFrame, pl.DataFrame, dict]:
    """
    Filter out bad data based on extrapolation thresholds.

    Returns:
        (filtered_moneyness_df, filtered_term_extrap_df, drop_report)
    """
    drop_report = {
        "config": {
            "max_moneyness_extrap_gap": config.max_moneyness_extrap_gap,
            "min_expiries": config.min_expiries,
        },
        "moneyness_drops": [],
        "term_drops": [],
        "summary": {},
    }

    # --- Filter 1: Moneyness extrapolation ---
    # For each curve (date, expiration), check if any target point
    # requires extrapolation beyond the threshold

    # Calculate extrapolation gap for each point
    # For deep OTM (negative moneyness), extrapolation happens when target < min_actual
    # Gap = min_actual - target (positive when extrapolating)
    deep_otm_target = config.interpolation_points[0]  # -0.30

    # Get curves where deep OTM point is extrapolated
    deep_otm_extrap = (
        moneyness_extrap_df
        .filter(pl.col("moneyness") == deep_otm_target)
        .filter(pl.col("is_extrapolated") == True)
        .with_columns([
            (pl.col("min_actual_moneyness") - pl.lit(deep_otm_target)).alias("extrap_gap")
        ])
    )

    # Identify curves to drop (gap exceeds threshold)
    curves_to_drop_moneyness = (
        deep_otm_extrap
        .filter(pl.col("extrap_gap").abs() > config.max_moneyness_extrap_gap)
        .select(["date", "expiration", "min_actual_moneyness", "max_actual_moneyness", "n_actual_points", "extrap_gap"])
    )

    # Log dropped curves
    for row in curves_to_drop_moneyness.iter_rows(named=True):
        drop_report["moneyness_drops"].append({
            "date": str(row["date"]),
            "expiration": str(row["expiration"]),
            "min_actual_moneyness": row["min_actual_moneyness"],
            "max_actual_moneyness": row["max_actual_moneyness"],
            "n_actual_points": row["n_actual_points"],
            "extrap_gap": row["extrap_gap"],
            "reason": f"Moneyness extrapolation gap {row['extrap_gap']:.4f} > {config.max_moneyness_extrap_gap}",
        })

    # Filter moneyness_df
    drop_keys = curves_to_drop_moneyness.select(["date", "expiration"])
    if drop_keys.height > 0:
        filtered_moneyness_df = moneyness_df.join(
            drop_keys,
            on=["date", "expiration"],
            how="anti"
        )
    else:
        filtered_moneyness_df = moneyness_df

    # --- Filter 2: Term structure (insufficient expiries) ---
    # Identify (date, moneyness) with < min_expiries
    insufficient_expiries = (
        term_extrap_df
        .filter(pl.col("n_expiries") < config.min_expiries)
        .select(["date", "moneyness", "n_expiries", "min_years", "max_years"])
        .unique()
    )

    # Get unique dates to drop (if any moneyness point on that date has insufficient expiries)
    dates_to_drop_term = insufficient_expiries.select("date").unique()

    # Log dropped dates
    for row in insufficient_expiries.iter_rows(named=True):
        drop_report["term_drops"].append({
            "date": str(row["date"]),
            "moneyness": row["moneyness"],
            "n_expiries": row["n_expiries"],
            "min_years": row["min_years"],
            "max_years": row["max_years"],
            "reason": f"Only {row['n_expiries']} expiry available, need {config.min_expiries}+ for interpolation",
        })

    # Filter term_extrap_df (for downstream use)
    if dates_to_drop_term.height > 0:
        filtered_term_extrap_df = term_extrap_df.join(
            dates_to_drop_term,
            on="date",
            how="anti"
        )
        # Also filter moneyness_df for these dates
        filtered_moneyness_df = filtered_moneyness_df.join(
            dates_to_drop_term,
            on="date",
            how="anti"
        )
    else:
        filtered_term_extrap_df = term_extrap_df

    # Summary
    original_curves = moneyness_df.select(["date", "expiration"]).unique().height
    original_dates = moneyness_df.select("date").unique().height
    filtered_curves = filtered_moneyness_df.select(["date", "expiration"]).unique().height
    filtered_dates = filtered_moneyness_df.select("date").unique().height

    drop_report["summary"] = {
        "original_curves": original_curves,
        "original_dates": original_dates,
        "dropped_curves_moneyness": curves_to_drop_moneyness.height,
        "dropped_dates_term": dates_to_drop_term.height,
        "final_curves": filtered_curves,
        "final_dates": filtered_dates,
        "pct_curves_dropped": (original_curves - filtered_curves) / original_curves * 100 if original_curves > 0 else 0,
        "pct_dates_dropped": (original_dates - filtered_dates) / original_dates * 100 if original_dates > 0 else 0,
    }

    return filtered_moneyness_df, filtered_term_extrap_df, drop_report


def write_drop_report(drop_report: dict, config: SurfaceConfig) -> None:
    """Write drop report to a log file."""
    import json

    log_path = config.log_dir / "data_quality_report.json"

    with open(log_path, "w") as f:
        json.dump(drop_report, f, indent=2, default=str)

    print(f"\nDrop report written to: {log_path}")

    # Also print summary to console
    summary = drop_report["summary"]
    print("\n" + "="*60)
    print("DATA QUALITY FILTER SUMMARY")
    print("="*60)
    print(f"Original curves: {summary['original_curves']:,}")
    print(f"Original dates:  {summary['original_dates']:,}")
    print(f"Dropped curves (moneyness extrap): {summary['dropped_curves_moneyness']:,}")
    print(f"Dropped dates (insufficient expiries): {summary['dropped_dates_term']:,}")
    print(f"Final curves: {summary['final_curves']:,}")
    print(f"Final dates:  {summary['final_dates']:,}")
    print(f"Curves dropped: {summary['pct_curves_dropped']:.2f}%")
    print(f"Dates dropped:  {summary['pct_dates_dropped']:.2f}%")


def build_final_surface(
    fixedterm_df: pl.DataFrame,
    config: SurfaceConfig,
) -> pl.DataFrame:
    """Pivot and demean the surface."""
    # Ensure unique (date, moneyness) by taking mean of any duplicates
    fixedterm_df = (
        fixedterm_df
        .group_by(["date", "ticker", "t_years", "moneyness"])
        .agg([
            pl.col("surface_vol").mean().alias("surface_vol"),
            pl.col("u_price").mean().alias("u_price"),
            pl.col("f_price").mean().alias("f_price"),
        ])
    )

    # Pivot to wide format
    pivot_df = fixedterm_df.pivot(
        on="moneyness",
        index=["date", "ticker", "u_price", "f_price", "t_years"],
        values=["surface_vol"]
    )

    # Rename columns
    metadata_cols = {"date", "ticker", "u_price", "f_price", "t_years"}
    target_cols = [c for c in pivot_df.columns if c not in metadata_cols]
    rename_map = {col: f"int_surface_vol_{col}" for col in target_cols}
    int_df = pivot_df.rename(rename_map)

    # Demean by ATM
    atm_col = "int_surface_vol_0.0"
    if atm_col in int_df.columns:
        df = int_df.with_columns([
            (pl.col(c) - pl.col(atm_col)).alias(c) for c in rename_map.values()
        ])
    else:
        df = int_df
        print(f"WARNING: ATM column {atm_col} not found, skipping demeaning")

    return df


def main():
    """Main pipeline."""
    parser = argparse.ArgumentParser(description="Build vol surface parquet")
    parser.add_argument(
        "--target-years",
        type=float,
        default=None,
        help="Target maturity in years (e.g. 0.25 for 3m, 0.0833 for 1m). Default: 3/12.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output parquet path. Default: from SurfaceConfig.",
    )
    args = parser.parse_args()

    config = SurfaceConfig.default()
    if args.target_years is not None:
        config.target_years = args.target_years
    if args.output_path is not None:
        config.surface_output_path = args.output_path

    print("Loading option data...")
    df = load_option_data(config)
    print(f"Loaded {df.height:,} rows")

    # Check for duplicate strikes
    tmp = df.with_columns(
        pl.col('strike').len().over("date", "expiration", "strike").alias('numStrikes')
    ).filter(pl.col('numStrikes') > 1)
    if tmp.height > 0:
        print(f"WARNING: {tmp.height} rows have multiple contracts for same strike")

    print("\nBuilding moneyness-interpolated surface...")
    moneyness_df, moneyness_extrap_df = build_moneyness_surface(df, config)
    print(f"Moneyness surface: {moneyness_df.height:,} rows")

    # Print extrapolation diagnostics (before filtering)
    print("\n" + "="*60)
    print("PRE-FILTER EXTRAPOLATION DIAGNOSTICS")
    print("="*60)

    # Quick pre-filter stats
    deep_otm = config.interpolation_points[0]
    deep_otm_stats = moneyness_extrap_df.filter(pl.col("moneyness") == deep_otm)
    extrap_pct = deep_otm_stats["is_extrapolated"].mean() * 100
    print(f"Deep OTM ({deep_otm}) extrapolation rate: {extrap_pct:.1f}%")

    # Build term structure to get term extrapolation stats
    print("\nBuilding fixed-term surface (variance-space interpolation)...")
    fixedterm_df_prefilt, term_extrap_df = build_fixedterm_surface(moneyness_df, config)
    print(f"Fixed-term surface (pre-filter): {fixedterm_df_prefilt.height:,} rows")

    # Apply data quality filters
    print("\nApplying data quality filters...")
    filtered_moneyness_df, filtered_term_extrap_df, drop_report = filter_bad_data(
        moneyness_df, moneyness_extrap_df, term_extrap_df, config
    )

    # Write drop report
    write_drop_report(drop_report, config)

    # Rebuild fixed-term surface with filtered data
    print("\nRebuilding fixed-term surface with filtered data...")
    fixedterm_df, _ = build_fixedterm_surface(filtered_moneyness_df, config)
    print(f"Fixed-term surface (post-filter): {fixedterm_df.height:,} rows")

    # Surface-level arbitrage diagnostic (IV → BS prices → check)
    print("\nSurface arbitrage diagnostics (interpolated data)...")
    surf_arb_summary, surf_arb_violations = diagnose_surface_arbitrage(fixedterm_df)
    print(f"  Dates checked: {surf_arb_summary['n_dates_checked']:,}")
    print(f"  Dates with violations: {surf_arb_summary['n_dates_with_violations']} "
          f"({surf_arb_summary['pct_dates_violated']:.1f}%)")
    print(f"  Put spread violations: {surf_arb_summary['n_put_spread_violations']}")
    print(f"  Butterfly violations: {surf_arb_summary['n_butterfly_violations']}")
    if surf_arb_violations.height > 0:
        print("  Worst 10 violations:")
        print(surf_arb_violations.sort('violation_amount', descending=True).head(10))

    # Print post-filter diagnostics
    print_extrapolation_diagnostics(
        moneyness_extrap_df.join(
            filtered_moneyness_df.select(["date", "expiration"]).unique(),
            on=["date", "expiration"],
            how="semi"
        ),
        filtered_term_extrap_df,
        config
    )

    print("\nBuilding final pivoted surface...")
    final_df = build_final_surface(fixedterm_df, config)

    # Check for dates with wrong number of points
    dates_check = final_df.group_by('date').len().filter(pl.col('len') != 1)
    if dates_check.height > 0:
        print(f"WARNING: {dates_check.height} dates have wrong number of rows")

    # Null summary
    print("\nNull count summary:")
    nulls_summary = (
        final_df.null_count()
        .transpose(include_header=True, header_name="column")
        .filter(pl.col("column_0") > 0)
        .rename({"column_0": "null_count"})
    )
    if nulls_summary.height > 0:
        print(nulls_summary)
    else:
        print("No nulls found")

    # Save
    final_df.write_parquet(config.surface_output_path)
    print(f"\nSaved to {config.surface_output_path}")
    print(f"Final shape: {final_df.shape}")

    return final_df, moneyness_extrap_df, term_extrap_df, drop_report


if __name__ == "__main__":
    final_df, moneyness_extrap_df, term_extrap_df, drop_report = main()
