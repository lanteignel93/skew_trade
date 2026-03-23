import polars as pl
import datetime
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


class OptionType(str, Enum):
    PUT = "Put"
    CALL = "Call"


class LegSide(str, Enum):
    LONG = "Long"
    SHORT = "Short"


@dataclass(frozen=True)
class OptionLeg:
    """A single leg of an option structure.

    Attributes:
        label: Unique identifier within the structure (e.g. "long_25d_put").
               Used as the tracking key in the portfolio skeleton and forward-fill.
        option_type: Put or Call.
        target_delta: Absolute delta target (e.g. 0.25 for 25-delta).
        side: Long or Short (determines P&L sign).
        ratio: Number of contracts for this leg.
    """
    label: str
    option_type: OptionType
    target_delta: float
    side: LegSide
    ratio: int = 1

    def pnl_sign(self) -> int:
        return 1 if self.side == LegSide.LONG else -1


@dataclass(frozen=True)
class OptionStructure:
    """Multi-leg option structure where all legs share the same expiry.

    Attributes:
        name: Descriptive name, e.g. "put_ratio_1x2".
        legs: Tuple of OptionLeg definitions.
    """
    name: str
    legs: tuple[OptionLeg, ...]

    def leg_labels(self) -> list[str]:
        return [leg.label for leg in self.legs]

    def n_legs(self) -> int:
        return len(self.legs)


PUT_RATIO_1x2 = OptionStructure(
    name="put_ratio_1x2",
    legs=(
        OptionLeg("long_20d_put", OptionType.PUT, 0.20, LegSide.LONG, 1),
        OptionLeg("short_10d_put", OptionType.PUT, 0.10, LegSide.SHORT, 2),
    ),
)

PUT_BWB_SHORT_SKEW = OptionStructure(
    name="put_bwb_short_skew",
    legs=(
        OptionLeg("long_05d_put", OptionType.PUT, 0.05, LegSide.LONG, 1),
        OptionLeg("short_15d_put", OptionType.PUT, 0.15, LegSide.SHORT, 2),
        OptionLeg("long_25d_put", OptionType.PUT, 0.25, LegSide.LONG, 1),
    ),
)

PUT_BWB_LONG_SKEW = OptionStructure(
    name="put_bwb_long_skew",
    legs=(
        OptionLeg("short_05d_put", OptionType.PUT, 0.05, LegSide.SHORT, 1),
        OptionLeg("long_15d_put", OptionType.PUT, 0.15, LegSide.LONG, 2),
        OptionLeg("short_25d_put", OptionType.PUT, 0.25, LegSide.SHORT, 1),
    ),
)
PUT_BACKSPREAD_2x1 = OptionStructure(
    name="put_backspread_2x1",
    legs=(
        OptionLeg("short_20d_put", OptionType.PUT, 0.20, LegSide.SHORT, 1),
        OptionLeg("long_10d_put", OptionType.PUT, 0.10, LegSide.LONG, 2),
    ),
)

NEW = OptionStructure(
    name="new",
    legs=(
        OptionLeg("short_5d_put", OptionType.PUT, 0.05, LegSide.SHORT, 5),
        OptionLeg("long_25d_put", OptionType.PUT, 0.25, LegSide.LONG, 1),
    ),
)
PUT_BWB_LONG_SKEW_FAR = OptionStructure(
    name="put_bwb_long_skew_far",
    legs=(
        OptionLeg("short_5d_put", OptionType.PUT, 0.05, LegSide.SHORT, 1),
        OptionLeg("long_10d_put", OptionType.PUT, 0.10, LegSide.LONG, 2),
        OptionLeg("short_15d_put", OptionType.PUT, 0.15, LegSide.SHORT, 1),
    ),
)

# ── Target structures for focused analysis ─────────────────────────────────────

BWB_5_10_15 = OptionStructure(
    name="bwb_5_10_15",
    legs=(
        OptionLeg("long_5d_put", OptionType.PUT, 0.05, LegSide.LONG, 1),
        OptionLeg("short_10d_put", OptionType.PUT, 0.10, LegSide.SHORT, 2),
        OptionLeg("long_15d_put", OptionType.PUT, 0.15, LegSide.LONG, 1),
    ),
)

BWB_10_15_20 = OptionStructure(
    name="bwb_10_15_20",
    legs=(
        OptionLeg("long_10d_put", OptionType.PUT, 0.10, LegSide.LONG, 1),
        OptionLeg("short_15d_put", OptionType.PUT, 0.15, LegSide.SHORT, 2),
        OptionLeg("long_20d_put", OptionType.PUT, 0.20, LegSide.LONG, 1),
    ),
)
BWB_5_15_25 = OptionStructure(
    name="bwb_5_15_25",
    legs=(
        OptionLeg("long_5d_put", OptionType.PUT, 0.05, LegSide.LONG, 1),
        OptionLeg("short_15d_put", OptionType.PUT, 0.15, LegSide.SHORT, 2),
        OptionLeg("long_25d_put", OptionType.PUT, 0.25, LegSide.LONG, 1),
    ),
)

# Convenience alias so notebooks can import BWB_10_20_30 by explicit name
BWB_10_20_30 = PUT_BWB_SHORT_SKEW

# Convenience alias so notebooks can import RATIO_20_10 by explicit name
RATIO_20_10 = PUT_RATIO_1x2


@dataclass
class StrategyParams:
    """Strategy parameters for single-expiry option structures."""
    structure: OptionStructure = field(default_factory=lambda: PUT_RATIO_1x2)
    target_dte: float = 63 / 252
    roll_weeks: int = 4
    start_date: datetime.date = datetime.date(2016, 1, 1)
    ticker: str = "SPXW"



def load_market_data() -> pl.LazyFrame:
    """Load SPXW option data from local parquet."""
    return pl.scan_parquet(DATA_DIR / "options_raw.parquet")


def load_vix_data() -> pl.DataFrame:
    return pl.read_parquet(DATA_DIR / "vol_features.parquet")



def _select_legs(
    df_signals: pl.DataFrame,
    roll_dates: list,
    params: StrategyParams,
) -> pl.DataFrame:
    """Select option contracts for each leg on roll dates.

    Two-phase selection (all legs share one expiry):
      1. Find the expiry closest to target_dte for each roll date.
      2. Within that expiry, find the strike closest to each leg's target_delta.

    Returns:
        DataFrame with columns [date, leg_label, strike, expiration, cp, side, ratio]
    """
    candidates = (
        df_signals
        .filter(pl.col('date').is_in(roll_dates))
        .with_columns(de_abs=pl.col('delta').abs())
    )

    best_expiries = (
        candidates
        .select(["date", "expiration", "t_years"])
        .unique()
        .with_columns(
            (pl.col("t_years") - params.target_dte).abs().alias("distance_dte")
        )
        .sort(["date", "distance_dte"])
        .group_by("date", maintain_order=True)
        .agg(pl.col("expiration").first().alias("best_expiration"))
    )

    candidates = (
        candidates
        .join(best_expiries, on="date")
        .filter(pl.col("expiration") == pl.col("best_expiration"))
        .drop("best_expiration")
    )

    all_legs = []
    for leg in params.structure.legs:
        leg_df = (
            candidates
            .filter(pl.col("cp") == leg.option_type.value)
            .with_columns(
                (pl.col("de_abs") - leg.target_delta).abs().alias("distance_de")
            )
            .sort(["date", "distance_de"])
            .group_by("date", maintain_order=True)
            .agg([
                pl.col("strike").first(),
                pl.col("expiration").first(),
                pl.col("cp").first(),
            ])
            .with_columns([
                pl.lit(leg.label).alias("leg_label"),
                pl.lit(leg.side.value).alias("side"),
                pl.lit(leg.ratio).alias("ratio"),
            ])
        )
        all_legs.append(leg_df)


    return pl.concat(all_legs).sort(["date", "cp", "side", "strike"])



def build_trade_df(
    df_market: pl.DataFrame,
    vol_features: pl.DataFrame,
    params: StrategyParams,
) -> pl.DataFrame:
    """Build trade dataframe for an arbitrary option structure.

    Args:
        df_market: Option market data from load_market_data()
        vol_features: Volatility features (joined on date)
        params: Strategy parameters including structure definition

    Returns:
        trade_df with per-leg daily P&L and volatility features
    """
    df_market = df_market.filter(pl.col('date') >= params.start_date)

    df_signals = (
        df_market
        .filter(
            (pl.col('cp') == 'Put') & (pl.col('u_price') > pl.col('strike'))
        )
        .with_columns((pl.col('ask_vol') - pl.col('bid_vol')).abs().alias('spreadIV'))
        .filter(
            ((pl.col('spreadIV') / pl.col('surface_vol')) < 0.1) &
            (pl.col('bid_vol') > 0) & (pl.col('ask_vol') > 0)
        )
    )

    weekly_last_business_dates = (
        df_market.group_by(pl.col("date").dt.truncate("1w"))
        .agg(trade_open_days=pl.col("date").max())
        .sort("trade_open_days")
        .with_columns([
            (pl.int_range(0, pl.len()) % params.roll_weeks == 0).alias("b_roll"),
        ])
    )

    roll_dates = (
        weekly_last_business_dates
        .filter(pl.col("b_roll"))
        .get_column("trade_open_days")
        .to_list()
    )

    tradeDates_df = (
        pl.DataFrame(df_market.get_column('date').unique())
        .sort("date")
        .with_columns(nextTradingDate=pl.col('date').shift(-1))
        .drop_nulls()
        .join(weekly_last_business_dates, left_on=['date'], right_on=['trade_open_days'], how='left')
        .with_columns([
            pl.col('b_roll').fill_null(False),
        ])
        .drop('date_right')
    )

    first_roll = (
        tradeDates_df
        .filter(pl.col("b_roll"))
        .get_column("date")
        .min()
    )
    tradeDates_df = tradeDates_df.filter(
        pl.col("date") >= first_roll
    )

    all_legs_df = _select_legs(df_signals, roll_dates, params)

    leg_meta = pl.DataFrame({
        "leg_label": [leg.label for leg in params.structure.legs],
        "side": [leg.side.value for leg in params.structure.legs],
        "cp": [leg.option_type.value for leg in params.structure.legs],
        "ratio": [leg.ratio for leg in params.structure.legs],
    })
    daily_skeleton = tradeDates_df.join(leg_meta, how="cross")

    portfolio_df = (
        daily_skeleton
        .join(
            all_legs_df.select(["date", "leg_label", "strike", "expiration"]),
            on=["date", "leg_label"],
            how="left",
        )
        .sort("date", "leg_label")
        .with_columns(
            pl.col(["strike", "expiration"])
            .fill_null(strategy="forward")
            .over("leg_label")
        )
        .drop_nulls(subset=["strike"])
    )

    closing_legs = (
        portfolio_df
        .sort("date", "leg_label")
        .with_columns([
            pl.col("strike").shift(1).over("leg_label").alias("closing_xx"),
            pl.col("expiration").shift(1).over("leg_label").alias("closing_date"),
        ])
        .filter(pl.col("b_roll"))
        .filter(pl.col("closing_xx").is_not_null())
        .with_columns(pl.lit("Close").alias("position_action"))
        .drop(["strike", "expiration"])
        .rename({"closing_xx": "strike", "closing_date": "expiration"})
    )

    active_legs = (
        portfolio_df
        .with_columns(
            position_action=pl.when(pl.col("b_roll"))
            .then(pl.lit("Open"))
            .otherwise(pl.lit("Hold"))
        )
    )

    final_transaction_df = (
        pl.concat([active_legs, closing_legs], how="diagonal")
        .sort("date", "leg_label", "position_action")
    )

    filter_df = final_transaction_df.select(["date", "cp", "strike", "expiration", "leg_label"]).unique()

    trade_df = (
        df_market
        .join(
            filter_df,
            on=["date", "cp", "strike", "expiration"],
        )
    )
    trade_df = trade_df.join(tradeDates_df.select(["date", "nextTradingDate"]), on='date')

    next_close_price = (
        trade_df
        .select(["date", "ticker", "expiration", "strike", "cp", "leg_label", "c_price"])
        .rename({'c_price': 'nextPrc'})
    )

    trade_df = (
        trade_df.join(
            next_close_price,
            left_on=["nextTradingDate", "ticker", "expiration", "strike", "cp", "leg_label"],
            right_on=["date", "ticker", "expiration", "strike", "cp", "leg_label"],
            how='left',
        )
        .unique()
        .sort('date')
        .filter(pl.col('nextPrc').is_finite())
    )

    expected_legs = params.structure.n_legs()
    valid_dates = (
        trade_df
        .group_by('date').len()
        .filter(pl.col('len') == expected_legs)
        .get_column('date')
    )
    trade_df = trade_df.filter(pl.col('date').is_in(valid_dates.to_list()))

    trade_df = trade_df.join(
        active_legs.select(['date', 'leg_label', 'strike', 'expiration', 'side', 'ratio', 'position_action']),
        on=['date', 'leg_label', 'strike', 'expiration'],
    )

    pnl_sign_df = pl.DataFrame({
        "leg_label": [leg.label for leg in params.structure.legs],
        "pnl_sign": [leg.pnl_sign() for leg in params.structure.legs],
    })

    trade_df = (
        trade_df
        .join(pnl_sign_df, on="leg_label")
        .with_columns(
            (pl.col("pnl_sign") * pl.col("ratio") * (pl.col("nextPrc") - pl.col("c_price")))
            .alias("pnl")
        )
    )

    # --- Portfolio greeks (signed by direction and ratio) ---
    trade_df = trade_df.with_columns([
        (pl.col("pnl_sign") * pl.col("ratio") * pl.col("delta")).alias("pos_delta"),
        (pl.col("pnl_sign") * pl.col("ratio") * pl.col("gamma")).alias("pos_gamma"),
        (pl.col("pnl_sign") * pl.col("ratio") * pl.col("theta")).alias("pos_theta"),
        (pl.col("pnl_sign") * pl.col("ratio") * pl.col("vega")).alias("pos_vega"),
    ])

    trade_df = trade_df.join(vol_features, on="date")

    return trade_df



# ---------------------------------------------------------------------------
# Data validation
# ---------------------------------------------------------------------------

def validate_trade_df(trade_df: pl.DataFrame, params: StrategyParams) -> list[str]:
    """Run data quality checks on trade_df. Returns list of failures (empty = all passed)."""
    failures = []
    expected_labels = set(params.structure.leg_labels())
    n_legs = params.structure.n_legs()

    # 1. No duplicate (date, leg_label)
    dupes = (
        trade_df
        .group_by(["date", "leg_label"]).len()
        .filter(pl.col("len") > 1)
    )
    if dupes.height > 0:
        failures.append(
            f"DUPLICATE ROWS: {dupes.height} (date, leg_label) pairs have >1 row. "
            f"First offender: {dupes.row(0, named=True)}"
        )

    # 2. Exactly n_legs per date
    legs_per_date = trade_df.group_by("date").len()
    bad_dates = legs_per_date.filter(pl.col("len") != n_legs)
    if bad_dates.height > 0:
        failures.append(
            f"LEG COUNT: {bad_dates.height} dates have != {n_legs} legs. "
            f"Counts found: {bad_dates['len'].unique().sort().to_list()}"
        )

    # 3. All expected leg_labels present on every date
    actual_labels = set(trade_df["leg_label"].unique().to_list())
    missing = expected_labels - actual_labels
    extra = actual_labels - expected_labels
    if missing:
        failures.append(f"MISSING LEGS: {missing} never appear in trade_df")
    if extra:
        failures.append(f"EXTRA LEGS: {extra} are not in the structure definition")

    # 4. All legs share the same expiry (expiration) on each date
    expiries_per_date = (
        trade_df
        .group_by("date")
        .agg(pl.col("expiration").n_unique().alias("n_expiries"))
        .filter(pl.col("n_expiries") > 1)
    )
    if expiries_per_date.height > 0:
        failures.append(
            f"MIXED EXPIRIES: {expiries_per_date.height} dates have legs on different expirations"
        )

    # 5. No nulls in critical columns
    critical_cols = ["date", "leg_label", "strike", "expiration", "c_price", "nextPrc", "delta", "pnl"]
    existing_critical = [c for c in critical_cols if c in trade_df.columns]
    null_counts = trade_df.select(existing_critical).null_count()
    for col in existing_critical:
        n_null = null_counts[col][0]
        if n_null > 0:
            failures.append(f"NULL VALUES: {col} has {n_null:,} nulls")

    # 6. Delta values are reasonable for each leg (within 2x of target)
    for leg in params.structure.legs:
        leg_data = trade_df.filter(pl.col("leg_label") == leg.label)
        if leg_data.height == 0:
            continue

        avg_abs_delta = leg_data["delta"].abs().mean()
        if avg_abs_delta > leg.target_delta * 3:
            failures.append(
                f"DELTA MISMATCH: {leg.label} avg |delta|={avg_abs_delta:.4f}, "
                f"target={leg.target_delta:.2f} (>3x off)"
            )

    # 7. Strikes ordered correctly (higher delta = closer to ATM = higher strike for puts)
    #    Check on each date that legs with higher target_delta have higher strikes
    put_legs_sorted = sorted(
        [l for l in params.structure.legs if l.option_type == OptionType.PUT],
        key=lambda l: l.target_delta,
    )
    if len(put_legs_sorted) >= 2:
        low_delta_leg = put_legs_sorted[0]
        high_delta_leg = put_legs_sorted[-1]
        strike_check = (
            trade_df
            .filter(pl.col("leg_label").is_in([low_delta_leg.label, high_delta_leg.label]))
            .pivot(on="leg_label", index="date", values="strike")
        )
        if low_delta_leg.label in strike_check.columns and high_delta_leg.label in strike_check.columns:
            violations = strike_check.filter(
                pl.col(low_delta_leg.label) >= pl.col(high_delta_leg.label)
            )
            if violations.height > 0:
                pct = violations.height / strike_check.height * 100
                failures.append(
                    f"STRIKE ORDER: {violations.height} dates ({pct:.1f}%) where "
                    f"{low_delta_leg.label} strike >= {high_delta_leg.label} strike "
                    f"(lower delta put should have lower strike)"
                )

    # 8. Roll frequency: check Open events are spaced by ~roll_weeks
    open_dates = (
        trade_df
        .filter(pl.col("position_action") == "Open")
        .select("date")
        .unique()
        .sort("date")
        .with_columns(
            (pl.col("date") - pl.col("date").shift(1)).dt.total_days().alias("gap_days")
        )
        .filter(pl.col("gap_days").is_not_null())
    )
    if open_dates.height > 0:
        expected_gap = params.roll_weeks * 7
        tolerance = 5  # allow some slack for holidays
        bad_gaps = open_dates.filter(
            (pl.col("gap_days") < expected_gap - tolerance) |
            (pl.col("gap_days") > expected_gap + tolerance)
        )
        if bad_gaps.height > 0:
            failures.append(
                f"ROLL FREQUENCY: {bad_gaps.height}/{open_dates.height} roll gaps outside "
                f"{expected_gap}d +/- {tolerance}d. "
                f"Gap range: [{open_dates['gap_days'].min()}, {open_dates['gap_days'].max()}]"
            )

    # 9. P&L sign consistency (pnl_sign matches side)
    sign_check = trade_df.select(["leg_label", "side", "pnl_sign"]).unique()
    for row in sign_check.iter_rows(named=True):
        expected_sign = 1 if row["side"] == "Long" else -1
        if row["pnl_sign"] != expected_sign:
            failures.append(
                f"PNL SIGN: {row['leg_label']} has side={row['side']} "
                f"but pnl_sign={row['pnl_sign']}"
            )

    # 10. No infinite or extreme P&L values
    extreme_pnl = trade_df.filter(pl.col("pnl").abs() > 500)
    if extreme_pnl.height > 0:
        pct = extreme_pnl.height / trade_df.height * 100
        max_abs = extreme_pnl["pnl"].abs().max()
        failures.append(
            f"EXTREME PNL: {extreme_pnl.height} rows ({pct:.1f}%) with |pnl| > 500. "
            f"Max |pnl|={max_abs:,.2f}"
        )

    return failures


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

def detect_outliers(
    trade_df: pl.DataFrame,
    params: StrategyParams,
    pnl_zscore_threshold: float = 4.0,
    pnl_rolling_window: int = 60,
    delta_jump_threshold: float = 0.15,
    iv_bounds: tuple[float, float] = (0.03, 2.0),
    iv_zscore_threshold: float = 4.0,
    price_pct_change_threshold: float = 0.80,
    underlying_return_threshold: float = 0.05,
) -> pl.DataFrame:
    """Detect outliers in trade data across multiple dimensions.

    Flags observations that are suspicious and may indicate data quality issues
    vs genuine market events. Returns a DataFrame of flagged rows with reasons.

    Checks:
      1. Per-leg P&L z-score (rolling window, avoids look-ahead)
      2. Structure-level daily P&L z-score
      3. Delta jumps on non-roll dates (stale data / bad forward-fill)
      4. IV absolute bounds + rolling z-score
      5. Option price day-over-day discontinuities
      6. Large underlying moves (informational, not necessarily errors)

    Args:
        trade_df: Output of build_trade_df()
        params: Strategy parameters
        pnl_zscore_threshold: Flag if |z| > this (default 4.0)
        pnl_rolling_window: Rolling window for z-score (days per leg)
        delta_jump_threshold: Flag if |delta change| > this on non-roll dates
        iv_bounds: (min, max) absolute IV bounds
        iv_zscore_threshold: Flag if IV rolling z-score |z| > this
        price_pct_change_threshold: Flag if |pct change in c_price| > this
        underlying_return_threshold: Flag if |underlying daily return| > this

    Returns:
        DataFrame with columns [date, leg_label, check, value, threshold, detail]
    """
    flags: list[dict] = []

    def _flag(date, leg: str, check: str, value: float, threshold, detail: str):
        flags.append({
            "date": date,
            "leg_label": leg,
            "check": check,
            "value": round(value, 4),
            "threshold": str(threshold),
            "detail": detail,
        })

    # --- 1. Per-leg P&L z-score (rolling) ---
    for leg in params.structure.legs:
        leg_data = (
            trade_df
            .filter(pl.col("leg_label") == leg.label)
            .sort("date")
            .with_columns([
                pl.col("pnl")
                    .rolling_mean(window_size=pnl_rolling_window, min_samples=20)
                    .alias("pnl_rmean"),
                pl.col("pnl")
                    .rolling_std(window_size=pnl_rolling_window, min_samples=20)
                    .alias("pnl_rstd"),
            ])
            .with_columns(
                ((pl.col("pnl") - pl.col("pnl_rmean")) / pl.col("pnl_rstd"))
                .alias("pnl_zscore")
            )
            .filter(pl.col("pnl_zscore").abs() > pnl_zscore_threshold)
        )
        for row in leg_data.iter_rows(named=True):
            _flag(row["date"], leg.label, "pnl_zscore", row["pnl_zscore"],
                  pnl_zscore_threshold,
                  f"pnl={row['pnl']:.4f}, rolling_mean={row['pnl_rmean']:.4f}, rolling_std={row['pnl_rstd']:.4f}")

    # --- 2. Structure-level daily P&L z-score ---
    daily_pnl = (
        trade_df
        .group_by("date")
        .agg(pl.col("pnl").sum().alias("structure_pnl"))
        .sort("date")
        .with_columns([
            pl.col("structure_pnl")
                .rolling_mean(window_size=pnl_rolling_window, min_samples=20)
                .alias("pnl_rmean"),
            pl.col("structure_pnl")
                .rolling_std(window_size=pnl_rolling_window, min_samples=20)
                .alias("pnl_rstd"),
        ])
        .with_columns(
            ((pl.col("structure_pnl") - pl.col("pnl_rmean")) / pl.col("pnl_rstd"))
            .alias("pnl_zscore")
        )
        .filter(pl.col("pnl_zscore").abs() > pnl_zscore_threshold)
    )
    for row in daily_pnl.iter_rows(named=True):
        _flag(row["date"], "_structure_", "structure_pnl_zscore", row["pnl_zscore"],
              pnl_zscore_threshold, f"structure_pnl={row['structure_pnl']:.4f}")

    # --- 3. Delta jumps on non-roll dates ---
    for leg in params.structure.legs:
        leg_data = (
            trade_df
            .filter(pl.col("leg_label") == leg.label)
            .sort("date")
            .with_columns(
                (pl.col("delta") - pl.col("delta").shift(1)).alias("delta_change")
            )
            .filter(
                (pl.col("delta_change").abs() > delta_jump_threshold) &
                (pl.col("position_action") != "Open")
            )
        )
        for row in leg_data.iter_rows(named=True):
            _flag(row["date"], leg.label, "delta_jump_non_roll", row["delta_change"],
                  delta_jump_threshold,
                  f"delta={row['delta']:.4f}, prev_delta={row['delta'] - row['delta_change']:.4f}, NOT a roll date")

    # --- 4. IV outliers (absolute bounds + rolling z-score) ---
    iv_min, iv_max = iv_bounds
    iv_bound_violations = trade_df.filter(
        (pl.col("surface_vol") < iv_min) | (pl.col("surface_vol") > iv_max)
    )
    for row in iv_bound_violations.iter_rows(named=True):
        _flag(row["date"], row["leg_label"], "iv_bounds", row["surface_vol"],
              f"[{iv_min}, {iv_max}]", f"surface_vol={row['surface_vol']:.4f} outside absolute bounds")

    for leg in params.structure.legs:
        leg_data = (
            trade_df
            .filter(pl.col("leg_label") == leg.label)
            .sort("date")
            .with_columns([
                pl.col("surface_vol")
                    .rolling_mean(window_size=pnl_rolling_window, min_samples=20)
                    .alias("iv_rmean"),
                pl.col("surface_vol")
                    .rolling_std(window_size=pnl_rolling_window, min_samples=20)
                    .alias("iv_rstd"),
            ])
            .with_columns(
                ((pl.col("surface_vol") - pl.col("iv_rmean")) / pl.col("iv_rstd"))
                .alias("iv_zscore")
            )
            .filter(pl.col("iv_zscore").abs() > iv_zscore_threshold)
        )
        for row in leg_data.iter_rows(named=True):
            _flag(row["date"], leg.label, "iv_zscore", row["iv_zscore"],
                  iv_zscore_threshold, f"surface_vol={row['surface_vol']:.4f}, rolling_mean={row['iv_rmean']:.4f}")

    # --- 5. Option price day-over-day discontinuities ---
    for leg in params.structure.legs:
        leg_data = (
            trade_df
            .filter(pl.col("leg_label") == leg.label)
            .sort("date")
            .with_columns(
                pl.col("c_price").shift(1).alias("prevPrc")
            )
            .filter(pl.col("prevPrc") > 0)
            .with_columns(
                ((pl.col("c_price") - pl.col("prevPrc")) / pl.col("prevPrc"))
                .abs()
                .alias("prc_pct_change")
            )
            .filter(
                (pl.col("prc_pct_change") > price_pct_change_threshold) &
                (pl.col("position_action") != "Open")
            )
        )
        for row in leg_data.iter_rows(named=True):
            _flag(row["date"], leg.label, "price_discontinuity", row["prc_pct_change"],
                  price_pct_change_threshold, f"c_price={row['c_price']:.2f}, prevPrc={row['prevPrc']:.2f}")

    # --- 6. Large underlying moves (informational) ---
    # Use first() per date to avoid duplicate rows when u_price differs across legs
    underlying_returns = (
        trade_df
        .group_by("date")
        .agg(pl.col("u_price").first())
        .sort("date")
        .with_columns(
            ((pl.col("u_price") / pl.col("u_price").shift(1)) - 1).alias("underlying_ret")
        )
        .filter(pl.col("underlying_ret").abs() > underlying_return_threshold)
    )
    for row in underlying_returns.iter_rows(named=True):
        _flag(row["date"], "_market_", "large_underlying_move", row["underlying_ret"],
              underlying_return_threshold, f"u_price={row['u_price']:.2f}, return={row['underlying_ret']*100:.2f}%")

    if not flags:
        return pl.DataFrame(schema={
            "date": pl.Date, "leg_label": pl.Utf8, "check": pl.Utf8,
            "value": pl.Float64, "threshold": pl.Utf8, "detail": pl.Utf8,
        })

    return pl.DataFrame(flags).sort(["date", "check", "leg_label"])


# ---------------------------------------------------------------------------
# uClose diagnostics
# ---------------------------------------------------------------------------

def diagnose_trade_arbitrage(
    trade_df: pl.DataFrame,
    params: StrategyParams,
    price_tol: float = 0.10,
) -> tuple[dict, pl.DataFrame]:
    """Diagnose no-arbitrage violations on the actual trading legs.

    For each trading date, checks:
      1. Put spread monotonicity between legs (lower delta = lower price)
      2. Butterfly convexity for 3-leg structures (BWB)

    Returns:
        (summary, violations_df)
    """
    structure = params.structure
    put_legs = sorted(
        [l for l in structure.legs if l.option_type == OptionType.PUT],
        key=lambda l: l.target_delta,
    )

    if len(put_legs) < 2:
        return {'message': 'Need >= 2 put legs'}, pl.DataFrame()

    # Pivot to wide: one row per date
    wide_prc = trade_df.pivot(on='leg_label', index='date', values='c_price')
    wide_xx = trade_df.pivot(on='leg_label', index='date', values='strike')

    diag_parts: list[pl.DataFrame] = []
    n_dates = wide_prc.height
    bfly_stats: dict = {}

    # --- Put spread: lower delta → lower strike → cheaper ---
    for i in range(len(put_legs) - 1):
        low = put_legs[i]
        high = put_legs[i + 1]
        if low.label in wide_prc.columns and high.label in wide_prc.columns:
            v = wide_prc.filter(
                pl.col(low.label) > pl.col(high.label) + price_tol
            ).select([
                'date',
                pl.lit('put_spread').alias('check'),
                (pl.col(low.label) - pl.col(high.label)).alias('violation_amount'),
                pl.concat_str([
                    pl.lit(f'{low.label}=$'), pl.col(low.label).round(2).cast(pl.Utf8),
                    pl.lit(f' > {high.label}=$'), pl.col(high.label).round(2).cast(pl.Utf8),
                ]).alias('detail'),
            ])
            if v.height > 0:
                diag_parts.append(v)

    # --- Butterfly for 3-leg structures ---
    if len(put_legs) == 3:
        l1, l2, l3 = put_legs
        labels = [l1.label, l2.label, l3.label]
        xx_labels = [f'{l}_xx' for l in labels]
        if all(l in wide_prc.columns for l in labels) and all(l in wide_xx.columns for l in labels):
            combined = wide_prc.join(
                wide_xx.select([
                    'date',
                    pl.col(l1.label).alias(xx_labels[0]),
                    pl.col(l2.label).alias(xx_labels[1]),
                    pl.col(l3.label).alias(xx_labels[2]),
                ]),
                on='date',
            )

            combined = combined.with_columns(
                ((pl.col(xx_labels[2]) - pl.col(xx_labels[1]))
                 / (pl.col(xx_labels[2]) - pl.col(xx_labels[0]))).alias('_lam')
            ).with_columns(
                (pl.col('_lam') * pl.col(l1.label)
                 + (1.0 - pl.col('_lam')) * pl.col(l3.label)
                 - pl.col(l2.label)).alias('bfly_val')
            )

            # Full distribution stats
            bfly_vals = combined['bfly_val'].drop_nulls()
            bfly_stats = {
                'mean': bfly_vals.mean(),
                'median': bfly_vals.median(),
                'std': bfly_vals.std(),
                'min': bfly_vals.min(),
                'max': bfly_vals.max(),
                'pct_negative': (bfly_vals < 0).mean() * 100,
            }

            bfly_v = combined.filter(pl.col('bfly_val') < -price_tol)
            if bfly_v.height > 0:
                diag_parts.append(bfly_v.select([
                    'date',
                    pl.lit('butterfly').alias('check'),
                    (-pl.col('bfly_val')).alias('violation_amount'),
                    pl.concat_str([
                        pl.lit(f'{l1.label}=$'), pl.col(l1.label).round(2).cast(pl.Utf8),
                        pl.lit(f' {l2.label}=$'), pl.col(l2.label).round(2).cast(pl.Utf8),
                        pl.lit(f' {l3.label}=$'), pl.col(l3.label).round(2).cast(pl.Utf8),
                        pl.lit(' bfly=$'), pl.col('bfly_val').round(4).cast(pl.Utf8),
                    ]).alias('detail'),
                ]))

    if diag_parts:
        violations_df = pl.concat(diag_parts).sort(['date', 'check'])
        n_bad = violations_df['date'].n_unique()
    else:
        violations_df = pl.DataFrame(schema={
            'date': pl.Date, 'check': pl.Utf8,
            'violation_amount': pl.Float64, 'detail': pl.Utf8,
        })
        n_bad = 0

    summary = {
        'structure': structure.name,
        'n_dates_checked': n_dates,
        'n_dates_with_violations': n_bad,
        'pct_dates_violated': n_bad / n_dates * 100 if n_dates > 0 else 0,
        'total_violations': violations_df.height,
        'bfly_stats': bfly_stats,
    }

    return summary, violations_df


def diagnose_uclose(trade_df: pl.DataFrame) -> dict:
    """Diagnose u_price data quality issues.

    Checks:
      1. Consistency across legs on the same date (should be identical).
      2. Zero, negative, null, or infinite values.
      3. Day-over-day jumps suggesting stale or bad data.

    Returns:
        dict with keys:
          "inconsistent_dates": dates where u_price differs across legs
          "bad_value_dates":    dates with u_price = 0, null, negative, or inf
          "jump_dates":         dates with |daily return| > 10%
          "bad_dates":          union of all problem dates (set)
          "summary":            printable summary dict
    """
    # 1. Consistency: u_price should be identical across all legs on a date
    uclose_per_date = (
        trade_df
        .group_by("date")
        .agg([
            pl.col("u_price").min().alias("u_price_min"),
            pl.col("u_price").max().alias("u_price_max"),
            pl.col("u_price").n_unique().alias("u_price_nunique"),
            pl.col("u_price").first().alias("u_price_first"),
        ])
        .sort("date")
    )

    inconsistent = uclose_per_date.filter(pl.col("u_price_nunique") > 1)
    inconsistent_dates = inconsistent["date"].to_list()

    # 2. Bad values: zero, negative, null, inf
    bad_values = uclose_per_date.filter(
        (pl.col("u_price_min") <= 0) |
        (pl.col("u_price_min").is_null()) |
        (pl.col("u_price_min").is_nan()) |
        (pl.col("u_price_min").is_infinite()) |
        (pl.col("u_price_max").is_infinite()) |
        (pl.col("u_price_max").is_nan())
    )
    bad_value_dates = bad_values["date"].to_list()

    # 3. Day-over-day jumps (using first leg's u_price per date as reference)
    daily_uclose = (
        uclose_per_date
        .select(["date", "u_price_first"])
        .with_columns(
            (pl.col("u_price_first") / pl.col("u_price_first").shift(1) - 1)
            .abs()
            .alias("abs_ret")
        )
    )

    jump_dates_df = daily_uclose.filter(
        (pl.col("abs_ret") > 0.10) | pl.col("abs_ret").is_infinite() | pl.col("abs_ret").is_nan()
    )
    jump_dates = jump_dates_df["date"].to_list()

    bad_dates = set(inconsistent_dates) | set(bad_value_dates) | set(jump_dates)

    # Build spread stats for inconsistent dates
    if inconsistent.height > 0:
        spread_stats = inconsistent.with_columns(
            (pl.col("u_price_max") - pl.col("u_price_min")).alias("spread")
        )
        max_spread = spread_stats["spread"].max()
        mean_spread = spread_stats["spread"].mean()
    else:
        max_spread = 0.0
        mean_spread = 0.0

    summary = {
        "total_dates": uclose_per_date.height,
        "inconsistent_dates": len(inconsistent_dates),
        "bad_value_dates": len(bad_value_dates),
        "jump_dates": len(jump_dates),
        "total_bad_dates": len(bad_dates),
        "max_spread": max_spread,
        "mean_spread": mean_spread,
    }

    return {
        "inconsistent_dates": inconsistent_dates,
        "bad_value_dates": bad_value_dates,
        "jump_dates": jump_dates,
        "bad_dates": bad_dates,
        "summary": summary,
        "inconsistent_df": inconsistent,
        "jump_df": jump_dates_df,
    }


# ---------------------------------------------------------------------------
# Delta hedging
# ---------------------------------------------------------------------------

def compute_delta_hedge_pnl(trade_df: pl.DataFrame) -> pl.DataFrame:
    """Compute daily delta-hedged P&L.

    Assumes continuous daily rebalancing of a hedge in the underlying.
    Hedge position = -net_delta (neutralize portfolio delta each day).
    Hedge P&L = -net_delta_t * (S_{t+1} - S_t).

    Returns:
        Daily DataFrame with columns:
          date, option_pnl, net_delta, hedge_pnl, total_pnl,
          u_price, nextUClose
    """
    # Aggregate to daily level
    daily = (
        trade_df
        .group_by("date")
        .agg([
            pl.col("pnl").sum().alias("option_pnl"),
            pl.col("pos_delta").sum().alias("net_delta"),
            pl.col("pos_gamma").sum().alias("net_gamma"),
            pl.col("pos_theta").sum().alias("net_theta"),
            pl.col("pos_vega").sum().alias("net_vega"),
            pl.col("u_price").first(),
            pl.col("nextTradingDate").first(),
        ])
        .sort("date")
    )

    # Get next day's u_price for hedge P&L
    uclose_map = daily.select(["date", "u_price"]).rename({"u_price": "nextUClose"})

    daily = (
        daily
        .join(uclose_map, left_on="nextTradingDate", right_on="date", how="left")
        .filter(pl.col("nextUClose").is_not_null())
    )

    # Daily underlying change
    daily = daily.with_columns(
        (pl.col("nextUClose") - pl.col("u_price")).alias("dS")
    )

    # Hedge P&L: short the underlying by net_delta amount
    # hedge_position = -net_delta (offset the portfolio delta)
    # hedge_pnl = -net_delta * (nextUClose - u_price)
    daily = daily.with_columns(
        (-pl.col("net_delta") * pl.col("dS")).alias("hedge_pnl")
    )

    daily = daily.with_columns(
        (pl.col("option_pnl") + pl.col("hedge_pnl")).alias("total_pnl")
    )

    return daily.select([
        "date", "option_pnl", "net_delta", "net_gamma", "net_theta", "net_vega",
        "hedge_pnl", "total_pnl", "u_price", "nextUClose", "dS",
    ])


def load_vol_features() -> pl.DataFrame:
    """Load pre-computed vol features from local parquet."""
    return pl.read_parquet(DATA_DIR / "vol_features.parquet")



def main():
    structures = {
        "PUT_RATIO_1x2": PUT_RATIO_1x2,
        "PUT_BWB_SHORT_SKEW": PUT_BWB_SHORT_SKEW,
    }

    # Load data once
    print("Loading market data...")
    df_market = load_market_data().collect()

    print(f"  {df_market.height:,} rows, date range: "
          f"{df_market['date'].min()} -> {df_market['date'].max()}")

    print("Loading vol features...")
    vol_features = load_vol_features()
    print(f"  {vol_features.height:,} rows")

    for name, structure in structures.items():
        print(f"\n{'='*60}")
        print(f"Structure: {name} ({structure.n_legs()} legs)")
        print(f"  Legs: {structure.leg_labels()}")
        print(f"{'='*60}")

        params = StrategyParams(structure=structure)
        trade_df = build_trade_df(df_market, vol_features, params)

        # --- Data validation ---
        print("\n--- Data Validation ---")
        failures = validate_trade_df(trade_df, params)
        if failures:
            for f in failures:
                print(f"  FAIL: {f}")
        else:
            print("  All checks passed.")

        print(f"\nResult shape: {trade_df.shape}")
        print(f"Date range: {trade_df['date'].min()} -> {trade_df['date'].max()}")
        print(f"Unique dates: {trade_df['date'].n_unique():,}")

        # Leg-level summary
        print("\n--- Leg Summary ---")
        leg_summary = (
            trade_df
            .group_by("leg_label")
            .agg([
                pl.len().alias("n_rows"),
                pl.col("strike").mean().alias("avg_strike"),
                pl.col("delta").mean().alias("avg_delta"),
                pl.col("ratio").first(),
                pl.col("side").first(),
                pl.col("pnl").sum().alias("total_pnl"),
                pl.col("pnl").mean().alias("mean_daily_pnl"),
            ])
            .sort("leg_label")
        )
        print(leg_summary)

        # --- Unhedged P&L ---
        daily_pnl = (
            trade_df
            .group_by("date")
            .agg(pl.col("pnl").sum().alias("structure_pnl"))
            .sort("date")
        )

        total_pnl = daily_pnl["structure_pnl"].sum()
        mean_pnl = daily_pnl["structure_pnl"].mean()
        std_pnl = daily_pnl["structure_pnl"].std()
        sharpe = (mean_pnl / std_pnl * (252 ** 0.5)) if std_pnl > 0 else 0.0

        print(f"\n--- Unhedged P&L ---")
        print(f"Total P&L:      ${total_pnl:,.2f}")
        print(f"Mean daily P&L: ${mean_pnl:,.4f}")
        print(f"Std daily P&L:  ${std_pnl:,.4f}")
        print(f"Sharpe (ann.):  {sharpe:.3f}")

        # --- Delta-hedged P&L ---
        hedged_daily = compute_delta_hedge_pnl(trade_df)

        h_total = hedged_daily["total_pnl"].sum()
        h_mean = hedged_daily["total_pnl"].mean()
        h_std = hedged_daily["total_pnl"].std()
        h_sharpe = (h_mean / h_std * (252 ** 0.5)) if h_std > 0 else 0.0

        print(f"\n--- Delta-Hedged P&L ---")
        print(f"Total P&L:      ${h_total:,.2f}")
        print(f"  Option comp:  ${hedged_daily['option_pnl'].sum():,.2f}")
        print(f"  Hedge comp:   ${hedged_daily['hedge_pnl'].sum():,.2f}")
        print(f"Mean daily P&L: ${h_mean:,.4f}")
        print(f"Std daily P&L:  ${h_std:,.4f}")
        print(f"Sharpe (ann.):  {h_sharpe:.3f}")

        # --- Daily portfolio greeks ---
        print(f"\n--- Daily Portfolio Greeks (avg / std / min / max) ---")
        for greek_name, greek_col in [("Delta", "net_delta"), ("Gamma", "net_gamma"),
                                       ("Theta", "net_theta"), ("Vega", "net_vega")]:
            s = hedged_daily[greek_col]
            print(f"  {greek_name:6s}:  avg={s.mean():+.4f}  std={s.std():.4f}  "
                  f"min={s.min():+.4f}  max={s.max():+.4f}")

        # --- Underlying price stats ---
        dS = hedged_daily["dS"]
        print(f"\n--- Underlying Daily Change (dS) ---")
        print(f"  avg dS:     {dS.mean():+.2f} pts")
        print(f"  std dS:     {dS.std():.2f} pts")
        print(f"  total dS:   {dS.sum():+.2f} pts  "
              f"({hedged_daily['u_price'].head(1).item():.0f} -> {hedged_daily['nextUClose'].tail(1).item():.0f})")

        # --- Hedge sanity check ---
        # If delta is constant, hedge P&L ≈ -mean_delta * total_dS
        mean_delta = hedged_daily["net_delta"].mean()
        total_dS = dS.sum()
        expected_hedge = -mean_delta * total_dS
        actual_hedge = hedged_daily["hedge_pnl"].sum()
        print(f"\n--- Hedge Sanity Check ---")
        print(f"  -mean_delta * total_dS = {-mean_delta:.4f} * {total_dS:.2f} = ${expected_hedge:,.2f}  (if delta were constant)")
        print(f"  Actual hedge P&L:  ${actual_hedge:,.2f}")
        print(f"  Discrepancy:       ${actual_hedge - expected_hedge:,.2f}  (from delta variation / gamma rebalancing)")

        # Correlation between net_delta and dS (if correlated, hedge timing matters)
        corr = hedged_daily.select(
            pl.corr("net_delta", "dS").alias("corr")
        )["corr"].item()
        print(f"  Corr(net_delta, dS): {corr:+.4f}")

        # Roll date sanity check
        roll_dates = trade_df.filter(pl.col("position_action") == "Open")
        print(f"\n--- Roll Sanity ---")
        print(f"Total Open events: {roll_dates.height}")
        print(f"First 5 roll dates: {roll_dates['date'].unique().sort().head(5).to_list()}")

        # --- Trade-level arbitrage diagnostic ---
        print(f"\n--- Trade Arbitrage Diagnostic ---")
        arb_summary, arb_violations = diagnose_trade_arbitrage(trade_df, params)
        print(f"  Dates checked: {arb_summary['n_dates_checked']}")
        print(f"  Dates with violations: {arb_summary['n_dates_with_violations']} "
              f"({arb_summary['pct_dates_violated']:.1f}%)")
        print(f"  Total violations: {arb_summary['total_violations']}")
        if arb_summary.get('bfly_stats'):
            bs = arb_summary['bfly_stats']
            print(f"  Butterfly value distribution:")
            print(f"    mean=${bs['mean']:.4f}  median=${bs['median']:.4f}  "
                  f"std=${bs['std']:.4f}  min=${bs['min']:.4f}  max=${bs['max']:.4f}")
            print(f"    % negative: {bs['pct_negative']:.1f}%")
        if arb_violations.height > 0:
            print(f"  Worst violations:")
            print(arb_violations.sort('violation_amount', descending=True).head(10))

        # --- u_price diagnostics ---
        print(f"\n--- u_price Diagnostics ---")
        diag = diagnose_uclose(trade_df)
        s = diag["summary"]
        print(f"  Total trading dates:         {s['total_dates']}")
        print(f"  Inconsistent u_price dates:  {s['inconsistent_dates']}  (u_price differs across legs)")
        print(f"  Bad value dates (0/null/inf): {s['bad_value_dates']}")
        print(f"  Jump dates (|ret| > 10%):    {s['jump_dates']}")
        print(f"  Total bad dates:             {s['total_bad_dates']}")

        if s["inconsistent_dates"] > 0:
            print(f"  Max u_price spread across legs: {s['max_spread']:.2f} pts")
            print(f"  Mean u_price spread:            {s['mean_spread']:.2f} pts")
            inc_df = diag["inconsistent_df"]
            print(f"  First 10 inconsistent dates:")
            for row in inc_df.head(10).iter_rows(named=True):
                print(f"    {row['date']}  min={row['u_price_min']:.2f}  max={row['u_price_max']:.2f}  "
                      f"spread={row['u_price_max'] - row['u_price_min']:.2f}")

        if diag["jump_df"].height > 0:
            print(f"  Jump dates detail:")
            for row in diag["jump_df"].iter_rows(named=True):
                ret_str = f"{row['abs_ret']*100:.2f}%" if row["abs_ret"] is not None else "N/A"
                print(f"    {row['date']}  u_price={row['u_price_first']:.2f}  |ret|={ret_str}")

        # --- Filtered hedge P&L (excluding bad u_price dates) ---
        bad_dates = diag["bad_dates"]
        if bad_dates:
            bad_dates_list = list(bad_dates)
            # Also exclude the day *after* each bad date (since nextPrc / dS reference the next day)
            all_dates_sorted = trade_df["date"].unique().sort().to_list()
            dates_to_exclude = set(bad_dates_list)
            for d in bad_dates_list:
                idx = None
                for i, td in enumerate(all_dates_sorted):
                    if td == d:
                        idx = i
                        break
                if idx is not None and idx + 1 < len(all_dates_sorted):
                    dates_to_exclude.add(all_dates_sorted[idx + 1])

            trade_df_clean = trade_df.filter(~pl.col("date").is_in(list(dates_to_exclude)))
            n_removed = trade_df["date"].n_unique() - trade_df_clean["date"].n_unique()
            print(f"\n--- Filtered Hedge P&L (excluding {n_removed} bad/adjacent dates) ---")

            if trade_df_clean.height > 0:
                hedged_clean = compute_delta_hedge_pnl(trade_df_clean)
                hc_total = hedged_clean["total_pnl"].sum()
                hc_mean = hedged_clean["total_pnl"].mean()
                hc_std = hedged_clean["total_pnl"].std()
                hc_sharpe = (hc_mean / hc_std * (252 ** 0.5)) if hc_std > 0 else 0.0

                print(f"  Total P&L:      ${hc_total:,.2f}")
                print(f"    Option comp:  ${hedged_clean['option_pnl'].sum():,.2f}")
                print(f"    Hedge comp:   ${hedged_clean['hedge_pnl'].sum():,.2f}")
                print(f"  Mean daily P&L: ${hc_mean:,.4f}")
                print(f"  Std daily P&L:  ${hc_std:,.4f}")
                print(f"  Sharpe (ann.):  {hc_sharpe:.3f}")

                # Filtered dS stats
                dS_clean = hedged_clean["dS"]
                print(f"  dS after filter:  avg={dS_clean.mean():+.2f}  std={dS_clean.std():.2f}")

                # Filtered hedge sanity
                md = hedged_clean["net_delta"].mean()
                td = dS_clean.sum()
                exp_h = -md * td
                act_h = hedged_clean["hedge_pnl"].sum()
                print(f"  Hedge sanity:  expected=${exp_h:,.2f}  actual=${act_h:,.2f}  "
                      f"discrepancy=${act_h - exp_h:,.2f}")
            else:
                print("  No data left after filtering!")
        else:
            print("\n  No bad u_price dates found — hedge P&L already clean.")

        # --- Outlier detection ---
        print(f"\n--- Outlier Detection ---")
        outliers = detect_outliers(trade_df, params)
        if outliers.height == 0:
            print("  No outliers detected.")
        else:
            # Summary by check type
            summary = (
                outliers
                .group_by("check")
                .agg([
                    pl.len().alias("count"),
                    pl.col("date").min().alias("first_date"),
                    pl.col("date").max().alias("last_date"),
                ])
                .sort("count", descending=True)
            )
            print(f"  {outliers.height} total flags across {summary.height} check types:")
            for row in summary.iter_rows(named=True):
                print(f"    {row['check']:30s}  n={row['count']:4d}  ({row['first_date']} -> {row['last_date']})")

            # Worst offenders (top 10 by absolute value)
            worst = (
                outliers
                .with_columns(pl.col("value").cast(pl.Float64, strict=False).abs().alias("abs_value"))
                .sort("abs_value", descending=True)
                .head(10)
                .select(["date", "leg_label", "check", "value", "detail"])
            )
            print(f"\n  Top 10 most extreme flags:")
            for row in worst.iter_rows(named=True):
                print(f"    {row['date']}  {row['leg_label']:20s}  {row['check']:30s}  val={row['value']}  {row['detail']}")

    return trade_df


if __name__ == "__main__":
    trade_df = main()
