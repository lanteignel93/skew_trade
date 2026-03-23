# Skew Trade — PCA-Based Volatility Skew Strategy

A systematic strategy that exploits mean reversion in equity volatility skew using PCA decomposition of the SPXW put-wing implied volatility surface.

## Hypothesis

Equity volatility skew fluctuates with fear/complacency cycles. PCA decomposition of the ATM-demeaned put-wing vol surface reveals PC2 as a clean measure of skew steepness. When PC2 is elevated (steep skew, D7–D10), skew tends to compress — creating a systematic short-skew opportunity via ratio spreads and broken-wing butterflies.

**OU half-life**: ~26.7 days [CI: 19.9, 41.6]. Walk-forward validated: D9-10 fires with 67% hit rate (t = 4.07, p = 0.00017).

## Results

| Metric | Value |
|---|---|
| OOS total Sharpe | **+2.53** |
| Positive windows | **13 / 15 (87%)** |
| Test period | 2019-01 → 2026-03 |
| Structure | Ratio_20_10 (primary) |
| Conditioning | Five-state regime sizing |


## Strategy Design

- **Signal**: PC2 of ATM-demeaned 3-month SPXW put-wing IV (17 moneyness points, −0.30 to 0.00)
- **PCA window**: Rolling 756 days (~3 years)
- **Rebalance**: Monthly (month-end)
- **Gate**: D7–D10 → short skew; D1–D6 → no position
- **Hold period**: ~3 weeks (enter ~90 DTE, exit ~70 DTE)
- **Delta hedge**: Daily continuous
- **Vega normalization**: 100 aggregate |vega| at entry
- **Sizing**: Five-state regime grid (PC2 decile × VVIX/VIX ratio × term structure shape)

## Project Structure

```
skew_trade/
├── src/
│   ├── main.py            # Vol surface construction pipeline
│   ├── trade_data.py      # Option structure framework, trade building, validation
│   └── walk_forward.py    # Walk-forward PCA (756d rolling window, deciles)
├── notebooks/
│   ├── 00_data_quality.ipynb              # Surface data validation
│   ├── 01_raw_skew.ipynb                  # Simple skew metrics, ACF, OU half-life
│   ├── 02_pca_factors.ipynb               # PCA extraction, loading stability
│   ├── 03_walk_forward.ipynb              # Walk-forward validation, stat tests
│   ├── 04_structure_selection.ipynb       # PC2 correlation screening (12 structures)
│   ├── 05_trade_mechanics.ipynb           # Trade building, validation, baselines
│   ├── 06_pnl_analysis.ipynb              # Vega-normalized PnL decomposition (18 sections)
│   ├── 07_signal_strategy.ipynb           # PC2 conditioned strategy (D9-10 gate)
│   ├── 08_conditioned_strategy.ipynb      # Five-state conditioned backtest
│   └── 09_walk_forward_validation.ipynb   # Expanding-window OOS validation
├── pyproject.toml         # Project config and dependencies
└── README.md              # This file
```

## Data Requirements

### Running notebooks (no special access needed)

All notebooks read from pre-generated parquet files in `data/`. If `data/` is populated, everything runs without external dependencies.

| File | Size | Description |
|---|---|---|
| `data/options_raw.parquet` | ~580 MB | SPXW daily option chain (2016+) |
| `data/vol_features.parquet` | ~112 KB | VIX, VVIX, VIX9D, VIX3M + rolling quintiles |
| `data/surface_data.parquet` | ~364 KB | Interpolated 3-month vol surface |
| `data/walk_forward_results_month_end.parquet` | ~9 KB | PC2 walk-forward deciles |


## Notebook Pipeline

```
00 Data Quality → 01 Raw Skew → 02 PCA Factors → 03 Walk-Forward → 04 Structure Selection
                                                                             ↓
05 Trade Mechanics → 06 PnL Analysis → 07 Signal Strategy → 08 Conditioned Strategy
                                                                             ↓
                                                             09 Walk-Forward Validation
```

## Installation

```bash
git clone https://github.com/lanteignel93/skew_trade.git
cd skew_trade
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

> Requires [uv](https://github.com/astral-sh/uv) and Python 3.10+.

## Dependencies

Core: `polars`, `numpy`, `scipy`, `scikit-learn`, `statsmodels`, `matplotlib`, `seaborn`

Dev: `ruff`, `mypy`, `pre-commit`, `jupyter`, `notebook`

Full dependency spec in `pyproject.toml`.

## License

Private research. Not for redistribution without permission.
