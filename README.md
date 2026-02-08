# DeleverageSentinel

**Institutional-grade early warning system for systemic deleveraging events.**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

DeleverageSentinel monitors credit spreads, tail stress, macro indicators, and cross-asset correlation clustering to detect early signs of synchronised market deleveraging — before it becomes front-page news.

## Why This Exists

The 2008 financial crisis, the 2020 COVID crash, and every major deleveraging event share a common signature: assets that appeared uncorrelated suddenly move together. Diversification fails precisely when it's needed most.

This tool operationalises two key results from the [Applied Probabilistic Systems](https://jasongething.substack.com) research series:

- **Axiom IV — Emergent Coupling**: Correlation intensifies under stress. Independence is a calm-weather illusion.
- **The Homogeneity Threshold**: When cross-asset correlations spike, behavioural diversity collapses and systemic failure becomes likely.

The **HOMOGENEITY_SPIKE** signal — which detects correlation clustering across equities, volatility, credit, and the dollar — is the system's core innovation and carries the highest weight (32%).

## Signals

| Signal | Weight | What It Detects |
|--------|--------|-----------------|
| `HY_OAS_JUMP` | 18% | High-yield spreads widening fast (30 d) |
| `TAIL_DIVERGENCE` | 18% | CCC−HY spread widening — worst credits deteriorating faster |
| `CCC_OAS_HIGH` | 12% | Absolute CCC spread stress level |
| `UK_APPROVALS_DOWN` | 12% | UK mortgage approvals declining (3 m) |
| `HOMOGENEITY_SPIKE` | 32% | Cross-asset correlation clustering — diversity collapse |
| `GBP_WEAK` | 8% | GBP/USD weakness as UK macro proxy |

## Regimes

| Score | Regime | Meaning |
|-------|--------|---------|
| 0–24 | 🟢 GREEN | Normal conditions |
| 25–49 | 🟡 AMBER | Elevated stress — monitor closely |
| 50–74 | 🔴 RED | Significant stress — alerts fire |
| 75–100 | ⚫ BLACK | Systemic deleveraging likely underway |

## Quick Start

```bash
# 1. Install
pip install requests pandas numpy pyyaml python-dateutil

# 2. Get free FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html
export FRED_API_KEY="your_key_here"

# 3. Run
python deleverage_sentinel.py run --config config.yaml

# 4. Open out/index.html in your browser
```

## Usage

```bash
# Live scoring
python deleverage_sentinel.py run --config config.yaml

# Backtest (2018–present)
python deleverage_sentinel.py backtest --config config.yaml \
    --start 2018-01-01 --end 2026-02-01

# Validate config without API calls
python deleverage_sentinel.py validate --config config.yaml

# Offline mode (local CSVs from data/ directory)
python deleverage_sentinel.py run --config config.yaml --offline

# Verbose logging
python deleverage_sentinel.py -v run --config config.yaml
```

## Outputs

```
out/
├── latest.json    # Machine-readable snapshot (schema-versioned)
├── index.html     # Self-contained dashboard
└── history.csv    # Backtest time series
```

Host the `out/` folder on Netlify, Vercel, or any static host for a live dashboard.

## Architecture

```
Config (YAML)
    │
    ▼
Fetch ─────────────────────────────────────────┐
    │  FRED: HY OAS, CCC OAS, SP500, VIX, DXY │
    │  FRED: GBP/USD (replaces exchangerate.host)
    │  BoE:  UK mortgage approvals              │
    │  (or local CSVs in offline mode)          │
    ▼                                           │
Compute ◄──────────────────────────────────────┘
    │  6 independent signals
    │  Correlation clustering (homogeneity proxy)
    │  Severity scoring (0–1 per signal)
    ▼
Score
    │  Weighted composite (0–100)
    │  Regime classification (GREEN → BLACK)
    ▼
Output
    │  JSON + HTML dashboard + CSV
    │  Telegram / Email alerts (on RED/BLACK)
    ▼
```

## Data Sources

All free. Only a FRED API key is required.

| Source | Series | Frequency |
|--------|--------|-----------|
| FRED | ICE BofA HY OAS (`BAMLH0A0HYM2`) | Daily |
| FRED | ICE BofA CCC OAS (`BAMLH0A3HYC`) | Daily |
| FRED | S&P 500 (`SP500`) | Daily |
| FRED | VIX (`VIXCLS`) | Daily |
| FRED | Trade-Weighted USD (`DTWEXBGS`) | Weekly |
| FRED | GBP/USD (`DEXUSUK`) | Daily |
| BoE | UK Mortgage Approvals (`LPMVYVA`) | Monthly |

## Production Features

- **Retry with exponential backoff** — network blips don't kill the run
- **Graceful degradation** — non-critical feeds fail without crashing
- **Data staleness detection** — warns when data is > 5 days old
- **Config validation** — catches typos before API calls
- **Custom exceptions** — `ConfigError` vs `FetchError` for clear debugging
- **Structured logging** — to stderr, cron-safe, debug-level available
- **Schema-versioned JSON** — machine consumers detect format changes
- **Backtest engine** — historical evaluation with no lookahead bias
- **O(log n) lookups** — binary search replaces O(n) slicing in backtest

## Alerts

```yaml
alerting:
  alert_on_regimes: [RED, BLACK]
  telegram:
    enabled: true
    token_env: TELEGRAM_BOT_TOKEN
    chat_id_env: TELEGRAM_CHAT_ID
```

## Offline Mode

For air-gapped environments, provide CSVs in `data/`:

```csv
date,value
2025-01-02,3.45
2025-01-03,3.52
```

## Cron

```cron
0 22 * * 1-5 cd /path/to/sentinel && python deleverage_sentinel.py run --config config.yaml >> /var/log/sentinel.log 2>&1
```

## Theoretical Foundation

| Paper | Title | Key Insight |
|-------|-------|-------------|
| I | [The Reflexive Stagnation Trap](https://jasongething.substack.com) | AI epistemic convergence undermines innovation |
| II | [The Homogeneity Threshold](https://jasongething.substack.com) | Critical diversity level below which systems become fragile |
| III | [The First Principles of Uncertainty](https://jasongething.substack.com) | Five axioms + seven theorems for the science of risk |

## License

MIT

## Author

**Jason Gething** — Founder, [FishIntel Global](https://fishintelglobal.com)

Not financial advice.
