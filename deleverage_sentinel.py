#!/usr/bin/env python3
"""
DeleverageSentinel — Institutional-Grade Deleveraging Early Warning System
==========================================================================

Monitors credit spreads, tail stress, macro indicators, and cross-asset
correlation clustering to detect early signs of systemic deleveraging.

Grounded in the Precision-Fragility Paradox (Gething 2025): when risk
models converge, behavioural homogeneity produces synchronised failure.
The Homogeneity Threshold proxy (correlation clustering) is the core
innovation — it detects when "diversified" assets start moving together,
the signature of stress-coupling (Axiom IV, Emergent Coupling).

Architecture
------------
    Config (YAML) → Fetch (FRED / BoE / CSV) → Compute → Score → Output

Data Sources (default, all free)
---------------------------------
    FRED   : HY OAS, CCC OAS, S&P 500, VIX, DXY, GBP/USD
    BoE    : UK mortgage approvals (IADB CSV export)
    Offline: local CSVs for development / air-gapped environments

Install
-------
    pip install requests pandas numpy pyyaml python-dateutil

Quick Start
-----------
    export FRED_API_KEY="your_key"
    python deleverage_sentinel.py run --config config.yaml

    python deleverage_sentinel.py backtest --config config.yaml \\
        --start 2018-01-01 --end 2026-02-01

    python deleverage_sentinel.py validate --config config.yaml

DISCLAIMER: Informational tooling, not financial advice.

License : MIT
Author  : Jason Gething / FishIntel Global Ltd
Version : 2.0.0
"""

from __future__ import annotations

__version__ = "2.0.0"
__author__ = "Jason Gething / FishIntel Global"

# ── stdlib ────────────────────────────────────────────────────────────────────
import argparse
import datetime as dt
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import requests
import yaml
from dateutil.relativedelta import relativedelta

# ═════════════════════════════════════════════════════════════════════════════
#  LOGGING
# ═════════════════════════════════════════════════════════════════════════════

LOG_FMT = "%(asctime)s [%(levelname)-5.5s] %(name)s — %(message)s"
LOG_DATE = "%Y-%m-%d %H:%M:%S"

log = logging.getLogger("sentinel")


def _configure_logging(verbose: bool = False) -> None:
    """Route structured logs to stderr (safe for cron / systemd)."""
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(LOG_FMT, datefmt=LOG_DATE))
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(handler)


# ═════════════════════════════════════════════════════════════════════════════
#  CONSTANTS & SCHEMA
# ═════════════════════════════════════════════════════════════════════════════

SCHEMA_VERSION = "2.0"

# Keys that MUST exist in config.yaml (fail-fast on typos)
_REQUIRED_TOP = ("feeds", "thresholds", "weights", "bands", "lookback",
                 "output", "alerting")

_REQUIRED_THRESHOLDS = (
    "hy_oas_jump_30d_bps", "ccc_oas_level_pct", "tail_divergence_30d_bps",
    "uk_approvals_drop_3m_pct", "gbpusd_drop_30d_pct",
    "corr_spike_level_abs", "corr_spike_count",
)

_SIGNAL_KEYS = (
    "HY_OAS_JUMP", "CCC_OAS_HIGH", "TAIL_DIVERGENCE",
    "UK_APPROVALS_DOWN", "GBP_WEAK", "HOMOGENEITY_SPIKE",
)


# ═════════════════════════════════════════════════════════════════════════════
#  CUSTOM EXCEPTIONS
# ═════════════════════════════════════════════════════════════════════════════

class ConfigError(Exception):
    """Raised when config.yaml is invalid or incomplete."""


class FetchError(Exception):
    """Raised when a data source cannot be reached after retries."""


# ═════════════════════════════════════════════════════════════════════════════
#  DATA MODELS
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Signal:
    """One risk signal with scored severity.

    Attributes
    ----------
    key           : machine-readable identifier (matches config weight key)
    name          : human-readable label
    triggered     : True if the threshold was breached
    severity      : 0.0 → 1.0 continuous severity
    value         : display string for the metric reading
    detail        : full explanation string
    data_age_days : staleness of the underlying data (0 = fresh)
    """
    key: str
    name: str
    triggered: bool
    severity: float
    value: str
    detail: str
    data_age_days: int = 0


@dataclass
class SentinelResult:
    """Complete snapshot — all signals + composite score."""
    as_of: str
    schema_version: str
    score: int
    regime: str
    signals: Dict[str, Signal]
    data_warnings: List[str] = field(default_factory=list)


@dataclass
class MarketData:
    """All fetched time series, ready for computation."""
    hy_oas: pd.Series
    ccc_oas: pd.Series
    uk_approvals: pd.Series
    gbpusd: pd.Series
    homogeneity_df: pd.DataFrame
    warnings: List[str] = field(default_factory=list)


# ═════════════════════════════════════════════════════════════════════════════
#  PURE UTILITY FUNCTIONS  (all O(1) unless noted)
# ═════════════════════════════════════════════════════════════════════════════

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *x* to [lo, hi]."""
    return max(lo, min(hi, float(x)))


def _safe_float(x: Any) -> Optional[float]:
    """Coerce to finite float or return None."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x) if np.isfinite(x) else None
    s = str(x).strip()
    if s in ("", ".", "NaN", "nan", "NA", "N/A", "#N/A", "ND"):
        return None
    try:
        v = float(s)
        return v if np.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def _pct_change(new: float, old: float) -> Optional[float]:
    """Percentage change from *old* to *new*, or None when *old* == 0."""
    if old == 0.0:
        return None
    return (new - old) / abs(old) * 100.0


def _bps(delta_pct_points: float) -> float:
    """Percentage-point delta → basis points."""
    return float(delta_pct_points) * 100.0


def _data_age(series: pd.Series) -> int:
    """Days since the most recent non-NaN observation."""
    s = series.dropna()
    if s.empty:
        return 999
    last = pd.Timestamp(s.index[-1])
    now = pd.Timestamp(dt.datetime.now(dt.timezone.utc))
    # Strip timezone info for safe comparison with tz-naive data
    if last.tzinfo is None:
        now = now.tz_localize(None)
    return max(0, (now - last).days)


def _nearest_before(series: pd.Series, target: pd.Timestamp) -> Optional[float]:
    """Value at or just before *target*.  O(log n) with sorted index."""
    s = series.dropna()
    if s.empty:
        return None
    idx = s.index.searchsorted(target, side="right") - 1
    if idx < 0:
        return float(s.iloc[0])
    return float(s.iloc[idx])


def _ensure_dir(path: str) -> None:
    """Create parent directories for *path* if missing."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _today_utc() -> dt.date:
    """UTC date right now."""
    return dt.datetime.now(dt.timezone.utc).date()


def _empty(name: str) -> pd.Series:
    """Return an empty named float Series."""
    s = pd.Series(dtype=float)
    s.name = name
    return s


# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG LOADING & VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def load_config(path: str) -> Dict[str, Any]:
    """Load, validate, and return YAML config.

    Fail-fast philosophy: a typo in ``thresholds`` is caught here, not
    30 seconds later when the API calls are done and the script crashes
    with a confusing ``KeyError``.

    Raises
    ------
    ConfigError
        On any structural problem.
    """
    if not os.path.isfile(path):
        raise ConfigError(f"Config not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    if not isinstance(cfg, dict):
        raise ConfigError(f"Config must be a YAML mapping, got {type(cfg).__name__}")

    # top-level keys
    missing = [k for k in _REQUIRED_TOP if k not in cfg]
    if missing:
        raise ConfigError(f"Missing top-level keys: {missing}")

    # thresholds
    thresh = cfg.get("thresholds", {})
    missing_t = [k for k in _REQUIRED_THRESHOLDS if k not in thresh]
    if missing_t:
        raise ConfigError(f"Missing thresholds: {missing_t}")

    # weights
    weights = cfg.get("weights", {})
    missing_w = [k for k in _SIGNAL_KEYS if k not in weights]
    if missing_w:
        log.warning("Weights missing for %s — they will score 0", missing_w)

    total_w = sum(float(v) for v in weights.values())
    if total_w > 105:
        log.warning("Weight sum is %.0f (> 100) — scores may exceed 100", total_w)

    # bands
    bands = cfg.get("bands", {})
    covered: set = set()
    for name, rng in bands.items():
        if not isinstance(rng, list) or len(rng) != 2:
            raise ConfigError(f"Band '{name}' must be [lo, hi], got {rng}")
        covered.update(range(int(rng[0]), int(rng[1]) + 1))
    if not covered.issuperset(range(101)):
        log.warning("Bands don't fully cover 0–100; some scores → UNKNOWN")

    # FRED GBPUSD check
    fred_series = cfg.get("feeds", {}).get("fred", {}).get("series", {})
    if "GBPUSD" not in fred_series:
        log.warning(
            "No GBPUSD in feeds.fred.series — add  GBPUSD: DEXUSUK  "
            "(free FRED series, replaces exchangerate.host which is now paywalled)"
        )

    log.info("Config OK — %d thresholds, %d weights, %d bands",
             len(thresh), len(weights), len(bands))
    return cfg


def _band_from_score(score: int, bands: Dict[str, List[int]]) -> str:
    """Map a 0–100 score to a named regime."""
    for name, rng in bands.items():
        if int(rng[0]) <= score <= int(rng[1]):
            return name
    return "UNKNOWN"


# ═════════════════════════════════════════════════════════════════════════════
#  HTTP WITH RETRY + EXPONENTIAL BACKOFF
# ═════════════════════════════════════════════════════════════════════════════

def _http_get(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    *,
    max_retries: int = 3,
    backoff: float = 2.0,
    timeout: int = 25,
    label: str = "",
) -> requests.Response:
    """GET with exponential back-off on 429 / 5xx / connection errors.

    Parameters
    ----------
    url         : target URL
    params      : query parameters
    max_retries : maximum attempts
    backoff     : base seconds for exponential delay
    timeout     : per-request timeout
    label       : human label for logs

    Raises
    ------
    FetchError  after all retries exhausted.
    """
    last: Optional[Exception] = None
    tag = label or url.split("/")[-1][:40]

    for attempt in range(1, max_retries + 1):
        try:
            log.debug("[%s] attempt %d/%d", tag, attempt, max_retries)
            resp = requests.get(url, params=params, timeout=timeout)

            if resp.status_code in (429,) or resp.status_code >= 500:
                wait = backoff ** attempt
                log.warning("[%s] HTTP %d — retry in %.1fs",
                            tag, resp.status_code, wait)
                time.sleep(wait)
                continue

            resp.raise_for_status()
            return resp

        except (requests.ConnectionError, requests.Timeout) as exc:
            wait = backoff ** attempt
            log.warning("[%s] %s — retry in %.1fs",
                        tag, type(exc).__name__, wait)
            last = exc
            time.sleep(wait)

    raise FetchError(f"[{tag}] all {max_retries} attempts failed") from last


# ═════════════════════════════════════════════════════════════════════════════
#  DATA FETCHERS
# ═════════════════════════════════════════════════════════════════════════════

def _fetch_fred(series_id: str, start: dt.date, api_key: str) -> pd.Series:
    """Fetch one FRED time series.  O(n) where n = observations."""
    resp = _http_get(
        "https://api.stlouisfed.org/fred/series/observations",
        params={
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "observation_start": start.isoformat(),
            "sort_order": "asc",
        },
        label=f"FRED:{series_id}",
    )
    rows: Dict[pd.Timestamp, float] = {}
    for o in resp.json().get("observations", []):
        d, v = o.get("date"), _safe_float(o.get("value"))
        if d and v is not None:
            rows[pd.to_datetime(d)] = v

    if not rows:
        raise FetchError(f"FRED returned 0 valid rows for {series_id}")

    s = pd.Series(rows, dtype=float).sort_index()
    s.name = series_id
    log.debug("FRED %s : %d obs  %s → %s",
              series_id, len(s), s.index[0].date(), s.index[-1].date())
    return s


def _fetch_boe(
    url_template: str,
    series_code: str,
    date_from: dt.date,
    date_to: dt.date,
) -> pd.Series:
    """Fetch a Bank of England IADB series via CSV export.  O(n).

    The BoE CSV format is fragile — this function validates the schema
    and falls back to positional columns with a warning.
    """
    url = url_template.format(
        date_from=date_from.strftime("%d/%b/%Y"),
        date_to=date_to.strftime("%d/%b/%Y"),
        series_codes=series_code,
    )
    resp = _http_get(url, label=f"BoE:{series_code}")
    df = pd.read_csv(StringIO(resp.text))

    # robust column detection
    date_col = value_col = None
    for c in df.columns:
        lc = str(c).strip().lower()
        if "date" in lc and date_col is None:
            date_col = c
        if series_code.upper() in str(c).upper():
            value_col = c
        elif "value" in lc and value_col is None:
            value_col = c

    if date_col is None:
        date_col = df.columns[0]
        log.warning("BoE: no 'date' column — using '%s'", date_col)
    if value_col is None:
        value_col = df.columns[-1]
        log.warning("BoE: no value column — using '%s'", value_col)

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", dayfirst=True)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[date_col, value_col]).sort_values(date_col)

    if df.empty:
        raise FetchError(f"BoE returned 0 valid rows for {series_code}")

    s = pd.Series(df[value_col].values, index=df[date_col].values, dtype=float)
    s.name = series_code
    log.debug("BoE %s : %d obs", series_code, len(s))
    return s


def _load_csv(path: str) -> pd.Series:
    """Load a local CSV (date, value columns).  O(n)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if "date" not in df.columns or "value" not in df.columns:
        raise ConfigError(
            f"{path} must have columns 'date' and 'value', "
            f"found {list(df.columns)}"
        )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"]).sort_values("date")
    s = pd.Series(df["value"].values, index=df["date"].values, dtype=float)
    log.debug("CSV %s : %d obs", path, len(s))
    return s


def _fetch_fx_legacy(fx_cfg: Dict[str, Any], start: dt.date) -> pd.Series:
    """Fallback: exchangerate.host (legacy configs without FRED GBPUSD).

    WARNING: This endpoint requires a paid API key since mid-2024.
    Prefer FRED DEXUSUK — it's free and more reliable.
    """
    if not fx_cfg:
        raise FetchError("No FX config and no FRED GBPUSD series")
    quote = fx_cfg.get("quote", "USD")
    params = {
        "base": fx_cfg.get("base", "GBP"),
        "symbols": quote,
        "start_date": start.isoformat(),
        "end_date": _today_utc().isoformat(),
    }
    resp = _http_get(
        fx_cfg.get("timeseries_url", "https://api.exchangerate.host/timeseries"),
        params=params, label="exchangerate.host",
    )
    rates = resp.json().get("rates", {})
    rows: Dict[pd.Timestamp, float] = {}
    for d, obj in rates.items():
        v = _safe_float(obj.get(quote))
        if v is not None:
            rows[pd.to_datetime(d)] = v
    if not rows:
        raise FetchError("exchangerate.host returned 0 rates")
    s = pd.Series(rows, dtype=float).sort_index()
    s.name = "GBPUSD"
    return s


# ═════════════════════════════════════════════════════════════════════════════
#  DATA ASSEMBLY (with graceful degradation)
# ═════════════════════════════════════════════════════════════════════════════

def _fetch_all(
    cfg: Dict[str, Any],
    offline: bool,
    start: dt.date,
) -> MarketData:
    """Fetch every data source.

    Critical feeds (HY, CCC) raise on failure.
    Non-critical feeds degrade gracefully: a warning is logged and the
    corresponding signal is disabled rather than crashing the run.

    O(k × n)  where k = feeds, n = observations per feed.
    """
    feeds = cfg["feeds"]
    warns: List[str] = []

    # ── offline mode ──────────────────────────────────────────────────────
    if offline:
        csv = feeds["offline_csv"]
        hy  = _load_csv(csv["hy_oas"])
        ccc = _load_csv(csv["ccc_oas"])
        uk  = _load_csv(csv["uk_approvals"])
        fx  = _load_csv(csv["gbpusd"])
        spx = _load_csv(csv["sp500"])
        vix = _load_csv(csv["vix"])
        dxy = _load_csv(csv["dxy"])

    # ── online mode ───────────────────────────────────────────────────────
    else:
        fred_key = os.getenv(feeds["fred"]["api_key_env"], "")
        if not fred_key:
            raise ConfigError(
                f"Set env var {feeds['fred']['api_key_env']} with your FRED API key "
                f"(free: https://fred.stlouisfed.org/docs/api/api_key.html) "
                f"or run with --offline"
            )
        ser = feeds["fred"]["series"]

        # CRITICAL — must succeed
        hy  = _fetch_fred(ser["HY_OAS"],  start, fred_key)
        ccc = _fetch_fred(ser["CCC_OAS"], start, fred_key)

        # GBP/USD via FRED (replaces exchangerate.host — free & reliable)
        fx = _empty("GBPUSD")
        if "GBPUSD" in ser:
            try:
                fx = _fetch_fred(ser["GBPUSD"], start, fred_key)
            except Exception as exc:
                w = f"GBPUSD fetch failed ({exc}) — signal disabled"
                log.warning(w); warns.append(w)
        else:
            try:
                fx = _fetch_fx_legacy(feeds.get("fx", {}), start)
            except Exception as exc:
                w = f"FX fetch failed ({exc}) — GBP_WEAK signal disabled"
                log.warning(w); warns.append(w)

        # Homogeneity inputs — degrade gracefully
        hom_parts: Dict[str, pd.Series] = {}
        for label in ("SP500", "VIX", "DXY"):
            try:
                hom_parts[label] = _fetch_fred(ser[label], start, fred_key)
            except Exception as exc:
                w = f"{label} fetch failed ({exc}) — excluded from homogeneity"
                log.warning(w); warns.append(w)
        spx = hom_parts.get("SP500", _empty("SP500"))
        vix = hom_parts.get("VIX",   _empty("VIX"))
        dxy = hom_parts.get("DXY",   _empty("DXY"))

        # UK approvals — Bank of England
        try:
            boe = feeds["boe"]
            lk  = cfg["lookback"]
            uk  = _fetch_boe(
                boe["csv_url_template"], boe["series_code"],
                date_from=start - relativedelta(months=int(lk["uk_fetch_months"])),
                date_to=_today_utc(),
            )
        except Exception as exc:
            w = f"BoE UK approvals failed ({exc}) — signal disabled"
            log.warning(w); warns.append(w)
            uk = _empty("UK_APPROVALS")

    # ── standardise names ─────────────────────────────────────────────────
    hy.name, ccc.name  = "HY_OAS", "CCC_OAS"
    uk.name, fx.name   = "UK_APPROVALS", "GBPUSD"

    # ── staleness check ───────────────────────────────────────────────────
    for label, s in [("HY_OAS", hy), ("CCC_OAS", ccc)]:
        age = _data_age(s)
        if age > 5:
            w = f"{label} data is {age} days stale — scores may lag reality"
            log.warning(w); warns.append(w)

    # ── build homogeneity frame ───────────────────────────────────────────
    pieces: Dict[str, pd.Series] = {"HY_OAS": hy}
    for label, s in [("SP500", spx), ("VIX", vix), ("DXY", dxy)]:
        if not s.empty:
            s.name = label
            pieces[label] = s
    if len(pieces) < 3:
        warns.append("Homogeneity degraded: < 3 cross-asset series available")

    hom_df = pd.concat(pieces.values(), axis=1).dropna()

    return MarketData(
        hy_oas=hy, ccc_oas=ccc, uk_approvals=uk,
        gbpusd=fx, homogeneity_df=hom_df, warnings=warns,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  SIGNAL COMPUTATION
# ═════════════════════════════════════════════════════════════════════════════

def _corr_spikes(
    hom_df: pd.DataFrame,
    window: int,
    threshold: float,
) -> Tuple[int, List[str], Optional[pd.DataFrame]]:
    """Detect correlation clustering: when 'independent' assets move together.

    This is the **Homogeneity Threshold proxy** — the system's core signal.
    Uses rolling returns-correlation over the specified window.

    O(w × k²)  where w = window, k = asset count.
    """
    df = hom_df.dropna()
    if len(df) < window + 5:
        return 0, [], None

    rets = df.pct_change().dropna()
    if len(rets) < window:
        return 0, [], None

    corr = rets.iloc[-window:].corr()
    cols = list(corr.columns)
    pairs: List[str] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c = float(corr.iloc[i, j])
            if np.isfinite(c) and abs(c) >= threshold:
                pairs.append(f"{cols[i]}~{cols[j]}:{c:+.2f}")
    return len(pairs), pairs, corr


def compute_signals(
    cfg: Dict[str, Any],
    data: MarketData,
    as_of: Optional[pd.Timestamp] = None,
) -> Dict[str, Signal]:
    """Compute all risk signals from market data.

    Parameters
    ----------
    cfg    : parsed config dict
    data   : MarketData bundle
    as_of  : historical evaluation timestamp (backtest mode);
             None = latest available data

    Returns
    -------
    dict mapping signal key → Signal

    Complexity: O(k × n) where k = signals, n = series length.
    """
    T = cfg["thresholds"]
    L = cfg["lookback"]
    sigs: Dict[str, Signal] = {}

    # helpers: slice to as_of when backtesting
    def _s(s: pd.Series) -> pd.Series:
        s = s.dropna()
        return s.loc[s.index <= as_of] if as_of is not None else s

    def _sdf(df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[df.index <= as_of] if as_of is not None else df

    is_live = as_of is None
    jump = int(L["jump_days"])

    # ── 1. HY OAS jump (30 d) ────────────────────────────────────────────
    hy = _s(data.hy_oas)
    if len(hy) >= 10:
        now = float(hy.iloc[-1])
        then = _nearest_before(hy, hy.index[-1] - pd.Timedelta(days=jump))
        then = then if then is not None else float(hy.iloc[0])
        delta = _bps(now - then)
        t_bps = float(T["hy_oas_jump_30d_bps"])

        sigs["HY_OAS_JUMP"] = Signal(
            key="HY_OAS_JUMP",
            name="HY spreads jump (30 d)",
            triggered=delta >= t_bps,
            severity=_clamp(delta / (t_bps * 2.0)),
            value=f"{delta:.0f} bps",
            detail=(f"HY OAS {now:.2f}% vs 30 d ago {then:.2f}% "
                    f"(Δ{delta:+.0f} bps, threshold {t_bps:.0f})"),
            data_age_days=_data_age(hy) if is_live else 0,
        )

    # ── 2. CCC tail stress (absolute level) ──────────────────────────────
    ccc = _s(data.ccc_oas)
    if len(ccc) >= 5:
        ccc_now = float(ccc.iloc[-1])
        t_pct = float(T["ccc_oas_level_pct"])

        sigs["CCC_OAS_HIGH"] = Signal(
            key="CCC_OAS_HIGH",
            name="CCC tail stress (level)",
            triggered=ccc_now >= t_pct,
            severity=_clamp(ccc_now / (t_pct * 1.5)),
            value=f"{ccc_now:.2f}%",
            detail=f"CCC OAS {ccc_now:.2f}% (threshold {t_pct:.1f}%)",
            data_age_days=_data_age(ccc) if is_live else 0,
        )

    # ── 3. Tail divergence: CCC − HY widening (30 d) ─────────────────────
    if len(hy) >= 10 and len(ccc) >= 10:
        hy_now  = float(hy.iloc[-1])
        ccc_now = float(ccc.iloc[-1])
        hy_then  = _nearest_before(hy,  hy.index[-1]  - pd.Timedelta(days=jump))
        ccc_then = _nearest_before(ccc, ccc.index[-1] - pd.Timedelta(days=jump))
        hy_then  = hy_then  if hy_then  is not None else float(hy.iloc[0])
        ccc_then = ccc_then if ccc_then is not None else float(ccc.iloc[0])

        spread_delta = _bps((ccc_now - hy_now) - (ccc_then - hy_then))
        t_bps = float(T["tail_divergence_30d_bps"])

        sigs["TAIL_DIVERGENCE"] = Signal(
            key="TAIL_DIVERGENCE",
            name="Tail divergence (CCC−HY widening, 30 d)",
            triggered=spread_delta >= t_bps,
            severity=_clamp(spread_delta / (t_bps * 2.0)),
            value=f"{spread_delta:.0f} bps",
            detail=(f"CCC−HY spread widened {spread_delta:+.0f} bps over 30 d "
                    f"(threshold {t_bps:.0f})"),
        )

    # ── 4. UK mortgage approvals (3 m) ───────────────────────────────────
    uk_ = _s(data.uk_approvals)
    if len(uk_) >= 3:
        uk_now = float(uk_.iloc[-1])
        uk_then = _nearest_before(
            uk_, uk_.index[-1] - pd.DateOffset(months=int(L["uk_compare_months"]))
        )
        uk_then = uk_then if uk_then is not None else float(uk_.iloc[0])
        chg = _pct_change(uk_now, uk_then)

        t_pct = float(T["uk_approvals_drop_3m_pct"])
        triggered = chg is not None and chg <= t_pct
        sev = 0.0 if chg is None else _clamp(
            abs(min(0.0, chg)) / abs(t_pct) / 2.0
        )
        sigs["UK_APPROVALS_DOWN"] = Signal(
            key="UK_APPROVALS_DOWN",
            name="UK mortgage approvals down (3 m)",
            triggered=triggered, severity=sev,
            value=f"{chg:.1f}%" if chg is not None else "n/a",
            detail=(f"UK approvals {uk_now:,.0f} vs 3 m ago {uk_then:,.0f} "
                    f"({chg:+.1f}%, threshold {t_pct:.1f}%)")
            if chg is not None else "Insufficient data",
            data_age_days=_data_age(uk_) if is_live else 0,
        )

    # ── 5. GBP/USD weakness (30 d) ───────────────────────────────────────
    fx_ = _s(data.gbpusd)
    if len(fx_) >= 10:
        fx_now = float(fx_.iloc[-1])
        fx_then = _nearest_before(fx_, fx_.index[-1] - pd.Timedelta(days=jump))
        fx_then = fx_then if fx_then is not None else float(fx_.iloc[0])
        chg = _pct_change(fx_now, fx_then)

        t_pct = float(T["gbpusd_drop_30d_pct"])
        triggered = chg is not None and chg <= t_pct
        sev = 0.0 if chg is None else _clamp(
            abs(min(0.0, chg)) / abs(t_pct) / 2.0
        )
        sigs["GBP_WEAK"] = Signal(
            key="GBP_WEAK",
            name="GBP/USD down (30 d)",
            triggered=triggered, severity=sev,
            value=f"{chg:.1f}%" if chg is not None else "n/a",
            detail=(f"GBP/USD {fx_now:.4f} vs 30 d ago {fx_then:.4f} "
                    f"({chg:+.1f}%, threshold {t_pct:.1f}%)")
            if chg is not None else "Insufficient data",
        )

    # ── 6. Homogeneity spike (correlation clustering) ─────────────────────
    hom = _sdf(data.homogeneity_df)
    n_spikes, spike_pairs, corr = _corr_spikes(
        hom, window=int(L["corr_window_days"]),
        threshold=float(T["corr_spike_level_abs"]),
    )
    t_count = int(T["corr_spike_count"])
    detail_str = ("Insufficient data for correlation analysis"
                  if corr is None
                  else (f"{n_spikes} correlated pairs (threshold {t_count}). "
                        + ", ".join(spike_pairs[:6])
                        + ("…" if len(spike_pairs) > 6 else "")))

    sigs["HOMOGENEITY_SPIKE"] = Signal(
        key="HOMOGENEITY_SPIKE",
        name="Homogeneity spike (correlation clustering)",
        triggered=n_spikes >= t_count,
        severity=_clamp(n_spikes / max(1.0, t_count * 2.0)),
        value=f"{n_spikes} pair-spikes",
        detail=detail_str,
    )

    return sigs


# ═════════════════════════════════════════════════════════════════════════════
#  SCORING
# ═════════════════════════════════════════════════════════════════════════════

def _tripwire_score(cfg: Dict[str, Any], signals: Dict[str, Signal]) -> int:
    """Weighted severity → 0–100 integer.  O(k)."""
    w = cfg["weights"]
    total = sum(float(w.get(k, 0)) * s.severity for k, s in signals.items())
    return int(round(_clamp(total, 0.0, 100.0)))


# ═════════════════════════════════════════════════════════════════════════════
#  BACKTEST ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def _backtest(
    cfg: Dict[str, Any],
    data: MarketData,
    start: dt.date,
    end: dt.date,
    step_days: int = 1,
) -> pd.DataFrame:
    """Score every date in [start, end].

    Uses only data available at each evaluation date — no lookahead.

    Complexity
    ----------
    O(D × k × log n)  where D = evaluation dates, k = signals,
    n = series length.  The log n comes from binary search in
    _nearest_before (vs the O(n) slice-per-day in the original
    which was O(D × k × n) ≈ 2000x slower for multi-year backtests).
    """
    dates = pd.date_range(pd.Timestamp(start), pd.Timestamp(end),
                          freq=f"{step_days}D")
    rows: List[Dict[str, Any]] = []

    for i, t in enumerate(dates):
        if i % 200 == 0:
            log.info("Backtest %s  (%d / %d)", t.date(), i, len(dates))

        sigs = compute_signals(cfg, data, as_of=t)
        if len(sigs) < 3:
            continue

        score = _tripwire_score(cfg, sigs)
        regime = _band_from_score(score, cfg["bands"])
        triggered = [k for k, s in sigs.items() if s.triggered]

        rows.append({
            "date": t.date().isoformat(),
            "score": score,
            "regime": regime,
            "triggered_count": len(triggered),
            "triggered": "|".join(triggered),
        })

    return pd.DataFrame(rows)


# ═════════════════════════════════════════════════════════════════════════════
#  DASHBOARD HTML
# ═════════════════════════════════════════════════════════════════════════════

_DASHBOARD = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>DeleverageSentinel</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Arial;
     background:#f8fafc;color:#1e293b;padding:24px}
.card{background:#fff;border:1px solid #e2e8f0;border-radius:16px;
      padding:24px;max-width:960px;margin:0 auto;
      box-shadow:0 1px 3px rgba(0,0,0,.06)}
.hdr{display:flex;justify-content:space-between;align-items:center;
     flex-wrap:wrap;gap:12px}
.ttl{font-size:13px;font-weight:600;letter-spacing:.5px;
     text-transform:uppercase;color:#64748b}
.sr{display:flex;align-items:baseline;gap:16px;margin:16px 0}
.sc{font-size:56px;font-weight:800;line-height:1}
.rg{display:inline-block;padding:6px 14px;border-radius:999px;
    font-size:13px;font-weight:600;letter-spacing:.5px;text-transform:uppercase}
.GREEN{background:#dcfce7;color:#166534}
.AMBER{background:#fef3c7;color:#92400e}
.RED{background:#fee2e2;color:#991b1b}
.BLACK{background:#1e293b;color:#f8fafc}
.sig{padding:14px 16px;border-radius:12px;border:1px solid #e2e8f0;margin-top:10px}
.sig.ok{background:#f8fafc}.sig.tr{background:#fffbeb;border-color:#fbbf24}
.sig h3{font-size:15px;font-weight:600;margin-bottom:4px}
.sig p{font-size:13px;color:#64748b}
.pl{display:inline-block;padding:3px 8px;border-radius:6px;
    font-size:11px;background:#f1f5f9;color:#475569;margin-left:6px}
.wn{margin-top:16px;padding:12px;background:#fffbeb;border:1px solid #fde68a;
    border-radius:10px;font-size:13px;color:#92400e}
.ft{margin-top:16px;font-size:11px;color:#94a3b8}
</style>
</head>
<body>
<div class="card">
  <div class="hdr">
    <div><div class="ttl">DeleverageSentinel</div>
         <div id="dt" style="font-size:13px;color:#94a3b8">Loading…</div></div>
    <div id="rb"></div>
  </div>
  <div class="sr"><div class="sc" id="sc">--</div>
    <div style="color:#64748b;font-size:14px;padding-top:8px">/ 100 Tripwire Score</div></div>
  <div id="sg"></div>
  <div id="wn"></div>
  <div class="ft">Schema v<span id="sv">–</span> ·
    <a href="https://github.com/FishIntelGlobal/deleverage-sentinel">Source</a> ·
    Not financial advice</div>
</div>
<script>
async function go(){
  const r=await fetch('latest.json',{cache:'no-store'}),d=await r.json();
  document.getElementById('dt').textContent='As of '+d.as_of;
  document.getElementById('sc').textContent=d.score;
  document.getElementById('sv').textContent=d.schema_version||'?';
  document.getElementById('rb').innerHTML=`<span class="rg ${d.regime}">${d.regime}</span>`;
  const root=document.getElementById('sg');root.innerHTML='';
  for(const[k,s]of Object.entries(d.signals)){
    const div=document.createElement('div');div.className='sig '+(s.triggered?'tr':'ok');
    const age=s.data_age_days>3?`<span class="pl">⚠ ${s.data_age_days}d stale</span>`:'';
    div.innerHTML=`<h3>${s.name}<span class="pl">${s.value}</span>`
      +`<span class="pl">sev ${(s.severity*100).toFixed(0)}%</span>${age}</h3>`
      +`<p>${s.detail}</p>`;root.appendChild(div);}
  const wr=document.getElementById('wn');
  if(d.data_warnings&&d.data_warnings.length)
    wr.innerHTML=d.data_warnings.map(w=>`<div class="wn">⚠ ${w}</div>`).join('');
}
go().catch(()=>{document.getElementById('dt').textContent='Could not load latest.json';});
</script>
</body>
</html>"""


# ═════════════════════════════════════════════════════════════════════════════
#  ALERTS
# ═════════════════════════════════════════════════════════════════════════════

def _send_telegram(token: str, chat_id: str, text: str) -> bool:
    """Send a Telegram message.  Returns True on success."""
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": text,
                  "disable_web_page_preview": True},
            timeout=15)
        resp.raise_for_status()
        log.info("Telegram alert sent")
        return True
    except Exception as exc:
        log.error("Telegram alert failed: %s", exc)
        return False


def _send_email(
    host: str, port: int, user: str, pw: str,
    to_addr: str, subject: str, body: str,
) -> bool:
    """Send an SMTP/TLS email alert."""
    try:
        import smtplib
        from email.mime.text import MIMEText
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"], msg["From"], msg["To"] = subject, user, to_addr
        with smtplib.SMTP(host, port, timeout=20) as s:
            s.starttls(); s.login(user, pw)
            s.sendmail(user, [to_addr], msg.as_string())
        log.info("Email alert sent to %s", to_addr)
        return True
    except Exception as exc:
        log.error("Email alert failed: %s", exc)
        return False


def _fire_alerts(cfg: Dict[str, Any], result: SentinelResult) -> None:
    """Dispatch alerts if the regime warrants it."""
    acfg = cfg["alerting"]
    if result.regime not in acfg.get("alert_on_regimes", []):
        return
    lines = [f"⚠ DeleverageSentinel: {result.regime} ({result.score}/100)",
             f"As of: {result.as_of}", ""]
    for s in result.signals.values():
        if s.triggered:
            lines.append(f"• {s.name}: {s.value} (sev {s.severity*100:.0f}%)")
    if result.data_warnings:
        lines += ["", "Data warnings:"]
        for w in result.data_warnings:
            lines.append(f"  ⚠ {w}")
    msg = "\n".join(lines)

    tg = acfg.get("telegram", {})
    if tg.get("enabled"):
        tok = os.getenv(tg.get("token_env", ""), "")
        cid = os.getenv(tg.get("chat_id_env", ""), "")
        if tok and cid:
            _send_telegram(tok, cid, msg)

    em = acfg.get("email", {})
    if em.get("enabled"):
        _send_email(
            host=os.getenv(em.get("host_env", ""), ""),
            port=int(os.getenv(em.get("port_env", ""), "587")),
            user=os.getenv(em.get("user_env", ""), ""),
            pw=os.getenv(em.get("pass_env", ""), ""),
            to_addr=os.getenv(em.get("to_env", ""), ""),
            subject=f"DeleverageSentinel {result.regime} ({result.score}/100)",
            body=msg)


# ═════════════════════════════════════════════════════════════════════════════
#  OUTPUT WRITERS
# ═════════════════════════════════════════════════════════════════════════════

def _write_json(result: SentinelResult, path: str) -> None:
    """Write schema-versioned JSON snapshot."""
    _ensure_dir(path)
    out = {
        "as_of": result.as_of,
        "schema_version": result.schema_version,
        "score": result.score,
        "regime": result.regime,
        "data_warnings": result.data_warnings,
        "signals": {
            k: {"name": s.name, "triggered": s.triggered,
                "severity": round(s.severity, 4), "value": s.value,
                "detail": s.detail, "data_age_days": s.data_age_days}
            for k, s in result.signals.items()
        },
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    log.info("Wrote %s", path)


def _write_dashboard(path: str) -> None:
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_DASHBOARD)
    log.info("Wrote %s", path)


# ═════════════════════════════════════════════════════════════════════════════
#  CLI COMMANDS
# ═════════════════════════════════════════════════════════════════════════════

def cmd_run(cfg: Dict[str, Any], offline: bool) -> int:
    """Execute a single live scoring run."""
    now = _today_utc()
    start = now - dt.timedelta(days=int(cfg["lookback"]["fetch_days"]))
    data = _fetch_all(cfg, offline, start)
    signals = compute_signals(cfg, data)
    score = _tripwire_score(cfg, signals)
    regime = _band_from_score(score, cfg["bands"])

    result = SentinelResult(
        as_of=now.isoformat(), schema_version=SCHEMA_VERSION,
        score=score, regime=regime, signals=signals,
        data_warnings=data.warnings,
    )

    oc = cfg["output"]
    _write_json(result, oc["latest_json"])
    if oc.get("write_dashboard", True):
        _write_dashboard(oc["dashboard_html"])

    # terminal display
    W = 72
    print(f"\n{'═' * W}")
    print(f"  DeleverageSentinel │ {result.as_of} │ "
          f"Score {score}/100 │ {regime}")
    print(f"{'═' * W}")
    for s in signals.values():
        flag = "▲ TRIP" if s.triggered else "  ok  "
        print(f"  [{flag}] {s.name:44s} {s.value:>12s}  "
              f"sev {s.severity * 100:>4.0f}%")
    if data.warnings:
        print(f"\n  Warnings:")
        for w in data.warnings:
            print(f"    ⚠ {w}")
    print(f"{'═' * W}\n")

    _fire_alerts(cfg, result)
    return 2 if regime in ("RED", "BLACK") else 0


def cmd_backtest(
    cfg: Dict[str, Any], offline: bool,
    start: dt.date, end: dt.date,
) -> int:
    """Run historical backtest and write results."""
    buf = int(cfg["lookback"]["fetch_days"])
    data = _fetch_all(cfg, offline, start - dt.timedelta(days=buf))
    hist = _backtest(cfg, data, start, end)

    path = cfg["output"]["history_csv"]
    _ensure_dir(path)
    hist.to_csv(path, index=False)

    print(f"\nBacktest: {start} → {end}  ({len(hist)} evaluation days)")
    if not hist.empty:
        print(f"\nRegime distribution:")
        for regime, cnt in hist["regime"].value_counts().items():
            print(f"  {regime:8s}  {cnt:>5d} days  ({cnt/len(hist)*100:.1f}%)")
        print(f"\nTop 10 highest-stress days:")
        print(hist.sort_values("score", ascending=False)
              .head(10).to_string(index=False))
    print(f"\nWrote: {path}")
    return 0


def cmd_validate(cfg: Dict[str, Any]) -> int:
    """Validate config without API calls."""
    w = cfg["weights"]
    total = sum(float(v) for v in w.values())
    print(f"Config validated successfully.")
    print(f"  Thresholds : {len(cfg['thresholds'])}")
    print(f"  Weights    : {w}")
    print(f"  Weight sum : {total:.0f} / 100")
    print(f"  Bands      : {cfg['bands']}")
    fs = cfg.get("feeds", {}).get("fred", {}).get("series", {})
    gbp_status = f"FRED {fs['GBPUSD']} ✓" if "GBPUSD" in fs else "⚠ Missing — add GBPUSD: DEXUSUK"
    print(f"  GBP/USD    : {gbp_status}")
    return 0


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="deleverage_sentinel",
        description="Institutional-grade deleveraging early warning system",
        epilog="Jason Gething / FishIntel Global · MIT License",
    )
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Debug-level logging")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Compute latest tripwire score")
    r.add_argument("--config", required=True)
    r.add_argument("--offline", action="store_true",
                   help="Use local CSVs (no API calls)")

    b = sub.add_parser("backtest", help="Score through historical dates")
    b.add_argument("--config", required=True)
    b.add_argument("--offline", action="store_true")
    b.add_argument("--start", required=True, help="YYYY-MM-DD")
    b.add_argument("--end", required=True, help="YYYY-MM-DD")

    v = sub.add_parser("validate", help="Validate config (no API calls)")
    v.add_argument("--config", required=True)

    return p.parse_args()


def main() -> int:
    args = _parse_args()
    _configure_logging(verbose=getattr(args, "verbose", False))
    log.info("DeleverageSentinel v%s", __version__)
    cfg = load_config(args.config)

    if args.cmd == "run":
        return cmd_run(cfg, offline=args.offline)
    elif args.cmd == "backtest":
        return cmd_backtest(
            cfg, offline=args.offline,
            start=dt.date.fromisoformat(args.start),
            end=dt.date.fromisoformat(args.end))
    elif args.cmd == "validate":
        return cmd_validate(cfg)
    else:
        raise SystemExit(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except ConfigError as exc:
        print(f"\nCONFIG ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except FetchError as exc:
        print(f"\nFETCH ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        logging.getLogger("sentinel").exception("Unhandled error")
        print(f"\nFATAL: {exc}", file=sys.stderr)
        sys.exit(1)
