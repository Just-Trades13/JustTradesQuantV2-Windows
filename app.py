# ═══════════════════════════════════════════════════════════════════════════════
# MOTIONCIRCLE — QUANT STRATEGY ANALYTICS ENGINE - JUST TRADES CONFIDENTIAL PROPERTY
# ═══════════════════════════════════════════════════════════════════════════════
# Run with:  streamlit run app.py
# Install:   pip install streamlit pandas numpy plotly scipy
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import time
from scipy.stats import (
    skew, kurtosis, norm, jarque_bera, shapiro, kstest,
    anderson, ttest_1samp,
)
from scipy.special import comb as _scipy_comb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import io
import re
import base64
import os
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable

warnings.filterwarnings("ignore")

from pine_optimizer import PineOptimizer, InputParam, OptimizationResult
from pine_interpreter import PineInterpreter as ExtPineInterpreter, lexer as ext_lexer, Parser as ExtParser

# ── Load .env file if present ────────────────────────────────────────────────
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _key, _val = _line.split('=', 1)
                os.environ.setdefault(_key.strip(), _val.strip())

# ═══════════════════════════════════════════════════════════════════════════════
# BRANDING
# ═══════════════════════════════════════════════════════════════════════════════

BRAND = {
    "name": "Just Trades",
    "tagline": "Hedge Fund–Grade Strategy Analytics",
    "teal": "#19b8ff",
    "teal_light": "#19b8ff",
    "teal_dark": "#0c6bbd",
    "dark_bg": "#0b1221",
    "card_bg": "rgba(11,18,33,0.6)",
}

def _load_logo_b64():
    """Load logo as base64 for inline HTML rendering."""
    logo_paths = ["just_trades_logo.png", "logo.png",
                  os.path.join(os.path.dirname(__file__), "just_trades_logo.png")]
    for p in logo_paths:
        if os.path.exists(p):
            with open(p, "rb") as f:
                return base64.b64encode(f.read()).decode()
    return None

LOGO_B64 = _load_logo_b64()

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Just Trades Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── MotionCircle Dark Theme CSS ────────────────────────────────────────────────
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Poppins:wght@300;400;500;600;700&display=swap');

    /* ── Hide Streamlit branding ── */
    #MainMenu {{visibility: hidden;}}
    header {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    [data-testid="stToolbar"] {{display: none;}}
    [data-testid="stDecoration"] {{display: none;}}
    .stDeployButton {{display: none;}}
    div[data-testid="stStatusWidget"] {{display: none;}}

    .stApp {{
        font-family: 'Poppins', sans-serif;
        background-color: #0b1221;
    }}
    h1, h2, h3, h4 {{
        font-family: 'JetBrains Mono', monospace !important;
    }}

    /* ── Metric cards with JT blue accent ── */
    div[data-testid="stMetric"] {{
        background: linear-gradient(135deg, rgba(25,184,255,0.08), rgba(11,18,33,0.4));
        border: 1px solid rgba(25,184,255,0.12);
        border-radius: 10px;
        padding: 14px 18px;
        transition: border-color 0.2s ease;
    }}
    div[data-testid="stMetric"]:hover {{
        border-color: rgba(25,184,255,0.3);
    }}
    div[data-testid="stMetric"] label {{
        font-size: 0.72rem !important;
        opacity: 0.6;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.35rem !important;
    }}

    /* ── Sidebar branding ── */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0f172a 0%, #020617 100%);
        border-right: 1px solid rgba(25,184,255,0.1);
    }}

    .block-container {{
        padding-top: 1rem;
    }}

    /* ── Expander styling ── */
    div[data-testid="stExpander"] {{
        border: 1px solid rgba(25,184,255,0.08);
        border-radius: 10px;
    }}

    /* ── Tab styling ── */
    button[data-baseweb="tab"] {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.78rem !important;
    }}

    /* ── MotionCircle branded header ── */
    .jt-header {{
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 8px 0 16px 0;
    }}
    .jt-header img {{
        width: 52px;
        height: 52px;
        border-radius: 12px;
        border: 2px solid rgba(25,184,255,0.3);
    }}
    .jt-header-text h1 {{
        margin: 0;
        font-size: 1.75rem;
        background: linear-gradient(135deg, {BRAND['teal_light']}, {BRAND['teal']});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.2;
    }}
    .jt-header-text p {{
        margin: 2px 0 0 0;
        font-size: 0.82rem;
        opacity: 0.5;
        letter-spacing: 0.5px;
    }}

    /* ── Footer ── */
    .jt-footer {{
        text-align: center;
        padding: 32px 0 16px 0;
        opacity: 0.35;
        font-size: 0.72rem;
        letter-spacing: 1px;
        border-top: 1px solid rgba(25,184,255,0.08);
        margin-top: 40px;
    }}
</style>
""", unsafe_allow_html=True)

# ── Plotly template ──────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="JetBrains Mono, monospace", size=11),
    margin=dict(l=50, r=30, t=40, b=40),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.08)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zerolinecolor="rgba(255,255,255,0.08)"),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)

COLORS = {
    "equity": "#19b8ff",     # JT blue
    "median": "#a78bfa",
    "band":   "#0c6bbd",     # JT dark blue
    "win":    "#00ff88",
    "loss":   "#ff4466",
    "neutral":"#94a3b8",
    "accent": "#2dd4bf",
    "accent2":"#ec4899",
    "teal":   "#19b8ff",
}


# ═══════════════════════════════════════════════════════════════════════════════
# TRADINGVIEW CSV PARSER — AUTO-DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def parse_tradingview_csv(df_raw):
    """
    Intelligently parse TradingView strategy report CSVs.
    Handles the actual TV export format including:
    - Entry/Exit row pairs (collapses into single trade rows)
    - BOM characters
    - All known TV column naming conventions
    - Currency suffixes (USD, EUR, etc.)
    """
    df = df_raw.copy()

    # Strip BOM and whitespace from column names
    df.columns = df.columns.str.replace('\ufeff', '').str.strip()

    # ── Detect Entry/Exit row format ──────────────────────────────────────
    # TradingView exports each trade as 2 rows: "Entry long" + "Exit long"
    # We need to collapse these into single trade rows
    type_col = None
    lower_cols = {c.lower().strip(): c for c in df.columns}
    for alias in ['type', 'side', 'direction', 'trade type', 'action']:
        if alias in lower_cols:
            type_col = lower_cols[alias]
            break

    if type_col and df[type_col].astype(str).str.lower().str.contains('entry|exit').any():
        df = _collapse_entry_exit_rows(df, type_col, lower_cols)

    # ── Column mapping with comprehensive aliases ─────────────────────────
    col_map = {}
    # Rebuild lower_cols after potential column changes
    lower_cols = {c.lower().strip(): c for c in df.columns}

    # Also build a partial matching system for multi-word columns
    def _find(aliases):
        # Exact match first
        for a in aliases:
            if a in lower_cols:
                return lower_cols[a]
        # Substring/contains match as fallback
        for a in aliases:
            for lc, orig in lower_cols.items():
                if a in lc:
                    return orig
        return None

    profit_aliases = [
        "net p&l usd", "net p&l", "net pl", "net profit", "profit",
        "profit/loss", "p&l", "pnl", "p/l", "trade profit",
        "profit usd", "realized p&l", "net p&l eur", "net p&l gbp",
    ]
    cum_profit_aliases = [
        "cumulative p&l usd", "cumulative p&l", "cum. profit", "cumulative profit",
        "cum profit", "cumprofit", "running profit", "cumulative p&l eur",
    ]
    date_aliases = [
        "date and time", "date/time", "datetime", "date", "time",
        "entry time", "entry date", "open time", "open date",
        "entry date/time", "entry date and time",
    ]
    exit_date_aliases = [
        "exit date/time", "exit date", "exit time", "close time",
        "close date", "exit date and time",
    ]
    type_aliases = ["type", "side", "direction", "trade type", "action"]
    price_aliases = [
        "price usd", "price", "entry price", "open price", "avg price",
        "price eur", "price gbp",
    ]
    exit_price_aliases = ["exit price", "close price", "exit price usd"]
    contracts_aliases = [
        "position size (qty)", "contracts", "qty", "quantity", "size",
        "shares", "position size", "lots", "amount",
    ]
    runup_aliases = [
        "favorable excursion usd", "favorable excursion", "run-up",
        "runup", "run up", "max favorable", "mfe",
        "favorable excursion eur", "favorable excursion gbp",
    ]
    drawdown_aliases = [
        "adverse excursion usd", "adverse excursion", "drawdown",
        "draw-down", "draw down", "max adverse", "mae",
        "adverse excursion eur", "adverse excursion gbp",
    ]
    signal_aliases = ["signal", "entry signal", "strategy", "entry reason"]
    trade_num_aliases = ["trade #", "trade no", "trade number", "#", "no"]
    commission_aliases = ["commission", "fees", "comm", "trading fees"]
    duration_aliases = ["duration", "holding time", "hold time", "bars"]

    col_map["profit"]      = _find(profit_aliases)
    col_map["cum_profit"]   = _find(cum_profit_aliases)
    col_map["entry_date"]   = _find(date_aliases)
    col_map["exit_date"]    = _find(exit_date_aliases)
    col_map["type"]         = _find(type_aliases)
    col_map["entry_price"]  = _find(price_aliases)
    col_map["exit_price"]   = _find(exit_price_aliases)
    col_map["contracts"]    = _find(contracts_aliases)
    col_map["runup"]        = _find(runup_aliases)
    col_map["drawdown_col"] = _find(drawdown_aliases)
    col_map["signal"]       = _find(signal_aliases)
    col_map["trade_num"]    = _find(trade_num_aliases)
    col_map["commission"]   = _find(commission_aliases)
    col_map["duration"]     = _find(duration_aliases)

    return df, col_map


def _collapse_entry_exit_rows(df, type_col, lower_cols):
    """
    TradingView exports Entry and Exit as separate rows.
    Collapse them into single trade rows with entry/exit prices and dates.
    """
    # Find trade number column
    trade_num_col = None
    for alias in ['trade #', 'trade no', 'trade number', '#']:
        if alias in lower_cols:
            trade_num_col = lower_cols[alias]
            break

    if trade_num_col is None:
        return df  # Can't collapse without trade numbers

    # Find date and price columns
    date_col = None
    for alias in ['date and time', 'date/time', 'datetime', 'date', 'time']:
        if alias in lower_cols:
            date_col = lower_cols[alias]
            break

    price_col = None
    for alias in ['price usd', 'price', 'entry price', 'price eur', 'price gbp']:
        if alias in lower_cols:
            price_col = lower_cols[alias]
            break

    trades = []
    type_lower = df[type_col].astype(str).str.lower().str.strip()

    # Group by trade number
    for trade_num in df[trade_num_col].unique():
        group = df[df[trade_num_col] == trade_num]
        type_vals = group[type_col].astype(str).str.lower().str.strip()

        entry_row = group[type_vals.str.contains('entry')].iloc[0] if type_vals.str.contains('entry').any() else None
        exit_row = group[type_vals.str.contains('exit')].iloc[0] if type_vals.str.contains('exit').any() else None

        if entry_row is None and exit_row is None:
            continue

        # Use whichever row has the P&L data (usually both have same values)
        base_row = entry_row if entry_row is not None else exit_row

        trade = base_row.to_dict()

        # Determine direction from entry type
        if entry_row is not None:
            entry_type = str(entry_row[type_col]).lower()
            if 'long' in entry_type:
                trade[type_col] = 'Long'
            elif 'short' in entry_type:
                trade[type_col] = 'Short'

        # Set entry/exit dates and prices
        if entry_row is not None and date_col:
            trade['_entry_date'] = entry_row[date_col]
        if exit_row is not None and date_col:
            trade['_exit_date'] = exit_row[date_col]
        if entry_row is not None and price_col:
            trade['_entry_price'] = entry_row[price_col]
        if exit_row is not None and price_col:
            trade['_exit_price'] = exit_row[price_col]

        trades.append(trade)

    if not trades:
        return df

    result = pd.DataFrame(trades)

    # Add the computed entry/exit columns
    if '_entry_date' in result.columns:
        result['Entry Date'] = result['_entry_date']
    if '_exit_date' in result.columns:
        result['Exit Date'] = result['_exit_date']
    if '_entry_price' in result.columns:
        result['Entry Price'] = result['_entry_price']
    if '_exit_price' in result.columns:
        result['Exit Price'] = result['_exit_price']

    # Clean up temp columns
    for col in ['_entry_date', '_exit_date', '_entry_price', '_exit_price']:
        if col in result.columns:
            result.drop(columns=[col], inplace=True)

    return result


def clean_numeric(series):
    """Strip currency symbols, commas, %, and convert to float.
    Handles negatives in parentheses like (500) → -500."""
    dtype_name = str(series.dtype)
    is_str_type = dtype_name in ("object", "string", "str", "string[python]", "string[pyarrow]")
    if is_str_type or not np.issubdtype(series.dtype, np.number):
        s = series.astype(str).str.strip()
        # Detect parenthesized negatives: (123.45) → -123.45
        is_paren_neg = s.str.match(r"^\(.*\)$").fillna(False)
        s = s.str.replace(r"[$€£¥₹,]", "", regex=True)
        s = s.str.replace(r"%", "", regex=True)
        s = s.str.replace(r"[()]", "", regex=True)
        s = s.str.strip()
        result = pd.to_numeric(s, errors="coerce")
        result = result.where(~is_paren_neg, -result.abs())
        return result
    return pd.to_numeric(series, errors="coerce")


def parse_tv_date(series):
    """Try multiple datetime formats common in TradingView exports."""
    for fmt in [None, "%Y-%m-%d %H:%M", "%m/%d/%Y %H:%M", "%d/%m/%Y %H:%M",
                "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"]:
        try:
            return pd.to_datetime(series, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.to_datetime(series, errors="coerce")


# ═══════════════════════════════════════════════════════════════════════════════
# CORE ANALYTICS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class StrategyAnalytics:
    """
    Comprehensive quantitative analytics for a trade-level backtest.
    Computes 70+ metrics natively — no external quant library dependency.
    """

    ANNUAL_TRADING_DAYS = 252
    RISK_FREE_RATE = 0.05  # 5% default, overrideable

    def __init__(self, profits, initial_capital, dates=None, contracts=None,
                 entry_prices=None, exit_prices=None, trade_types=None,
                 runups=None, drawdowns_per_trade=None, commissions=None,
                 signals=None, risk_free_rate=0.05,
                 trades_per_year_override=None):
        self.raw_profits = np.array(profits, dtype=float)
        self.n_trades = len(self.raw_profits)
        self.initial_capital = float(initial_capital)
        self.RISK_FREE_RATE = risk_free_rate

        # Optional arrays
        self.dates = dates
        self.contracts = np.array(contracts, dtype=float) if contracts is not None else None
        self.entry_prices = np.array(entry_prices, dtype=float) if entry_prices is not None else None
        self.exit_prices = np.array(exit_prices, dtype=float) if exit_prices is not None else None
        self.trade_types = trade_types
        self.runups = np.array(runups, dtype=float) if runups is not None else None
        self.drawdowns_per_trade = np.array(drawdowns_per_trade, dtype=float) if drawdowns_per_trade is not None else None
        self.commissions = np.array(commissions, dtype=float) if commissions is not None else None
        self.signals = signals

        # ── Derived series ────────────────────────────────────────────────
        self.equity = np.cumsum(self.raw_profits) + self.initial_capital
        self.peak = np.maximum.accumulate(self.equity)
        self.drawdown = self.equity - self.peak
        self.drawdown_pct = np.where(self.peak > 0, self.drawdown / self.peak, 0.0)

        # Per-trade returns relative to equity BEFORE each trade
        equity_before = np.concatenate([[self.initial_capital], self.equity[:-1]])
        # Guard against division by zero if equity ever hits zero
        safe_eb = np.where(equity_before != 0, equity_before, np.nan)
        self.returns = np.nan_to_num(self.raw_profits / safe_eb, nan=0.0, posinf=0.0, neginf=0.0)
        # Guard log1p against returns <= -1 (total wipeout trades)
        clipped_returns = np.clip(self.returns, -0.9999, None)
        self.log_returns = np.log1p(clipped_returns)

        # Estimate annualization
        if trades_per_year_override:
            self.trades_per_year = trades_per_year_override
        elif self.dates is not None and len(self.dates) >= 2:
            valid_dates = self.dates.dropna()
            if len(valid_dates) >= 2:
                span_days = (valid_dates.iloc[-1] - valid_dates.iloc[0]).days
                if span_days > 0:
                    self.trades_per_year = self.n_trades / (span_days / 365.25)
                else:
                    self.trades_per_year = self.n_trades
            else:
                self.trades_per_year = self.n_trades
        else:
            self.trades_per_year = self.n_trades

        # Win / loss splits
        self.wins_mask = self.raw_profits > 0
        self.losses_mask = self.raw_profits < 0
        self.flat_mask = self.raw_profits == 0
        self.wins = self.raw_profits[self.wins_mask]
        self.losses = self.raw_profits[self.losses_mask]

    # ── Basic metrics ─────────────────────────────────────────────────────
    @property
    def total_profit(self):
        return self.raw_profits.sum()

    @property
    def total_return_pct(self):
        return (self.equity[-1] / self.initial_capital - 1) * 100

    @property
    def win_rate(self):
        return len(self.wins) / self.n_trades if self.n_trades > 0 else 0

    @property
    def loss_rate(self):
        return len(self.losses) / self.n_trades if self.n_trades > 0 else 0

    @property
    def avg_win(self):
        return self.wins.mean() if len(self.wins) > 0 else 0

    @property
    def avg_loss(self):
        return abs(self.losses.mean()) if len(self.losses) > 0 else 0

    @property
    def largest_win(self):
        return self.wins.max() if len(self.wins) > 0 else 0

    @property
    def largest_loss(self):
        return self.losses.min() if len(self.losses) > 0 else 0

    @property
    def median_win(self):
        return np.median(self.wins) if len(self.wins) > 0 else 0

    @property
    def median_loss(self):
        return np.median(self.losses) if len(self.losses) > 0 else 0

    @property
    def avg_trade(self):
        return self.raw_profits.mean()

    @property
    def median_trade(self):
        return np.median(self.raw_profits)

    @property
    def std_trade(self):
        return self.raw_profits.std(ddof=1) if self.n_trades > 1 else 0

    # ── Ratios & Risk Metrics ─────────────────────────────────────────────
    @property
    def profit_factor(self):
        if len(self.losses) == 0:
            return np.inf
        return self.wins.sum() / abs(self.losses.sum())

    @property
    def payoff_ratio(self):
        """Average win / average loss."""
        return self.avg_win / self.avg_loss if self.avg_loss > 0 else np.inf

    @property
    def expectancy(self):
        """Expected $ per trade."""
        return (self.win_rate * self.avg_win) - (self.loss_rate * self.avg_loss)

    @property
    def expectancy_pct(self):
        """Expectancy as % of average trade size."""
        avg_size = (self.avg_win + self.avg_loss) / 2
        return (self.expectancy / avg_size * 100) if avg_size > 0 else 0

    @property
    def kelly_criterion(self):
        """Full Kelly fraction: W - (1-W)/R."""
        R = self.payoff_ratio
        W = self.win_rate
        if R <= 0 or R == np.inf:
            return 0
        return W - (1 - W) / R

    @property
    def half_kelly(self):
        return self.kelly_criterion / 2

    @property
    def max_drawdown(self):
        return self.drawdown.min()

    @property
    def max_drawdown_pct(self):
        return self.drawdown_pct.min() * 100

    @property
    def avg_drawdown(self):
        dd_neg = self.drawdown[self.drawdown < 0]
        return dd_neg.mean() if len(dd_neg) > 0 else 0

    @property
    def avg_drawdown_pct(self):
        dd_neg = self.drawdown_pct[self.drawdown_pct < 0]
        return dd_neg.mean() * 100 if len(dd_neg) > 0 else 0

    def _drawdown_periods(self):
        """Identify individual drawdown periods (start, trough, recovery)."""
        periods = []
        in_dd = False
        start = 0
        for i in range(len(self.drawdown)):
            if self.drawdown[i] < 0 and not in_dd:
                in_dd = True
                start = i
            elif self.drawdown[i] >= 0 and in_dd:
                in_dd = False
                trough_idx = start + np.argmin(self.drawdown[start:i])
                periods.append({
                    "start": start,
                    "trough": trough_idx,
                    "recovery": i,
                    "depth": self.drawdown[trough_idx],
                    "depth_pct": self.drawdown_pct[trough_idx] * 100,
                    "length": i - start,
                })
        if in_dd:
            trough_idx = start + np.argmin(self.drawdown[start:])
            periods.append({
                "start": start,
                "trough": trough_idx,
                "recovery": None,
                "depth": self.drawdown[trough_idx],
                "depth_pct": self.drawdown_pct[trough_idx] * 100,
                "length": self.n_trades - start,
            })
        return periods

    @property
    def max_drawdown_duration(self):
        periods = self._drawdown_periods()
        return max((p["length"] for p in periods), default=0)

    @property
    def avg_drawdown_duration(self):
        periods = self._drawdown_periods()
        return np.mean([p["length"] for p in periods]) if periods else 0

    @property
    def calmar_ratio(self):
        if self.max_drawdown_pct == 0:
            return np.inf
        ann_ret = self._cagr()
        return ann_ret / abs(self.max_drawdown_pct)

    def _cagr(self):
        """Compound annual growth rate."""
        total_ret = self.equity[-1] / self.initial_capital
        years = self.n_trades / self.trades_per_year if self.trades_per_year > 0 else 1
        if years <= 0 or total_ret <= 0:
            return 0
        return (total_ret ** (1 / years) - 1) * 100

    @property
    def cagr(self):
        return self._cagr()

    @property
    def sharpe_ratio(self):
        """Annualized Sharpe ratio."""
        # If raw profit has zero variability, Sharpe is meaningless
        if self.std_trade < 1e-12:
            return 0
        std = self.returns.std(ddof=1) if self.n_trades > 1 else 0
        if not np.isfinite(std) or std < 1e-12:
            return 0
        excess = self.returns.mean() - self.RISK_FREE_RATE / self.trades_per_year
        return (excess / std) * np.sqrt(self.trades_per_year)

    @property
    def sortino_ratio(self):
        """Uses downside deviation (returns below target=0)."""
        downside = self.returns[self.returns < 0]
        if len(downside) == 0:
            return np.inf
        downside_std = np.sqrt(np.mean(downside ** 2))
        if downside_std == 0:
            return np.inf
        excess = self.returns.mean() - self.RISK_FREE_RATE / self.trades_per_year
        return (excess / downside_std) * np.sqrt(self.trades_per_year)

    @property
    def omega_ratio(self):
        """Omega ratio at threshold=0."""
        gains = self.returns[self.returns > 0].sum()
        losses_abs = abs(self.returns[self.returns < 0].sum())
        return gains / losses_abs if losses_abs > 0 else np.inf

    @property
    def tail_ratio(self):
        """95th percentile / abs(5th percentile) — measures tail asymmetry."""
        p95 = np.percentile(self.returns, 95)
        p5 = abs(np.percentile(self.returns, 5))
        return p95 / p5 if p5 > 0 else np.inf

    @property
    def common_sense_ratio(self):
        """Profit factor × tail ratio."""
        tr = self.tail_ratio
        pf = self.profit_factor
        if tr == np.inf or pf == np.inf:
            return np.inf
        return pf * tr

    @property
    def ulcer_index(self):
        """Measures depth & duration of drawdowns."""
        dd_pct_sq = self.drawdown_pct ** 2
        return np.sqrt(dd_pct_sq.mean()) * 100

    @property
    def ulcer_performance_index(self):
        """(CAGR - Rf) / Ulcer Index."""
        ui = self.ulcer_index
        return (self.cagr - self.RISK_FREE_RATE * 100) / ui if ui > 0 else np.inf

    @property
    def gain_to_pain_ratio(self):
        """Sum of returns / sum of absolute losses."""
        abs_neg = abs(self.returns[self.returns < 0].sum())
        return self.returns.sum() / abs_neg if abs_neg > 0 else np.inf

    @property
    def cpc_index(self):
        """CPC = profit factor × win rate × payoff ratio. > 1 is good."""
        return self.profit_factor * self.win_rate * self.payoff_ratio

    @property
    def recovery_factor(self):
        """Net profit / max drawdown."""
        if self.max_drawdown == 0:
            return np.inf
        return self.total_profit / abs(self.max_drawdown)

    @property
    def mar_ratio(self):
        """CAGR / Max Drawdown %."""
        if self.max_drawdown_pct == 0:
            return np.inf
        return self.cagr / abs(self.max_drawdown_pct)

    @property
    def sqn(self):
        """System Quality Number = sqrt(N) * expectancy / std."""
        if self.std_trade < 1e-12:
            return 0
        return np.sqrt(self.n_trades) * self.avg_trade / self.std_trade

    @property
    def t_stat_profit(self):
        """T-statistic: is mean trade profit != 0?"""
        if self.n_trades < 2 or self.std_trade < 1e-12:
            return 0
        return self.avg_trade / (self.std_trade / np.sqrt(self.n_trades))

    @property
    def p_value_profit(self):
        if self.n_trades < 2:
            return 1.0
        _, p = ttest_1samp(self.raw_profits, 0)
        return p

    # ── Streak analysis ───────────────────────────────────────────────────
    def _streaks(self):
        signs = np.sign(self.raw_profits)
        win_streaks, loss_streaks = [], []
        current, count = 0, 0
        for s in signs:
            if s == current:
                count += 1
            else:
                if current == 1:
                    win_streaks.append(count)
                elif current == -1:
                    loss_streaks.append(count)
                current, count = s, 1
        if current == 1:
            win_streaks.append(count)
        elif current == -1:
            loss_streaks.append(count)
        return win_streaks, loss_streaks

    @property
    def max_consecutive_wins(self):
        ws, _ = self._streaks()
        return max(ws) if ws else 0

    @property
    def max_consecutive_losses(self):
        _, ls = self._streaks()
        return max(ls) if ls else 0

    @property
    def avg_consecutive_wins(self):
        ws, _ = self._streaks()
        return np.mean(ws) if ws else 0

    @property
    def avg_consecutive_losses(self):
        _, ls = self._streaks()
        return np.mean(ls) if ls else 0

    # ── Distribution tests ────────────────────────────────────────────────
    @property
    def skewness(self):
        return skew(self.raw_profits)

    @property
    def excess_kurtosis(self):
        return kurtosis(self.raw_profits)  # scipy default = excess

    def normality_tests(self):
        results = {}
        if self.n_trades >= 8:
            stat, p = jarque_bera(self.raw_profits)
            results["Jarque-Bera"] = {"statistic": stat, "p_value": p}
        if 3 <= self.n_trades <= 5000:
            stat, p = shapiro(self.raw_profits)
            results["Shapiro-Wilk"] = {"statistic": stat, "p_value": p}
        if self.n_trades >= 5:
            stat, p = kstest(self.raw_profits, "norm",
                             args=(self.raw_profits.mean(), self.raw_profits.std()))
            results["Kolmogorov-Smirnov"] = {"statistic": stat, "p_value": p}
        if self.n_trades >= 3:
            try:
                result = anderson(self.raw_profits, dist="norm")
                results["Anderson-Darling"] = {
                    "statistic": result.statistic,
                    "critical_values": dict(zip(result.significance_level, result.critical_values)),
                }
            except Exception:
                pass
        return results

    # ── Monte Carlo ───────────────────────────────────────────────────────
    def monte_carlo(self, n_sims=2000, seed=42):
        """
        Bootstrap Monte Carlo with replacement — fully vectorized.
        Returns dict with equity paths and summary stats.
        """
        rng = np.random.default_rng(seed)

        # Vectorized: draw all indices at once (n_sims × n_trades matrix)
        idx = rng.choice(self.n_trades, size=(n_sims, self.n_trades), replace=True)
        sim_profits = self.raw_profits[idx]  # shape: (n_sims, n_trades)

        all_paths = np.cumsum(sim_profits, axis=1) + self.initial_capital
        final_equities = all_paths[:, -1]

        # Vectorized max drawdown per simulation
        peaks = np.maximum.accumulate(all_paths, axis=1)
        dd_pct = np.where(peaks > 0, (all_paths - peaks) / peaks, 0)
        max_dds = dd_pct.min(axis=1) * 100

        return {
            "paths": all_paths,
            "final_equities": final_equities,
            "max_drawdowns": max_dds,
            "percentiles": {
                "p1":  np.percentile(final_equities, 1),
                "p5":  np.percentile(final_equities, 5),
                "p10": np.percentile(final_equities, 10),
                "p25": np.percentile(final_equities, 25),
                "p50": np.percentile(final_equities, 50),
                "p75": np.percentile(final_equities, 75),
                "p90": np.percentile(final_equities, 90),
                "p95": np.percentile(final_equities, 95),
                "p99": np.percentile(final_equities, 99),
            },
            "dd_percentiles": {
                "p1":  np.percentile(max_dds, 1),
                "p5":  np.percentile(max_dds, 5),
                "p10": np.percentile(max_dds, 10),
                "p25": np.percentile(max_dds, 25),
                "p50": np.percentile(max_dds, 50),
                "p75": np.percentile(max_dds, 75),
                "p90": np.percentile(max_dds, 90),
                "p95": np.percentile(max_dds, 95),
                "p99": np.percentile(max_dds, 99),
            },
            "prob_profit": (final_equities > self.initial_capital).mean() * 100,
            "prob_2x":     (final_equities > 2 * self.initial_capital).mean() * 100,
            "prob_loss50":  (final_equities < 0.5 * self.initial_capital).mean() * 100,
            "prob_ruin":    (final_equities <= 0).mean() * 100,
        }

    def monte_carlo_block(self, n_sims=2000, block_size=10, seed=42):
        """
        Block bootstrap Monte Carlo — preserves trade clustering/autocorrelation.
        Resamples BLOCKS of consecutive trades instead of individual trades.
        Critical for DCA/averaging strategies where losses cluster.
        """
        rng = np.random.default_rng(seed)
        n = self.n_trades

        # Number of blocks needed to fill n trades
        n_blocks = int(np.ceil(n / block_size))

        all_paths = np.zeros((n_sims, n))
        for sim in range(n_sims):
            # Pick random starting positions for each block
            starts = rng.integers(0, n - block_size + 1, size=n_blocks)
            # Concatenate blocks and trim to exact trade count
            sim_trades = np.concatenate([self.raw_profits[s:s + block_size] for s in starts])[:n]
            all_paths[sim] = np.cumsum(sim_trades) + self.initial_capital

        final_equities = all_paths[:, -1]

        peaks = np.maximum.accumulate(all_paths, axis=1)
        dd_pct = np.where(peaks > 0, (all_paths - peaks) / peaks, 0)
        max_dds = dd_pct.min(axis=1) * 100

        return {
            "paths": all_paths,
            "final_equities": final_equities,
            "max_drawdowns": max_dds,
            "percentiles": {
                "p1":  np.percentile(final_equities, 1),
                "p5":  np.percentile(final_equities, 5),
                "p10": np.percentile(final_equities, 10),
                "p25": np.percentile(final_equities, 25),
                "p50": np.percentile(final_equities, 50),
                "p75": np.percentile(final_equities, 75),
                "p90": np.percentile(final_equities, 90),
                "p95": np.percentile(final_equities, 95),
                "p99": np.percentile(final_equities, 99),
            },
            "dd_percentiles": {
                "p1":  np.percentile(max_dds, 1),
                "p5":  np.percentile(max_dds, 5),
                "p10": np.percentile(max_dds, 10),
                "p25": np.percentile(max_dds, 25),
                "p50": np.percentile(max_dds, 50),
                "p75": np.percentile(max_dds, 75),
                "p90": np.percentile(max_dds, 90),
                "p95": np.percentile(max_dds, 95),
                "p99": np.percentile(max_dds, 99),
            },
            "prob_profit": (final_equities > self.initial_capital).mean() * 100,
            "prob_2x":     (final_equities > 2 * self.initial_capital).mean() * 100,
            "prob_loss50":  (final_equities < 0.5 * self.initial_capital).mean() * 100,
            "prob_ruin":    (final_equities <= 0).mean() * 100,
        }

    def trade_autocorrelation(self, max_lag=10):
        """Compute autocorrelation of trade PnL at various lags."""
        if self.n_trades < max_lag + 2:
            return {}
        mean = self.raw_profits.mean()
        var = np.sum((self.raw_profits - mean) ** 2)
        if var == 0:
            return {}
        autocorrs = {}
        for lag in range(1, max_lag + 1):
            c = np.sum((self.raw_profits[:-lag] - mean) * (self.raw_profits[lag:] - mean))
            autocorrs[lag] = c / var
        return autocorrs

    def var_cvar(self, confidence=0.95):
        """
        Value at Risk and Conditional VaR (Expected Shortfall).
        VaR = worst loss at the given confidence level.
        CVaR = average of all losses beyond VaR (the true tail risk).
        """
        sorted_profits = np.sort(self.raw_profits)
        idx = int((1 - confidence) * len(sorted_profits))
        var = sorted_profits[idx] if idx < len(sorted_profits) else sorted_profits[0]
        tail = sorted_profits[:max(idx, 1)]
        cvar = tail.mean()
        return {"VaR": var, "CVaR": cvar, "confidence": confidence}

    def tail_risk_metrics(self):
        """Comprehensive tail risk analysis."""
        var95 = self.var_cvar(0.95)
        var99 = self.var_cvar(0.99)

        # Estimate sigma events
        std = self.std_trade if self.std_trade > 0 else 1.0
        mean = self.avg_trade
        sigma3_loss = mean - 3 * std
        sigma5_loss = mean - 5 * std

        # Worst observed vs expected
        worst = self.largest_loss
        expected_worst_normal = mean - norm.ppf(1 - 1/self.n_trades) * std if self.n_trades > 0 else 0

        return {
            "var_95": var95,
            "var_99": var99,
            "sigma3_loss": sigma3_loss,
            "sigma5_loss": sigma5_loss,
            "worst_observed": worst,
            "worst_expected_normal": expected_worst_normal,
            "tail_severity": worst / sigma3_loss if sigma3_loss < 0 else 0,
        }

    # ── Risk of Ruin (analytical) ─────────────────────────────────────────
    def risk_of_ruin(self, ruin_level_pct=50):
        """
        Approximate risk of ruin using the formula:
        RoR ≈ ((1 - edge) / (1 + edge))^(capital_units)
        where edge = expectancy / avg_win
        """
        if self.avg_win <= 0:
            return 100.0
        edge = self.expectancy / self.avg_win
        if edge <= 0:
            return 100.0
        if edge >= 1:
            return 0.0
        capital_units = (self.initial_capital * (ruin_level_pct / 100)) / self.avg_trade if self.avg_trade > 0 else 0
        if capital_units <= 0:
            return 100.0
        base = (1 - edge) / (1 + edge)
        return min(max(base ** capital_units * 100, 0), 100)

    # ── Rolling metrics ───────────────────────────────────────────────────
    def rolling_metrics(self, window=20):
        """Compute rolling win rate, expectancy, Sharpe."""
        n = self.n_trades
        if n < window:
            return None
        r_winrate = np.full(n, np.nan)
        r_expectancy = np.full(n, np.nan)
        r_sharpe = np.full(n, np.nan)
        r_avg = np.full(n, np.nan)

        for i in range(window - 1, n):
            chunk = self.raw_profits[i - window + 1: i + 1]
            r_chunk = self.returns[i - window + 1: i + 1]
            w = chunk[chunk > 0]
            l = chunk[chunk < 0]
            wr = len(w) / window
            aw = w.mean() if len(w) > 0 else 0
            al = abs(l.mean()) if len(l) > 0 else 0
            r_winrate[i] = wr
            r_expectancy[i] = (wr * aw) - ((1 - wr) * al)
            r_avg[i] = chunk.mean()
            std = r_chunk.std(ddof=1)
            r_sharpe[i] = (r_chunk.mean() / std * np.sqrt(self.trades_per_year)) if std > 1e-12 else 0

        return {
            "win_rate": r_winrate,
            "expectancy": r_expectancy,
            "sharpe": r_sharpe,
            "avg_trade": r_avg,
        }

    # ── Time-based analysis ───────────────────────────────────────────────
    def time_analysis(self):
        """Group trades by day-of-week, month, hour, week, and half-hour if dates available."""
        if self.dates is None:
            return None
        df = pd.DataFrame({
            "date": self.dates,
            "profit": self.raw_profits,
            "return": self.returns,
        })
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        if df.empty:
            return None

        result = {}
        if not df.empty:
            def _win_rate(x):
                return (x > 0).mean()

            def _profit_factor(x):
                gross_win = x[x > 0].sum()
                gross_loss = abs(x[x < 0].sum())
                return gross_win / gross_loss if gross_loss > 0 else float('inf')

            def _largest_win(x):
                return x.max() if len(x) > 0 else 0

            def _largest_loss(x):
                return x.min() if len(x) > 0 else 0

            _agg = dict(
                total_profit="sum", avg_profit="mean", n_trades="count",
                win_rate=_win_rate, profit_factor=_profit_factor,
                largest_win=_largest_win, largest_loss=_largest_loss,
            )

            # Day of week
            df["dow"] = df["date"].dt.day_name()
            dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            g = df.groupby("dow")["profit"].agg(**_agg)
            g = g.reindex([d for d in dow_order if d in g.index])
            result["day_of_week"] = g

            # Month
            df["month"] = df["date"].dt.month_name()
            month_order = ["January", "February", "March", "April", "May", "June",
                           "July", "August", "September", "October", "November", "December"]
            gm = df.groupby("month")["profit"].agg(**_agg)
            gm = gm.reindex([m for m in month_order if m in gm.index])
            result["month"] = gm

            # Week number
            df["week_label"] = df["date"].dt.strftime("W%U (%b %d)")
            gw = df.groupby("week_label", sort=False)["profit"].agg(**_agg)
            result["week"] = gw

            # Hour
            if df["date"].dt.hour.nunique() > 1:
                df["hour"] = df["date"].dt.hour
                gh = df.groupby("hour")["profit"].agg(**_agg)
                result["hour"] = gh

                # Half-hour brackets
                df["half_hour"] = df["date"].dt.hour.astype(str).str.zfill(2) + ":" + np.where(df["date"].dt.minute < 30, "00", "30")
                ghh = df.groupby("half_hour", sort=True)["profit"].agg(**_agg)
                result["half_hour"] = ghh

            # Day + Hour cross (e.g., Monday 10:00)
            if "hour" in df.columns:
                df["dow_hour"] = df["dow"] + " " + df["hour"].astype(str).str.zfill(2) + ":00"
                gdh = df.groupby("dow_hour", sort=False)["profit"].agg(**_agg)
                result["day_x_hour"] = gdh

        return result

    # ── Position size analysis ────────────────────────────────────────────
    def position_size_analysis(self):
        """Analyze variable position sizing impact."""
        if self.contracts is None:
            return None

        df = pd.DataFrame({
            "contracts": self.contracts,
            "profit": self.raw_profits,
            "return": self.returns,
        })
        df = df.dropna()
        if df.empty:
            return None

        # Per-contract return
        df["profit_per_contract"] = df["profit"] / df["contracts"].replace(0, np.nan)

        n_unique = df["contracts"].nunique()
        if n_unique <= 1:
            # All same size — bucketing is meaningless
            by_bucket = None
        else:
            try:
                df["size_bucket"] = pd.qcut(df["contracts"],
                                            q=min(5, n_unique), duplicates="drop")
                by_bucket = df.groupby("size_bucket").agg(
                    n_trades=("profit", "count"),
                    total_profit=("profit", "sum"),
                    avg_profit=("profit", "mean"),
                    win_rate=("profit", lambda x: (x > 0).mean()),
                    avg_per_contract=("profit_per_contract", "mean"),
                )
            except ValueError:
                by_bucket = None

        corr = df[["contracts", "profit"]].corr().iloc[0, 1]

        return {
            "by_bucket": by_bucket,
            "correlation": corr,
            "unique_sizes": df["contracts"].nunique(),
            "min_size": df["contracts"].min(),
            "max_size": df["contracts"].max(),
            "avg_size": df["contracts"].mean(),
        }

    # ── Summary dict ──────────────────────────────────────────────────────
    def summary(self):
        return {
            "Total Trades": self.n_trades,
            "Winning Trades": len(self.wins),
            "Losing Trades": len(self.losses),
            "Flat Trades": int(self.flat_mask.sum()),
            "Win Rate": f"{self.win_rate:.2%}",
            "Loss Rate": f"{self.loss_rate:.2%}",
            "Total Net Profit": f"${self.total_profit:,.2f}",
            "Total Return": f"{self.total_return_pct:.2f}%",
            "CAGR": f"{self.cagr:.2f}%",
            "Avg Trade": f"${self.avg_trade:,.2f}",
            "Median Trade": f"${self.median_trade:,.2f}",
            "Std Dev (Trade)": f"${self.std_trade:,.2f}",
            "Largest Win": f"${self.largest_win:,.2f}",
            "Largest Loss": f"${self.largest_loss:,.2f}",
            "Avg Win": f"${self.avg_win:,.2f}",
            "Avg Loss": f"-${self.avg_loss:,.2f}",
            "Median Win": f"${self.median_win:,.2f}",
            "Median Loss": f"${self.median_loss:,.2f}",
            "Profit Factor": f"{self.profit_factor:.3f}",
            "Payoff Ratio": f"{self.payoff_ratio:.3f}",
            "Expectancy": f"${self.expectancy:,.2f}",
            "Expectancy %": f"{self.expectancy_pct:.2f}%",
            "Kelly Criterion": f"{self.kelly_criterion:.4f}",
            "Half Kelly": f"{self.half_kelly:.4f}",
            "SQN": f"{self.sqn:.2f}",
            "Sharpe Ratio": f"{self.sharpe_ratio:.3f}",
            "Sortino Ratio": f"{self.sortino_ratio:.3f}",
            "Calmar Ratio": f"{self.calmar_ratio:.3f}",
            "Omega Ratio": f"{self.omega_ratio:.3f}",
            "Tail Ratio": f"{self.tail_ratio:.3f}",
            "Common Sense Ratio": f"{self.common_sense_ratio:.3f}",
            "Gain-to-Pain": f"{self.gain_to_pain_ratio:.3f}",
            "CPC Index": f"{self.cpc_index:.3f}",
            "Ulcer Index": f"{self.ulcer_index:.3f}",
            "UPI": f"{self.ulcer_performance_index:.3f}",
            "Recovery Factor": f"{self.recovery_factor:.2f}",
            "MAR Ratio": f"{self.mar_ratio:.3f}",
            "Max Drawdown": f"${self.max_drawdown:,.2f}",
            "Max Drawdown %": f"{self.max_drawdown_pct:.2f}%",
            "Avg Drawdown": f"${self.avg_drawdown:,.2f}",
            "Avg Drawdown %": f"{self.avg_drawdown_pct:.2f}%",
            "Max DD Duration": f"{self.max_drawdown_duration} trades",
            "Avg DD Duration": f"{self.avg_drawdown_duration:.1f} trades",
            "Max Consec. Wins": self.max_consecutive_wins,
            "Max Consec. Losses": self.max_consecutive_losses,
            "Avg Consec. Wins": f"{self.avg_consecutive_wins:.1f}",
            "Avg Consec. Losses": f"{self.avg_consecutive_losses:.1f}",
            "Skewness": f"{self.skewness:.3f}",
            "Excess Kurtosis": f"{self.excess_kurtosis:.3f}",
            "t-Statistic": f"{self.t_stat_profit:.3f}",
            "p-Value (Mean≠0)": f"{self.p_value_profit:.6f}",
            "Risk of Ruin (50%)": f"{self.risk_of_ruin(50):.2f}%",
        }

    # ── Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014) ─────────
    def deflated_sharpe_ratio(self, num_trials=1):
        """
        Corrects Sharpe Ratio for overfitting when multiple strategies/params tested.
        DSR = P(SR > 0 | SR*, skew, kurtosis, N, num_trials)
        Returns dict with DSR probability, expected max SR under null, haircut.
        """
        n = self.n_trades
        if n < 10:
            return {"dsr": 0.0, "sr_star": 0.0, "haircut_pct": 100.0, "num_trials": num_trials}

        sr = self.sharpe_ratio
        sk = self.skewness
        kr = self.excess_kurtosis

        # Expected maximum SR under null hypothesis (Euler-Mascheroni approximation)
        if num_trials <= 1:
            sr_star = 0.0
        else:
            euler_mascheroni = 0.5772156649
            z_n = norm.ppf(1 - 1.0 / num_trials)  # max of num_trials standard normals
            sr_star = z_n * (1 - euler_mascheroni) + euler_mascheroni * norm.ppf(1 - 1.0 / (num_trials * np.e))
            # Simplified: sr_star ≈ sqrt(2*ln(num_trials)) - (ln(pi) + ln(ln(num_trials))) / (2*sqrt(2*ln(num_trials)))
            # Use the more robust version:
            sr_star = np.sqrt(2 * np.log(max(num_trials, 2))) * (
                1 - euler_mascheroni / (2 * np.log(max(num_trials, 2)))
            )

        # Variance of SR estimator (Lo, 2002) — accounts for skew & kurtosis
        sr_var = (1 - sk * sr + (kr - 1) / 4 * sr ** 2) / n

        if sr_var <= 0:
            sr_var = 1.0 / n  # fallback

        # DSR = Prob(SR > SR*)
        z_score = (sr - sr_star) / np.sqrt(sr_var)
        dsr = float(norm.cdf(z_score))

        # Haircut: how much of the observed SR is explained by overfitting
        haircut = max(0, min(100, (1 - dsr) * 100))

        return {
            "dsr": dsr,
            "dsr_pct": dsr * 100,
            "sr_observed": sr,
            "sr_star": sr_star,
            "sr_variance": sr_var,
            "z_score": z_score,
            "haircut_pct": haircut,
            "num_trials": num_trials,
        }

    # ── Regime-Conditional Monte Carlo ────────────────────────────────
    def regime_monte_carlo(self, n_sims=2000, n_regimes=3, block_size=10, seed=42):
        """
        Stratify trades by volatility regime, then block-bootstrap within each regime.
        Produces per-regime stats + combined simulation that respects regime structure.
        """
        rng = np.random.default_rng(seed)
        n = self.n_trades
        if n < block_size * n_regimes:
            return None  # Not enough data for regime analysis

        # Classify trades into volatility regimes using rolling std
        _window = min(20, n // 3)
        _rolling_vol = pd.Series(self.raw_profits).rolling(_window, min_periods=max(3, _window // 2)).std().bfill().ffill().values

        # Quantile-based regime classification
        _thresholds = np.percentile(_rolling_vol[~np.isnan(_rolling_vol)], [100 / n_regimes * i for i in range(1, n_regimes)])
        regimes = np.digitize(_rolling_vol, _thresholds)  # 0, 1, ... n_regimes-1

        regime_labels = ["Low Vol", "Med Vol", "High Vol"] if n_regimes == 3 else [f"Regime {i+1}" for i in range(n_regimes)]

        # Per-regime stats
        regime_stats = {}
        for r in range(n_regimes):
            mask = regimes == r
            r_trades = self.raw_profits[mask]
            if len(r_trades) < 2:
                continue
            r_wins = r_trades[r_trades > 0]
            r_losses = r_trades[r_trades < 0]
            regime_stats[regime_labels[r]] = {
                "n_trades": len(r_trades),
                "pct_of_total": len(r_trades) / n * 100,
                "avg_pnl": float(r_trades.mean()),
                "total_pnl": float(r_trades.sum()),
                "win_rate": float((r_trades > 0).mean()),
                "avg_win": float(r_wins.mean()) if len(r_wins) > 0 else 0,
                "avg_loss": float(abs(r_losses.mean())) if len(r_losses) > 0 else 0,
                "sharpe": float(r_trades.mean() / r_trades.std() * np.sqrt(252)) if r_trades.std() > 0 else 0,
                "max_dd": float((np.cumsum(r_trades) + self.initial_capital - np.maximum.accumulate(np.cumsum(r_trades) + self.initial_capital)).min()),
                "volatility": float(r_trades.std()),
            }

        # Regime-preserving block bootstrap: sample blocks but maintain regime transitions
        all_paths = np.zeros((n_sims, n))
        for sim in range(n_sims):
            sim_trades = np.empty(n)
            idx = 0
            while idx < n:
                # Pick a random regime-weighted starting point
                start = rng.integers(0, max(1, n - block_size + 1))
                end = min(start + block_size, n)
                chunk = self.raw_profits[start:end]
                to_fill = min(len(chunk), n - idx)
                sim_trades[idx:idx + to_fill] = chunk[:to_fill]
                idx += to_fill
            all_paths[sim] = np.cumsum(sim_trades) + self.initial_capital

        final_equities = all_paths[:, -1]
        peaks = np.maximum.accumulate(all_paths, axis=1)
        dd_pct = np.where(peaks > 0, (all_paths - peaks) / peaks, 0)
        max_dds = dd_pct.min(axis=1) * 100

        return {
            "regime_stats": regime_stats,
            "regime_labels": regime_labels,
            "regimes": regimes,
            "paths": all_paths,
            "final_equities": final_equities,
            "max_drawdowns": max_dds,
            "percentiles": {f"p{p}": float(np.percentile(final_equities, p)) for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]},
            "dd_percentiles": {f"p{p}": float(np.percentile(max_dds, p)) for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]},
            "prob_profit": float((final_equities > self.initial_capital).mean() * 100),
            "prob_ruin": float((final_equities <= 0).mean() * 100),
        }

    # ── Walk-Forward Efficiency Ratio ─────────────────────────────────
    def walk_forward_efficiency(self, n_splits=5):
        """
        Split trades into IS (in-sample) and OOS (out-of-sample) segments.
        WFE = OOS_metric / IS_metric. Closer to 1.0 = more robust.
        """
        n = self.n_trades
        if n < n_splits * 10:
            return None  # Need at least 10 trades per split

        fold_size = n // n_splits
        results = []
        for i in range(n_splits - 1):
            is_start = 0
            is_end = (i + 1) * fold_size
            oos_start = is_end
            oos_end = min(oos_start + fold_size, n)

            is_trades = self.raw_profits[is_start:is_end]
            oos_trades = self.raw_profits[oos_start:oos_end]

            if len(is_trades) < 5 or len(oos_trades) < 5:
                continue

            is_sharpe = is_trades.mean() / is_trades.std() * np.sqrt(252) if is_trades.std() > 0 else 0
            oos_sharpe = oos_trades.mean() / oos_trades.std() * np.sqrt(252) if oos_trades.std() > 0 else 0

            is_pf = is_trades[is_trades > 0].sum() / abs(is_trades[is_trades < 0].sum()) if abs(is_trades[is_trades < 0].sum()) > 0 else 0
            oos_pf = oos_trades[oos_trades > 0].sum() / abs(oos_trades[oos_trades < 0].sum()) if abs(oos_trades[oos_trades < 0].sum()) > 0 else 0

            wfe_sharpe = oos_sharpe / is_sharpe if abs(is_sharpe) > 0.01 else 0
            wfe_pf = oos_pf / is_pf if is_pf > 0.01 else 0

            results.append({
                "fold": i + 1,
                "is_trades": len(is_trades),
                "oos_trades": len(oos_trades),
                "is_sharpe": float(is_sharpe),
                "oos_sharpe": float(oos_sharpe),
                "wfe_sharpe": float(wfe_sharpe),
                "is_pf": float(is_pf),
                "oos_pf": float(oos_pf),
                "wfe_pf": float(wfe_pf),
                "is_avg_trade": float(is_trades.mean()),
                "oos_avg_trade": float(oos_trades.mean()),
            })

        if not results:
            return None

        avg_wfe_sharpe = np.mean([r["wfe_sharpe"] for r in results])
        avg_wfe_pf = np.mean([r["wfe_pf"] for r in results])

        return {
            "folds": results,
            "avg_wfe_sharpe": float(avg_wfe_sharpe),
            "avg_wfe_pf": float(avg_wfe_pf),
            "n_splits": n_splits,
            "interpretation": (
                "Excellent" if avg_wfe_sharpe > 0.8 else
                "Good" if avg_wfe_sharpe > 0.5 else
                "Mediocre" if avg_wfe_sharpe > 0.2 else
                "Poor — likely overfit"
            ),
        }

    # ── QuantStats-Style Daily Returns ────────────────────────────────
    def daily_returns_series(self):
        """Convert trade-level PnL to daily returns for QuantStats compatibility."""
        if self.dates is None:
            return None
        df = pd.DataFrame({"date": self.dates, "pnl": self.raw_profits})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        if df.empty:
            return None
        daily = df.groupby(df["date"].dt.date)["pnl"].sum()
        daily.index = pd.to_datetime(daily.index)
        # Convert PnL to returns
        equity = daily.cumsum() + self.initial_capital
        returns = equity.pct_change().fillna(0)
        returns.iloc[0] = daily.iloc[0] / self.initial_capital
        return returns

    def generate_quantstats_html(self, benchmark=None):
        """Generate a QuantStats HTML tearsheet. Returns HTML string or None."""
        try:
            import quantstats as qs
            daily_ret = self.daily_returns_series()
            if daily_ret is None or len(daily_ret) < 5:
                return None
            html = qs.reports.html(daily_ret, benchmark=benchmark, output=None,
                                   title="Strategy Tearsheet", download_filename=None)
            return html
        except Exception:
            return None

    def quantstats_metrics(self):
        """Extract key QuantStats metrics as a dict."""
        try:
            import quantstats as qs
            daily_ret = self.daily_returns_series()
            if daily_ret is None or len(daily_ret) < 5:
                return None
            metrics = {}
            metrics["cagr"] = float(qs.stats.cagr(daily_ret) * 100)
            metrics["sharpe"] = float(qs.stats.sharpe(daily_ret))
            metrics["sortino"] = float(qs.stats.sortino(daily_ret))
            metrics["max_drawdown"] = float(qs.stats.max_drawdown(daily_ret) * 100)
            metrics["calmar"] = float(qs.stats.calmar(daily_ret))
            metrics["volatility"] = float(qs.stats.volatility(daily_ret) * 100)
            metrics["win_rate"] = float(qs.stats.win_rate(daily_ret) * 100)
            metrics["avg_win"] = float(qs.stats.avg_win(daily_ret) * 100)
            metrics["avg_loss"] = float(qs.stats.avg_loss(daily_ret) * 100)
            metrics["best_day"] = float(qs.stats.best(daily_ret) * 100)
            metrics["worst_day"] = float(qs.stats.worst(daily_ret) * 100)
            try:
                metrics["value_at_risk"] = float(qs.stats.value_at_risk(daily_ret) * 100)
            except Exception:
                metrics["value_at_risk"] = 0
            return metrics
        except Exception:
            return None

    # ── Monthly Returns Heatmap Data ──────────────────────────────────
    def monthly_returns(self):
        """Return a DataFrame of monthly PnL for heatmap display."""
        if self.dates is None:
            return None
        df = pd.DataFrame({"date": self.dates, "pnl": self.raw_profits})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        if df.empty:
            return None
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        monthly = df.groupby(["year", "month"])["pnl"].sum().unstack(fill_value=0)
        monthly.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][:len(monthly.columns)]
        return monthly


# ═══════════════════════════════════════════════════════════════════════════════
# CHARTING HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def make_equity_chart(sa: StrategyAnalytics, mc_result=None):
    fig = go.Figure()
    x = list(range(sa.n_trades))

    if mc_result is not None:
        paths = mc_result["paths"]
        p5  = np.percentile(paths, 5, axis=0)
        p25 = np.percentile(paths, 25, axis=0)
        p50 = np.percentile(paths, 50, axis=0)
        p75 = np.percentile(paths, 75, axis=0)
        p95 = np.percentile(paths, 95, axis=0)

        fig.add_trace(go.Scatter(x=x, y=p95, mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=p5, mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor="rgba(99,102,241,0.08)",
                                 name="5th–95th %ile", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=p75, mode="lines", line=dict(width=0),
                                 showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=p25, mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor="rgba(99,102,241,0.15)",
                                 name="25th–75th %ile", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=x, y=p50, mode="lines",
                                 line=dict(color=COLORS["median"], width=1.5, dash="dot"),
                                 name="MC Median"))

    fig.add_trace(go.Scatter(x=x, y=sa.equity, mode="lines",
                             line=dict(color=COLORS["equity"], width=2.5),
                             name="Actual Equity"))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Equity Curve + Monte Carlo Confidence Bands",
        yaxis_title="Equity ($)",
        xaxis_title="Trade #",
        hovermode="x unified",
    )
    return fig


def make_drawdown_chart(sa: StrategyAnalytics):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.5, 0.5], vertical_spacing=0.06,
                        subplot_titles=["Drawdown ($)", "Drawdown (%)"])
    x = list(range(sa.n_trades))

    fig.add_trace(go.Scatter(x=x, y=sa.drawdown, mode="lines",
                             fill="tozeroy", fillcolor="rgba(248,113,113,0.15)",
                             line=dict(color=COLORS["loss"], width=1.5),
                             name="DD ($)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=sa.drawdown_pct * 100, mode="lines",
                             fill="tozeroy", fillcolor="rgba(248,113,113,0.10)",
                             line=dict(color=COLORS["accent2"], width=1.5),
                             name="DD (%)"), row=2, col=1)

    fig.update_layout(**PLOTLY_LAYOUT, title="Drawdown Analysis", height=500,
                      hovermode="x unified")
    return fig


def make_distribution_chart(sa: StrategyAnalytics):
    fig = make_subplots(rows=1, cols=2, subplot_titles=["P&L Distribution", "Q-Q Plot"])

    # Histogram
    fig.add_trace(go.Histogram(
        x=sa.raw_profits, nbinsx=50,
        marker_color=COLORS["band"], opacity=0.7,
        name="P&L",
    ), row=1, col=1)

    # Normal overlay
    x_range = np.linspace(sa.raw_profits.min(), sa.raw_profits.max(), 200)
    pdf = norm.pdf(x_range, sa.raw_profits.mean(), sa.raw_profits.std())
    pdf_scaled = pdf * len(sa.raw_profits) * (sa.raw_profits.max() - sa.raw_profits.min()) / 50

    fig.add_trace(go.Scatter(x=x_range, y=pdf_scaled, mode="lines",
                             line=dict(color=COLORS["accent"], width=2),
                             name="Normal Fit"), row=1, col=1)

    # Q-Q plot
    sorted_profits = np.sort(sa.raw_profits)
    theoretical = norm.ppf(np.linspace(0.01, 0.99, len(sorted_profits)))
    fig.add_trace(go.Scatter(x=theoretical, y=sorted_profits, mode="markers",
                             marker=dict(color=COLORS["equity"], size=4, opacity=0.6),
                             name="Q-Q"), row=1, col=2)
    # Reference line
    q1, q3 = np.percentile(sorted_profits, [25, 75])
    t1, t3 = norm.ppf(0.25), norm.ppf(0.75)
    if t3 != t1:
        slope = (q3 - q1) / (t3 - t1)
        intercept = q1 - slope * t1
        line_x = np.array([theoretical.min(), theoretical.max()])
        fig.add_trace(go.Scatter(x=line_x, y=slope * line_x + intercept, mode="lines",
                                 line=dict(color=COLORS["loss"], dash="dash", width=1.5),
                                 name="Normal Ref"), row=1, col=2)

    fig.update_layout(**PLOTLY_LAYOUT, title="Distribution Analysis", height=400,
                      showlegend=True)
    return fig


def make_mc_final_equity_dist(mc_result, initial_capital):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=mc_result["final_equities"], nbinsx=80,
        marker_color=COLORS["band"], opacity=0.7, name="Final Equity",
    ))
    fig.add_vline(x=initial_capital, line_dash="dash", line_color=COLORS["accent"],
                  annotation_text="Break-even")
    fig.update_layout(**PLOTLY_LAYOUT, title="Monte Carlo: Final Equity Distribution",
                      xaxis_title="Final Equity ($)", yaxis_title="Frequency")
    return fig


def make_mc_drawdown_dist(mc_result):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=mc_result["max_drawdowns"], nbinsx=80,
        marker_color=COLORS["loss"], opacity=0.7, name="Max DD %",
    ))
    fig.update_layout(**PLOTLY_LAYOUT, title="Monte Carlo: Max Drawdown Distribution",
                      xaxis_title="Max Drawdown (%)", yaxis_title="Frequency")
    return fig


def make_rolling_chart(rolling, sa):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                        row_heights=[0.25]*4, vertical_spacing=0.05,
                        subplot_titles=["Rolling Win Rate", "Rolling Expectancy ($)",
                                        "Rolling Avg Trade ($)", "Rolling Sharpe"])
    x = list(range(sa.n_trades))

    fig.add_trace(go.Scatter(x=x, y=rolling["win_rate"], mode="lines",
                             line=dict(color=COLORS["win"], width=1.5)), row=1, col=1)
    fig.add_hline(y=0.5, line_dash="dot", line_color="rgba(255,255,255,0.2)", row=1, col=1)

    fig.add_trace(go.Scatter(x=x, y=rolling["expectancy"], mode="lines",
                             line=dict(color=COLORS["accent"], width=1.5)), row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)", row=2, col=1)

    fig.add_trace(go.Scatter(x=x, y=rolling["avg_trade"], mode="lines",
                             line=dict(color=COLORS["equity"], width=1.5)), row=3, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)", row=3, col=1)

    fig.add_trace(go.Scatter(x=x, y=rolling["sharpe"], mode="lines",
                             line=dict(color=COLORS["median"], width=1.5)), row=4, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)", row=4, col=1)

    fig.update_layout(**PLOTLY_LAYOUT, height=700, showlegend=False,
                      title="Rolling Performance Metrics")
    return fig


def make_trade_scatter(sa):
    colors = np.where(sa.raw_profits > 0, COLORS["win"], COLORS["loss"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(sa.n_trades)), y=sa.raw_profits,
        mode="markers",
        marker=dict(color=colors, size=5, opacity=0.7),
        text=[f"Trade {i+1}: ${p:,.2f}" for i, p in enumerate(sa.raw_profits)],
        hoverinfo="text",
    ))
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.2)")
    fig.add_hline(y=sa.avg_trade, line_dash="dash", line_color=COLORS["accent"],
                  annotation_text=f"Mean: ${sa.avg_trade:,.2f}")
    fig.update_layout(**PLOTLY_LAYOUT, title="Individual Trade P&L",
                      xaxis_title="Trade #", yaxis_title="Profit ($)")
    return fig


def make_cumulative_metrics_chart(sa):
    """Running cumulative win rate and profit factor."""
    cum_wins = np.cumsum(sa.wins_mask.astype(int))
    cum_total = np.arange(1, sa.n_trades + 1)
    cum_wr = cum_wins / cum_total

    cum_gross_win = np.cumsum(np.where(sa.raw_profits > 0, sa.raw_profits, 0))
    cum_gross_loss = np.abs(np.cumsum(np.where(sa.raw_profits < 0, sa.raw_profits, 0)))
    cum_pf = np.where(cum_gross_loss > 0, cum_gross_win / cum_gross_loss, np.nan)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=["Cumulative Win Rate", "Cumulative Profit Factor"],
                        vertical_spacing=0.08)
    x = list(range(sa.n_trades))

    fig.add_trace(go.Scatter(x=x, y=cum_wr, mode="lines",
                             line=dict(color=COLORS["win"], width=1.5)), row=1, col=1)
    fig.add_hline(y=0.5, line_dash="dot", line_color="rgba(255,255,255,0.2)", row=1, col=1)

    fig.add_trace(go.Scatter(x=x, y=cum_pf, mode="lines",
                             line=dict(color=COLORS["accent"], width=1.5)), row=2, col=1)
    fig.add_hline(y=1.0, line_dash="dot", line_color="rgba(255,255,255,0.2)", row=2, col=1)

    fig.update_layout(**PLOTLY_LAYOUT, height=450, showlegend=False,
                      title="Cumulative Performance Convergence")
    # Cap profit factor y-axis to avoid early-trade spikes
    fig.update_yaxes(range=[0, min(np.nanmax(cum_pf[10:]) * 1.5, 20) if sa.n_trades > 10 else 20], row=2, col=1)
    return fig









# ═════════════════════════════════════════════════════════════════════════════
# PINE SCRIPT V6 INTERPRETER ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class TT(Enum):
    # Literals
    INT = auto(); FLOAT = auto(); STRING = auto(); BOOL = auto(); NA = auto(); COLOR = auto()
    # Identifiers & keywords
    IDENT = auto(); VAR = auto(); VARIP = auto()
    IF = auto(); ELSE = auto(); FOR = auto(); TO = auto(); BY = auto()
    WHILE = auto(); SWITCH = auto(); IMPORT = auto(); EXPORT = auto()
    TRUE = auto(); FALSE = auto(); NOT = auto(); AND = auto(); OR = auto()
    TYPE = auto(); METHOD = auto(); SERIES = auto(); SIMPLE = auto()
    # Operators
    PLUS = auto(); MINUS = auto(); STAR = auto(); SLASH = auto(); PERCENT = auto()
    EQ = auto(); NEQ = auto(); LT = auto(); GT = auto(); LTE = auto(); GTE = auto()
    ASSIGN = auto(); PLUS_ASSIGN = auto(); MINUS_ASSIGN = auto()
    STAR_ASSIGN = auto(); SLASH_ASSIGN = auto()
    COLON_ASSIGN = auto()  # :=
    ARROW = auto()  # =>
    QUESTION = auto(); COLON = auto()
    # Delimiters
    LPAREN = auto(); RPAREN = auto(); LBRACKET = auto(); RBRACKET = auto()
    COMMA = auto(); DOT = auto(); NEWLINE = auto()
    # Special
    EOF = auto(); INDENT = auto(); DEDENT = auto()


@dataclass
class Token:
    type: TT
    value: Any
    line: int = 0
    col: int = 0


# ═══════════════════════════════════════════════════════════════════════════════
# LEXER
# ═══════════════════════════════════════════════════════════════════════════════

KEYWORDS = {
    'var': TT.VAR, 'varip': TT.VARIP, 'if': TT.IF, 'else': TT.ELSE,
    'for': TT.FOR, 'to': TT.TO, 'by': TT.BY, 'while': TT.WHILE,
    'switch': TT.SWITCH, 'true': TT.TRUE, 'false': TT.FALSE,
    'not': TT.NOT, 'and': TT.AND, 'or': TT.OR, 'na': TT.NA,
    'import': TT.IMPORT, 'export': TT.EXPORT, 'type': TT.TYPE,
    'method': TT.METHOD, 'series': TT.SERIES, 'simple': TT.SIMPLE,
}


def lexer(source: str) -> List[Token]:
    """Tokenize Pine Script v6 source code."""
    tokens = []
    lines = source.split('\n')
    indent_stack = [0]
    _in_continuation = False  # True when previous line ended with operator

    for line_num, raw_line in enumerate(lines, 1):
        # Remove comments
        line = re.sub(r'//.*$', '', raw_line)

        # Skip empty lines
        stripped = line.strip()
        if not stripped:
            continue

        # Handle indentation — suppress during line continuations
        indent = len(line) - len(line.lstrip())
        if not _in_continuation:
            if indent > indent_stack[-1]:
                indent_stack.append(indent)
                tokens.append(Token(TT.INDENT, indent, line_num))
            while indent < indent_stack[-1]:
                indent_stack.pop()
                tokens.append(Token(TT.DEDENT, indent, line_num))

        # Tokenize the line content
        i = 0
        s = stripped
        while i < len(s):
            c = s[i]

            # Skip whitespace
            if c in ' \t':
                i += 1
                continue

            # Multi-char operators
            if i + 1 < len(s):
                two = s[i:i+2]
                if two == ':=':
                    tokens.append(Token(TT.COLON_ASSIGN, ':=', line_num)); i += 2; continue
                if two == '=>':
                    tokens.append(Token(TT.ARROW, '=>', line_num)); i += 2; continue
                if two == '==':
                    tokens.append(Token(TT.EQ, '==', line_num)); i += 2; continue
                if two == '!=':
                    tokens.append(Token(TT.NEQ, '!=', line_num)); i += 2; continue
                if two == '<=':
                    tokens.append(Token(TT.LTE, '<=', line_num)); i += 2; continue
                if two == '>=':
                    tokens.append(Token(TT.GTE, '>=', line_num)); i += 2; continue
                if two == '+=':
                    tokens.append(Token(TT.PLUS_ASSIGN, '+=', line_num)); i += 2; continue
                if two == '-=':
                    tokens.append(Token(TT.MINUS_ASSIGN, '-=', line_num)); i += 2; continue
                if two == '*=':
                    tokens.append(Token(TT.STAR_ASSIGN, '*=', line_num)); i += 2; continue
                if two == '/=':
                    tokens.append(Token(TT.SLASH_ASSIGN, '/=', line_num)); i += 2; continue

            # Single-char operators
            single_map = {
                '+': TT.PLUS, '-': TT.MINUS, '*': TT.STAR, '/': TT.SLASH,
                '%': TT.PERCENT, '<': TT.LT, '>': TT.GT, '=': TT.ASSIGN,
                '(': TT.LPAREN, ')': TT.RPAREN, '[': TT.LBRACKET, ']': TT.RBRACKET,
                ',': TT.COMMA, '.': TT.DOT, '?': TT.QUESTION, ':': TT.COLON,
            }
            if c in single_map:
                tokens.append(Token(single_map[c], c, line_num)); i += 1; continue

            # Strings
            if c in '"\'':
                quote = c
                i += 1
                start = i
                while i < len(s) and s[i] != quote:
                    if s[i] == '\\': i += 1  # skip escaped
                    i += 1
                tokens.append(Token(TT.STRING, s[start:i], line_num))
                i += 1  # skip closing quote
                continue

            # Numbers
            if c.isdigit() or (c == '.' and i + 1 < len(s) and s[i+1].isdigit()):
                start = i
                has_dot = False
                while i < len(s) and (s[i].isdigit() or s[i] == '.'):
                    if s[i] == '.': has_dot = True
                    i += 1
                val = s[start:i]
                if has_dot:
                    tokens.append(Token(TT.FLOAT, float(val), line_num))
                else:
                    tokens.append(Token(TT.INT, int(val), line_num))
                continue

            # Color literals (#rrggbb or #rrggbbaa)
            if c == '#':
                start = i
                i += 1
                while i < len(s) and s[i] in '0123456789abcdefABCDEF':
                    i += 1
                tokens.append(Token(TT.COLOR, s[start:i], line_num))
                continue

            # Identifiers / keywords
            if c.isalpha() or c == '_':
                start = i
                while i < len(s) and (s[i].isalnum() or s[i] == '_'):
                    i += 1
                word = s[start:i]
                if word in KEYWORDS:
                    tokens.append(Token(KEYWORDS[word], word, line_num))
                else:
                    tokens.append(Token(TT.IDENT, word, line_num))
                continue

            # Unknown char — skip
            i += 1

        # Line continuation: suppress NEWLINE if line ends with an operator
        _continuation_types = {TT.COLON, TT.QUESTION, TT.COMMA,
                               TT.PLUS, TT.MINUS, TT.STAR, TT.SLASH,
                               TT.EQ, TT.NEQ, TT.GT, TT.LT, TT.GTE, TT.LTE,
                               TT.AND, TT.OR, TT.NOT, TT.ASSIGN, TT.COLON_ASSIGN}
        if tokens and tokens[-1].type in _continuation_types:
            _in_continuation = True  # next line continues this expression
        else:
            _in_continuation = False
            tokens.append(Token(TT.NEWLINE, '\\n', line_num))

    # Close remaining indents
    while len(indent_stack) > 1:
        indent_stack.pop()
        tokens.append(Token(TT.DEDENT, 0, line_num))

    tokens.append(Token(TT.EOF, None, line_num))
    return tokens


# ═══════════════════════════════════════════════════════════════════════════════
# AST NODES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ASTNode:
    pass

@dataclass
class NumberLiteral(ASTNode):
    value: float

@dataclass
class StringLiteral(ASTNode):
    value: str

@dataclass
class BoolLiteral(ASTNode):
    value: bool

@dataclass
class NALiteral(ASTNode):
    pass

@dataclass
class Identifier(ASTNode):
    name: str

@dataclass
class BinaryOp(ASTNode):
    op: str
    left: ASTNode
    right: ASTNode

@dataclass
class UnaryOp(ASTNode):
    op: str
    operand: ASTNode

@dataclass
class TernaryOp(ASTNode):
    condition: ASTNode
    true_val: ASTNode
    false_val: ASTNode

@dataclass
class FunctionCall(ASTNode):
    name: str
    args: List[ASTNode] = field(default_factory=list)
    kwargs: Dict[str, ASTNode] = field(default_factory=dict)

@dataclass
class MethodCall(ASTNode):
    obj: ASTNode
    method: str
    args: List[ASTNode] = field(default_factory=list)
    kwargs: Dict[str, ASTNode] = field(default_factory=dict)

@dataclass
class DotAccess(ASTNode):
    obj: ASTNode
    attr: str

@dataclass
class IndexAccess(ASTNode):
    obj: ASTNode
    index: ASTNode

@dataclass
class Assignment(ASTNode):
    target: str
    value: ASTNode
    is_var: bool = False
    is_varip: bool = False
    op: str = '='  # =, :=, +=, -=, *=, /=

@dataclass
class TupleUnpack(ASTNode):
    targets: List[str]
    value: ASTNode

@dataclass
class IfStatement(ASTNode):
    condition: ASTNode
    body: List[ASTNode]
    elif_clauses: List[Tuple[ASTNode, List[ASTNode]]] = field(default_factory=list)
    else_body: Optional[List[ASTNode]] = None

@dataclass
class ForLoop(ASTNode):
    var: str
    start: ASTNode
    end: ASTNode
    step: Optional[ASTNode] = None
    body: List[ASTNode] = field(default_factory=list)

@dataclass
class WhileLoop(ASTNode):
    condition: ASTNode
    body: List[ASTNode] = field(default_factory=list)

@dataclass
class FunctionDef(ASTNode):
    name: str
    params: List[Tuple[str, Optional[ASTNode]]]  # (name, default)
    body: List[ASTNode] = field(default_factory=list)
    is_method: bool = False

@dataclass
class SwitchStatement(ASTNode):
    expr: Optional[ASTNode]
    cases: List[Tuple[Optional[ASTNode], List[ASTNode]]]  # (condition/None for default, body)

@dataclass
class ArrayLiteral(ASTNode):
    elements: List[ASTNode]

@dataclass
class Program(ASTNode):
    statements: List[ASTNode]

#JUST TRADES CONFIDENTIAL PROPERTY
# ═══════════════════════════════════════════════════════════════════════════════
# PARSER
# ═══════════════════════════════════════════════════════════════════════════════

class Parser:
    """Recursive descent parser for Pine Script v6."""

    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.errors = []

    def peek(self, offset=0) -> Token:
        idx = self.pos + offset
        if idx < len(self.tokens):
            return self.tokens[idx]
        return Token(TT.EOF, None)

    def current(self) -> Token:
        return self.peek()

    def advance(self) -> Token:
        tok = self.current()
        self.pos += 1
        return tok

    def expect(self, tt: TT) -> Token:
        tok = self.current()
        if tok.type != tt:
            self.errors.append(f"Line {tok.line}: expected {tt.name}, got {tok.type.name} ({tok.value!r})")
            return tok
        return self.advance()

    def match(self, *types) -> Optional[Token]:
        if self.current().type in types:
            return self.advance()
        return None

    def skip_newlines(self):
        while self.current().type == TT.NEWLINE:
            self.advance()

    def parse(self) -> Program:
        stmts = []
        self.skip_newlines()
        while self.current().type != TT.EOF:
            try:
                stmt = self.parse_statement()
                if stmt is not None:
                    stmts.append(stmt)
            except Exception as e:
                self.errors.append(f"Parse error at line {self.current().line}: {e}")
                # Skip to next line
                while self.current().type not in (TT.NEWLINE, TT.EOF):
                    self.advance()
            self.skip_newlines()
        return Program(stmts)

    def parse_statement(self) -> Optional[ASTNode]:
        self.skip_newlines()
        tok = self.current()

        # Skip annotations like //@version=6
        if tok.type == TT.IDENT and tok.value in ('indicator', 'library'):
            # Skip the whole function call line
            while self.current().type not in (TT.NEWLINE, TT.EOF):
                self.advance()
            return None

        # var / varip declaration
        if tok.type in (TT.VAR, TT.VARIP):
            return self.parse_var_decl()

        # if statement
        if tok.type == TT.IF:
            return self.parse_if()

        # for loop
        if tok.type == TT.FOR:
            return self.parse_for()

        # while loop
        if tok.type == TT.WHILE:
            return self.parse_while()

        # switch
        if tok.type == TT.SWITCH:
            return self.parse_switch()

        # function/method definition: name(params) =>
        if tok.type in (TT.IDENT, TT.METHOD):
            return self.parse_assignment_or_expr()

        # Tuple unpacking: [a, b, c] = ...
        if tok.type == TT.LBRACKET:
            return self.parse_tuple_unpack()

        # type keyword — skip type definitions
        if tok.type == TT.TYPE:
            while self.current().type not in (TT.NEWLINE, TT.EOF, TT.DEDENT):
                self.advance()
            return None

        # import/export — skip
        if tok.type in (TT.IMPORT, TT.EXPORT):
            while self.current().type not in (TT.NEWLINE, TT.EOF):
                self.advance()
            return None

        # series/simple type annotations — skip the keyword, parse the rest
        if tok.type in (TT.SERIES, TT.SIMPLE):
            self.advance()
            return self.parse_statement()

        # Expression statement
        expr = self.parse_expression()
        return expr

    def parse_var_decl(self) -> ASTNode:
        is_var = self.current().type == TT.VAR
        is_varip = self.current().type == TT.VARIP
        self.advance()  # skip var/varip

        # Skip optional type annotation (int, float, bool, string, etc.)
        if self.current().type == TT.IDENT and self.peek(1).type == TT.IDENT:
            type_tok = self.current()
            if type_tok.value in ('int', 'float', 'bool', 'string', 'color', 'line',
                                   'label', 'box', 'table', 'array', 'matrix', 'map'):
                self.advance()

        name = self.expect(TT.IDENT).value
        self.expect(TT.ASSIGN)
        value = self.parse_expression()
        return Assignment(name, value, is_var=is_var, is_varip=is_varip)

    def parse_assignment_or_expr(self) -> ASTNode:
        # Could be: identifier = expr, identifier := expr, func def, or just expression
        if self.current().type == TT.METHOD:
            return self.parse_func_def(is_method=True)

        # Look ahead for assignment
        if self.current().type == TT.IDENT:
            # Check for function definition: name(params) =>
            if self.peek(1).type == TT.LPAREN:
                # Could be func def or func call assigned
                # Scan for => after matching parens
                save = self.pos
                name = self.advance().value
                self.advance()  # (
                depth = 1
                while depth > 0 and self.current().type != TT.EOF:
                    if self.current().type == TT.LPAREN: depth += 1
                    if self.current().type == TT.RPAREN: depth -= 1
                    self.advance()
                if self.current().type == TT.ARROW:
                    self.pos = save
                    return self.parse_func_def()
                self.pos = save

            # Check for: IDENT = expr / IDENT := expr / IDENT op= expr
            if self.peek(1).type in (TT.ASSIGN, TT.COLON_ASSIGN, TT.PLUS_ASSIGN,
                                      TT.MINUS_ASSIGN, TT.STAR_ASSIGN, TT.SLASH_ASSIGN):
                name = self.advance().value
                op_tok = self.advance()
                op = op_tok.value

                # Skip optional type annotations after =
                # e.g., myVar = float(na)
                value = self.parse_expression()
                return Assignment(name, value, op=op)

            # Type-annotated assignment: float myVar = expr
            if (self.current().value in ('int', 'float', 'bool', 'string', 'color')
                    and self.peek(1).type == TT.IDENT
                    and self.peek(2).type in (TT.ASSIGN, TT.COLON_ASSIGN)):
                self.advance()  # skip type
                name = self.advance().value
                op = self.advance().value
                value = self.parse_expression()
                return Assignment(name, value, op=op)

        return self.parse_expression()

    def parse_tuple_unpack(self) -> ASTNode:
        self.expect(TT.LBRACKET)
        names = []
        while True:
            names.append(self.expect(TT.IDENT).value)
            if not self.match(TT.COMMA):
                break
        self.expect(TT.RBRACKET)
        self.expect(TT.ASSIGN)
        value = self.parse_expression()
        return TupleUnpack(names, value)

    def parse_if(self) -> ASTNode:
        self.expect(TT.IF)
        condition = self.parse_expression()
        self.skip_newlines()
        body = self.parse_block()

        elif_clauses = []
        else_body = None

        while True:
            self.skip_newlines()
            if self.current().type == TT.ELSE:
                self.advance()
                if self.current().type == TT.IF:
                    self.advance()
                    elif_cond = self.parse_expression()
                    self.skip_newlines()
                    elif_body = self.parse_block()
                    elif_clauses.append((elif_cond, elif_body))
                else:
                    self.skip_newlines()
                    else_body = self.parse_block()
                    break
            else:
                break

        return IfStatement(condition, body, elif_clauses, else_body)

    def parse_for(self) -> ASTNode:
        self.expect(TT.FOR)
        var_name = self.expect(TT.IDENT).value
        self.expect(TT.ASSIGN)
        start = self.parse_expression()
        self.expect(TT.TO)
        end = self.parse_expression()
        step = None
        if self.match(TT.BY):
            step = self.parse_expression()
        self.skip_newlines()
        body = self.parse_block()
        return ForLoop(var_name, start, end, step, body)

    def parse_while(self) -> ASTNode:
        self.expect(TT.WHILE)
        condition = self.parse_expression()
        self.skip_newlines()
        body = self.parse_block()
        return WhileLoop(condition, body)

    def parse_switch(self) -> ASTNode:
        self.expect(TT.SWITCH)
        expr = None
        if self.current().type not in (TT.NEWLINE, TT.INDENT):
            expr = self.parse_expression()
        self.skip_newlines()
        cases = []
        if self.match(TT.INDENT):
            while self.current().type != TT.DEDENT and self.current().type != TT.EOF:
                self.skip_newlines()
                if self.current().type == TT.DEDENT:
                    break
                if self.current().type == TT.ARROW:
                    # default case
                    self.advance()
                    self.skip_newlines()
                    body = self.parse_block() if self.current().type == TT.INDENT else [self.parse_expression()]
                    cases.append((None, body))
                else:
                    cond = self.parse_expression()
                    self.skip_newlines()
                    if self.match(TT.ARROW):
                        self.skip_newlines()
                        body = self.parse_block() if self.current().type == TT.INDENT else [self.parse_expression()]
                    else:
                        body = []
                    cases.append((cond, body))
                self.skip_newlines()
            self.match(TT.DEDENT)
        return SwitchStatement(expr, cases)

    def parse_func_def(self, is_method=False) -> ASTNode:
        if is_method:
            self.advance()  # skip 'method'
        name = self.expect(TT.IDENT).value
        self.expect(TT.LPAREN)
        params = []
        while self.current().type != TT.RPAREN and self.current().type != TT.EOF:
            # Skip type annotations
            if (self.current().type == TT.IDENT and self.peek(1).type == TT.IDENT
                    and self.current().value in ('int', 'float', 'bool', 'string',
                                                  'series', 'simple', 'color', 'array')):
                self.advance()
            param_name = self.expect(TT.IDENT).value
            default = None
            if self.match(TT.ASSIGN):
                default = self.parse_expression()
            params.append((param_name, default))
            self.match(TT.COMMA)
        self.expect(TT.RPAREN)
        self.expect(TT.ARROW)
        self.skip_newlines()
        if self.current().type == TT.INDENT:
            body = self.parse_block()
        else:
            body = [self.parse_expression()]
        return FunctionDef(name, params, body, is_method)

    def parse_block(self) -> List[ASTNode]:
        stmts = []
        if self.match(TT.INDENT):
            while self.current().type != TT.DEDENT and self.current().type != TT.EOF:
                self.skip_newlines()
                if self.current().type == TT.DEDENT:
                    break
                stmt = self.parse_statement()
                if stmt is not None:
                    stmts.append(stmt)
                self.skip_newlines()
            self.match(TT.DEDENT)
        else:
            # Single-line block
            stmt = self.parse_statement()
            if stmt is not None:
                stmts.append(stmt)
        return stmts

    # ── Expression parsing (precedence climbing) ──────────────────────────

    def parse_expression(self) -> ASTNode:
        return self.parse_ternary()

    def parse_ternary(self) -> ASTNode:
        expr = self.parse_or()
        if self.match(TT.QUESTION):
            true_val = self.parse_expression()
            self.expect(TT.COLON)
            false_val = self.parse_expression()
            return TernaryOp(expr, true_val, false_val)
        return expr

    def parse_or(self) -> ASTNode:
        left = self.parse_and()
        while self.match(TT.OR):
            right = self.parse_and()
            left = BinaryOp('or', left, right)
        return left

    def parse_and(self) -> ASTNode:
        left = self.parse_not()
        while self.match(TT.AND):
            right = self.parse_not()
            left = BinaryOp('and', left, right)
        return left

    def parse_not(self) -> ASTNode:
        if self.match(TT.NOT):
            return UnaryOp('not', self.parse_not())
        return self.parse_comparison()

    def parse_comparison(self) -> ASTNode:
        left = self.parse_addition()
        ops = {TT.EQ: '==', TT.NEQ: '!=', TT.LT: '<', TT.GT: '>',
               TT.LTE: '<=', TT.GTE: '>='}
        while self.current().type in ops:
            op = ops[self.advance().type]
            right = self.parse_addition()
            left = BinaryOp(op, left, right)
        return left

    def parse_addition(self) -> ASTNode:
        left = self.parse_multiplication()
        while self.current().type in (TT.PLUS, TT.MINUS):
            op = '+' if self.advance().type == TT.PLUS else '-'
            right = self.parse_multiplication()
            left = BinaryOp(op, left, right)
        return left

    def parse_multiplication(self) -> ASTNode:
        left = self.parse_unary()
        while self.current().type in (TT.STAR, TT.SLASH, TT.PERCENT):
            tok = self.advance()
            op = {TT.STAR: '*', TT.SLASH: '/', TT.PERCENT: '%'}[tok.type]
            right = self.parse_unary()
            left = BinaryOp(op, left, right)
        return left

    def parse_unary(self) -> ASTNode:
        if self.current().type == TT.MINUS:
            self.advance()
            return UnaryOp('-', self.parse_unary())
        if self.current().type == TT.PLUS:
            self.advance()
            return self.parse_unary()
        return self.parse_postfix()

    def parse_postfix(self) -> ASTNode:
        node = self.parse_primary()
        while True:
            if self.current().type == TT.DOT:
                self.advance()
                attr = self.expect(TT.IDENT).value
                # Check for method call: obj.method(args)
                if self.current().type == TT.LPAREN:
                    self.advance()
                    args, kwargs = self.parse_arg_list()
                    self.expect(TT.RPAREN)
                    node = MethodCall(node, attr, args, kwargs)
                else:
                    node = DotAccess(node, attr)
            elif self.current().type == TT.LBRACKET:
                self.advance()
                index = self.parse_expression()
                self.expect(TT.RBRACKET)
                node = IndexAccess(node, index)
            elif self.current().type == TT.LPAREN and isinstance(node, Identifier):
                self.advance()
                args, kwargs = self.parse_arg_list()
                self.expect(TT.RPAREN)
                node = FunctionCall(node.name, args, kwargs)
            else:
                break
        return node

    def parse_primary(self) -> ASTNode:
        tok = self.current()

        if tok.type == TT.INT:
            self.advance(); return NumberLiteral(float(tok.value))
        if tok.type == TT.FLOAT:
            self.advance(); return NumberLiteral(tok.value)
        if tok.type == TT.STRING:
            self.advance(); return StringLiteral(tok.value)
        if tok.type in (TT.TRUE, TT.FALSE):
            self.advance(); return BoolLiteral(tok.type == TT.TRUE)
        if tok.type == TT.NA:
            # If followed by '(', treat as function call na(x), not literal
            if self.pos + 1 < len(self.tokens) and self.tokens[self.pos + 1].type == TT.LPAREN:
                self.advance()
                return Identifier('na')
            self.advance(); return NALiteral()
        if tok.type == TT.COLOR:
            self.advance(); return StringLiteral(tok.value)  # treat color as string

        if tok.type == TT.IDENT:
            self.advance()
            return Identifier(tok.value)

        if tok.type == TT.LPAREN:
            self.advance()
            expr = self.parse_expression()
            self.expect(TT.RPAREN)
            return expr

        # Array literal: [expr, expr, ...]
        if tok.type == TT.LBRACKET:
            self.advance()
            elements = []
            while self.current().type != TT.RBRACKET and self.current().type != TT.EOF:
                elements.append(self.parse_expression())
                self.match(TT.COMMA)
            self.expect(TT.RBRACKET)
            return ArrayLiteral(elements)

        # Type cast expressions like int(x), float(x)
        if tok.type in (TT.SERIES, TT.SIMPLE):
            self.advance()
            return self.parse_postfix()

        self.advance()  # skip unknown
        return NALiteral()

    def parse_arg_list(self) -> Tuple[List[ASTNode], Dict[str, ASTNode]]:
        args = []
        kwargs = {}
        while self.current().type != TT.RPAREN and self.current().type != TT.EOF:
            # Check for kwarg: name = expr
            if (self.current().type == TT.IDENT and self.peek(1).type == TT.ASSIGN):
                name = self.advance().value
                self.advance()  # skip =
                val = self.parse_expression()
                kwargs[name] = val
            else:
                args.append(self.parse_expression())
            self.match(TT.COMMA)
        return args, kwargs


# ═══════════════════════════════════════════════════════════════════════════════
# INTERPRETER (bar-by-bar execution)
# ═══════════════════════════════════════════════════════════════════════════════

class PineInterpreter:
    """
    Executes a parsed Pine Script AST bar-by-bar on OHLCV data.
    Maintains proper variable scoping, var/varip persistence,
    and history operator ([N]) support.
    """

    def __init__(self, ast: Program, df: pd.DataFrame,
                 initial_capital: float = 10000,
                 commission_pct: float = 0.0,
                 default_qty: int = 1,
                 pyramiding: int = 1,
                 slippage: int = 0,
                 mintick: float = 0.01,
                 margin_long: int = 0,
                 margin_short: int = 0,
                 input_overrides: dict = None):
        self.ast = ast
        self.df = df.copy()
        self.n_bars = len(df)
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.commission_cash = 0.0  # cash per contract (set by strategy() declaration)
        self.default_qty = default_qty

        # State
        self.bar_index = 0
        self.globals = {}       # variable name -> current value
        self.var_inited = {}    # var declarations — True if already initialized
        self.history = {}       # variable name -> list of past values
        self.functions = {}     # user-defined functions
        self.trades = []
        self.warnings = []
        self.strategy_name = "Pine Strategy"
        self.strategy_params = {}

        # ── Input system ──────────────────────────────────────────────────
        self.input_defs = []       # list of input definitions for UI generation
        self.input_overrides = input_overrides or {}  # user-set values from UI

        # ── Position management ───────────────────────────────────────────
        self.position = 0          # 1=long, -1=short, 0=flat
        self.entry_price = 0.0
        self.entry_qty = default_qty
        self.entry_bar = 0
        self.open_qty = 0          # remaining qty (for partial exits)
        self.positions = []        # for pyramiding: list of {dir, price, qty, bar}

        # ── Strategy settings ─────────────────────────────────────────────
        self.pyramiding = pyramiding        # max entries in same direction
        self.slippage = slippage            # ticks of slippage per trade
        self.calc_on_every_tick = False
        self.process_orders_on_close = False

        # ── TP/SL from strategy.exit ──────────────────────────────────────
        self.exit_rules = {}       # from_entry -> {tp, sl, tp_pct, sl_pct, trail, qty_pct}

        # ── Pending orders from strategy.order() with limit/stop ─────────
        self.pending_orders = {}   # name -> {dir, qty, limit, stop, name}

        # ── Pending limit/stop orders from strategy.order() ──────────────
        self.pending_orders = {}   # name -> {dir, qty, limit, stop, comment}

        # ── request.security fallback values ─────────────────────────────
        # When a strategy calls request.security("VIX", ..., close), the
        # interpreter can't fetch cross-symbol data. Instead of returning
        # None (which breaks regime logic), return a sensible default.
        # Users can override via input_overrides with key "_security_<TICKER>"
        self.security_defaults = {
            'VIX': 16.0,     # Calm market default
            'VIX1!': 16.0,
            'CBOE:VIX': 16.0,
            'TVC:VIX': 16.0,
            'VXX': 20.0,
            'MOVE': 100.0,   # Bond volatility index
            'DXY': 104.0,    # Dollar index
            'US10Y': 4.3,    # 10yr yield
            'TNX': 43.0,
        }

        # ── syminfo ───────────────────────────────────────────────────────
        self.mintick = mintick
        self.margin_long = margin_long
        self.margin_short = margin_short


        self._call_counter = {}
        self._bar_call_reset = -1
        self._setup_builtins()


    def _call_key(self, prefix):
        """Generate a stable key for ta.* history buffers.
        Same call site produces same key across bars."""
        if self._bar_call_reset != self.bar_index:
            self._call_counter = {}
            self._bar_call_reset = self.bar_index
        count = self._call_counter.get(prefix, 0)
        self._call_counter[prefix] = count + 1
        return f"_{prefix}_{count}"

    def _setup_builtins(self):
        """Register all built-in Pine Script functions and variables."""
        self.builtins = {}

        # ── math.* ──
        self.builtins['math.abs'] = lambda args, kw: abs(self._num(args[0]))
        self.builtins['math.max'] = lambda args, kw: max(self._num(args[0]), self._num(args[1]))
        self.builtins['math.min'] = lambda args, kw: min(self._num(args[0]), self._num(args[1]))
        self.builtins['math.sqrt'] = lambda args, kw: np.sqrt(self._num(args[0]))
        self.builtins['math.pow'] = lambda args, kw: self._num(args[0]) ** self._num(args[1])
        self.builtins['math.log'] = lambda args, kw: np.log(self._num(args[0]))
        self.builtins['math.log10'] = lambda args, kw: np.log10(self._num(args[0]))
        self.builtins['math.ceil'] = lambda args, kw: int(np.ceil(self._num(args[0])))
        self.builtins['math.floor'] = lambda args, kw: int(np.floor(self._num(args[0])))
        self.builtins['math.round'] = lambda args, kw: round(self._num(args[0]), int(args[1]) if len(args) > 1 else 0)
        self.builtins['math.sign'] = lambda args, kw: np.sign(self._num(args[0]))
        self.builtins['math.avg'] = lambda args, kw: np.mean([self._num(a) for a in args])
        self.builtins['math.sum'] = lambda args, kw: self._ta_sum(args, kw)

        # ── nz, na, fixnan ──
        self.builtins['nz'] = lambda args, kw: args[0] if args[0] is not None and not self._is_na(args[0]) else (args[1] if len(args) > 1 else 0)
        self.builtins['na'] = lambda args, kw: self._is_na(args[0]) if args else None
        self.builtins['fixnan'] = lambda args, kw: args[0] if not self._is_na(args[0]) else self._prev(str(id(args)), args[0])

        # ── Type casts ──
        self.builtins['int'] = lambda args, kw: int(self._num(args[0])) if not self._is_na(args[0]) else None
        self.builtins['float'] = lambda args, kw: float(self._num(args[0])) if not self._is_na(args[0]) else None
        self.builtins['bool'] = lambda args, kw: bool(args[0])
        self.builtins['str.tostring'] = lambda args, kw: str(args[0])
        self.builtins['string'] = lambda args, kw: str(args[0]) if args else ""

        # ── ta.* indicators ──
        self.builtins['ta.sma'] = lambda args, kw: self._ta_sma(args, kw)
        self.builtins['ta.ema'] = lambda args, kw: self._ta_ema(args, kw)
        self.builtins['ta.wma'] = lambda args, kw: self._ta_wma(args, kw)
        self.builtins['ta.vwma'] = lambda args, kw: self._ta_vwma(args, kw)
        self.builtins['ta.hma'] = lambda args, kw: self._ta_hma(args, kw)
        self.builtins['ta.rma'] = lambda args, kw: self._ta_rma(args, kw)
        self.builtins['ta.rsi'] = lambda args, kw: self._ta_rsi(args, kw)
        self.builtins['ta.macd'] = lambda args, kw: self._ta_macd(args, kw)
        self.builtins['ta.bb'] = lambda args, kw: self._ta_bb(args, kw)
        self.builtins['ta.bbw'] = lambda args, kw: self._ta_bbw(args, kw)
        self.builtins['ta.atr'] = lambda args, kw: self._ta_atr(args, kw)
        self.builtins['ta.tr'] = lambda args, kw: self._ta_tr(args, kw)
        self.builtins['ta.stoch'] = lambda args, kw: self._ta_stoch(args, kw)
        self.builtins['ta.cci'] = lambda args, kw: self._ta_cci(args, kw)
        self.builtins['ta.mfi'] = lambda args, kw: self._ta_mfi(args, kw)
        self.builtins['ta.adx'] = lambda args, kw: self._ta_adx(args, kw)
        self.builtins['ta.dmi'] = lambda args, kw: self._ta_adx(args, kw)
        self.builtins['ta.supertrend'] = lambda args, kw: self._ta_supertrend(args, kw)
        self.builtins['ta.crossover'] = lambda args, kw: self._ta_crossover(args, kw)
        self.builtins['ta.crossunder'] = lambda args, kw: self._ta_crossunder(args, kw)
        self.builtins['ta.cross'] = lambda args, kw: self._ta_cross(args, kw)
        self.builtins['ta.highest'] = lambda args, kw: self._ta_highest(args, kw)
        self.builtins['ta.lowest'] = lambda args, kw: self._ta_lowest(args, kw)
        self.builtins['ta.highestbars'] = lambda args, kw: self._ta_highestbars(args, kw)
        self.builtins['ta.lowestbars'] = lambda args, kw: self._ta_lowestbars(args, kw)
        self.builtins['ta.change'] = lambda args, kw: self._ta_change(args, kw)
        self.builtins['ta.mom'] = lambda args, kw: self._ta_change(args, kw)
        self.builtins['ta.roc'] = lambda args, kw: self._ta_roc(args, kw)
        self.builtins['ta.valuewhen'] = lambda args, kw: self._ta_valuewhen(args, kw)
        self.builtins['ta.barssince'] = lambda args, kw: self._ta_barssince(args, kw)
        self.builtins['ta.cum'] = lambda args, kw: self._ta_cum(args, kw)
        self.builtins['ta.rising'] = lambda args, kw: self._ta_rising(args, kw)
        self.builtins['ta.falling'] = lambda args, kw: self._ta_falling(args, kw)
        self.builtins['ta.pivothigh'] = lambda args, kw: self._ta_pivothigh(args, kw)
        self.builtins['ta.pivotlow'] = lambda args, kw: self._ta_pivotlow(args, kw)
        self.builtins['ta.percentrank'] = lambda args, kw: self._ta_percentrank(args, kw)
        self.builtins['ta.percentile_nearest_rank'] = lambda args, kw: self._ta_percentile(args, kw)
        self.builtins['ta.correlation'] = lambda args, kw: self._ta_correlation(args, kw)
        self.builtins['ta.dev'] = lambda args, kw: self._ta_dev(args, kw)
        self.builtins['ta.stdev'] = lambda args, kw: self._ta_dev(args, kw)
        self.builtins['ta.variance'] = lambda args, kw: self._ta_variance(args, kw)
        self.builtins['ta.median'] = lambda args, kw: self._ta_median(args, kw)
        self.builtins['ta.mode'] = lambda args, kw: self._ta_mode(args, kw)
        self.builtins['ta.linreg'] = lambda args, kw: self._ta_linreg(args, kw)
        self.builtins['ta.swma'] = lambda args, kw: self._ta_swma(args, kw)
        self.builtins['ta.alma'] = lambda args, kw: self._ta_alma(args, kw)
        self.builtins['ta.vwap'] = lambda args, kw: self._ta_vwap(args, kw)
        self.builtins['ta.obv'] = lambda args, kw: self._ta_obv(args, kw)

        # ── strategy.* ──
        self.builtins['strategy'] = lambda args, kw: self._strategy_decl(args, kw)
        self.builtins['strategy.entry'] = lambda args, kw: self._strategy_entry(args, kw)
        self.builtins['strategy.close'] = lambda args, kw: self._strategy_close(args, kw)
        self.builtins['strategy.close_all'] = lambda args, kw: self._strategy_close(["all"], kw)
        self.builtins['strategy.exit'] = lambda args, kw: self._strategy_exit(args, kw)
        self.builtins['strategy.order'] = lambda args, kw: self._strategy_order(args, kw)
        self.builtins['strategy.cancel'] = lambda args, kw: self._strategy_cancel(args, kw)
        self.builtins['strategy.cancel_all'] = lambda args, kw: self._strategy_cancel_all(args, kw)

        # ── input.* ──
        self.builtins['input'] = lambda args, kw: self._input(args, kw)
        self.builtins['input.int'] = lambda args, kw: self._input(args, kw, cast=int)
        self.builtins['input.float'] = lambda args, kw: self._input(args, kw, cast=float)
        self.builtins['input.bool'] = lambda args, kw: self._input(args, kw, cast=bool)
        self.builtins['input.string'] = lambda args, kw: self._input(args, kw, cast=str)
        self.builtins['input.source'] = lambda args, kw: self._input_source(args, kw)
        self.builtins['input.timeframe'] = lambda args, kw: self._input(args, kw, cast=str)
        self.builtins['input.symbol'] = lambda args, kw: self._input(args, kw, cast=str)
        self.builtins['input.color'] = lambda args, kw: self._input(args, kw, cast=str)

        # ── v3/v4/v5 compatibility — bare function names without ta. prefix ──
        for _fn in ('sma', 'ema', 'wma', 'vwma', 'hma', 'rma', 'rsi', 'macd', 'bb', 'bbw',
                     'atr', 'tr', 'stoch', 'cci', 'mfi', 'adx', 'dmi', 'supertrend',
                     'crossover', 'crossunder', 'cross',
                     'highest', 'lowest', 'highestbars', 'lowestbars',
                     'change', 'mom', 'roc', 'valuewhen', 'barssince', 'cum',
                     'rising', 'falling', 'pivothigh', 'pivotlow',
                     'percentrank', 'percentile_nearest_rank', 'correlation',
                     'dev', 'stdev', 'variance', 'median', 'mode', 'linreg',
                     'swma', 'alma', 'vwap', 'obv'):
            _ta_key = f'ta.{_fn}'
            if _ta_key in self.builtins:
                self.builtins[_fn] = self.builtins[_ta_key]

        # ── color.* / plot / drawing — no-ops ──
        for fn in ('plot', 'plotshape', 'plotchar', 'plotarrow', 'plotcandle',
                    'plotbar', 'bgcolor', 'barcolor', 'fill', 'hline',
                    'label.new', 'label.delete', 'label.set_xy',
                    'line.new', 'line.delete', 'line.set_xy1', 'line.set_xy2',
                    'box.new', 'box.delete', 'box.set_lefttop', 'box.set_rightbottom',
                    'table.new', 'table.cell', 'table.delete',
                    'alert', 'alertcondition', 'runtime.error',
                    'log.info', 'log.warning', 'log.error',
                    'color.new', 'color.rgb', 'color.from_gradient',
                    'request.financial', 'request.quandl',
                    'request.dividends', 'request.splits', 'request.earnings'):
            self.builtins[fn] = lambda args, kw, _fn=fn: None

        # ── request.security — return fallback value instead of None ──
        # Cross-symbol data isn't available, but returning None breaks
        # strategies that use VIX/other tickers for regime detection.
        # Strategy: return the expression value if it's an OHLCV reference
        # (same-symbol approximation), otherwise return a configurable default.
        self.builtins['request.security'] = lambda args, kw: self._request_security(args, kw)

        # ── array.* ──#JUST TRADES CONFIDENTIAL PROPERTY
        self.builtins['array.new_float'] = lambda args, kw: [self._num(args[1]) if len(args) > 1 else 0.0] * (int(self._num(args[0])) if args else 0)
        self.builtins['array.new_int'] = lambda args, kw: [int(self._num(args[1])) if len(args) > 1 else 0] * (int(self._num(args[0])) if args else 0)
        self.builtins['array.new_bool'] = lambda args, kw: [bool(args[1]) if len(args) > 1 else False] * (int(self._num(args[0])) if args else 0)
        self.builtins['array.new_string'] = lambda args, kw: [str(args[1]) if len(args) > 1 else ""] * (int(self._num(args[0])) if args else 0)
        self.builtins['array.new'] = self.builtins['array.new_float']
        self.builtins['array.size'] = lambda args, kw: len(args[0]) if isinstance(args[0], list) else 0
        self.builtins['array.get'] = lambda args, kw: args[0][int(self._num(args[1]))] if isinstance(args[0], list) and int(self._num(args[1])) < len(args[0]) else None
        self.builtins['array.set'] = lambda args, kw: self._array_set(args)
        self.builtins['array.push'] = lambda args, kw: args[0].append(args[1]) if isinstance(args[0], list) else None
        self.builtins['array.pop'] = lambda args, kw: args[0].pop() if isinstance(args[0], list) and args[0] else None
        self.builtins['array.remove'] = lambda args, kw: args[0].pop(int(self._num(args[1]))) if isinstance(args[0], list) else None
        self.builtins['array.insert'] = lambda args, kw: args[0].insert(int(self._num(args[1])), args[2]) if isinstance(args[0], list) else None
        self.builtins['array.clear'] = lambda args, kw: args[0].clear() if isinstance(args[0], list) else None
        self.builtins['array.sort'] = lambda args, kw: args[0].sort() if isinstance(args[0], list) else None
        self.builtins['array.reverse'] = lambda args, kw: args[0].reverse() if isinstance(args[0], list) else None
        self.builtins['array.slice'] = lambda args, kw: args[0][int(self._num(args[1])):int(self._num(args[2]))] if isinstance(args[0], list) else []
        self.builtins['array.includes'] = lambda args, kw: args[1] in args[0] if isinstance(args[0], list) else False
        self.builtins['array.indexof'] = lambda args, kw: args[0].index(args[1]) if isinstance(args[0], list) and args[1] in args[0] else -1
        self.builtins['array.max'] = lambda args, kw: max(args[0]) if isinstance(args[0], list) and args[0] else None
        self.builtins['array.min'] = lambda args, kw: min(args[0]) if isinstance(args[0], list) and args[0] else None
        self.builtins['array.avg'] = lambda args, kw: np.mean(args[0]) if isinstance(args[0], list) and args[0] else None
        self.builtins['array.sum'] = lambda args, kw: sum(args[0]) if isinstance(args[0], list) else 0
        self.builtins['array.stdev'] = lambda args, kw: np.std(args[0]) if isinstance(args[0], list) and len(args[0]) > 1 else 0
        self.builtins['array.median'] = lambda args, kw: np.median(args[0]) if isinstance(args[0], list) and args[0] else None
        self.builtins['array.copy'] = lambda args, kw: list(args[0]) if isinstance(args[0], list) else []
        self.builtins['array.concat'] = lambda args, kw: (args[0].extend(args[1]) if isinstance(args[0], list) and isinstance(args[1], list) else None) or args[0]
        self.builtins['array.from'] = lambda args, kw: list(args)
        self.builtins['array.fill'] = lambda args, kw: self._array_fill(args)
        self.builtins['array.join'] = lambda args, kw: str(args[1] if len(args) > 1 else ",").join(str(x) for x in args[0]) if isinstance(args[0], list) else ""

        # str.* basics
        self.builtins['str.contains'] = lambda args, kw: str(args[1]) in str(args[0]) if len(args) >= 2 else False
        self.builtins['str.length'] = lambda args, kw: len(str(args[0])) if args else 0
        self.builtins['str.tonumber'] = lambda args, kw: float(args[0]) if args else None
        self.builtins['str.format'] = lambda args, kw: str(args[0]) if args else ""
        self.builtins['str.tostring'] = lambda args, kw: str(args[0]) if args else ""
        self.builtins['str.substring'] = lambda args, kw: str(args[0])[int(self._num(args[1])):int(self._num(args[2]))] if len(args) >= 3 else str(args[0])
        self.builtins['str.replace'] = lambda args, kw: str(args[0]).replace(str(args[1]), str(args[2])) if len(args) >= 3 else str(args[0])
        self.builtins['str.lower'] = lambda args, kw: str(args[0]).lower() if args else ""
        self.builtins['str.upper'] = lambda args, kw: str(args[0]).upper() if args else ""
        self.builtins['str.startswith'] = lambda args, kw: str(args[0]).startswith(str(args[1])) if len(args) >= 2 else False
        self.builtins['str.endswith'] = lambda args, kw: str(args[0]).endswith(str(args[1])) if len(args) >= 2 else False
        self.builtins['str.trim'] = lambda args, kw: str(args[0]).strip() if args else ""
        self.builtins['str.split'] = lambda args, kw: str(args[0]).split(str(args[1])) if len(args) >= 2 else [str(args[0])]
        self.builtins['str.pos'] = lambda args, kw: str(args[0]).find(str(args[1])) if len(args) >= 2 else -1

        # time / session functions
        self.builtins['time'] = lambda args, kw: self._time_func(args, kw)
        self.builtins['time_close'] = lambda args, kw: self._time_func(args, kw)
        self.builtins['hour'] = lambda args, kw: self._hour_func(args, kw)
        self.builtins['minute'] = lambda args, kw: self._minute_func(args, kw)
        self.builtins['second'] = lambda args, kw: 0
        self.builtins['year'] = lambda args, kw: self._year_func(args, kw)
        self.builtins['month'] = lambda args, kw: self._month_func(args, kw)
        self.builtins['dayofmonth'] = lambda args, kw: self._dom_func(args, kw)
        self.builtins['timestamp'] = lambda args, kw: self._timestamp_func(args, kw)

    def _array_set(self, args):
        if isinstance(args[0], list) and int(self._num(args[1])) < len(args[0]):
            args[0][int(self._num(args[1]))] = args[2]

    def _array_fill(self, args):
        if isinstance(args[0], list):
            val = args[1] if len(args) > 1 else 0
            for i in range(len(args[0])):
                args[0][i] = val

    # ── Helper methods ────────────────────────────────────────────────────

    def _num(self, v):
        if v is None: return 0.0
        if isinstance(v, bool): return 1.0 if v else 0.0
        try: return float(v)
        except (TypeError, ValueError): return 0.0

    def _is_na(self, v):
        if v is None: return True
        if isinstance(v, float) and np.isnan(v): return True
        return False

    def _get_history(self, name, offset=0):
        hist = self.history.get(name, [])
        idx = len(hist) - 1 - offset
        if idx < 0 or idx >= len(hist):
            return None
        return hist[idx]

    def _get_ohlcv(self, name):
        """Get OHLCV value for current bar — reads from pre-set globals (fast)."""
        if name in self.globals:
            return self.globals[name]
        if name == 'hlcc4':
            return (self.globals.get('high', 0) + self.globals.get('low', 0) + self.globals.get('close', 0) * 2) / 4
        return None

    def _get_source_history(self, source_name, length):
        """Get last `length` values of a source from history."""
        hist = self.history.get(source_name, [])
        if len(hist) < length:
            return None
        return hist[-length:]

    # ── ta.* implementations (bar-by-bar with history) ────────────────────

    def _vec_get(self, key, source_arr, length, compute_fn):
        """Get precomputed vectorized indicator value for current bar."""
        if key not in self._vec_cache:
            self._vec_cache[key] = compute_fn(source_arr, length)
        arr = self._vec_cache[key]
        i = self.bar_index
        if i < len(arr):
            v = arr[i]
            return None if np.isnan(v) else float(v)
        return None

    def _vec_source(self, source_val):
        """Map a source value to the correct precomputed array."""
        if isinstance(source_val, str):
            m = {'close': self._vec_close, 'open': self._vec_open,
                 'high': self._vec_high, 'low': self._vec_low, 'volume': self._vec_volume}
            return m.get(source_val, self._vec_close)
        # If it's a number, it's the current bar's value — need to use history-based approach
        return None

    def _ta_sma(self, args, kw):
        src_val = args[0]; length = int(self._num(args[1] if len(args) > 1 else kw.get('length', 14)))
        # Try vectorized path
        if hasattr(self, '_vec_cache'):
            key = self._call_key("sma")
            if key not in self._vec_cache:
                # Check if source is a simple OHLCV reference
                src_key = key + "_src"
                if src_key not in self.history: self.history[src_key] = []
                self.history[src_key].append(self._num(src_val))
                vals = self.history[src_key]
                if len(vals) < length: return None
                return float(np.mean(vals[-length:]))
            return self._vec_get(key, None, length, lambda s, l: None)
        # Fallback
        src_key = self._call_key("src")
        if src_key not in self.history: self.history[src_key] = []
        self.history[src_key].append(self._num(src_val))
        vals = self.history[src_key]
        if len(vals) < length: return None
        return float(np.mean(vals[-length:]))

    def _ta_ema(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else kw.get('length', 14)))
        key = self._call_key("ema")
        prev = self.globals.get(key)
        if prev is None or self._is_na(prev):
            # Initialize with SMA
            buf_key = self._call_key("ema_buf")
            if buf_key not in self.history: self.history[buf_key] = []
            self.history[buf_key].append(source)
            if len(self.history[buf_key]) >= length:
                result = np.mean(self.history[buf_key][-length:])
                self.globals[key] = result
                return result
            return None
        alpha = 2.0 / (length + 1)
        result = alpha * source + (1 - alpha) * prev
        self.globals[key] = result
        return result

    def _ta_rma(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else kw.get('length', 14)))
        key = self._call_key("rma")
        prev = self.globals.get(key)
        if prev is None or self._is_na(prev):
            buf_key = self._call_key("rma_buf")
            if buf_key not in self.history: self.history[buf_key] = []
            self.history[buf_key].append(source)
            if len(self.history[buf_key]) >= length:
                result = np.mean(self.history[buf_key][-length:])
                self.globals[key] = result
                return result
            return None
        alpha = 1.0 / length
        result = alpha * source + (1 - alpha) * prev
        self.globals[key] = result
        return result

    def _ta_wma(self, args, kw):
        source = args[0]; length = int(self._num(args[1] if len(args) > 1 else 14))
        key = self._call_key("wma")
        if key not in self.history: self.history[key] = []
        self.history[key].append(self._num(source))
        vals = self.history[key]
        if len(vals) < length: return None
        w = np.arange(1, length+1, dtype=float)
        return np.dot(vals[-length:], w) / w.sum()

    def _ta_vwma(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else 14))
        vol = self._get_ohlcv('volume') or 1.0
        key = self._call_key("vwma_s"); vkey = self._call_key("vwma_v")
        if key not in self.history: self.history[key] = []; self.history[vkey] = []
        self.history[key].append(source * vol); self.history[vkey].append(vol)
        if len(self.history[key]) < length: return None
        return sum(self.history[key][-length:]) / sum(self.history[vkey][-length:])

    def _ta_hma(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else 14))
        key = self._call_key("hma")
        if key not in self.history: self.history[key] = []
        self.history[key].append(source)
        vals = self.history[key]
        n = length
        if len(vals) < n: return None
        s = pd.Series(vals[-max(n, int(np.sqrt(n))+n):])
        half = s.rolling(n//2).apply(lambda x: np.dot(x, np.arange(1,n//2+1))/np.arange(1,n//2+1).sum(), raw=True)
        full = s.rolling(n).apply(lambda x: np.dot(x, np.arange(1,n+1))/np.arange(1,n+1).sum(), raw=True)
        diff = 2*half - full
        sq = int(np.sqrt(n))
        hma = diff.rolling(sq).apply(lambda x: np.dot(x, np.arange(1,sq+1))/np.arange(1,sq+1).sum(), raw=True)
        return hma.iloc[-1] if not np.isnan(hma.iloc[-1]) else None

    def _ta_rsi(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else 14))
        key = self._call_key("rsi")
        buf_key = self._call_key("rsi_buf")
        if buf_key not in self.history: self.history[buf_key] = []
        self.history[buf_key].append(source)
        vals = self.history[buf_key]
        if len(vals) < length + 1: return None
        deltas = np.diff(vals[-(length+1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        prev = self.globals.get(key)
        if prev is None:
            avg_gain = np.mean(gains); avg_loss = np.mean(losses)
        else:
            prev_ag, prev_al = prev
            avg_gain = (prev_ag * (length-1) + gains[-1]) / length
            avg_loss = (prev_al * (length-1) + losses[-1]) / length
        self.globals[key] = (avg_gain, avg_loss)
        if avg_loss == 0: return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1.0 + rs)

    def _ta_macd(self, args, kw):
        source = self._num(args[0])
        fast_len = int(self._num(args[1] if len(args) > 1 else kw.get('fastlen', 12)))
        slow_len = int(self._num(args[2] if len(args) > 2 else kw.get('slowlen', 26)))
        sig_len = int(self._num(args[3] if len(args) > 3 else kw.get('siglen', 9)))
        # Need to compute EMA of source at two lengths, then EMA of diff
        buf = self._call_key("macd_buf")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(source)
        vals = self.history[buf]
        if len(vals) < slow_len: return (None, None, None)
        s = pd.Series(vals)
        fast_ema = s.ewm(span=fast_len, adjust=False).mean().iloc[-1]
        slow_ema = s.ewm(span=slow_len, adjust=False).mean().iloc[-1]
        macd_line = fast_ema - slow_ema
        # Signal line needs macd history
        macd_buf = self._call_key("macd_line")
        if macd_buf not in self.history: self.history[macd_buf] = []
        self.history[macd_buf].append(macd_line)
        if len(self.history[macd_buf]) >= sig_len:
            signal = pd.Series(self.history[macd_buf]).ewm(span=sig_len, adjust=False).mean().iloc[-1]
        else:
            signal = None
        hist = (macd_line - signal) if signal is not None else None
        return (macd_line, signal, hist)

    def _ta_bb(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else 20))
        mult = self._num(args[2] if len(args) > 2 else kw.get('mult', 2.0))
        buf = self._call_key("bb")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(source)
        if len(self.history[buf]) < length: return (None, None, None)
        vals = self.history[buf][-length:]
        basis = np.mean(vals); dev = np.std(vals)
        return (basis + mult*dev, basis, basis - mult*dev)

    def _ta_bbw(self, args, kw):
        result = self._ta_bb(args, kw)
        if result[0] is None: return None
        upper, basis, lower = result
        return (upper - lower) / basis if basis != 0 else 0

    def _ta_atr(self, args, kw):
        length = int(self._num(args[0] if args else kw.get('length', 14)))
        h = self._get_ohlcv('high'); l = self._get_ohlcv('low'); c_prev = self._get_history('close', 1)
        if c_prev is None: c_prev = self._get_ohlcv('close')
        tr = max(h-l, abs(h-c_prev), abs(l-c_prev))
        buf = self._call_key("atr")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(tr)
        prev = self.globals.get(buf)
        if prev is None:
            if len(self.history[buf]) < length: return None
            result = np.mean(self.history[buf][-length:])
        else:
            result = (prev * (length-1) + tr) / length
        self.globals[buf] = result
        return result

    def _ta_tr(self, args, kw):
        h = self._get_ohlcv('high'); l = self._get_ohlcv('low')
        c_prev = self._get_history('close', 1)
        if c_prev is None: c_prev = self._get_ohlcv('close')
        return max(h-l, abs(h-c_prev), abs(l-c_prev))

    def _ta_stoch(self, args, kw):
        high = self._num(args[0]); low = self._num(args[1]); close = self._num(args[2])
        k_len = int(self._num(args[3] if len(args) > 3 else 14))
        hbuf = self._call_key("stoch_h"); lbuf = self._call_key("stoch_l")
        if hbuf not in self.history: self.history[hbuf] = []; self.history[lbuf] = []
        self.history[hbuf].append(high); self.history[lbuf].append(low)
        if len(self.history[hbuf]) < k_len: return None
        hh = max(self.history[hbuf][-k_len:]); ll = min(self.history[lbuf][-k_len:])
        if hh == ll: return 50.0
        return 100.0 * (close - ll) / (hh - ll)

    def _ta_cci(self, args, kw):
        length = int(self._num(args[0] if args else 20))
        tp = (self._get_ohlcv('high') + self._get_ohlcv('low') + self._get_ohlcv('close')) / 3
        buf = self._call_key("cci")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(tp)
        if len(self.history[buf]) < length: return None
        vals = self.history[buf][-length:]
        mean = np.mean(vals); mad = np.mean(np.abs(np.array(vals) - mean))
        if mad == 0: return 0
        return (tp - mean) / (0.015 * mad)

    def _ta_mfi(self, args, kw):
        length = int(self._num(args[1] if len(args) > 1 else args[0] if args else 14))
        tp = (self._get_ohlcv('high') + self._get_ohlcv('low') + self._get_ohlcv('close')) / 3
        vol = self._get_ohlcv('volume') or 1.0
        mf = tp * vol
        buf = self._call_key("mfi"); tp_buf = self._call_key("mfi_tp")
        if buf not in self.history: self.history[buf] = []; self.history[tp_buf] = []
        self.history[buf].append(mf); self.history[tp_buf].append(tp)
        if len(self.history[buf]) < length + 1: return None
        pos = sum(self.history[buf][-(length):][i] for i in range(length) if self.history[tp_buf][-(length):][i] > self.history[tp_buf][-(length+1):-1][i])
        neg = sum(self.history[buf][-(length):][i] for i in range(length) if self.history[tp_buf][-(length):][i] <= self.history[tp_buf][-(length+1):-1][i])
        if neg == 0: return 100.0
        return 100.0 - 100.0 / (1.0 + pos/neg)

    def _ta_adx(self, args, kw):
        """Returns (plus_di, minus_di, adx) tuple — matches ta.dmi() in Pine Script."""
        di_len = int(self._num(args[0] if args else 14))
        adx_smooth = int(self._num(args[1] if len(args) > 1 else di_len))
        h = self._get_ohlcv('high'); l = self._get_ohlcv('low')
        prev_h = self._get_history('high', 1) or h; prev_l = self._get_history('low', 1) or l
        up_move = h - prev_h; down_move = prev_l - l
        plus_dm = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm = down_move if down_move > up_move and down_move > 0 else 0

        # Use RMA for smoothing (like TradingView)
        buf_p = self._call_key("dmi_p"); buf_m = self._call_key("dmi_m")
        buf_atr = self._call_key("dmi_atr"); buf_dx = self._call_key("dmi_dx")
        for b in (buf_p, buf_m, buf_atr, buf_dx):
            if b not in self.history: self.history[b] = []

        # True Range for ATR
        c_prev = self._get_history('close', 1) or self._get_ohlcv('close')
        tr = max(h - l, abs(h - c_prev), abs(l - c_prev))

        self.history[buf_p].append(plus_dm)
        self.history[buf_m].append(minus_dm)
        self.history[buf_atr].append(tr)

        if len(self.history[buf_p]) < di_len:
            return (None, None, None)

        # RMA smoothing
        prev_state = self.globals.get(buf_p + '_rma')
        if prev_state is None:
            sm_plus = np.mean(self.history[buf_p][-di_len:])
            sm_minus = np.mean(self.history[buf_m][-di_len:])
            sm_atr = np.mean(self.history[buf_atr][-di_len:])
        else:
            sm_plus, sm_minus, sm_atr = prev_state
            alpha = 1.0 / di_len
            sm_plus = alpha * plus_dm + (1 - alpha) * sm_plus
            sm_minus = alpha * minus_dm + (1 - alpha) * sm_minus
            sm_atr = alpha * tr + (1 - alpha) * sm_atr

        self.globals[buf_p + '_rma'] = (sm_plus, sm_minus, sm_atr)

        plus_di = 100 * sm_plus / sm_atr if sm_atr > 0 else 0
        minus_di = 100 * sm_minus / sm_atr if sm_atr > 0 else 0
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) > 0 else 0

        self.history[buf_dx].append(dx)

        # Smooth ADX
        prev_adx = self.globals.get(buf_dx + '_rma')
        if prev_adx is None:
            if len(self.history[buf_dx]) < adx_smooth:
                return (plus_di, minus_di, None)
            adx = np.mean(self.history[buf_dx][-adx_smooth:])
        else:
            alpha = 1.0 / adx_smooth
            adx = alpha * dx + (1 - alpha) * prev_adx
        self.globals[buf_dx + '_rma'] = adx

        return (plus_di, minus_di, adx)

    def _ta_supertrend(self, args, kw):
        factor = self._num(args[0] if args else 3.0); length = int(self._num(args[1] if len(args) > 1 else 10))
        atr_val = self._ta_atr([length], kw) or 0
        hl2 = (self._get_ohlcv('high') + self._get_ohlcv('low')) / 2
        up = hl2 + factor * atr_val; dn = hl2 - factor * atr_val
        close = self._get_ohlcv('close')
        key = self._call_key("st")
        prev = self.globals.get(key, {'dir': 1, 'up': up, 'dn': dn})
        if prev['dir'] == 1:
            dn = max(dn, prev['dn']) if close > prev['dn'] else dn
            direction = 1 if close >= dn else -1
        else:
            up = min(up, prev['up']) if close < prev['up'] else up
            direction = -1 if close <= up else 1
        self.globals[key] = {'dir': direction, 'up': up, 'dn': dn}
        st_val = dn if direction == 1 else up
        return (st_val, direction)

    def _ta_crossover(self, args, kw):
        a = self._num(args[0]); b = self._num(args[1])
        key_a = self._call_key("co_a"); key_b = self._call_key("co_b")
        if key_a not in self.history: self.history[key_a] = []; self.history[key_b] = []
        # Read previous values BEFORE appending current
        a_prev = self.history[key_a][-1] if self.history[key_a] else None
        b_prev = self.history[key_b][-1] if self.history[key_b] else None
        self.history[key_a].append(a); self.history[key_b].append(b)
        if a_prev is None or b_prev is None: return False
        return a > b and a_prev <= b_prev

    def _ta_crossunder(self, args, kw):
        a = self._num(args[0]); b = self._num(args[1])
        key_a = self._call_key("cu_a"); key_b = self._call_key("cu_b")
        if key_a not in self.history: self.history[key_a] = []; self.history[key_b] = []
        # Read previous values BEFORE appending current
        a_prev = self.history[key_a][-1] if self.history[key_a] else None
        b_prev = self.history[key_b][-1] if self.history[key_b] else None
        self.history[key_a].append(a); self.history[key_b].append(b)
        if a_prev is None or b_prev is None: return False
        return a < b and a_prev >= b_prev

    def _ta_cross(self, args, kw):
        a = self._num(args[0]); b = self._num(args[1])
        key_a = self._call_key("cx_a"); key_b = self._call_key("cx_b")
        if key_a not in self.history: self.history[key_a] = []; self.history[key_b] = []
        a_prev = self.history[key_a][-1] if self.history[key_a] else None
        b_prev = self.history[key_b][-1] if self.history[key_b] else None
        self.history[key_a].append(a); self.history[key_b].append(b)
        if a_prev is None or b_prev is None: return False
        return (a > b and a_prev <= b_prev) or (a < b and a_prev >= b_prev)

    def _ta_highest(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else 14))
        buf = self._call_key("highest")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(source)
        if len(self.history[buf]) < length: return None
        return max(self.history[buf][-length:])

    def _ta_lowest(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else 14))
        buf = self._call_key("lowest")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(source)
        if len(self.history[buf]) < length: return None
        return min(self.history[buf][-length:])

    def _ta_highestbars(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else 14))
        buf = self._call_key("hbars")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(source)
        if len(self.history[buf]) < length: return None
        vals = self.history[buf][-length:]
        return -(length - 1 - np.argmax(vals))

    def _ta_lowestbars(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else 14))
        buf = self._call_key("lbars")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(source)
        if len(self.history[buf]) < length: return None
        vals = self.history[buf][-length:]
        return -(length - 1 - np.argmin(vals))

    def _ta_change(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else 1))
        buf = self._call_key("change")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(source)
        if len(self.history[buf]) <= length: return None
        return source - self.history[buf][-(length+1)]

    def _ta_roc(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else 9))
        buf = self._call_key("roc")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(source)
        if len(self.history[buf]) <= length: return None
        prev = self.history[buf][-(length+1)]
        return 100 * (source - prev) / prev if prev != 0 else 0

    def _ta_sum(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else 14))
        buf = self._call_key("sum")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(source)
        if len(self.history[buf]) < length: return None
        return sum(self.history[buf][-length:])

    def _ta_valuewhen(self, args, kw):
        cond = bool(args[0]); source = args[1]; occur = int(self._num(args[2])) if len(args) > 2 else 0
        buf = self._call_key("vw")
        if buf not in self.history: self.history[buf] = []
        if cond: self.history[buf].append(source)
        idx = -(occur + 1)
        return self.history[buf][idx] if abs(idx) <= len(self.history[buf]) else None

    def _ta_barssince(self, args, kw):
        cond = bool(args[0])
        buf = self._call_key("bs")
        if cond: self.globals[buf] = 0
        elif buf in self.globals: self.globals[buf] += 1
        return self.globals.get(buf)

    def _ta_cum(self, args, kw):
        source = self._num(args[0])
        buf = self._call_key("cum")
        self.globals[buf] = self.globals.get(buf, 0) + source
        return self.globals[buf]

    def _ta_rising(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else 1))
        buf = self._call_key("rising")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(source)
        if len(self.history[buf]) <= length: return False
        return all(self.history[buf][-(i+1)] > self.history[buf][-(i+2)] for i in range(length))

    def _ta_falling(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else 1))
        buf = self._call_key("falling")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(source)
        if len(self.history[buf]) <= length: return False
        return all(self.history[buf][-(i+1)] < self.history[buf][-(i+2)] for i in range(length))

    def _ta_pivothigh(self, args, kw):
        source = self._num(args[0]) if len(args) > 2 else self._get_ohlcv('high')
        lb = int(self._num(args[-2] if len(args) >= 2 else 5))
        rb = int(self._num(args[-1] if len(args) >= 1 else 5))
        buf = self._call_key("ph")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(source)
        if len(self.history[buf]) < lb + rb + 1: return None
        idx = -(rb + 1)
        pivot = self.history[buf][idx]
        left = self.history[buf][idx-lb:idx]
        right = self.history[buf][idx+1:] if rb > 0 else []
        if all(pivot >= v for v in left) and all(pivot >= v for v in right): return pivot
        return None

    def _ta_pivotlow(self, args, kw):
        source = self._num(args[0]) if len(args) > 2 else self._get_ohlcv('low')
        lb = int(self._num(args[-2] if len(args) >= 2 else 5))
        rb = int(self._num(args[-1] if len(args) >= 1 else 5))
        buf = self._call_key("pl")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(source)
        if len(self.history[buf]) < lb + rb + 1: return None
        idx = -(rb + 1)
        pivot = self.history[buf][idx]
        left = self.history[buf][idx-lb:idx]
        right = self.history[buf][idx+1:] if rb > 0 else []
        if all(pivot <= v for v in left) and all(pivot <= v for v in right): return pivot
        return None

    def _ta_percentrank(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else 14))
        buf = self._call_key("pr")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(source)
        if len(self.history[buf]) < length: return None
        vals = self.history[buf][-length:]
        return 100 * sum(1 for v in vals if v <= source) / length

    def _ta_percentile(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else 14))
        pct = self._num(args[2] if len(args) > 2 else 50)
        buf = self._call_key("pctl")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(source)
        if len(self.history[buf]) < length: return None
        return np.percentile(self.history[buf][-length:], pct)

    def _ta_correlation(self, args, kw):
        s1 = self._num(args[0]); s2 = self._num(args[1])
        length = int(self._num(args[2] if len(args) > 2 else 14))
        b1 = self._call_key("corr1"); b2 = self._call_key("corr2")
        if b1 not in self.history: self.history[b1] = []; self.history[b2] = []
        self.history[b1].append(s1); self.history[b2].append(s2)
        if len(self.history[b1]) < length: return None
        return np.corrcoef(self.history[b1][-length:], self.history[b2][-length:])[0,1]

    def _ta_dev(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else 14))
        buf = self._call_key("dev")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(source)
        if len(self.history[buf]) < length: return None
        return np.std(self.history[buf][-length:])

    def _ta_variance(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else 14))
        buf = self._call_key("var")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(source)
        if len(self.history[buf]) < length: return None
        return np.var(self.history[buf][-length:])

    def _ta_median(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else 14))
        buf = self._call_key("med")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(source)
        if len(self.history[buf]) < length: return None
        return np.median(self.history[buf][-length:])

    def _ta_mode(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else 14))
        buf = self._call_key("mode")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(round(source, 4))
        if len(self.history[buf]) < length: return None
        from collections import Counter
        c = Counter(self.history[buf][-length:])
        return c.most_common(1)[0][0]

    def _ta_linreg(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else 14))
        offset = int(self._num(args[2] if len(args) > 2 else 0))
        buf = self._call_key("lr")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(source)
        if len(self.history[buf]) < length: return None
        y = np.array(self.history[buf][-length:]); x = np.arange(length)
        slope, intercept = np.polyfit(x, y, 1)
        return intercept + slope * (length - 1 - offset)

    def _ta_swma(self, args, kw):
        source = self._num(args[0])
        buf = self._call_key("swma")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(source)
        if len(self.history[buf]) < 4: return None
        v = self.history[buf][-4:]
        return (v[0]*1 + v[1]*2 + v[2]*2 + v[3]*1) / 6

    def _ta_alma(self, args, kw):
        source = self._num(args[0]); length = int(self._num(args[1] if len(args) > 1 else 9))
        offset = self._num(args[2] if len(args) > 2 else 0.85)
        sigma = self._num(args[3] if len(args) > 3 else 6)
        buf = self._call_key("alma")
        if buf not in self.history: self.history[buf] = []
        self.history[buf].append(source)
        if len(self.history[buf]) < length: return None
        vals = np.array(self.history[buf][-length:])
        m = offset * (length - 1); s = length / sigma
        w = np.exp(-((np.arange(length) - m)**2) / (2*s*s))
        return np.dot(vals, w) / w.sum()

    def _ta_vwap(self, args, kw):
        tp = (self._get_ohlcv('high') + self._get_ohlcv('low') + self._get_ohlcv('close')) / 3
        vol = self._get_ohlcv('volume') or 1.0
        buf_tv = self._call_key("vwap_tv"); buf_v = self._call_key("vwap_v")
        self.globals[buf_tv] = self.globals.get(buf_tv, 0) + tp * vol
        self.globals[buf_v] = self.globals.get(buf_v, 0) + vol
        return self.globals[buf_tv] / self.globals[buf_v] if self.globals[buf_v] > 0 else tp

    def _ta_obv(self, args, kw):
        close = self._get_ohlcv('close'); vol = self._get_ohlcv('volume') or 0
        prev_close = self._get_history('close', 1)
        buf = self._call_key("obv")
        prev_obv = self.globals.get(buf, 0)
        if prev_close is not None:
            if close > prev_close: prev_obv += vol
            elif close < prev_close: prev_obv -= vol
        self.globals[buf] = prev_obv
        return prev_obv

    # ── input.* ───────────────────────────────────────────────────────────

    # ── Time / Session functions ────────────────────────────────────────

    def _get_bar_timestamp(self):
        """Get current bar's timestamp."""
        if isinstance(self.df.index, pd.DatetimeIndex):
            return self.df.index[self.bar_index]
        return None

    def _time_func(self, args, kw):
        """time(timeframe, session, timezone) — returns timestamp if bar is in session, na otherwise."""
        ts = self._get_bar_timestamp()
        if ts is None:
            return self.globals.get('time', 0)

        # If called with session string, check if current bar is within session
        if len(args) >= 2:
            session = str(args[1]) if args[1] else None
            if session and '-' in session:
                try:
                    parts = session.split('-')
                    start_str = parts[0].strip()
                    end_str = parts[1].strip()
                    sh = int(start_str[:2]); sm = int(start_str[2:4]) if len(start_str) >= 4 else 0
                    eh = int(end_str[:2]); em = int(end_str[2:4]) if len(end_str) >= 4 else 0
                    bar_minutes = ts.hour * 60 + ts.minute
                    start_minutes = sh * 60 + sm
                    end_minutes = eh * 60 + em
                    if start_minutes <= bar_minutes < end_minutes:
                        return int(ts.timestamp() * 1000)
                    return None
                except (ValueError, IndexError):
                    pass

        return int(ts.timestamp() * 1000)

    def _hour_func(self, args, kw):
        """hour(time, timezone) — extract hour from timestamp."""
        if args:
            # hour(time, "America/New_York") — just return hour from bar
            ts = self._get_bar_timestamp()
            return ts.hour if ts else 0
        return self.globals.get('hour', 0)

    def _minute_func(self, args, kw):
        if args:
            ts = self._get_bar_timestamp()
            return ts.minute if ts else 0
        return self.globals.get('minute', 0)

    def _year_func(self, args, kw):
        if args:
            ts = self._get_bar_timestamp()
            return ts.year if ts else 0
        return self.globals.get('year', 0)

    def _month_func(self, args, kw):
        if args:
            ts = self._get_bar_timestamp()
            return ts.month if ts else 0
        return self.globals.get('month', 0)

    def _dom_func(self, args, kw):
        if args:
            ts = self._get_bar_timestamp()
            return ts.day if ts else 0
        return self.globals.get('dayofmonth', 0)

    def _timestamp_func(self, args, kw):
        """timestamp(year, month, day, hour, minute) or timestamp(datestring)."""
        if len(args) >= 5:
            try:
                from datetime import datetime
                dt = datetime(int(self._num(args[0])), int(self._num(args[1])),
                             int(self._num(args[2])), int(self._num(args[3])),
                             int(self._num(args[4])))
                return int(dt.timestamp() * 1000)
            except (ValueError, TypeError):
                return 0
        return 0

    def _input(self, args, kw, cast=None):
        """Handle input.*() calls — capture definition + return value (or override)."""
        defval = kw.get('defval', args[0] if args else 0)
        title = kw.get('title', args[1] if len(args) > 1 and isinstance(args[1], str) else str(args[0]) if args and isinstance(args[0], str) and cast is None else "")
        if isinstance(title, str) and title.startswith(("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")):
            title = ""  # numeric first arg isn't a title

        group = kw.get('group', '')
        tooltip = kw.get('tooltip', '')
        options = kw.get('options')
        minval = kw.get('minval')
        maxval = kw.get('maxval')
        step = kw.get('step')

        # Build input definition for UI
        input_def = {
            'title': str(title) if title else f"Input {len(self.input_defs)+1}",
            'defval': defval,
            'type': 'int' if cast == int else 'float' if cast == float else 'bool' if cast == bool else 'string',
            'group': str(group) if group else '',
            'tooltip': str(tooltip) if tooltip else '',
            'options': options if isinstance(options, list) else None,
            'minval': self._num(minval) if minval is not None else None,
            'maxval': self._num(maxval) if maxval is not None else None,
            'step': self._num(step) if step is not None else None,
        }

        # Assign a stable key based on definition count
        input_key = input_def['title']
        input_def['key'] = input_key

        # Only register on first bar to avoid duplicates
        if self.bar_index == 0:
            self.input_defs.append(input_def)

        # Check for user override
        if input_key in self.input_overrides:
            override_val = self.input_overrides[input_key]
            if cast:
                try:
                    return cast(self._num(override_val) if cast in (int, float) else override_val)
                except (ValueError, TypeError):
                    return override_val
            return override_val

        # Return default
        if cast:
            try:
                return cast(self._num(defval) if cast in (int, float) else defval)
            except (ValueError, TypeError):
                return defval
        return defval

    def _request_security(self, args, kw):
        """Handle request.security() — return fallback values for cross-symbol data.

        Pine syntax: request.security(symbol, timeframe, expression, ...)
        Since we only have single-symbol OHLCV data, we can't fetch other tickers.
        Strategy:
          1. Check if user provided an override via input_overrides["_security_<TICKER>"]
          2. Check built-in defaults for known tickers (VIX, DXY, etc.)
          3. If the expression arg is an OHLCV reference, return current bar's value
          4. Fall back to 0.0
        """
        ticker = str(args[0]).upper().strip() if args else ""
        # Clean ticker — strip exchange prefix
        clean_ticker = ticker.split(":")[-1] if ":" in ticker else ticker

        # Check user override first (allows optimizer to sweep VIX values)
        override_key = f"_security_{clean_ticker}"
        if override_key in self.input_overrides:
            return self._num(self.input_overrides[override_key])

        # Check built-in defaults
        for key, val in self.security_defaults.items():
            if clean_ticker == key.upper().split(":")[-1]:
                # Register as an optimizable input so the optimizer can find it
                if self.bar_index == 0:
                    self.input_defs.append({
                        'title': f"_security_{clean_ticker}",
                        'defval': val,
                        'type': 'float',
                        'group': 'External Data Fallbacks',
                        'tooltip': f'Fallback value for request.security("{ticker}", ...). '
                                   f'Real data not available — using static default.',
                        'key': f"_security_{clean_ticker}",
                        'minval': None, 'maxval': None, 'step': None, 'options': None,
                    })
                return val

        # Unknown ticker — try to use current symbol's OHLCV
        # (the expression arg is typically 'close', 'open', etc.)
        if len(args) >= 3:
            expr = args[2]
            if isinstance(expr, str) and expr.lower() in ('close', 'open', 'high', 'low', 'volume', 'hl2', 'hlc3', 'ohlc4'):
                return self._get_ohlcv(expr.lower())
            # If it's already a numeric value (pre-evaluated), return it
            if isinstance(expr, (int, float)):
                return expr

        return 0.0

    def _input_source(self, args, kw):
        defval = kw.get('defval', args[0] if args else 'close')
        title = kw.get('title', 'Source')
        group = kw.get('group', '')
        tooltip = kw.get('tooltip', '')

        if self.bar_index == 0:
            self.input_defs.append({
                'title': str(title), 'defval': str(defval) if isinstance(defval, str) else 'close',
                'type': 'source', 'group': str(group), 'tooltip': str(tooltip),
                'options': ['close', 'open', 'high', 'low', 'hl2', 'hlc3', 'ohlc4'],
                'key': str(title),
            })

        key = str(title)
        if key in self.input_overrides:
            return str(self.input_overrides[key])
        if isinstance(defval, str):
            return defval
        return 'close'

    # ── strategy.* ────────────────────────────────────────────────────────

    def _strategy_decl(self, args, kw):
        if args:
            self.strategy_name = str(args[0])
        self.strategy_params = kw
        # Parse strategy() params
        if 'pyramiding' in kw:
            p = int(self._num(kw['pyramiding']))
            self.pyramiding = max(1, p)  # Pine: pyramiding=0 means no pyramiding = 1 entry max
        if 'slippage' in kw:
            self.slippage = int(self._num(kw['slippage']))
        if 'default_qty_value' in kw:
            self.default_qty = int(self._num(kw['default_qty_value']))
            self.entry_qty = self.default_qty
        if 'calc_on_every_tick' in kw:
            self.calc_on_every_tick = bool(kw['calc_on_every_tick'])
        if 'process_orders_on_close' in kw:
            self.process_orders_on_close = bool(kw['process_orders_on_close'])
        if 'commission_value' in kw:
            comm_type = kw.get('commission_type', '')
            comm_val = self._num(kw['commission_value'])
            if isinstance(comm_type, str) and 'cash' in str(comm_type).lower():
                # cash_per_contract or cash_per_order — store as fixed $ amount
                self.commission_pct = 0.0
                self.commission_cash = comm_val
            else:
                self.commission_pct = comm_val / 100.0
                self.commission_cash = 0.0
        if 'initial_capital' in kw:
            self.initial_capital = self._num(kw['initial_capital'])
        if 'margin_long' in kw:
            self.margin_long = int(self._num(kw['margin_long']))
        if 'margin_short' in kw:
            self.margin_short = int(self._num(kw['margin_short']))

    def _strategy_entry(self, args, kw):
        if len(args) < 2: return
        name = str(args[0])
        direction_val = args[1]
        if isinstance(direction_val, str):
            new_dir = 1 if 'long' in direction_val.lower() else -1
        else:
            new_dir = 1 if self._num(direction_val) >= 0 else -1

        qty = int(self._num(kw.get('qty', args[2] if len(args) > 2 else self.default_qty)))
        if qty <= 0: qty = self.default_qty

        # When condition
        when = kw.get('when', True)
        if isinstance(when, bool) and not when: return
        if self._is_na(when): return

        # Use globals instead of df.iloc
        close = self.globals['close']

        # Apply slippage
        if self.slippage > 0:
            slip = self.slippage * self.mintick
            close = close + slip if new_dir == 1 else close - slip

        # Pyramiding check: how many open positions in same direction
        same_dir_count = sum(1 for p in self.positions if p['dir'] == new_dir)

        # Close opposite positions first
        opposite = [p for p in self.positions if p['dir'] != new_dir]
        if opposite:
            for p in opposite:
                self._close_single_position(p, close, 'Reverse')
            self.positions = [p for p in self.positions if p['dir'] == new_dir]

        # Check pyramiding limit
        if same_dir_count >= self.pyramiding:
            return

        #JUST TRADES CONFIDENTIAL PROPERTY
        # Open new position
        pos = {'dir': new_dir, 'price': close, 'qty': qty, 'bar': self.bar_index,
               'name': name, 'initial_qty': qty, 'be_activated': False, 'sl': None}

        # Apply exit rules if already registered for this entry name
        if name in self.exit_rules:
            rules = self.exit_rules[name]
            if rules.get('sl_ticks'):
                if new_dir == 1:
                    pos['sl'] = close - rules['sl_ticks'] * self.mintick
                else:
                    pos['sl'] = close + rules['sl_ticks'] * self.mintick

        self.positions.append(pos)
        self._update_position_state()

    def _strategy_close(self, args, kw):
        if not self.positions: return
        when = kw.get('when', True)
        if isinstance(when, bool) and not when: return
        if self._is_na(when): return

        name = str(args[0]) if args else "all"
        close = self.globals['close']

        # Apply slippage
        if self.slippage > 0:
            slip = self.slippage * self.mintick
            for p in self.positions:
                exit_price = close - slip if p['dir'] == 1 else close + slip
                self._close_single_position(p, exit_price, 'Signal Exit')
        else:
            if name == "all":
                for p in list(self.positions):
                    self._close_single_position(p, close, 'Signal Exit')
            else:
                matching = [p for p in self.positions if p.get('name') == name]
                for p in matching:
                    self._close_single_position(p, close, 'Signal Exit')

        self.positions = [p for p in self.positions if p.get('qty', 0) > 0]
        self._update_position_state()

    def _strategy_exit(self, args, kw):
        """Register exit rules (TP/SL/trailing) for a named entry."""
        exit_name = str(args[0]) if args else ""
        from_entry = str(args[1]) if len(args) > 1 else kw.get('from_entry', '')

        rules = self.exit_rules.get(from_entry, {})

        if 'profit' in kw:
            rules['tp_ticks'] = self._num(kw['profit'])
        if 'loss' in kw:
            rules['sl_ticks'] = self._num(kw['loss'])
        if 'limit' in kw:
            rules['tp_price'] = self._num(kw['limit'])
        if 'stop' in kw:
            rules['sl_price'] = self._num(kw['stop'])
        if 'trail_points' in kw:
            rules['trail_points'] = self._num(kw['trail_points'])
        if 'trail_offset' in kw:
            rules['trail_offset'] = self._num(kw['trail_offset'])
        if 'qty' in kw:
            rules['exit_qty'] = int(self._num(kw['qty']))
        if 'qty_percent' in kw:
            rules['exit_qty_pct'] = self._num(kw['qty_percent'])

        self.exit_rules[from_entry] = rules

        # Also register without from_entry as fallback
        if from_entry:
            self.exit_rules[from_entry] = rules
        if not from_entry:
            self.exit_rules['_default'] = rules

    def _strategy_order(self, args, kw):
        """Handle strategy.order() — supports limit orders as pending orders."""
        if len(args) < 2:
            return
        name = str(args[0])
        direction_val = args[1]
        if isinstance(direction_val, str):
            new_dir = 1 if 'long' in direction_val.lower() else -1
        else:
            new_dir = 1 if self._num(direction_val) >= 0 else -1

        qty = int(self._num(kw.get('qty', args[2] if len(args) > 2 else self.default_qty)))
        if qty <= 0:
            qty = self.default_qty

        when = kw.get('when', True)
        if isinstance(when, bool) and not when:
            return
        if self._is_na(when):
            return

        limit_price = kw.get('limit')
        stop_price = kw.get('stop')

        # If no limit/stop, execute as market order (same as strategy.entry)
        if limit_price is None and stop_price is None:
            return self._strategy_entry(args, kw)

        # Store as pending order — checked each bar in _check_pending_orders
        self.pending_orders[name] = {
            'dir': new_dir,
            'qty': qty,
            'limit': self._num(limit_price) if limit_price is not None else None,
            'stop': self._num(stop_price) if stop_price is not None else None,
            'comment': kw.get('comment', name),
        }

    def _strategy_cancel(self, args, kw):
        """Cancel a pending order by name."""
        if args:
            name = str(args[0])
            self.pending_orders.pop(name, None)

    def _strategy_cancel_all(self, args, kw):
        """Cancel all pending orders."""
        self.pending_orders.clear()

    def _check_pending_orders(self):
        """Check if any pending limit/stop orders should fill on this bar."""
        if not self.pending_orders:
            return
        h = self.globals['high']
        l = self.globals['low']
        filled = []
        for name, order in self.pending_orders.items():
            limit_price = order.get('limit')
            stop_price = order.get('stop')
            fill_price = None

            if limit_price is not None:
                # Limit buy: fills if low <= limit
                # Limit sell: fills if high >= limit
                if order['dir'] == 1 and l <= limit_price:
                    fill_price = limit_price
                elif order['dir'] == -1 and h >= limit_price:
                    fill_price = limit_price

            if stop_price is not None and fill_price is None:
                # Stop buy: fills if high >= stop
                # Stop sell: fills if low <= stop
                if order['dir'] == 1 and h >= stop_price:
                    fill_price = stop_price
                elif order['dir'] == -1 and l <= stop_price:
                    fill_price = stop_price

            if fill_price is not None:
                # Check if this is closing an opposite position (TP order)
                if self.positions:
                    pos_dir = self.positions[0]['dir']
                    if order['dir'] != pos_dir:
                        # This is a closing order (e.g., TP)
                        qty_to_close = order['qty']
                        total_open = sum(p['qty'] for p in self.positions)
                        if qty_to_close >= total_open:
                            # Close all positions at limit price
                            for p in list(self.positions):
                                if p.get('qty', 0) > 0:
                                    self._close_single_position(p, fill_price, order.get('comment', 'Limit Fill'))
                            self.positions = [p for p in self.positions if p.get('qty', 0) > 0]
                            self._update_position_state()
                        else:
                            # Partial close
                            remaining = qty_to_close
                            for p in list(self.positions):
                                if remaining <= 0:
                                    break
                                close_qty = min(p['qty'], remaining)
                                self._partial_close(p, fill_price, close_qty, order.get('comment', 'Limit Fill'))
                                remaining -= close_qty
                            self.positions = [p for p in self.positions if p.get('qty', 0) > 0]
                            self._update_position_state()
                        filled.append(name)
                        continue

                # Opening order — check pyramiding
                same_dir_count = sum(1 for p in self.positions if p['dir'] == order['dir'])
                if same_dir_count >= self.pyramiding:
                    filled.append(name)
                    continue

                # Close opposite positions first
                opposite = [p for p in self.positions if p['dir'] != order['dir']]
                if opposite:
                    for p in opposite:
                        self._close_single_position(p, fill_price, 'Reverse')
                    self.positions = [p for p in self.positions if p['dir'] == order['dir']]

                pos = {'dir': order['dir'], 'price': fill_price, 'qty': order['qty'],
                       'bar': self.bar_index, 'name': name, 'initial_qty': order['qty'],
                       'be_activated': False, 'sl': None}
                self.positions.append(pos)
                self._update_position_state()
                filled.append(name)

        for name in filled:
            self.pending_orders.pop(name, None)

    def _close_single_position(self, pos, exit_price, signal='Signal Exit'):
        """Close a single position and record the trade."""
        qty = pos.get('qty', pos.get('initial_qty', self.default_qty))
        if qty <= 0: return

        entry_price = pos['price']
        entry_bar = pos['bar']
        direction = pos['dir']

        if direction == 1:
            pnl = (exit_price - entry_price) * qty
        else:
            pnl = (entry_price - exit_price) * qty

        commission = (entry_price * qty * self.commission_pct) + (qty * self.commission_cash)

        # Compute run-up / drawdown
        if entry_bar < self.bar_index:
            bars = self.df.iloc[entry_bar:self.bar_index+1]
            if direction == 1:
                runup = (bars['High'].max() - entry_price) * qty
                dd = (entry_price - bars['Low'].min()) * qty
            else:
                runup = (entry_price - bars['Low'].min()) * qty
                dd = (bars['High'].max() - entry_price) * qty
        else:
            runup = max(abs(pnl), 0)
            dd = max(abs(pnl), 0) if pnl < 0 else 0

        idx = self.df.index
        entry_date = str(idx[entry_bar]) if isinstance(idx, pd.DatetimeIndex) else str(entry_bar)
        exit_date = str(idx[self.bar_index]) if isinstance(idx, pd.DatetimeIndex) else str(self.bar_index)

        self.trades.append({
            'Date/Time': entry_date,
            'Exit Date/Time': exit_date,
            'Type': 'Long' if direction == 1 else 'Short',
            'Signal': signal,
            'Price': round(entry_price, 2),
            'Exit Price': round(exit_price, 2),
            'Contracts': qty,
            'Profit': round(pnl - commission, 2),
            'Run-up': round(max(runup, 0), 2),
            'Drawdown': round(max(dd, 0), 2),
            'Commission': round(commission, 2),
        })

        pos['qty'] = 0  # mark as closed

    def _partial_close(self, pos, exit_price, qty_to_close, signal='Partial TP'):
        """Close part of a position."""
        if qty_to_close <= 0 or qty_to_close > pos['qty']:
            qty_to_close = pos['qty']

        entry_price = pos['price']
        entry_bar = pos['bar']
        direction = pos['dir']

        if direction == 1:
            pnl = (exit_price - entry_price) * qty_to_close
        else:
            pnl = (entry_price - exit_price) * qty_to_close

        commission = (entry_price * qty_to_close * self.commission_pct) + (qty_to_close * self.commission_cash)

        # Run-up/drawdown
        if entry_bar < self.bar_index:
            bars = self.df.iloc[entry_bar:self.bar_index+1]
            if direction == 1:
                runup = (bars['High'].max() - entry_price) * qty_to_close
                dd = (entry_price - bars['Low'].min()) * qty_to_close
            else:
                runup = (entry_price - bars['Low'].min()) * qty_to_close
                dd = (bars['High'].max() - entry_price) * qty_to_close
        else:
            runup = max(abs(pnl), 0)
            dd = max(abs(pnl), 0) if pnl < 0 else 0

        idx = self.df.index
        entry_date = str(idx[entry_bar]) if isinstance(idx, pd.DatetimeIndex) else str(entry_bar)
        exit_date = str(idx[self.bar_index]) if isinstance(idx, pd.DatetimeIndex) else str(self.bar_index)

        self.trades.append({
            'Date/Time': entry_date,
            'Exit Date/Time': exit_date,
            'Type': 'Long' if direction == 1 else 'Short',
            'Signal': signal,
            'Price': round(entry_price, 2),
            'Exit Price': round(exit_price, 2),
            'Contracts': qty_to_close,
            'Profit': round(pnl - commission, 2),
            'Run-up': round(max(runup, 0), 2),
            'Drawdown': round(max(dd, 0), 2),
            'Commission': round(commission, 2),
        })

        pos['qty'] -= qty_to_close

    def _update_position_state(self):
        """Update aggregate position state from individual positions."""
        self.positions = [p for p in self.positions if p.get('qty', 0) > 0]
        if not self.positions:
            self.position = 0
            self.entry_price = 0.0
            self.entry_qty = self.default_qty
            self.open_qty = 0
        else:
            self.position = self.positions[0]['dir']
            total_qty = sum(p['qty'] for p in self.positions)
            self.entry_price = sum(p['price'] * p['qty'] for p in self.positions) / total_qty if total_qty > 0 else 0
            self.entry_qty = total_qty
            self.open_qty = total_qty
            self.entry_bar = self.positions[0]['bar']

    # ── AST Execution ─────────────────────────────────────────────────────

    def execute(self):
        """Run the strategy bar-by-bar across all OHLCV data."""
        # ── Pre-compute numpy arrays for fast access (avoid df.iloc per bar) ──
        _open = self.df['Open'].values.astype(float)
        _high = self.df['High'].values.astype(float)
        _low = self.df['Low'].values.astype(float)
        _close = self.df['Close'].values.astype(float)
        _volume = self.df['Volume'].values.astype(float) if 'Volume' in self.df.columns else np.zeros(self.n_bars)

        # ── Precompute vectorized indicator cache ─────────────────────
        # This runs common ta.* functions on the FULL series once using
        # pandas vectorized ops (milliseconds), then bar-by-bar just looks up values.
        self._vec_cache = {}
        self._vec_open = _open
        self._vec_high = _high
        self._vec_low = _low
        self._vec_close = _close
        self._vec_volume = _volume

        _has_dt_index = isinstance(self.df.index, pd.DatetimeIndex)
        if _has_dt_index:
            _timestamps = self.df.index
            _ts_ms = (_timestamps.astype(np.int64) // 10**6).tolist()
            _hours = _timestamps.hour.tolist()
            _minutes = _timestamps.minute.tolist()
            _dows = (_timestamps.weekday + 2).tolist()  # Pine: Sun=1, Mon=2...
            _months = _timestamps.month.tolist()
            _years = _timestamps.year.tolist()

        # ── Set static globals once (not every bar) ──
        g = self.globals
        g['strategy.long'] = 1
        g['strategy.short'] = -1
        g['strategy.fixed'] = 'fixed'
        g['strategy.cash'] = 'cash'
        g['strategy.percent_of_equity'] = 'percent_of_equity'
        g['strategy.commission.percent'] = 'commission_percent'
        g['strategy.commission.cash_per_contract'] = 'commission_cash_per_contract'
        g['strategy.commission.cash_per_order'] = 'commission_cash_per_order'
        # Pine dayofweek constants: Sun=1, Mon=2 ... Sat=7
        g['dayofweek.sunday'] = 1
        g['dayofweek.monday'] = 2
        g['dayofweek.tuesday'] = 3
        g['dayofweek.wednesday'] = 4
        g['dayofweek.thursday'] = 5
        g['dayofweek.friday'] = 6
        g['dayofweek.saturday'] = 7
        g['strategy.initial_capital'] = self.initial_capital
        g['last_bar_index'] = self.n_bars - 1
        g['barstate.isconfirmed'] = True
        g['syminfo.mintick'] = self.mintick
        g['syminfo.pointvalue'] = 1.0
        g['syminfo.tickerid'] = ''
        g['syminfo.ticker'] = ''
        g['syminfo.type'] = 'stock'
        g['syminfo.timezone'] = 'UTC'
        g['timeframe.period'] = '1D'
        g['timeframe.multiplier'] = 1
        g['timeframe.isweekly'] = False
        g['timeframe.isdaily'] = True
        g['timeframe.isintraday'] = False

        # ── Track which user variables need history (lazy — add on first [N] access) ──
        self._tracked_vars = getattr(self, '_tracked_vars', set())
        # Pre-scan AST for [N] history references so tracking starts from bar 0
        self._prescan_history_refs(self.ast)
        _tracked_vars = self._tracked_vars
        _statements = self.ast.statements
        _running_pnl = 0.0

        for i in range(self.n_bars):
            self.bar_index = i

            # ── Fast OHLCV set (numpy arrays, no iloc) ──
            o = _open[i]; h = _high[i]; l = _low[i]; c = _close[i]; v = _volume[i]
            g['open'] = o; g['high'] = h; g['low'] = l; g['close'] = c; g['volume'] = v
            g['hl2'] = (h + l) * 0.5
            g['hlc3'] = (h + l + c) / 3.0
            g['ohlc4'] = (o + h + l + c) * 0.25
            g['bar_index'] = i
            g['barstate.isfirst'] = (i == 0)
            g['barstate.islast'] = (i == self.n_bars - 1)

            # ── Position state (avoid recomputing sum every bar) ──
            _pos_size = 0
            _open_profit = 0.0
            if self.positions:
                # Signed position size: positive for long, negative for short
                _pos_size = self.open_qty * self.position
                _bar_close = c
                for _p in self.positions:
                    if _p.get('qty', 0) > 0:
                        if _p['dir'] == 1:
                            _open_profit += (_bar_close - _p['price']) * _p['qty']
                        else:
                            _open_profit += (_p['price'] - _bar_close) * _p['qty']
            g['strategy.position_size'] = _pos_size
            g['strategy.position_avg_price'] = self.entry_price if self.positions else 0
            g['strategy.opentrades'] = len(self.positions)
            g['strategy.closedtrades'] = len(self.trades)
            g['strategy.openprofit'] = _open_profit
            g['strategy.equity'] = self.initial_capital + _running_pnl + _open_profit
            g['strategy.netprofit'] = _running_pnl

            # ── Time (pre-computed lists, no .iloc) ──
            if _has_dt_index:
                g['time'] = _ts_ms[i]
                g['hour'] = _hours[i]
                g['minute'] = _minutes[i]
                g['dayofweek'] = _dows[i]
                g['month'] = _months[i]
                g['year'] = _years[i]
                g['dayofmonth'] = _timestamps[i].day

            # Check pending limit/stop orders (strategy.order with limit=)
            if self.pending_orders:
                self._check_pending_orders()

            # Check TP/SL on open positions (strategy.exit rules)
            if self.positions:
                self._check_tp_sl()

            # Execute all statements
            _prev_trade_count = len(self.trades)
            for stmt in _statements:
                try:
                    self._exec(stmt)
                except Exception as e:
                    if i == 0:
                        self.warnings.append(f"Bar {i}: {type(e).__name__}: {e}")

            # After bar 0, strategy() may have updated initial_capital
            if i == 0:
                g['strategy.initial_capital'] = self.initial_capital
                _running_pnl = 0.0

            # Update running PnL if new trades were closed
            if len(self.trades) > _prev_trade_count:
                _running_pnl = sum(t['Profit'] for t in self.trades)

            # ── Record history only for variables used with [N] operator ──
            # Always record OHLCV + any user variable that was accessed via history
            for var_name in ('open', 'high', 'low', 'close', 'volume'):
                if var_name not in self.history:
                    self.history[var_name] = []
                self.history[var_name].append(g[var_name])

            for var_name in _tracked_vars:
                if var_name not in self.history:
                    self.history[var_name] = []
                self.history[var_name].append(g.get(var_name))

        # Close any open positions at end
        if self.positions:
            close = _close[-1]
            for pos in list(self.positions):
                if pos.get('qty', 0) > 0:
                    self._close_single_position(pos, close, 'End of Data')
            self.positions = []
            self._update_position_state()

        return self.trades

    def _prescan_history_refs(self, node):
        """Walk AST to find all variable[N] references and register them for tracking."""
        if isinstance(node, IndexAccess) and isinstance(node.obj, Identifier):
            self._tracked_vars.add(node.obj.name)
        if isinstance(node, Program):
            for s in node.statements: self._prescan_history_refs(s)
        for attr in ('left', 'right', 'operand', 'condition', 'true_val', 'false_val',
                      'value', 'obj', 'index', 'expr', 'start', 'end', 'step'):
            child = getattr(node, attr, None)
            if child and isinstance(child, ASTNode): self._prescan_history_refs(child)
        for attr in ('body', 'else_body', 'statements', 'args', 'elements', 'targets'):
            children = getattr(node, attr, None)
            if isinstance(children, list):
                for c in children:
                    if isinstance(c, ASTNode): self._prescan_history_refs(c)
        for attr in ('elif_clauses', 'cases'):
            items = getattr(node, attr, None)
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, tuple):
                        for elem in item:
                            if isinstance(elem, ASTNode): self._prescan_history_refs(elem)
                            elif isinstance(elem, list):
                                for e in elem:
                                    if isinstance(e, ASTNode): self._prescan_history_refs(e)
        kw = getattr(node, 'kwargs', None)
        if isinstance(kw, dict):
            for v in kw.values():
                if isinstance(v, ASTNode): self._prescan_history_refs(v)

    def _check_tp_sl(self):
        """Check TP/SL/trailing/breakeven for all open positions."""
        # Use globals instead of df.iloc
        closed_any = False

        for pos in list(self.positions):
            if pos.get('qty', 0) <= 0:
                continue

            entry_name = pos.get('name', '')
            rules = self.exit_rules.get(entry_name, self.exit_rules.get('_default', {}))
            if not rules:
                continue

            direction = pos['dir']
            entry_price = pos['price']
            qty = pos['qty']

            # ── Stop Loss ─────────────────────────────────────────────────
            sl_price = pos.get('sl')  # dynamic SL (breakeven, trailing)
            if sl_price is None and rules.get('sl_ticks'):
                if direction == 1:
                    sl_price = entry_price - rules['sl_ticks'] * self.mintick
                else:
                    sl_price = entry_price + rules['sl_ticks'] * self.mintick
            if sl_price is None and rules.get('sl_price'):
                sl_price = rules['sl_price']

            if sl_price is not None:
                hit = (direction == 1 and self.globals['low'] <= sl_price) or \
                      (direction == -1 and self.globals['high'] >= sl_price)
                if hit:
                    self._close_single_position(pos, sl_price, 'SL Hit')
                    closed_any = True
                    continue

            # ── Take Profit (full or partial) ─────────────────────────────
            tp_ticks = rules.get('tp_ticks')
            tp_price = rules.get('tp_price')

            if tp_ticks:
                if direction == 1:
                    tp_target = entry_price + tp_ticks * self.mintick
                else:
                    tp_target = entry_price - tp_ticks * self.mintick
            elif tp_price:
                tp_target = tp_price
            else:
                tp_target = None

            if tp_target is not None:
                hit = (direction == 1 and self.globals['high'] >= tp_target) or \
                      (direction == -1 and self.globals['low'] <= tp_target)
                if hit:
                    exit_qty_pct = rules.get('exit_qty_pct')
                    exit_qty = rules.get('exit_qty')
                    if exit_qty_pct and exit_qty_pct < 100:
                        qty_close = max(1, int(pos['initial_qty'] * exit_qty_pct / 100))
                        qty_close = min(qty_close, pos['qty'])
                        self._partial_close(pos, tp_target, qty_close, 'Partial TP')
                        # Move stop to breakeven after partial TP
                        pos['sl'] = entry_price + (self.slippage * self.mintick if direction == 1 else -self.slippage * self.mintick)
                        pos['be_activated'] = True
                    elif exit_qty and exit_qty < pos['qty']:
                        self._partial_close(pos, tp_target, exit_qty, 'Partial TP')
                        pos['sl'] = entry_price
                        pos['be_activated'] = True
                    else:
                        self._close_single_position(pos, tp_target, 'TP Hit')
                    closed_any = True
                    continue

            # ── Trailing Stop ─────────────────────────────────────────────
            trail_points = rules.get('trail_points')
            trail_offset = rules.get('trail_offset', 0)
            if trail_points:
                trail_key = f"_trail_{id(pos)}"
                if direction == 1:
                    highest = self.globals.get(trail_key, entry_price)
                    if self.globals['high'] > highest:
                        highest = self.globals['high']
                    self.globals[trail_key] = highest
                    activation = entry_price + trail_points * self.mintick
                    if highest >= activation:
                        trail_sl = highest - trail_offset * self.mintick
                        if self.globals['low'] <= trail_sl:
                            self._close_single_position(pos, trail_sl, 'Trail Stop')
                            closed_any = True
                            continue
                else:
                    lowest = self.globals.get(trail_key, entry_price)
                    if self.globals['low'] < lowest:
                        lowest = self.globals['low']
                    self.globals[trail_key] = lowest
                    activation = entry_price - trail_points * self.mintick
                    if lowest <= activation:
                        trail_sl = lowest + trail_offset * self.mintick
                        if self.globals['high'] >= trail_sl:
                            self._close_single_position(pos, trail_sl, 'Trail Stop')
                            closed_any = True
                            continue

        if closed_any:
            self._update_position_state()

    def _exec(self, node) -> Any:
        if node is None: return None

        if isinstance(node, NumberLiteral): return node.value
        if isinstance(node, StringLiteral): return node.value
        if isinstance(node, BoolLiteral): return node.value
        if isinstance(node, NALiteral): return None
        if isinstance(node, ArrayLiteral): return [self._exec(e) for e in node.elements]

        if isinstance(node, Identifier):
            name = node.name
            # Check globals
            if name in self.globals: return self.globals[name]
            # Check OHLCV
            v = self._get_ohlcv(name)
            if v is not None: return v
            return None

        if isinstance(node, BinaryOp):
            left = self._exec(node.left)
            right = self._exec(node.right)
            return self._binary_op(node.op, left, right)

        if isinstance(node, UnaryOp):
            val = self._exec(node.operand)
            if node.op == '-': return -self._num(val)
            if node.op == 'not': return not val
            return val

        if isinstance(node, TernaryOp):
            cond = self._exec(node.condition)
            return self._exec(node.true_val) if cond else self._exec(node.false_val)

        if isinstance(node, Assignment):
            val = self._exec(node.value)
            if node.is_var:
                if node.target not in self.var_inited:
                    self.globals[node.target] = val
                    self.var_inited[node.target] = True
                return self.globals.get(node.target)
            elif node.is_varip:
                if node.target not in self.var_inited:
                    self.globals[node.target] = val
                    self.var_inited[node.target] = True
                return self.globals.get(node.target)
            elif node.op == ':=':
                self.globals[node.target] = val
            elif node.op == '+=':
                self.globals[node.target] = self._num(self.globals.get(node.target, 0)) + self._num(val)
            elif node.op == '-=':
                self.globals[node.target] = self._num(self.globals.get(node.target, 0)) - self._num(val)
            elif node.op == '*=':
                self.globals[node.target] = self._num(self.globals.get(node.target, 0)) * self._num(val)
            elif node.op == '/=':
                d = self._num(val)
                self.globals[node.target] = self._num(self.globals.get(node.target, 0)) / d if d != 0 else None
            else:
                self.globals[node.target] = val
            return val

        if isinstance(node, TupleUnpack):
            result = self._exec(node.value)
            if isinstance(result, tuple):
                for i, name in enumerate(node.targets):
                    self.globals[name] = result[i] if i < len(result) else None
            return result

        if isinstance(node, IfStatement):
            cond = self._exec(node.condition)
            if cond:
                return self._exec_block(node.body)
            for elif_cond, elif_body in node.elif_clauses:
                if self._exec(elif_cond):
                    return self._exec_block(elif_body)
            if node.else_body:
                return self._exec_block(node.else_body)
            return None

        if isinstance(node, ForLoop):
            start = int(self._num(self._exec(node.start)))
            end = int(self._num(self._exec(node.end)))
            step = int(self._num(self._exec(node.step))) if node.step else 1
            if step == 0: step = 1
            result = None
            count = 0
            for val in range(start, end + (1 if step > 0 else -1), step):
                self.globals[node.var] = val
                result = self._exec_block(node.body)
                count += 1
                if count > 10000: break  # safety
            return result

        if isinstance(node, WhileLoop):
            result = None
            count = 0
            while self._exec(node.condition):
                result = self._exec_block(node.body)
                count += 1
                if count > 10000: break
            return result

        if isinstance(node, SwitchStatement):
            expr_val = self._exec(node.expr) if node.expr else None
            for case_cond, case_body in node.cases:
                if case_cond is None:  # default
                    return self._exec_block(case_body)
                case_val = self._exec(case_cond)
                if expr_val is not None:
                    if case_val == expr_val:
                        return self._exec_block(case_body)
                else:
                    if case_val:
                        return self._exec_block(case_body)
            return None

        if isinstance(node, FunctionDef):
            self.functions[node.name] = node
            return None

        if isinstance(node, FunctionCall):
            return self._call_function(node.name, node.args, node.kwargs)

        if isinstance(node, MethodCall):
            obj = self._exec(node.obj)
            # Try as dotted builtin: obj.method
            if isinstance(node.obj, Identifier):
                full_name = f"{node.obj.name}.{node.method}"
                if full_name in self.builtins:
                    args = [self._exec(a) for a in node.args]
                    kwargs = {k: self._exec(v) for k, v in node.kwargs.items()}
                    return self.builtins[full_name](args, kwargs)
                # Two-level dot: ta.ema etc is handled as DotAccess -> MethodCall
            if isinstance(node.obj, DotAccess):
                full_name = f"{self._dot_name(node.obj)}.{node.method}"
                if full_name in self.builtins:
                    args = [self._exec(a) for a in node.args]
                    kwargs = {k: self._exec(v) for k, v in node.kwargs.items()}
                    return self.builtins[full_name](args, kwargs)
            # Array method calls
            if isinstance(obj, list):
                full_name = f"array.{node.method}"
                if full_name in self.builtins:
                    args = [obj] + [self._exec(a) for a in node.args]
                    kwargs = {k: self._exec(v) for k, v in node.kwargs.items()}
                    return self.builtins[full_name](args, kwargs)
            return None

        if isinstance(node, DotAccess):
            name = self._dot_name_full(node)
            if name in self.globals: return self.globals[name]
            if name in self.builtins:
                # It's a namespace, not a call — return the name for later resolution
                return name
            obj = self._exec(node.obj)
            if isinstance(obj, dict):
                return obj.get(node.attr)
            return self.globals.get(name)

        if isinstance(node, IndexAccess):
            obj_name = node.obj.name if isinstance(node.obj, Identifier) else None
            idx = int(self._num(self._exec(node.index)))
            # History operator: variable[N] means N bars ago
            if obj_name:
                # Register for history tracking
                if hasattr(self, '_tracked_vars'):
                    self._tracked_vars.add(obj_name)
                obj_val = self._exec(node.obj)
                if isinstance(obj_val, list):
                    return obj_val[idx] if 0 <= idx < len(obj_val) else None
                # Bar lookback
                return self._get_history(obj_name, idx)
            obj = self._exec(node.obj)
            if isinstance(obj, list):
                return obj[idx] if 0 <= idx < len(obj) else None
            return None

        return None

    def _dot_name(self, node):
        if isinstance(node, DotAccess):
            return f"{self._dot_name(node.obj)}.{node.attr}"
        if isinstance(node, Identifier):
            return node.name
        return str(node)

    def _dot_name_full(self, node):
        return self._dot_name(node)

    def _exec_block(self, stmts):
        result = None
        for s in stmts:
            result = self._exec(s)
        return result

    def _call_function(self, name, arg_nodes, kwarg_nodes):
        # Resolve dotted name for builtins
        if name in self.builtins:
            args = [self._exec(a) for a in arg_nodes]
            kwargs = {k: self._exec(v) for k, v in kwarg_nodes.items()}
            return self.builtins[name](args, kwargs)

        # User-defined functions
        if name in self.functions:
            func = self.functions[name]
            # Save globals, set params
            saved = {}
            args = [self._exec(a) for a in arg_nodes]
            kwargs = {k: self._exec(v) for k, v in kwarg_nodes.items()}
            for i, (param_name, default) in enumerate(func.params):
                saved[param_name] = self.globals.get(param_name)
                if param_name in kwargs:
                    self.globals[param_name] = kwargs[param_name]
                elif i < len(args):
                    self.globals[param_name] = args[i]
                elif default is not None:
                    self.globals[param_name] = self._exec(default)
            result = self._exec_block(func.body)
            # Restore
            for param_name, old_val in saved.items():
                if old_val is None:
                    self.globals.pop(param_name, None)
                else:
                    self.globals[param_name] = old_val
            return result

        # Type cast functions
        if name in ('int', 'float', 'bool', 'string', 'color'):
            if name in self.builtins:
                args = [self._exec(a) for a in arg_nodes]
                return self.builtins[name](args, {})

        return None

    def _binary_op(self, op, left, right):
        if op == 'and':
            if self._is_na(left) or self._is_na(right): return False
            return bool(left) and bool(right)
        if op == 'or':
            if self._is_na(left) and self._is_na(right): return False
            return bool(left if not self._is_na(left) else False) or bool(right if not self._is_na(right) else False)

        # Comparisons with None/na always return False (like Pine Script)
        if op in ('>', '<', '>=', '<='):
            if self._is_na(left) or self._is_na(right): return False

        l = self._num(left); r = self._num(right)

        if op == '+':
            if isinstance(left, str) or isinstance(right, str):
                return str(left if left is not None else '') + str(right if right is not None else '')
            if self._is_na(left) or self._is_na(right): return None
            return l + r
        if op == '-':
            if self._is_na(left) or self._is_na(right): return None
            return l - r
        if op == '*':
            if self._is_na(left) or self._is_na(right): return None
            return l * r
        if op == '/': return l / r if r != 0 and not self._is_na(left) else None
        if op == '%': return l % r if r != 0 and not self._is_na(left) else None
        if op == '>': return l > r
        if op == '<': return l < r
        if op == '>=': return l >= r
        if op == '<=': return l <= r
        if op == '==':
            if self._is_na(left) and self._is_na(right): return True
            if self._is_na(left) or self._is_na(right): return False
            # String comparison — compare directly, don't convert to numbers
            if isinstance(left, str) or isinstance(right, str):
                return str(left) == str(right)
            return l == r
        if op == '!=':
            if self._is_na(left) and self._is_na(right): return False
            if self._is_na(left) or self._is_na(right): return True
            # String comparison — compare directly, don't convert to numbers
            if isinstance(left, str) or isinstance(right, str):
                return str(left) != str(right)
            return l != r
        return None

    def to_dataframe(self):
        if not self.trades: return pd.DataFrame()
        df = pd.DataFrame(self.trades)
        df.insert(0, 'Trade #', range(1, len(df)+1))
        df['Cum. Profit'] = df['Profit'].cumsum().round(2)
        return df


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMLIT APP
# ═══════════════════════════════════════════════════════════════════════════════

# ── Branded header ────────────────────────────────────────────────────────────
if LOGO_B64:
    st.markdown(f"""
    <div class="jt-header">
        <img src="data:image/png;base64,{LOGO_B64}" alt="Just Trades">
        <div class="jt-header-text">
            <h1>Just Trades Analytics</h1>
            <p>{BRAND['tagline']}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="jt-header">
        <div class="jt-header-text">
            <h1>Just Trades Analytics</h1>
            <p>{BRAND['tagline']}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    if LOGO_B64:
        st.markdown(f"""
        <div style="text-align:center; padding: 8px 0 16px 0;">
            <img src="data:image/png;base64,{LOGO_B64}" style="width:80px; border-radius:16px; border: 2px solid rgba(25,184,255,0.25);">
            <div style="font-family:'JetBrains Mono',monospace; font-size:1.1rem; font-weight:700;
                        background: linear-gradient(135deg, #19b8ff, #2dd4bf);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                        margin-top:8px;">Just Trades</div>
            <div style="font-size:0.65rem; opacity:0.4; letter-spacing:1px; margin-top:2px;">STRATEGY ANALYTICS</div>
        </div>
        """, unsafe_allow_html=True)
    st.header("⚙️ Configuration")

    input_mode = st.radio("Input Mode", ["📁 Upload CSV/XLSX", "🌲 Pine Script"],
                          horizontal=True, label_visibility="collapsed")

    if input_mode == "📁 Upload CSV/XLSX":
        uploaded_file = st.file_uploader("Upload TradingView CSV or Excel", type=["csv", "xlsx", "xls"])
    else:
        uploaded_file = None

    initial_capital = st.number_input("Initial Capital ($)", value=10000, min_value=1, step=1000)
    risk_free_rate = st.number_input("Risk-Free Rate (%)", value=5.0, min_value=0.0,
                                     max_value=20.0, step=0.25) / 100
    mc_sims = st.slider("Monte Carlo Simulations", min_value=500, max_value=10000,
                         value=2000, step=500)
    rolling_window = st.slider("Rolling Window (trades)", min_value=5, max_value=100,
                                value=20, step=5)

    if input_mode == "📁 Upload CSV/XLSX":
        st.divider()
        st.markdown("**Column Override** *(leave on Auto if unsure)*")
        manual_profit_col = st.checkbox("Manually select profit column", value=False)
    else:
        manual_profit_col = False

    if input_mode == "🌲 Pine Script":
        st.divider()
        st.markdown("**📊 Data Settings**")
        pine_ticker = st.text_input("Symbol", value="NQ", 
            help="Futures: NQ, ES, SI, GC, CL, ZB, YM, RTY, HG, NG — Stocks: SPY, AAPL, TSLA etc.")
        pine_timeframe = st.selectbox("Timeframe", [
            "1min", "3min", "5min", "15min", "30min", "1h", "1d", "1W"
        ], index=2, help="Databento provides years of tick/minute futures data.")

        # Flexible date range: days, weeks, months, or years
        pine_range_unit = st.selectbox("Data Range", ["Days", "Weeks", "Months", "Years"], index=3)
        if pine_range_unit == "Days":
            pine_range_val = st.slider("Days", min_value=7, max_value=365, value=90, step=7)
            pine_offset = pd.DateOffset(days=pine_range_val)
            pine_range_label = f"{pine_range_val}d"
        elif pine_range_unit == "Weeks":
            pine_range_val = st.slider("Weeks", min_value=1, max_value=52, value=12)
            pine_offset = pd.DateOffset(weeks=pine_range_val)
            pine_range_label = f"{pine_range_val}w"
        elif pine_range_unit == "Months":
            pine_range_val = st.slider("Months", min_value=1, max_value=24, value=6)
            pine_offset = pd.DateOffset(months=pine_range_val)
            pine_range_label = f"{pine_range_val}mo"
        else:
            pine_range_val = st.slider("Years", min_value=1, max_value=5, value=2)
            pine_offset = pd.DateOffset(years=pine_range_val)
            pine_range_label = f"{pine_range_val}yr"

        # API keys from env only — no UI fields
        db_key = os.environ.get("DATABENTO_API_KEY", "")
        gemini_key = os.environ.get("GEMINI_API_KEY", "")

        # Auto-detect provider from env
        if db_key:
            data_provider = "Databento (futures + stocks)"
        else:
            data_provider = "Yahoo Finance (stocks only, no key)"

        st.divider()
        st.markdown("**⚙️ Execution Settings**")
        pine_commission = st.number_input("Commission %", value=0.0, min_value=0.0,
                                           max_value=5.0, step=0.01,
                                           help="Default 0 matches TradingView.") / 100
        pine_qty = st.number_input("Default Qty", value=1, min_value=1, step=1)
        pine_pyramiding = st.number_input("Pyramiding", value=1, min_value=1, max_value=50, step=1,
                                           help="Default 1 = no pyramiding (TradingView default).")
        pine_slippage = st.number_input("Slippage (ticks)", value=0, min_value=0, max_value=50, step=1)
        pine_mintick = st.number_input("Tick Size", value=0.25, min_value=0.0001, max_value=10.0, step=0.01,
                                        format="%.4f", help="0.25 for NQ/ES, 0.005 for SI, 0.10 for GC, 0.01 for stocks.")
        pine_margin_long = st.number_input("Margin Long %", value=0, min_value=0, max_value=100, step=1,
                                            help="0 = cash (TradingView default). 100 can cause 0 trades.")
        pine_margin_short = st.number_input("Margin Short %", value=0, min_value=0, max_value=100, step=1)

    # ── Data Cache Manager ────────────────────────────────────────────
    if input_mode == "🌲 Pine Script":
        st.divider()
        st.markdown("**💾 Data Cache**")
        _cache_dir_sidebar = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_cache')
        if os.path.exists(_cache_dir_sidebar):
            _cached_files = [f for f in os.listdir(_cache_dir_sidebar) if f.endswith('.parquet')]
            if _cached_files:
                st.caption(f"{len(_cached_files)} cached dataset(s)")
                for cf in sorted(_cached_files):
                    _size = os.path.getsize(os.path.join(_cache_dir_sidebar, cf)) / (1024*1024)
                    st.caption(f"  📄 {cf.replace('.parquet','')} ({_size:.1f}MB)")
                if st.button("🗑️ Clear All Cache", key="_clear_cache"):
                    import shutil
                    shutil.rmtree(_cache_dir_sidebar)
                    st.success("Cache cleared")
                    st.rerun()
            else:
                st.caption("No cached data yet")

    # ── AI Quant Analyst ─────────────────────────────────────────────
    st.divider()
    st.markdown("**🤖 AI Quant Analyst**")
    import shutil as _shutil
    _cli_available = os.path.exists(_shutil.which('claude') or '/usr/local/bin/claude')
    if _cli_available:
        st.caption("✅ Claude CLI detected — using your Max plan")
    else:
        st.caption("⚠️ Claude CLI not found")
    _anthropic_key = os.environ.get('ANTHROPIC_API_KEY', '')
    with st.expander("API Key (optional fallback)", expanded=False):
        anthropic_api_key = st.text_input(
            "Anthropic API Key",
            value=_anthropic_key,
            type="password",
            key="_anthropic_api_key",
            help="Only needed if Claude CLI is not installed"
        )
        if anthropic_api_key:
            os.environ['ANTHROPIC_API_KEY'] = anthropic_api_key

#JUST TRADES CONFIDENTIAL PROPERTY
# ═══════════════════════════════════════════════════════════════════════════════
# PINE SCRIPT MODE
# ═══════════════════════════════════════════════════════════════════════════════

pine_generated_df = None

if input_mode == "🌲 Pine Script":
    st.markdown("""
    <div style="padding: 4px 0 12px 0;">
        <span style="font-family:'JetBrains Mono',monospace; font-size:0.9rem; opacity:0.7;">
            Paste your Pine Script strategy below
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Strategy file picker from strategies/ folder
    _strat_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'strategies')
    _strat_files = []
    if os.path.isdir(_strat_dir):
        _strat_files = sorted([f for f in os.listdir(_strat_dir) if f.endswith('.pine')])
    if _strat_files:
        _pick = st.selectbox("📂 Load from strategies/", ["— paste manually —"] + _strat_files,
                             key="_strat_picker")
        if _pick != "— paste manually —":
            with open(os.path.join(_strat_dir, _pick)) as _sf:
                st.session_state['_loaded_pine_code'] = _sf.read()

    # Pre-fill from loaded strategy if available
    _prefill = st.session_state.pop('_loaded_pine_code', '')

    pine_code = st.text_area(
        "Pine Script Code",
        value=_prefill,
        height=300,
        placeholder='''// Paste your Pine Script strategy here, e.g.:
//@version=6
strategy("My Strategy", overlay=true)

fastEma = ta.ema(close, 9)
slowEma = ta.ema(close, 21)

if ta.crossover(fastEma, slowEma)
    strategy.entry("Long", strategy.long)

if ta.crossunder(fastEma, slowEma)
    strategy.entry("Short", strategy.short)''',
        label_visibility="collapsed",
    )

    if pine_code and pine_code.strip():
        # ── AI Code Optimizer (automatic with Gemini) ─────────────────
        use_ai_optimize = False
        if gemini_key:
            use_ai_optimize = st.checkbox("🤖 AI-optimize code before parsing",
                value=False, help="Uses Gemini to strip plotting/drawing code and simplify syntax. Strategy logic stays 100% identical.")

        st.markdown("")

        # ── Mode selector: Backtest vs Optimizer vs TV Import ────────
        _pine_mode = st.radio(
            "Mode", ["🚀 Single Backtest", "📊 TradingView Import", "🔬 Parameter Optimizer"],
            horizontal=True, key="_pine_mode", label_visibility="collapsed",
        )

        if _pine_mode == "🚀 Single Backtest":
            # ── Dataset picker: use cached parquet OR fetch live ──────
            _bt_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_cache')
            _bt_datasets = {"⚡ Fetch live data (uses sidebar settings)": None}
            if os.path.exists(_bt_cache_dir):
                for _f in sorted(os.listdir(_bt_cache_dir)):
                    if _f.endswith('.parquet'):
                        _bt_datasets[_f.replace('.parquet', '')] = os.path.join(_bt_cache_dir, _f)
            _bt_ds_choice = st.selectbox(
                "📊 Data Source",
                list(_bt_datasets.keys()),
                index=0,
                key="_bt_dataset_pick",
                help="Pick a cached dataset to backtest against, or fetch live data using the sidebar Symbol/Timeframe/Range settings."
            )
            st.session_state['_bt_use_cached_dataset'] = _bt_datasets.get(_bt_ds_choice)

            _col_btn1, _col_btn2 = st.columns([3, 1])
            with _col_btn1:
                run_backtest = st.button("🚀 Run Backtest", type="primary", use_container_width=True)
            with _col_btn2:
                rerun_inputs = st.button("🔄 Apply", use_container_width=True,
                                          disabled='_pine_cached_data' not in st.session_state)
        elif _pine_mode == "📊 TradingView Import":
            # ── TradingView Import: real trade data + Pine code for AI ────
            run_backtest = False
            rerun_inputs = False
            st.markdown("---")
            st.info("📊 **TradingView Import** — Upload your real TradingView trade export. "
                    "The AI will analyze actual results while referencing your Pine code for fixes.")
            _tv_file = st.file_uploader(
                "Upload TradingView CSV or XLSX export",
                type=["csv", "xlsx", "xls"],
                key="_tv_import_file",
                help="Export from TradingView: Strategy Tester → right-click trade list → Export → CSV or Excel"
            )
            if _tv_file:
                try:
                    _tv_fname = _tv_file.name.lower()
                    if _tv_fname.endswith(('.xlsx', '.xls')):
                        _tv_file.seek(0)
                        _tv_xl = pd.ExcelFile(_tv_file)
                        _tv_sheet = None
                        for _s in _tv_xl.sheet_names:
                            if _s.lower().strip() == 'list of trades':
                                _tv_sheet = _s
                                break
                        if _tv_sheet is None and len(_tv_xl.sheet_names) > 1:
                            _best_s, _best_c = None, 0
                            for _s in _tv_xl.sheet_names:
                                _nc = len(_tv_xl.parse(_s, nrows=1).columns)
                                if _nc > _best_c:
                                    _best_c = _nc
                                    _best_s = _s
                            _tv_sheet = _best_s
                        _tv_raw = _tv_xl.parse(_tv_sheet if _tv_sheet else 0)
                    else:
                        _tv_file.seek(0)
                        _tv_raw = pd.read_csv(_tv_file)

                    _tv_df, _tv_col_map = parse_tradingview_csv(_tv_raw)

                    # Store as pine_generated_df so it flows into the analytics pipeline
                    pine_generated_df = _tv_df
                    st.session_state['_pine_trades_df'] = pine_generated_df
                    st.session_state['_pine_source_code'] = pine_code  # Keep Pine code for AI

                    # Count trades
                    _tv_profit_col = _tv_col_map.get('profit')
                    _tv_n = len(_tv_df) if _tv_profit_col else 0
                    st.success(f"✅ Loaded **{_tv_n:,}** trades from TradingView export. "
                              f"Pine code stored for AI analysis.")

                    # Preview
                    with st.expander("📋 TradingView Data Preview", expanded=False):
                        st.dataframe(_tv_df.head(20), use_container_width=True)
                        st.caption(f"Columns detected: {list(_tv_col_map.keys())}")
                except Exception as _tv_err:
                    st.error(f"Failed to parse TradingView export: {_tv_err}")
        else:
            run_backtest = False
            rerun_inputs = False

        # ══════════════════════════════════════════════════════════════════════
        # INLINE OPTIMIZER — runs when mode is "🔬 Parameter Optimizer"
        # ══════════════════════════════════════════════════════════════════════
        if _pine_mode == "🔬 Parameter Optimizer":
            st.markdown("---")
            st.markdown("### 🔬 Parameter Optimizer")
            st.caption("Brute-force grid search across all input parameter combinations to find the most profitable settings.")

            # ── Dataset picker ────────────────────────────────────────────
            _opt_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_cache')
            _opt_datasets = {}
            if os.path.exists(_opt_cache_dir):
                for f in sorted(os.listdir(_opt_cache_dir)):
                    if f.endswith('.parquet'):
                        _opt_datasets[f.replace('.parquet', '')] = os.path.join(_opt_cache_dir, f)
                    elif f.endswith('.csv'):
                        _opt_datasets[f.replace('.csv', '') + ' (CSV)'] = os.path.join(_opt_cache_dir, f)

            if not _opt_datasets:
                st.error("No datasets found in `data_cache/`. Add parquet or CSV files to optimize against.")
                st.stop()

            _oc1, _oc2, _oc3 = st.columns([2, 1, 1])
            with _oc1:
                _opt_dataset_choice = st.selectbox("📊 Dataset", list(_opt_datasets.keys()), key="_opt_ds")
            with _oc2:
                _opt_sort_by = st.selectbox("Rank By", [
                    'net_profit', 'sharpe_ratio', 'profit_factor', 'sortino_ratio',
                    'win_rate', 'sqn', 'calmar_ratio', 'expectancy',
                ], index=0, key="_opt_rank")
            with _oc3:
                _opt_top_n = st.number_input("Top N", value=10, min_value=1, max_value=50, key="_opt_topn")

            # ── Parse + Extract Inputs ────────────────────────────────────
            _opt_parse_key = f"_opt_parsed_{hash(pine_code)}"
            if _opt_parse_key not in st.session_state:
                try:
                    _opt_tokens = ext_lexer(pine_code)
                    _opt_ast = ExtParser(_opt_tokens).parse()
                    _opt_file_path = _opt_datasets[_opt_dataset_choice]
                    if _opt_file_path.endswith('.parquet'):
                        _opt_df = pd.read_parquet(_opt_file_path)
                    else:
                        _opt_df = pd.read_csv(_opt_file_path)
                        col_remap = {'open': 'Open', 'high': 'High', 'low': 'Low',
                                     'close': 'Close', 'volume': 'Volume'}
                        _opt_df.rename(columns=col_remap, inplace=True)
                    if 'Volume' not in _opt_df.columns:
                        _opt_df['Volume'] = 0.0
                    _opt_df = _opt_df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()

                    _opt_optimizer = PineOptimizer(
                        ast=_opt_ast, df_price=_opt_df,
                        initial_capital=initial_capital,
                        commission_pct=pine_commission, default_qty=pine_qty,
                        pyramiding=pine_pyramiding, slippage=pine_slippage,
                        mintick=pine_mintick, margin_long=pine_margin_long,
                        margin_short=pine_margin_short,
                    )
                    _opt_extracted = _opt_optimizer.extract_inputs(ExtPineInterpreter)

                    st.session_state[_opt_parse_key] = {
                        'ast': _opt_ast, 'df': _opt_df, 'params': _opt_extracted,
                    }
                except Exception as e:
                    st.error(f"Failed to parse strategy or load data: {e}")
                    st.stop()

            _opt_cache = st.session_state[_opt_parse_key]
            _opt_ast = _opt_cache['ast']
            _opt_df = _opt_cache['df']
            _opt_params = _opt_cache['params']

            if not _opt_params:
                st.warning("No `input.*()` parameters found. Nothing to optimize.")
                st.stop()

            st.info(f"**{len(_opt_df):,}** bars loaded from `{_opt_dataset_choice}` — "
                    f"**{len(_opt_params)}** optimizable inputs detected")

            # ── Parameter Configuration ───────────────────────────────────
            st.markdown("#### Configure Parameter Ranges")
            st.caption("Check the parameters you want to optimize. Unchecked parameters use their default. "
                       "**Tip:** Start with 3-5 key parameters to keep combos under 50K.")

            # Quick select buttons
            _qs1, _qs2, _qs3 = st.columns(3)
            with _qs1:
                if st.button("✅ Enable All", key="_opt_all_on"):
                    for p in _opt_params:
                        p.enabled = True
                    st.rerun()
            with _qs2:
                if st.button("❌ Disable All", key="_opt_all_off"):
                    for p in _opt_params:
                        p.enabled = False
                    st.rerun()
            with _qs3:
                if st.button("⚡ Smart Select (bools + key numerics)", key="_opt_smart"):
                    # Enable bools and string options (low combo cost), disable wide numerics
                    for p in _opt_params:
                        if p.param_type == 'bool':
                            p.enabled = True
                        elif p.param_type == 'string' and p.options:
                            p.enabled = True
                        else:
                            p.enabled = False
                    st.rerun()

            # Get group info from input_defs
            _opt_grouped = {}
            for idx, p in enumerate(_opt_params):
                group = 'General'
                if hasattr(p, '_group'):
                    group = p._group or 'General'
                if group not in _opt_grouped:
                    _opt_grouped[group] = []
                _opt_grouped[group].append((idx, p))

            for group_name, group_items in _opt_grouped.items():
                if len(_opt_grouped) > 1:
                    st.markdown(f"**{group_name}**")
                for idx, p in group_items:
                    if p.param_type == 'bool':
                        cols = st.columns([0.5, 3, 2, 1])
                        with cols[0]:
                            p.enabled = st.checkbox("", value=p.enabled, key=f"_oen_{idx}", label_visibility="collapsed")
                        with cols[1]:
                            st.markdown(f"`{p.name}` — *bool, default={p.default}*")
                        with cols[2]:
                            st.caption("True / False" if p.enabled else f"Fixed: {p.default}")
                        with cols[3]:
                            st.caption(f"**{2 if p.enabled else 1}** vals")
                    elif p.param_type in ('string', 'source') and p.options:
                        cols = st.columns([0.5, 3, 2, 1])
                        with cols[0]:
                            p.enabled = st.checkbox("", value=p.enabled, key=f"_oen_{idx}", label_visibility="collapsed")
                        with cols[1]:
                            st.markdown(f"`{p.name}` — *{p.param_type}, default=\"{p.default}\"*")
                        with cols[2]:
                            st.caption(f"Options: {', '.join(str(o) for o in p.options)}" if p.enabled else f"Fixed: \"{p.default}\"")
                        with cols[3]:
                            st.caption(f"**{len(p.options) if p.enabled else 1}** vals")
                    elif p.param_type in ('int', 'float'):
                        cols = st.columns([0.5, 2, 1.2, 1.2, 1.2, 0.8])
                        with cols[0]:
                            p.enabled = st.checkbox("", value=p.enabled, key=f"_oen_{idx}", label_visibility="collapsed")
                        with cols[1]:
                            st.markdown(f"`{p.name}` — *{p.param_type}, default={p.default}*")
                        if p.enabled:
                            _fmt = "%.4f" if p.param_type == 'float' else "%.0f"
                            with cols[2]:
                                _mn = st.number_input("Min", value=float(p.min_val) if p.min_val is not None else float(p.default or 1) * 0.5,
                                                      key=f"_omn_{idx}", format=_fmt, label_visibility="collapsed")
                                p.min_val = int(_mn) if p.param_type == 'int' else _mn
                            with cols[3]:
                                _mx = st.number_input("Max", value=float(p.max_val) if p.max_val is not None else float(p.default or 1) * 2.0,
                                                      key=f"_omx_{idx}", format=_fmt, label_visibility="collapsed")
                                p.max_val = int(_mx) if p.param_type == 'int' else _mx
                            with cols[4]:
                                _stp = p.step if p.step is not None else (1 if p.param_type == 'int' else 0.1)
                                _ns = st.number_input("Step", value=float(_stp), min_value=0.0001,
                                                      key=f"_ost_{idx}", format=_fmt, label_visibility="collapsed")
                                p.step = int(_ns) if p.param_type == 'int' and _ns >= 1 else _ns
                            p.generate_values()
                            with cols[5]:
                                st.caption(f"**{len(p.values)}** vals")
                        else:
                            with cols[2]:
                                st.caption(f"Fixed: {p.default}")
                    else:
                        cols = st.columns([0.5, 5, 1])
                        with cols[0]:
                            p.enabled = False
                        with cols[1]:
                            st.caption(f"`{p.name}` — fixed: {p.default}")

            # ── Combination count + Run button ────────────────────────────
            st.markdown("---")
            _total_combos = PineOptimizer.estimate_combinations(_opt_params)
            _enabled_count = sum(1 for p in _opt_params if p.enabled and p.values)

            _rc1, _rc2, _rc3 = st.columns([2, 2, 2])
            with _rc1:
                st.metric("Enabled Parameters", _enabled_count)
            with _rc2:
                st.metric("Total Combinations", f"{_total_combos:,}")
            with _rc3:
                _bars = len(_opt_df)
                _est = max(0.01, _bars / 50000) * _total_combos
                _est_label = f"~{_est:.0f}s" if _est < 60 else (f"~{_est/60:.1f}min" if _est < 3600 else f"~{_est/3600:.1f}hr")
                st.metric("Est. Time", _est_label)

            if _total_combos > 50_000:
                # Show which params contribute most to combo explosion
                _param_sizes = [(p.name, len(p.values)) for p in _opt_params if p.enabled and p.values]
                _param_sizes.sort(key=lambda x: x[1], reverse=True)
                _top_offenders = ", ".join(f"`{n}` ({v} vals)" for n, v in _param_sizes[:5])
                st.error(f"⚠️ **{_total_combos:,}** combinations exceeds 50K limit.\n\n"
                         f"**Biggest contributors:** {_top_offenders}\n\n"
                         f"Uncheck some parameters, increase step sizes, or narrow min/max ranges.")
            elif _total_combos == 0:
                st.warning("Enable at least one parameter to optimize.")

            _opt_min_trades = st.number_input("Min Trades Filter", value=5, min_value=1, max_value=100,
                                               help="Ignore parameter combos that produce fewer than this many trades",
                                               key="_opt_min_tr")

            _run_opt = st.button("🚀 Run Optimization", type="primary", use_container_width=True,
                                  key="_run_opt_main", disabled=(_total_combos > 50_000 or _total_combos == 0))

            if _run_opt:
                _opt_optimizer = PineOptimizer(
                    ast=_opt_ast, df_price=_opt_df,
                    initial_capital=initial_capital,
                    commission_pct=pine_commission, default_qty=pine_qty,
                    pyramiding=pine_pyramiding, slippage=pine_slippage,
                    mintick=pine_mintick, margin_long=pine_margin_long,
                    margin_short=pine_margin_short,
                )

                _opt_progress = st.progress(0, text="Starting optimization...")
                _opt_t0 = time.time()

                def _opt_cb(current, total):
                    pct = current / total
                    elapsed = time.time() - _opt_t0
                    rate = current / elapsed if elapsed > 0 else 0
                    eta = (total - current) / rate if rate > 0 else 0
                    _opt_progress.progress(pct, text=f"Combo {current:,}/{total:,} — {rate:.1f}/sec — ETA {eta:.0f}s")

                try:
                    _opt_results = _opt_optimizer.optimize(
                        params=_opt_params, PineInterpreterClass=ExtPineInterpreter,
                        sort_by=_opt_sort_by, top_n=_opt_top_n,
                        min_trades=_opt_min_trades, progress_callback=_opt_cb,
                    )
                    _opt_elapsed = time.time() - _opt_t0
                    _opt_progress.progress(1.0, text=f"✅ Done — {_total_combos:,} combos in {_opt_elapsed:.1f}s")
                    st.session_state['_opt_results_main'] = _opt_results
                    st.session_state['_opt_elapsed_main'] = _opt_elapsed
                except ValueError as ve:
                    st.error(str(ve))
                except Exception as e:
                    st.error(f"Optimization failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())

            # ── Display Results ───────────────────────────────────────────
            if '_opt_results_main' in st.session_state:
                _opt_results = st.session_state['_opt_results_main']
                _opt_elapsed = st.session_state.get('_opt_elapsed_main', 0)

                if not _opt_results:
                    st.warning("No combos met the minimum trade count. Lower the min trades filter.")
                else:
                    st.markdown("---")
                    st.markdown(f"### 🏆 Top {len(_opt_results)} Parameter Combinations")
                    st.caption(f"Ranked by **{_opt_sort_by}** | {_opt_elapsed:.1f}s | Min trades: {_opt_min_trades}")

                    # ── Results Table ─────────────────────────────────────
                    _opt_df_results = PineOptimizer.to_dataframe(_opt_results)
                    st.dataframe(_opt_df_results, use_container_width=True, hide_index=True)

                    # ── Profit Comparison Bar Chart ───────────────────────
                    _short_labels = [f"#{i+1}" for i in range(len(_opt_results))]
                    _profits = [r.net_profit for r in _opt_results]
                    _colors = ['#00C853' if p > 0 else '#FF1744' for p in _profits]

                    _bar_fig = go.Figure()
                    _bar_fig.add_trace(go.Bar(
                        x=_short_labels, y=_profits, marker_color=_colors, opacity=0.9,
                        text=[f"${p:,.0f}" for p in _profits], textposition='outside',
                    ))
                    _bar_fig.update_layout(title="Net Profit by Rank", yaxis_title="Net Profit ($)",
                                           xaxis_title="Rank", showlegend=False, height=400,
                                           template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
                                           plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(_bar_fig, use_container_width=True, key="opt_profit_inline")

                    # ── Sharpe vs Profit Scatter ──────────────────────────
                    if len(_opt_results) >= 2:
                        _sharpes = [min(r.sharpe_ratio, 10) if r.sharpe_ratio != float('inf') else 10.0 for r in _opt_results]
                        _net_profits = [r.net_profit for r in _opt_results]
                        _win_rates = [r.win_rate * 100 for r in _opt_results]
                        _hover = [f"#{i+1}<br>Profit: ${r.net_profit:,.0f}<br>Sharpe: {r.sharpe_ratio:.2f}<br>"
                                  f"Win: {r.win_rate:.1%}<br>Trades: {r.num_trades}<br>DD: {r.max_drawdown_pct:.1f}%<br>"
                                  + "<br>".join(f"{k}={v}" for k, v in r.params.items())
                                  for i, r in enumerate(_opt_results)]

                        _scat = go.Figure()
                        _scat.add_trace(go.Scatter(
                            x=_sharpes, y=_net_profits, mode='markers+text',
                            marker=dict(size=[max(10, w/3) for w in _win_rates], color=_win_rates,
                                        colorscale='Viridis', showscale=True, colorbar=dict(title="Win%")),
                            text=_short_labels, textposition='top center',
                            hovertext=_hover, hoverinfo='text',
                        ))
                        _scat.update_layout(title="Sharpe vs Profit (bubble = win rate)",
                                            xaxis_title="Sharpe Ratio", yaxis_title="Net Profit ($)",
                                            showlegend=False, height=400, template="plotly_dark",
                                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(_scat, use_container_width=True, key="opt_scatter_inline")

                    # ── Inspect Individual Results ────────────────────────
                    st.markdown("### 🔍 Inspect Result")
                    _inspect_i = st.selectbox(
                        "Select result", list(range(len(_opt_results))),
                        format_func=lambda i: f"#{i+1} — ${_opt_results[i].net_profit:,.0f} profit, "
                                              f"{_opt_results[i].num_trades} trades, "
                                              f"Sharpe {_opt_results[i].sharpe_ratio:.2f}",
                        key="_opt_inspect_inline",
                    )
                    _sel = _opt_results[_inspect_i]

                    # Settings to copy
                    st.markdown("**📋 Winning Settings (copy these to TradingView):**")
                    _settings_text = "\n".join(f"{k} = {v}" for k, v in _sel.params.items())
                    st.code(_settings_text, language="python")

                    # Metrics
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Net Profit", f"${_sel.net_profit:,.2f}")
                    m2.metric("Sharpe", f"{_sel.sharpe_ratio:.3f}")
                    m3.metric("Win Rate", f"{_sel.win_rate:.1%}")
                    m4.metric("Profit Factor", f"{_sel.profit_factor:.2f}" if _sel.profit_factor != float('inf') else "∞")
                    m5.metric("Max Drawdown", f"{_sel.max_drawdown_pct:.2f}%")

                    m6, m7, m8, m9, m10 = st.columns(5)
                    m6.metric("Trades", _sel.num_trades)
                    m7.metric("Sortino", f"{_sel.sortino_ratio:.3f}" if _sel.sortino_ratio != float('inf') else "∞")
                    m8.metric("Expectancy", f"${_sel.expectancy:,.2f}")
                    m9.metric("SQN", f"{_sel.sqn:.3f}")
                    m10.metric("Avg Trade", f"${_sel.avg_trade:,.2f}")

                    # ── Walk-Forward Validation ───────────────────────────
                    st.markdown("---")
                    st.markdown("### Walk-Forward Validation")
                    st.caption("Tests the winning settings across multiple time windows to check robustness.")

                    _wfc1, _wfc2 = st.columns([1, 4])
                    with _wfc1:
                        _wf_n = st.number_input("Splits", value=5, min_value=2, max_value=20, key="_wf_inline")
                    _wf_run = st.button("Run Walk-Forward", key="_wf_btn_inline")

                    if _wf_run:
                        _wf_opt = PineOptimizer(
                            ast=_opt_ast, df_price=_opt_df,
                            initial_capital=initial_capital,
                            commission_pct=pine_commission, default_qty=pine_qty,
                            pyramiding=pine_pyramiding, slippage=pine_slippage,
                            mintick=pine_mintick, margin_long=pine_margin_long,
                            margin_short=pine_margin_short,
                        )
                        with st.spinner("Running walk-forward..."):
                            try:
                                _wf_res = _wf_opt.walk_forward(_sel.params, ExtPineInterpreter, n_splits=_wf_n)
                                st.session_state['_wf_inline'] = _wf_res
                            except Exception as e:
                                st.error(f"Walk-forward failed: {e}")

                    if '_wf_inline' in st.session_state:
                        _wf = st.session_state['_wf_inline']
                        if _wf['splits']:
                            wm1, wm2, wm3 = st.columns(3)
                            wm1.metric("Avg Profit/Split", f"${_wf['avg_profit']:,.2f}")
                            wm2.metric("Avg Sharpe/Split", f"{_wf['avg_sharpe']:.3f}")
                            _cs = _wf['consistency_score']
                            wm3.metric("Consistency", f"{_cs:.0f}%",
                                       delta="Robust" if _cs >= 60 else "Fragile",
                                       delta_color="normal" if _cs >= 60 else "off")

                            _wf_profits = [s['net_profit'] for s in _wf['splits']]
                            _wf_colors = ['#00C853' if p > 0 else '#FF1744' for p in _wf_profits]
                            _wf_fig = go.Figure()
                            _wf_fig.add_trace(go.Bar(
                                x=[f"Split {s['split']}" for s in _wf['splits']],
                                y=_wf_profits, marker_color=_wf_colors,
                                text=[f"${p:,.0f}" for p in _wf_profits], textposition='outside',
                            ))
                            _wf_fig.update_layout(title="Profit by Walk-Forward Split",
                                                  yaxis_title="Net Profit ($)", showlegend=False,
                                                  height=350, template="plotly_dark",
                                                  paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                            st.plotly_chart(_wf_fig, use_container_width=True, key="wf_bar_inline")
                        else:
                            st.warning("Not enough data for walk-forward. Try fewer splits.")

        if run_backtest:
            with st.status("Running Pine Script backtest...", expanded=True) as status:

                # Step 0: AI Code Optimization (Gemini with retry + cache)
                code_to_parse = pine_code
                if use_ai_optimize and gemini_key:
                    import hashlib
                    code_hash = hashlib.md5(pine_code.encode()).hexdigest()
                    cache_key = f"_gemini_opt_{code_hash}"

                    if cache_key in st.session_state:
                        code_to_parse = st.session_state[cache_key]
                        st.write("🤖 Using cached AI optimization (same code)")
                    else:
                        st.write("🤖 AI optimizing code with Gemini...")
                        try:
                            import google.generativeai as genai
                            import time as _time
                            genai.configure(api_key=gemini_key)
                            model = genai.GenerativeModel('gemini-2.5-flash')

                            opt_prompt = f"""Convert this Pine Script to v6 and clean it for an external interpreter. Do both in one pass.

VERSION UPGRADE (if not already v6):
- Add //@version=6 at top
- sma() → ta.sma(), ema() → ta.ema(), rsi() → ta.rsi(), etc (all indicators get ta. prefix)
- highest() → ta.highest(), lowest() → ta.lowest()
- crossover() → ta.crossover(), crossunder() → ta.crossunder()
- stoch() → ta.stoch(), atr() → ta.atr(), macd() → ta.macd()
- nz() stays nz(), na() stays na()
- input(defval=x, title="y") → input.int(x, "y") or input.float(x, "y") based on type
- strategy.position_size, strategy.long, strategy.short stay the same

KEEP IDENTICAL: ALL strategy logic, conditions, math, variable names, entry/exit rules, strategy.entry/close/exit calls, if/else/for/while/switch, custom functions.

REMOVE: plot/plotshape/plotchar/plotarrow/plotcandle/plotbar, bgcolor/barcolor/fill/hline, label.*/line.*/box.*/table.*, alert/alertcondition, all comments, color.* not used in logic.

SIMPLIFY: Remove 'series float'/'simple float' type annotations.

Output ONLY clean v6 Pine Script. No explanations, no markdown, no code fences.

{pine_code}"""

                            # Retry with backoff for rate limits
                            for attempt in range(3):
                                try:
                                    response = model.generate_content(opt_prompt)
                                    cleaned = response.text.strip()
                                    if cleaned.startswith("```"):
                                        cleaned = '\n'.join(cleaned.split('\n')[1:])
                                    if cleaned.endswith("```"):
                                        cleaned = '\n'.join(cleaned.split('\n')[:-1])
                                    code_to_parse = cleaned
                                    st.session_state[cache_key] = cleaned
                                    st.write("✅ AI optimization complete")
                                    break
                                except Exception as retry_err:
                                    if '429' in str(retry_err) and attempt < 2:
                                        wait = (attempt + 1) * 15
                                        st.write(f"⏳ Rate limited — waiting {wait}s (attempt {attempt+2}/3)...")
                                        _time.sleep(wait)
                                    else:
                                        raise retry_err
                        except ImportError:
                            st.warning("Install: `pip install google-generativeai`. Using original code.")
                        except Exception as e:
                            st.warning(f"Gemini optimization skipped: {str(e)[:200]}. Using original code.")

                # Step 1: Parse Pine Script
                st.write("📝 Lexing & parsing Pine Script...")
                try:
                    tokens = lexer(code_to_parse)
                    ast = Parser(tokens).parse()
                    if ast.statements:
                        st.write(f"Parsed **{len(ast.statements)}** top-level statements")
                    else:
                        st.error("Could not parse any statements from the Pine Script. "
                                 "Make sure your code is valid Pine Script v6.")
                        st.stop()
                except Exception as e:
                    st.error(f"Parse error: {e}")
                    st.stop()

                # ── Step 1.5: DRY RUN — validate on synthetic data before fetching real data ──
                st.write("🧪 Dry-run validation (testing on synthetic data before fetching)...")
                try:
                    # Generate 200 bars of synthetic OHLCV
                    _rng = np.random.default_rng(42)
                    _n = 200
                    _close = 100 + np.cumsum(_rng.standard_normal(_n) * 1.5)
                    _syn_df = pd.DataFrame({
                        'Open': _close + _rng.standard_normal(_n) * 0.5,
                        'High': _close + np.abs(_rng.standard_normal(_n)) * 2,
                        'Low': _close - np.abs(_rng.standard_normal(_n)) * 2,
                        'Close': _close,
                        'Volume': _rng.integers(100000, 5000000, _n).astype(float),
                    }, index=pd.bdate_range("2023-01-03", periods=_n))

                    # Collect input overrides
                    input_overrides = {}
                    for key, val in st.session_state.items():
                        if key.startswith("pine_input_"):
                            # Key format: pine_input_N_Title — strip prefix and counter
                            _remainder = key[len("pine_input_"):]
                            _parts = _remainder.split("_", 1)
                            input_name = _parts[1] if len(_parts) > 1 and _parts[0].isdigit() else _remainder
                            input_overrides[input_name] = val

                    _dry_interp = PineInterpreter(
                        ast, _syn_df,
                        initial_capital=initial_capital,
                        commission_pct=pine_commission,
                        default_qty=pine_qty,
                        pyramiding=pine_pyramiding,
                        slippage=pine_slippage,
                        mintick=pine_mintick,
                        margin_long=pine_margin_long,
                        margin_short=pine_margin_short,
                        input_overrides=input_overrides,
                    )
                    _dry_trades = _dry_interp.execute()

                    # Store input defs from dry run
                    st.session_state['_pine_input_defs'] = _dry_interp.input_defs

                    if len(_dry_trades) > 0:
                        st.write(f"✅ Dry run passed — **{len(_dry_trades)}** trades on synthetic data. Strategy logic works.")
                    else:
                        st.warning("⚠️ **Dry run produced 0 trades on synthetic data.**")
                        if _dry_interp.warnings:
                            with st.expander("View warnings", expanded=False):
                                for w in _dry_interp.warnings[:10]:
                                    st.caption(f"⚠️ {w}")
                        st.info("This can be normal — many strategies need real market structure "
                                "(trends, support/resistance, volume) to trigger. Synthetic random data won't have that.")
                        if not st.checkbox("⚡ Proceed anyway — fetch real data and run", value=False, key="_force_fetch"):
                            st.stop()

                except Exception as dry_err:
                    st.error(f"⚠️ **Dry run failed:** {dry_err}\n\n"
                             f"The strategy code has execution errors. Fix the code before fetching data.")
                    st.stop()

                # Step 2: Fetch OHLCV data (with local cache)
                df_price = None
                end_date = pd.Timestamp.now()
                start_date = end_date - pine_offset

                # ── Direct dataset pick (bypasses fetch) ─────────────────
                _direct_ds = st.session_state.get('_bt_use_cached_dataset')
                if _direct_ds and os.path.exists(_direct_ds):
                    try:
                        df_price = pd.read_parquet(_direct_ds)
                        # Normalize columns to Title Case for interpreter
                        _col_remap = {}
                        for _c in df_price.columns:
                            if _c.lower() == 'open': _col_remap[_c] = 'Open'
                            elif _c.lower() == 'high': _col_remap[_c] = 'High'
                            elif _c.lower() == 'low': _col_remap[_c] = 'Low'
                            elif _c.lower() == 'close': _col_remap[_c] = 'Close'
                            elif _c.lower() == 'volume': _col_remap[_c] = 'Volume'
                        if _col_remap:
                            df_price = df_price.rename(columns=_col_remap)
                        # Ensure DatetimeIndex
                        if not isinstance(df_price.index, pd.DatetimeIndex):
                            for _tc in ['time', 'timestamp', 'date', 'datetime', 'ts_event']:
                                if _tc in df_price.columns:
                                    df_price[_tc] = pd.to_datetime(df_price[_tc])
                                    df_price = df_price.set_index(_tc)
                                    break
                        st.write(f"✅ Loaded **{len(df_price):,}** bars from **{os.path.basename(_direct_ds).replace('.parquet','')}**")
                    except Exception as _ds_err:
                        st.warning(f"Failed to load dataset: {_ds_err}. Falling back to fetch.")
                        df_price = None

                # ── Data cache system ─────────────────────────────────────
                _cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_cache')
                os.makedirs(_cache_dir, exist_ok=True)
                _cache_file = os.path.join(_cache_dir,
                    f"{pine_ticker.upper().replace('!','')}_{pine_timeframe}_{pine_range_label}.parquet")

                _use_cache = False
                if os.path.exists(_cache_file):
                    _cache_age = (pd.Timestamp.now() - pd.Timestamp(os.path.getmtime(_cache_file), unit='s'))
                    _cache_bars = 0
                    try:
                        _cache_meta = pd.read_parquet(_cache_file, columns=['Close'])
                        _cache_bars = len(_cache_meta)
                    except:
                        pass
                    st.info(f"📦 Cached data found: **{_cache_bars:,}** bars, saved {_cache_age.days}d ago")
                    _use_cache = st.checkbox("Use cached data (saves API calls)", value=True, key="_use_cache")

                if _use_cache and os.path.exists(_cache_file):
                    try:
                        df_price = pd.read_parquet(_cache_file)
                        st.write(f"✅ Loaded **{len(df_price):,}** bars from cache")
                    except Exception as ce:
                        st.warning(f"Cache read failed: {ce}. Fetching fresh data...")
                        df_price = None

                # ── Futures symbol mapping ────────────────────────────────
                # Map TradingView-style symbols to Databento dataset + symbol
                FUTURES_MAP = {
                    # CME E-mini / Micro — continuous front-month: ROOT.c.0
                    'NQ': ('GLBX.MDP3', 'NQ.c.0'),  'NQ1!': ('GLBX.MDP3', 'NQ.c.0'),
                    'ES': ('GLBX.MDP3', 'ES.c.0'),  'ES1!': ('GLBX.MDP3', 'ES.c.0'),
                    'YM': ('GLBX.MDP3', 'YM.c.0'),  'YM1!': ('GLBX.MDP3', 'YM.c.0'),
                    'RTY': ('GLBX.MDP3', 'RTY.c.0'), 'RTY1!': ('GLBX.MDP3', 'RTY.c.0'),
                    'MNQ': ('GLBX.MDP3', 'MNQ.c.0'), 'MES': ('GLBX.MDP3', 'MES.c.0'),
                    # Metals (COMEX)
                    'GC': ('GLBX.MDP3', 'GC.c.0'),  'GC1!': ('GLBX.MDP3', 'GC.c.0'),
                    'SI': ('GLBX.MDP3', 'SI.c.0'),  'SI1!': ('GLBX.MDP3', 'SI.c.0'),
                    'HG': ('GLBX.MDP3', 'HG.c.0'),  'HG1!': ('GLBX.MDP3', 'HG.c.0'),
                    'MGC': ('GLBX.MDP3', 'MGC.c.0'), 'SIL': ('GLBX.MDP3', 'SIL.c.0'),
                    # Energy (NYMEX)
                    'CL': ('GLBX.MDP3', 'CL.c.0'),  'CL1!': ('GLBX.MDP3', 'CL.c.0'),
                    'NG': ('GLBX.MDP3', 'NG.c.0'),  'NG1!': ('GLBX.MDP3', 'NG.c.0'),
                    'MCL': ('GLBX.MDP3', 'MCL.c.0'),
                    # Treasuries (CBOT)
                    'ZB': ('GLBX.MDP3', 'ZB.c.0'),  'ZN': ('GLBX.MDP3', 'ZN.c.0'),
                    'ZF': ('GLBX.MDP3', 'ZF.c.0'),  'ZT': ('GLBX.MDP3', 'ZT.c.0'),
                    # Grains (CBOT)
                    'ZC': ('GLBX.MDP3', 'ZC.c.0'),  'ZS': ('GLBX.MDP3', 'ZS.c.0'),
                    'ZW': ('GLBX.MDP3', 'ZW.c.0'),
                    # Forex
                    '6E': ('GLBX.MDP3', '6E.c.0'),  '6J': ('GLBX.MDP3', '6J.c.0'),
                    '6B': ('GLBX.MDP3', '6B.c.0'),
                }

                # Auto-detect tick size from symbol
                TICK_MAP = {
                    'NQ': 0.25, 'MNQ': 0.25, 'ES': 0.25, 'MES': 0.25,
                    'YM': 1.0, 'RTY': 0.10,
                    'GC': 0.10, 'MGC': 0.10, 'SI': 0.005, 'SIL': 0.005, 'HG': 0.0005,
                    'CL': 0.01, 'MCL': 0.01, 'NG': 0.001,
                    'ZB': 0.03125, 'ZN': 0.015625, 'ZF': 0.0078125, 'ZT': 0.00390625,
                    'ZC': 0.25, 'ZS': 0.25, 'ZW': 0.25,
                    '6E': 0.00005, '6J': 0.0000005, '6B': 0.0001,
                }

                # Clean ticker (remove ! suffix from TradingView format)
                clean_sym = pine_ticker.upper().replace('1!', '').replace('!', '').strip()
                is_futures = clean_sym in FUTURES_MAP

                # Auto-set tick size if futures
                if is_futures and clean_sym in TICK_MAP:
                    pine_mintick = TICK_MAP[clean_sym]

                # ── Databento fetch (skip if cache loaded) ────────────────
                if df_price is None and data_provider == "Databento (futures + stocks)" and db_key:
                    st.write(f"📊 Fetching {pine_range_label} of `{pine_ticker}` ({pine_timeframe}) from Databento...")
                    try:
                        import databento as db

                        client = db.Historical(db_key)

                        # Timeframe mapping
                        db_schema_map = {
                            '1min': 'ohlcv-1m', '3min': 'ohlcv-1m', '5min': 'ohlcv-1m',
                            '15min': 'ohlcv-1m', '30min': 'ohlcv-1m',
                            '1h': 'ohlcv-1h', '1d': 'ohlcv-1d', '1W': 'ohlcv-1d',
                        }
                        schema = db_schema_map.get(pine_timeframe, 'ohlcv-1m')

                        if is_futures:
                            dataset, symbol = FUTURES_MAP[clean_sym]
                        else:
                            dataset = 'XNAS.ITCH'  # US equities
                            symbol = clean_sym

                        data = client.timeseries.get_range(
                            dataset=dataset,
                            symbols=[symbol],
                            schema=schema,
                            start=start_date.strftime("%Y-%m-%d"),
                            end=end_date.strftime("%Y-%m-%d"),
                            stype_in='continuous' if is_futures else 'raw_symbol',
                        )

                        df_price = data.to_df()

                        if df_price.empty:
                            st.warning("Databento returned no data. Trying Yahoo Finance...")
                            df_price = None
                        else:
                            # Standardize columns
                            col_remap = {'open': 'Open', 'high': 'High', 'low': 'Low',
                                         'close': 'Close', 'volume': 'Volume'}
                            df_price.rename(columns=col_remap, inplace=True)

                            # Keep only OHLCV
                            keep_cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df_price.columns]
                            df_price = df_price[keep_cols]

                            # Resample if needed (e.g., 5min from 1min data)
                            resample_map = {
                                '3min': '3T', '5min': '5T', '15min': '15T', '30min': '30T',
                            }
                            if pine_timeframe in resample_map and schema == 'ohlcv-1m':
                                rule = resample_map[pine_timeframe]
                                df_price = df_price.resample(rule).agg({
                                    'Open': 'first', 'High': 'max', 'Low': 'min',
                                    'Close': 'last', 'Volume': 'sum',
                                }).dropna()

                            if pine_timeframe == '1W' and schema == 'ohlcv-1d':
                                df_price = df_price.resample('W').agg({
                                    'Open': 'first', 'High': 'max', 'Low': 'min',
                                    'Close': 'last', 'Volume': 'sum',
                                }).dropna()

                            st.write(f"✅ **{len(df_price):,}** bars from Databento "
                                     f"({df_price.index[0].strftime('%Y-%m-%d %H:%M')} → "
                                     f"{df_price.index[-1].strftime('%Y-%m-%d %H:%M')})")

                    except ImportError:
                        st.warning("**databento** not installed. Run: `pip install databento`. Falling back to Yahoo Finance...")
                        df_price = None
                    except Exception as e:
                        st.warning(f"Databento error: {e}. Falling back to Yahoo Finance...")
                        df_price = None

                # ── Yahoo Finance fallback ────────────────────────────────
                if df_price is None:
                    # Map futures symbols to yfinance format
                    yf_futures = {
                        'NQ': 'NQ=F', 'ES': 'ES=F', 'YM': 'YM=F', 'RTY': 'RTY=F',
                        'GC': 'GC=F', 'SI': 'SI=F', 'HG': 'HG=F',
                        'CL': 'CL=F', 'NG': 'NG=F',
                        'ZB': 'ZB=F', 'ZN': 'ZN=F', 'ZC': 'ZC=F', 'ZS': 'ZS=F',
                    }
                    yf_symbol = yf_futures.get(clean_sym, pine_ticker)

                    st.write(f"📊 Fetching `{yf_symbol}` ({pine_timeframe}) from Yahoo Finance...")
                    try:
                        import yfinance as yf
                        yf_tf_map = {
                            '1min': '1m', '3min': '5m', '5min': '5m',
                            '15min': '15m', '30min': '30m',
                            '1h': '1h', '1d': '1d', '1W': '1wk',
                        }
                        yf_tf = yf_tf_map.get(pine_timeframe, '1d')

                        yf_max_days = {'1m': 7, '5m': 60, '15m': 60, '30m': 60, '1h': 730}
                        max_days = yf_max_days.get(yf_tf, 365 * pine_range_val)
                        actual_start = max(start_date, end_date - pd.Timedelta(days=max_days))

                        if yf_tf in yf_max_days and max_days < 365 * pine_range_val:
                            st.info(f"⚠️ Yahoo Finance only provides {max_days} days of `{yf_tf}` data. "
                                    f"Use Databento for full futures intraday history.")

                        df_price = yf.Ticker(yf_symbol).history(
                            start=actual_start.strftime("%Y-%m-%d"),
                            end=end_date.strftime("%Y-%m-%d"),
                            interval=yf_tf,
                        )

                    except ImportError:
                        st.error("**yfinance** not installed. Run: `pip install yfinance`")
                        st.stop()
                    except Exception as e:
                        st.error(f"Failed to fetch data: {e}")
                        st.stop()

                if df_price is None or df_price.empty:
                    st.error(f"No data returned for `{pine_ticker}` with interval `{pine_timeframe}`. "
                             f"Check the symbol and try again.")
                    st.stop()

                # Standardize column names
                df_price.columns = [c.strip().title() for c in df_price.columns]
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if col not in df_price.columns:
                        for c in df_price.columns:
                            if c.lower() == col.lower():
                                df_price.rename(columns={c: col}, inplace=True)

                if not isinstance(df_price.index, pd.DatetimeIndex):
                    df_price.index = pd.to_datetime(df_price.index)

                df_price = df_price.dropna(subset=['Open', 'High', 'Low', 'Close'])

                st.write(f"✅ **{len(df_price):,}** bars loaded "
                         f"({df_price.index[0].strftime('%Y-%m-%d %H:%M')} → "
                         f"{df_price.index[-1].strftime('%Y-%m-%d %H:%M')})")

                # ── Save to cache for future runs ─────────────────────────
                if not _use_cache and len(df_price) > 0:
                    try:
                        df_price.to_parquet(_cache_file)
                        st.write(f"💾 Data cached to `data_cache/` — next run can skip the API call")
                    except Exception as save_err:
                        pass  # silently fail cache save

                # Step 3: Execute strategy bar-by-bar
                st.write("⚡ Executing strategy bar-by-bar...")

                # Collect input overrides from session state
                input_overrides = {}
                for key, val in st.session_state.items():
                    if key.startswith("pine_input_"):
                        _remainder = key[len("pine_input_"):]
                        _parts = _remainder.split("_", 1)
                        input_name = _parts[1] if len(_parts) > 1 and _parts[0].isdigit() else _remainder
                        input_overrides[input_name] = val

                interp = PineInterpreter(
                    ast, df_price,
                    initial_capital=initial_capital,
                    commission_pct=pine_commission,
                    default_qty=pine_qty,
                    pyramiding=pine_pyramiding,
                    slippage=pine_slippage,
                    mintick=pine_mintick,
                    margin_long=pine_margin_long,
                    margin_short=pine_margin_short,
                    input_overrides=input_overrides,
                )
                trades = interp.execute()

                # Store everything in session state for Apply re-runs
                st.session_state['_pine_input_defs'] = interp.input_defs
                st.session_state['_pine_ast'] = ast
                st.session_state['_pine_cached_data'] = df_price
                st.session_state['_pine_strategy_name'] = interp.strategy_name
                st.session_state['_pine_source_code'] = pine_code

                # Save interpreter's effective settings (strategy() overrides sidebar defaults)
                st.session_state['_pine_effective_settings'] = {
                    'initial_capital': interp.initial_capital,
                    'pyramiding': interp.pyramiding,
                    'slippage': interp.slippage,
                    'default_qty': interp.default_qty,
                    'commission_pct': interp.commission_pct,
                    'commission_cash': getattr(interp, 'commission_cash', 0.0),
                    'commission_type': getattr(interp, 'commission_type', ''),
                    'mintick': interp.mintick,
                }

                if len(trades) == 0:
                    st.error("The strategy generated **0 trades** on this data. "
                             "This usually means the entry conditions never triggered. "
                             "Try a different ticker, timeframe, or date range.")
                    if interp.warnings:
                        st.markdown("**⚠️ Interpreter Warnings:**")
                        for w in interp.warnings[:20]:
                            st.caption(w)
                    # Diagnostic: show key variable states from last bar
                    st.markdown("**🔍 Debug: Final Variable States**")
                    _debug_vars = ['regime', 'regimeAllowsEntry', 'canScale', 'canTrade',
                                   'inLossCooldown', 'isBullishEntry', 'isBearishEntry',
                                   'longSignal', 'shortSignal', 'isFlat', 'dcaReady',
                                   'activeMaxDCA', 'dollarStopHit', 'tpReady',
                                   'isTrend', 'isRange', 'isCaution', 'useRegime',
                                   'currentMode', 'filtersPass', 'volumeAllowed', 'timeAllowed',
                                   'strategy.position_size', 'strategy.openprofit']
                    _found = {}
                    for _dv in _debug_vars:
                        if _dv in interp.globals:
                            _found[_dv] = interp.globals[_dv]
                    if _found:
                        for k, v in _found.items():
                            st.code(f"{k} = {v}", language="python")
                    else:
                        st.info("No recognized strategy variables found — "
                                "the strategy may use different variable names.")
                    st.caption(f"Pyramiding: {interp.pyramiding} | "
                               f"Positions: {len(interp.positions)} | "
                               f"Pending orders: {len(interp.pending_orders)}")
                    st.stop()

                pine_generated_df = interp.to_dataframe()
                st.session_state['_pine_trades_df'] = pine_generated_df
                st.write(f"✅ Generated **{len(pine_generated_df)}** trades")

                if interp.warnings:
                    for w in interp.warnings[:5]:
                        st.warning(f"⚠️ {w}")

                status.update(label=f"✅ Backtest complete — {len(pine_generated_df)} trades generated",
                              state="complete")

        # ── Apply Inputs: fast re-run using cached AST + data ─────────
        if rerun_inputs and '_pine_ast' in st.session_state and '_pine_cached_data' in st.session_state:
            with st.status("Re-running with new inputs...", expanded=True) as status:
                progress_re = st.progress(0, text="Collecting inputs...")

                input_overrides = {}
                for key, val in st.session_state.items():
                    if key.startswith("pine_input_"):
                        _remainder = key[len("pine_input_"):]
                        _parts = _remainder.split("_", 1)
                        input_name = _parts[1] if len(_parts) > 1 and _parts[0].isdigit() else _remainder
                        input_overrides[input_name] = val

                progress_re.progress(20, text="Executing strategy...")

                _cached_ast = st.session_state['_pine_ast']
                _cached_data = st.session_state['_pine_cached_data']

                interp = PineInterpreter(
                    _cached_ast, _cached_data,
                    initial_capital=initial_capital,
                    commission_pct=pine_commission,
                    default_qty=pine_qty,
                    pyramiding=pine_pyramiding,
                    slippage=pine_slippage,
                    mintick=pine_mintick,
                    margin_long=pine_margin_long,
                    margin_short=pine_margin_short,
                    input_overrides=input_overrides,
                )
                trades = interp.execute()

                progress_re.progress(90, text="Building results...")

                st.session_state['_pine_input_defs'] = interp.input_defs
                st.session_state['_pine_strategy_name'] = interp.strategy_name
                st.session_state['_pine_source_code'] = pine_code

                # Save interpreter's effective settings (strategy() overrides sidebar defaults)
                st.session_state['_pine_effective_settings'] = {
                    'initial_capital': interp.initial_capital,
                    'pyramiding': interp.pyramiding,
                    'slippage': interp.slippage,
                    'default_qty': interp.default_qty,
                    'commission_pct': interp.commission_pct,
                    'commission_cash': getattr(interp, 'commission_cash', 0.0),
                    'commission_type': getattr(interp, 'commission_type', ''),
                    'mintick': interp.mintick,
                }

                if len(trades) == 0:
                    st.warning("⚠️ 0 trades with these input settings. Showing previous results.")
                    # Don't update session state — keep previous results
                    progress_re.progress(100, text="Done (0 trades)")
                    status.update(label="⚠️ 0 trades with new inputs — showing previous results",
                                  state="complete")
                else:
                    pine_generated_df = interp.to_dataframe()
                    st.session_state['_pine_trades_df'] = pine_generated_df

                    progress_re.progress(100, text="Done!")
                    status.update(label=f"✅ Re-run complete — {len(pine_generated_df)} trades with new inputs",
                                  state="complete")

        # ── Load from session state if available (persists across rerenders) ──
        if pine_generated_df is None and '_pine_trades_df' in st.session_state:
            pine_generated_df = st.session_state['_pine_trades_df']

        # ── Show results if we have them ──────────────────────────────
        if pine_generated_df is not None and not pine_generated_df.empty:
            # Execution details
            with st.expander("🔍 Execution Details", expanded=False):
                c1, c2, c3 = st.columns(3)
                c1.metric("Strategy", st.session_state.get('_pine_strategy_name', 'Strategy'))
                _cached_data = st.session_state.get('_pine_cached_data')
                c2.metric("Bars Processed", f"{len(_cached_data):,}" if _cached_data is not None else "—")
                c3.metric("Trades Generated", len(pine_generated_df))

                # Show effective strategy() settings (may differ from sidebar defaults)
                _eff_s = st.session_state.get('_pine_effective_settings')
                if _eff_s:
                    st.markdown("**Effective Strategy Settings** *(from Pine Script `strategy()` call)*")
                    e1, e2, e3, e4 = st.columns(4)
                    e1.metric("Initial Capital", f"${_eff_s['initial_capital']:,.0f}")
                    e2.metric("Pyramiding", f"{_eff_s['pyramiding']:,}")
                    if _eff_s.get('commission_cash', 0) > 0:
                        e3.metric("Commission", f"${_eff_s['commission_cash']}/contract")
                    elif _eff_s.get('commission_pct', 0) > 0:
                        e3.metric("Commission", f"{_eff_s['commission_pct']*100:.2f}%")
                    else:
                        e3.metric("Commission", "0")
                    e4.metric("Slippage", f"{_eff_s['slippage']} ticks")

            # ── Dynamic Strategy Inputs (use st.form to prevent rerenders) ──
            _input_defs = st.session_state.get('_pine_input_defs', [])
            if _input_defs:
                with st.expander(f"⚙️ Strategy Inputs ({len(_input_defs)} parameters)", expanded=False):
                    st.caption("Change values below, then click **🔄 Apply** above to re-run with new settings.")

                    grouped = {}
                    for inp in _input_defs:
                        g = inp.get('group', '') or 'General'
                        if g not in grouped: grouped[g] = []
                        grouped[g].append(inp)

                    _inp_counter = 0
                    for group_name, inputs in grouped.items():
                        st.markdown(f"**{group_name}**")
                        cols = st.columns(min(3, len(inputs)))
                        for idx, inp in enumerate(inputs):
                            col = cols[idx % len(cols)]
                            key = f"pine_input_{_inp_counter}_{inp['key']}"
                            _inp_counter += 1
                            help_text = inp.get('tooltip', '') or None

                            with col:
                                if inp['type'] == 'bool':
                                    st.checkbox(inp['title'], value=bool(inp['defval']),
                                               key=key, help=help_text)
                                elif inp['type'] == 'int':
                                    kw = {'label': inp['title'], 'value': int(inp.get('defval', 0) or 0), 'key': key, 'help': help_text}
                                    if inp.get('minval') is not None: kw['min_value'] = int(inp['minval'])
                                    if inp.get('maxval') is not None: kw['max_value'] = int(inp['maxval'])
                                    if inp.get('step') is not None: kw['step'] = int(inp['step'])
                                    st.number_input(**kw)
                                elif inp['type'] == 'float':
                                    kw = {'label': inp['title'], 'value': float(inp.get('defval', 0.0) or 0.0), 'key': key, 'help': help_text}
                                    if inp.get('minval') is not None: kw['min_value'] = float(inp['minval'])
                                    if inp.get('maxval') is not None: kw['max_value'] = float(inp['maxval'])
                                    if inp.get('step') is not None: kw['step'] = float(inp['step'])
                                    st.number_input(**kw)
                                elif inp['type'] == 'string' and inp.get('options'):
                                    options = inp['options']
                                    defval = str(inp.get('defval', ''))
                                    idx_default = options.index(defval) if defval in options else 0
                                    st.selectbox(inp['title'], options=options, index=idx_default,
                                                key=key, help=help_text)
                                elif inp['type'] == 'source' and inp.get('options'):
                                    options = inp['options']
                                    defval = str(inp.get('defval', 'close'))
                                    idx_default = options.index(defval) if defval in options else 0
                                    st.selectbox(inp['title'], options=options, index=idx_default,
                                                key=key, help=help_text)
                                else:
                                    st.text_input(inp['title'], value=str(inp.get('defval', '')),
                                                 key=key, help=help_text)
                        st.markdown("---")

            # ── Trade Chart (price + entries/exits) ───────────────────────
            _chart_data = st.session_state.get('_pine_cached_data')
            if _chart_data is not None:
                with st.expander("📈 Trade Chart", expanded=True):
                    _cd = _chart_data
                    if len(_cd) > 5000:
                        _step = max(1, len(_cd) // 5000)
                        _cd = _cd.iloc[::_step]

                    fig_chart = go.Figure()
                    fig_chart.add_trace(go.Scatter(
                        x=_cd.index, y=_cd['Close'], mode='lines',
                        line=dict(color='#8b8b8b', width=1), name='Price',
                    ))

                    for _, trade in pine_generated_df.iterrows():
                        try:
                            entry_dt = pd.to_datetime(trade.get('Date/Time', ''))
                            exit_dt = pd.to_datetime(trade.get('Exit Date/Time', ''))
                            is_long = str(trade.get('Type', '')).lower() == 'long'
                            profit = float(trade.get('Profit', 0))
                            fig_chart.add_trace(go.Scatter(
                                x=[entry_dt], y=[trade['Price']], mode='markers',
                                marker=dict(symbol='triangle-up' if is_long else 'triangle-down',
                                            size=8, color='#22d3c5' if is_long else '#ff4757'),
                                showlegend=False,
                                hovertemplate=f"{'Long' if is_long else 'Short'} @ {trade['Price']}<extra></extra>",
                            ))
                            fig_chart.add_trace(go.Scatter(
                                x=[exit_dt], y=[trade['Exit Price']], mode='markers',
                                marker=dict(symbol='x', size=7, color='#22d3c5' if profit > 0 else '#ff4757'),
                                showlegend=False,
                                hovertemplate=f"Exit P&L: ${profit:,.2f}<extra></extra>",
                            ))
                        except: pass

                    fig_chart.update_layout(
                        height=450, template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
                        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', title='Price'),
                        margin=dict(l=50, r=20, t=30, b=40), showlegend=False,
                    )
                    st.plotly_chart(fig_chart, use_container_width=True, key="pine_trade_chart")

            # ── Strategy Save/Load ────────────────────────────────────────
            with st.expander("💾 Save / Load Strategy", expanded=False):
                import json as _json
                _strat_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_strategies')
                os.makedirs(_strat_dir, exist_ok=True)

                _col_sv, _col_ld = st.columns(2)
                with _col_sv:
                    st.markdown("**Save Current**")
                    _save_name = st.text_input("Strategy Name", key="_strat_save_name", placeholder="My Strategy")
                    if st.button("💾 Save", key="_save_strat") and _save_name:
                        save_data = {'name': _save_name, 'code': pine_code,
                                     'inputs': {k: v for k, v in st.session_state.items() if k.startswith('pine_input_')}}
                        _sp = os.path.join(_strat_dir, f"{_save_name.replace(' ','_')}.json")
                        with open(_sp, 'w') as _sf: _json.dump(save_data, _sf, indent=2, default=str)
                        st.success(f"✅ Saved: {_save_name}")

                with _col_ld:
                    st.markdown("**Load Saved**")
                    _saved = [f.replace('.json','').replace('_',' ') for f in os.listdir(_strat_dir) if f.endswith('.json')] if os.path.exists(_strat_dir) else []
                    if _saved:
                        _load_sel = st.selectbox("Strategy", _saved, key="_strat_load_sel")
                        if st.button("📂 Load", key="_load_strat"):
                            _lp = os.path.join(_strat_dir, f"{_load_sel.replace(' ','_')}.json")
                            with open(_lp) as _lf: loaded = _json.load(_lf)
                            st.session_state['_loaded_pine_code'] = loaded.get('code', '')
                            for k, v in loaded.get('inputs', {}).items(): st.session_state[k] = v
                            st.success(f"✅ Loaded: {_load_sel}")
                            st.rerun()
                    else:
                        st.caption("No saved strategies yet")

    else:
        st.markdown("""
        <div style="text-align:center; padding: 40px 20px; opacity: 0.5;">
            <div style="font-size:2rem; margin-bottom:8px;">🌲</div>
            <div style="font-size:0.9rem;">Paste a Pine Script strategy to backtest it on real data</div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    if pine_generated_df is None or pine_generated_df.empty:
        st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED DATA PATH — CSV or Pine Script trades merge here
# ═══════════════════════════════════════════════════════════════════════════════

if input_mode == "📁 Upload CSV/XLSX" and not uploaded_file:
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px;">
        <div style="font-size:3rem; margin-bottom:8px;">📊</div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:1.2rem; opacity:0.8;">
            Upload a TradingView CSV or Excel file to begin
        </div>
        <div style="font-size:0.85rem; opacity:0.4; margin-top:6px;">
            Just Trades will auto-detect your columns and run full analytics
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📖 What CSV format is expected?"):
        st.markdown("""
        This tool auto-detects columns from TradingView strategy exports.
        Common column names it recognizes:

        **Required:** A profit/P&L column (e.g., `Profit`, `Net Profit`, `P&L`)

        **Optional (enhance analysis):**
        `Date/Time`, `Exit Date/Time`, `Type`, `Price`, `Exit Price`,
        `Contracts`/`Qty`, `Run-up`, `Drawdown`, `Signal`, `Commission`

        Export your strategy results from TradingView → Strategy Tester → Export trades as CSV.
        """)
    st.stop()

# Choose data source
if input_mode == "🌲 Pine Script" and pine_generated_df is not None:
    df_raw = pine_generated_df
else:
    file_name = uploaded_file.name.lower()
    if file_name.endswith(('.xlsx', '.xls')):
        # Try to find "List of trades" sheet (TradingView multi-sheet export)
        uploaded_file.seek(0)
        xl = pd.ExcelFile(uploaded_file)
        trade_sheet = None
        # Priority 1: exact "list of trades" match
        for s in xl.sheet_names:
            if s.lower().strip() == 'list of trades':
                trade_sheet = s
                break
        # Priority 2: sheet with most columns (trade list has 15, summaries have 7)
        if trade_sheet is None and len(xl.sheet_names) > 1:
            best_sheet, best_cols = None, 0
            for s in xl.sheet_names:
                ncols = len(xl.parse(s, nrows=1).columns)
                if ncols > best_cols:
                    best_cols = ncols
                    best_sheet = s
            trade_sheet = best_sheet
        df_raw = xl.parse(trade_sheet if trade_sheet else 0)
    else:
        df_raw = pd.read_csv(uploaded_file)
df, col_map = parse_tradingview_csv(df_raw)

# Show raw preview
with st.expander("📋 Raw Data Preview", expanded=False):
    st.dataframe(df.head(20), use_container_width=True)
    st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# ── Select profit column ──────────────────────────────────────────────────────
use_cum_diff = False
if manual_profit_col:
    profit_col = st.selectbox("Select Profit Column", df.columns)
    is_cumulative = st.checkbox("This column is cumulative (derive per-trade from diff)", value=False)
    if is_cumulative:
        use_cum_diff = True
else:
    profit_col = col_map.get("profit")
    if profit_col is None and col_map.get("cum_profit"):
        profit_col = col_map["cum_profit"]
        use_cum_diff = True
        st.info(f"No per-trade profit column found. Using cumulative column `{profit_col}` and computing differences.")
    elif profit_col is None:
        st.warning("Could not auto-detect a profit column. Please select manually.")
        profit_col = st.selectbox("Select Profit Column", df.columns)

profits_series = clean_numeric(df[profit_col])

if use_cum_diff:
    # Derive per-trade profits from cumulative column
    profits_series = profits_series.diff()
    profits_series.iloc[0] = clean_numeric(df[profit_col]).iloc[0]  # first trade is as-is

valid_mask = profits_series.notna()

if valid_mask.sum() == 0:
    st.error(f"No valid numeric values found in column '{profit_col}'.")
    st.stop()

if (~valid_mask).sum() > 0:
    st.warning(f"Dropped {(~valid_mask).sum()} rows with non-numeric values in '{profit_col}'.")

profits = profits_series[valid_mask].values

# ── Optional columns ──────────────────────────────────────────────────────────
dates = None
if col_map.get("entry_date"):
    d = parse_tv_date(df.loc[valid_mask, col_map["entry_date"]])
    if d.notna().sum() > 0:
        dates = d.reset_index(drop=True)

contracts = None
if col_map.get("contracts"):
    c = clean_numeric(df.loc[valid_mask, col_map["contracts"]])
    if c.notna().sum() > 0:
        contracts = c.values

entry_prices = None
if col_map.get("entry_price"):
    ep = clean_numeric(df.loc[valid_mask, col_map["entry_price"]])
    if ep.notna().sum() > 0:
        entry_prices = ep.values

exit_prices = None
if col_map.get("exit_price"):
    xp = clean_numeric(df.loc[valid_mask, col_map["exit_price"]])
    if xp.notna().sum() > 0:
        exit_prices = xp.values

trade_types = None
if col_map.get("type"):
    trade_types = df.loc[valid_mask, col_map["type"]].values

runups = None
if col_map.get("runup"):
    ru = clean_numeric(df.loc[valid_mask, col_map["runup"]])
    if ru.notna().sum() > 0:
        runups = ru.values

dd_per_trade = None
if col_map.get("drawdown_col"):
    ddt = clean_numeric(df.loc[valid_mask, col_map["drawdown_col"]])
    if ddt.notna().sum() > 0:
        dd_per_trade = ddt.values

commissions = None
if col_map.get("commission"):
    cm = clean_numeric(df.loc[valid_mask, col_map["commission"]])
    if cm.notna().sum() > 0:
        commissions = cm.values

signals = None
if col_map.get("signal"):
    signals = df.loc[valid_mask, col_map["signal"]].values


# ═══════════════════════════════════════════════════════════════════════════════
# RUN ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

def _fmt(val, fmt_str=".3f", prefix="", suffix=""):
    """Safely format a value that might be inf/nan."""
    if val is None or (isinstance(val, float) and (np.isinf(val) or np.isnan(val))):
        return "∞" if (isinstance(val, float) and np.isinf(val) and val > 0) else "N/A"
    return f"{prefix}{val:{fmt_str}}{suffix}"


def _safe_df(df_input):
    """Convert DataFrame to Arrow-safe format by ensuring clean dtypes.
    Fixes PyArrow serialization errors with mixed-type columns."""
    if df_input is None or df_input.empty:
        return df_input
    df_out = df_input.copy()
    for col in df_out.columns:
        if df_out[col].dtype == object:
            df_out[col] = df_out[col].astype(str)
        elif df_out[col].dtype.name.startswith('float'):
            df_out[col] = pd.to_numeric(df_out[col], errors='coerce')
        elif df_out[col].dtype.name.startswith('int'):
            df_out[col] = pd.to_numeric(df_out[col], errors='coerce')
    # Replace inf values
    df_out = df_out.replace([np.inf, -np.inf], np.nan)
    return df_out


# Use interpreter's actual capital if strategy() overrode the sidebar default
if input_mode == "🌲 Pine Script" and '_pine_effective_settings' in st.session_state:
    _eff = st.session_state['_pine_effective_settings']
    initial_capital = _eff['initial_capital']

sa = StrategyAnalytics(
    profits=profits,
    initial_capital=initial_capital,
    dates=dates,
    contracts=contracts,
    entry_prices=entry_prices,
    exit_prices=exit_prices,
    trade_types=trade_types,
    runups=runups,
    drawdowns_per_trade=dd_per_trade,
    commissions=commissions,
    signals=signals,
    risk_free_rate=risk_free_rate,
)

mc = sa.monte_carlo(n_sims=mc_sims)
# Block bootstrap — preserves DCA loss clustering
_block_size = max(5, min(20, int(np.sqrt(sa.n_trades))))  # adaptive block size
mc_block = sa.monte_carlo_block(n_sims=mc_sims, block_size=_block_size)
# Autocorrelation analysis
trade_autocorr = sa.trade_autocorrelation(max_lag=10)
# Tail risk
tail_risk = sa.tail_risk_metrics()
rolling = sa.rolling_metrics(window=rolling_window)

# ═══════════════════════════════════════════════════════════════════════════════
# AI QUANT ANALYST — GLOBAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

AI_SYSTEM_PROMPT = """You are an elite institutional quantitative analyst AND strategy engineer with 20+ years building and optimizing systematic trading strategies for tier-1 hedge funds and family offices. You are both a critic AND a builder — you diagnose problems AND prescribe exact fixes.

Your role: You are the user's tireless quant co-pilot. The client manages USD 1B+ in capital across futures markets. Your job is NOT just to point out what's wrong — it's to make the strategy BETTER. Every weakness you identify must come with a specific, actionable fix.

PERSONALITY:
- You are the hardest-working employee they've ever had. You don't just say "this is bad" — you say "this is bad, here's exactly how to fix it, here's the parameter change, here's what the improved metric should look like."
- You think in terms of SOLUTIONS, not just problems. For every risk you flag, propose a mitigation.
- You give specific numbers: "Change TP from 6 ticks to 12 ticks" not "consider widening your TP."
- You think about strategy ARCHITECTURE — entry logic, exit logic, position sizing, regime filters, risk overlays.
- When you see a pattern in the trade data (like repeating PnL values), you reverse-engineer what the strategy is doing and suggest structural improvements.

QUANTITATIVE PRINCIPLES:
- Backtest results are OPTIMISTIC. Real-world slippage, data gaps, regime changes degrade performance 30-50%.
- DCA strategies have asymmetric risk — steady small wins, catastrophic drawdowns. Always flag and propose max-loss circuit breakers.
- Monte Carlo bootstrapping assumes trade independence — flag autocorrelation.
- Sharpe ratios from backtests degrade 30-50% in live trading. A backtest Sharpe of 2.0 = live ~1.2.
- Parameter-optimized strategies are prone to overfitting — recommend walk-forward validation.
- Sample size matters: trades/year, calendar span, regime diversity all factor into statistical validity.
- Kelly criterion is an upper bound — real allocation should be 1/4 to 1/2 Kelly.
- Skewness < -1.0 combined with kurtosis > 6.0 = tail risk that Monte Carlo underestimates.

IMPROVEMENT FRAMEWORK — When suggesting improvements, think about these layers:
1. ENTRY LOGIC — Is the edge real? What filter would remove the worst trades?
2. EXIT LOGIC — Is TP too tight? Is SL too wide? What ratio fixes the payoff imbalance?
3. POSITION SIZING — Kelly, fractional Kelly, fixed fractional, vol-targeting?
4. REGIME FILTER — Should the strategy pause during trending/volatile markets?
5. RISK OVERLAY — Max daily loss, max consecutive losses, drawdown circuit breaker?
6. PORTFOLIO CONTEXT — Correlation with other strategies, diversification benefit?

PINE SCRIPT CODE RULES (when source code is provided):
- You have the ACTUAL Pine Script source code. Reference exact line numbers.
- For EVERY fix, show the exact code change: "Line 44: `rTPTicks = input.int(50, ...)` → change default to `input.int(100, ...)`"
- When adding new logic (regime filters, circuit breakers), write the COMPLETE Pine Script code block that the user can copy-paste directly into their strategy.
- New code blocks should indicate WHERE to insert them: "Add after line 68 (below the Regime Detector section):"
- Preserve the user's coding style (variable naming, spacing, groups).
- When suggesting a new parameter, include the full `input.int()` or `input.float()` call with group, minval, maxval, step.
- If the strategy needs a new section (e.g., circuit breakers), write it as a complete, self-contained block with comments matching the user's style.

PARAMETER OPTIMIZATION RULES (when parsed parameters are provided):
- You have every tunable parameter with its current value, type, min, max, step, and group.
- For parameter changes, specify: variable name, current value, recommended value, and the min/max range to test in the optimizer.
- Group related parameter changes together — don't suggest changing TP without also adjusting SL.
- Consider parameter interactions: wider TP may need adjusted DCA spacing, tighter SL may need adjusted cooldown.

FORMAT RULES:
- Use markdown headers, bullet points, bold for key numbers
- When writing dollar amounts, write 'USD X' not '$X' (rendering compatibility)
- Every claim must cite a specific metric from the data
- Improvement suggestions must include BEFORE value, AFTER target, and WHY
- When suggesting parameter changes, be as specific as possible: "lookbackPeriod: 100 → try 60-80" not "reduce lookback"
- Structure each fix as: PROBLEM → ROOT CAUSE (cite line #) → FIX (exact code) → EXPECTED RESULT (metric target)

MANDATORY OUTPUT BLOCK 1 — RECOMMENDED PARAMETERS:
After your analysis, include a JSON block with your recommended parameter changes.
This block is machine-parsed by the app to auto-run a "what-if" backtest with your fixes applied.
Format EXACTLY like this (use the parameter TITLE as the key, matching the input.int/input.float title string):

```json:recommended_params
{
  "TP Ticks": 200,
  "Dollar Stop ($)": 200,
  "Max DCA Entries": 3,
  "DCA Spacing (bars)": 20,
  "Cooldown Bars": 15,
  "ATR Period": 100
}
```

Rules for the JSON block:
- Only include parameters you want to CHANGE (not all parameters)
- Use the exact parameter TITLE string from the input.int()/input.float() call (e.g., "TP Ticks" not "i_r_tpTicks")
- If the same title exists in multiple groups (e.g., "TP Ticks" in Range AND Caution), include it once — it will be applied to ALL matching parameters
- Values must be numbers (int or float), not strings

MANDATORY OUTPUT BLOCK 2 — FULL REVISED PINE SCRIPT:
At the VERY END of your response (after the recommended_params JSON), you MUST output the COMPLETE revised Pine Script with ALL your fixes applied.
This is NON-NEGOTIABLE. The user needs to copy-paste this directly into TradingView to verify your recommendations.

Format it as a single fenced code block with the label `pine:revised_strategy`:

```pine:revised_strategy
//@version=5
strategy("...", ...)
// ... THE ENTIRE SCRIPT with all parameter changes, new logic blocks,
// circuit breakers, regime filters, etc. already applied.
// Every line of the original script must be present (modified where needed).
// Do NOT omit sections with "..." or "// rest unchanged" — output the FULL script.
```

Rules for the revised script:
- Include EVERY line of the original script — modified where your fixes apply, unchanged everywhere else
- All parameter default values must reflect your recommended changes (e.g., input.int(200, ...) not input.int(50, ...))
- Any NEW code blocks (circuit breakers, trailing stops, regime filters) must be inserted at the correct location
- The script must be syntactically valid Pine Script v5 that compiles in TradingView without errors
- Do NOT truncate, abbreviate, or use "..." placeholders — the user will copy-paste this DIRECTLY
- Preserve all original comments, formatting, and variable names except where your fix changes them

You are relentless, detail-oriented, and constructive. You don't stop at the diagnosis — you build the cure. You write code, not essays."""


def build_strategy_context():
    """Assemble all strategy metrics into structured text for the AI."""
    _strat_name = st.session_state.get('_pine_strategy_name', 'Strategy')
    lines = []
    lines.append(f"STRATEGY: {_strat_name}")
    lines.append(f"Initial Capital: ${sa.initial_capital:,.0f}")
    lines.append(f"Total Trades: {sa.n_trades}")
    lines.append(f"Trades/Year (est): {sa.trades_per_year:.1f}")
    lines.append("")
    lines.append("=== PROFITABILITY ===")
    lines.append(f"Total Net Profit: ${sa.total_profit:,.2f}")
    lines.append(f"Total Return: {sa.total_return_pct:.2f}%")
    lines.append(f"CAGR: {sa.cagr:.2f}%")
    lines.append(f"Profit Factor: {sa.profit_factor:.3f}")
    lines.append(f"Expectancy ($/trade): ${sa.expectancy:.2f}")
    lines.append(f"Expectancy %: {sa.expectancy_pct:.2f}%")
    lines.append(f"Avg Trade: ${sa.avg_trade:.2f}")
    lines.append(f"Median Trade: ${sa.median_trade:.2f}")
    lines.append(f"Std Dev (trade): ${sa.std_trade:.2f}")
    lines.append("")
    lines.append("=== WIN/LOSS ===")
    lines.append(f"Win Rate: {sa.win_rate*100:.1f}%")
    lines.append(f"Loss Rate: {sa.loss_rate*100:.1f}%")
    lines.append(f"Avg Win: ${sa.avg_win:.2f}")
    lines.append(f"Avg Loss: ${sa.avg_loss:.2f}")
    lines.append(f"Largest Win: ${sa.largest_win:.2f}")
    lines.append(f"Largest Loss: ${sa.largest_loss:.2f}")
    lines.append(f"Median Win: ${sa.median_win:.2f}")
    lines.append(f"Median Loss: ${sa.median_loss:.2f}")
    lines.append(f"Payoff Ratio: {sa.payoff_ratio:.3f}")
    lines.append("")
    lines.append("=== RISK METRICS ===")
    lines.append(f"Max Drawdown: ${sa.max_drawdown:,.2f}")
    lines.append(f"Max Drawdown %: {sa.max_drawdown_pct:.2f}%")
    lines.append(f"Avg Drawdown: ${sa.avg_drawdown:,.2f}")
    lines.append(f"Avg Drawdown %: {sa.avg_drawdown_pct:.2f}%")
    lines.append(f"Max DD Duration (trades): {sa.max_drawdown_duration}")
    lines.append(f"Avg DD Duration (trades): {sa.avg_drawdown_duration:.1f}")
    lines.append(f"Sharpe Ratio: {sa.sharpe_ratio:.3f}")
    lines.append(f"Sortino Ratio: {sa.sortino_ratio:.3f}")
    lines.append(f"Calmar Ratio: {sa.calmar_ratio:.3f}")
    lines.append(f"Omega Ratio: {sa.omega_ratio:.3f}")
    lines.append(f"Tail Ratio: {sa.tail_ratio:.3f}")
    lines.append(f"Ulcer Index: {sa.ulcer_index:.3f}")
    lines.append(f"Gain-to-Pain Ratio: {sa.gain_to_pain_ratio:.3f}")
    lines.append(f"Recovery Factor: {sa.recovery_factor:.3f}")
    lines.append(f"MAR Ratio: {sa.mar_ratio:.3f}")
    lines.append(f"CPC Index: {sa.cpc_index:.3f}")
    lines.append(f"Common Sense Ratio: {sa.common_sense_ratio:.3f}")
    lines.append("")
    lines.append("=== STATISTICAL VALIDITY ===")
    lines.append(f"SQN: {sa.sqn:.3f}")
    lines.append(f"T-Statistic: {sa.t_stat_profit:.3f}")
    lines.append(f"P-Value: {sa.p_value_profit:.6f}")
    lines.append(f"Skewness: {sa.skewness:.3f}")
    lines.append(f"Excess Kurtosis: {sa.excess_kurtosis:.3f}")
    lines.append(f"Kelly Criterion: {sa.kelly_criterion:.4f}")
    lines.append(f"Half Kelly: {sa.half_kelly:.4f}")
    lines.append("")
    lines.append("=== STREAKS ===")
    lines.append(f"Max Consecutive Wins: {sa.max_consecutive_wins}")
    lines.append(f"Max Consecutive Losses: {sa.max_consecutive_losses}")
    lines.append(f"Avg Consecutive Wins: {sa.avg_consecutive_wins:.1f}")
    lines.append(f"Avg Consecutive Losses: {sa.avg_consecutive_losses:.1f}")
    lines.append("")
    lines.append("=== MONTE CARLO — STANDARD (single-trade bootstrap) ===")
    lines.append(f"Probability of Profit: {mc['prob_profit']:.1f}%")
    lines.append(f"Probability of 2x Capital: {mc['prob_2x']:.1f}%")
    lines.append(f"Probability of 50% Loss: {mc['prob_loss50']:.1f}%")
    lines.append(f"Probability of Ruin: {mc['prob_ruin']:.1f}%")
    lines.append(f"MC Final Equity P5: ${mc['percentiles']['p5']:,.0f}")
    lines.append(f"MC Final Equity P50: ${mc['percentiles']['p50']:,.0f}")
    lines.append(f"MC Final Equity P95: ${mc['percentiles']['p95']:,.0f}")
    lines.append(f"MC Max DD P50: {mc['dd_percentiles']['p50']:.2f}%")
    lines.append(f"MC Max DD P95: {mc['dd_percentiles']['p95']:.2f}%")
    lines.append(f"MC Max DD P99: {mc['dd_percentiles']['p99']:.2f}%")
    lines.append("")
    lines.append(f"=== MONTE CARLO — BLOCK BOOTSTRAP (block={_block_size}, preserves clustering) ===")
    lines.append(f"Block P(Profit): {mc_block['prob_profit']:.1f}%")
    lines.append(f"Block P(2x Capital): {mc_block['prob_2x']:.1f}%")
    lines.append(f"Block P(50% Loss): {mc_block['prob_loss50']:.1f}%")
    lines.append(f"Block P(Ruin): {mc_block['prob_ruin']:.1f}%")
    lines.append(f"Block Final Equity P5: ${mc_block['percentiles']['p5']:,.0f}")
    lines.append(f"Block Final Equity P50: ${mc_block['percentiles']['p50']:,.0f}")
    lines.append(f"Block Final Equity P95: ${mc_block['percentiles']['p95']:,.0f}")
    lines.append(f"Block Max DD P50: {mc_block['dd_percentiles']['p50']:.2f}%")
    lines.append(f"Block Max DD P95: {mc_block['dd_percentiles']['p95']:.2f}%")
    lines.append(f"Block Max DD P99: {mc_block['dd_percentiles']['p99']:.2f}%")
    lines.append("")
    # ── Institutional Metrics ──
    lines.append("=== INSTITUTIONAL METRICS ===")
    _ctx_dsr = sa.deflated_sharpe_ratio(num_trials=1)
    lines.append(f"Deflated Sharpe Ratio (1 trial): {_ctx_dsr['dsr_pct']:.1f}%")
    lines.append(f"DSR Haircut: {_ctx_dsr['haircut_pct']:.1f}%")
    _ctx_wfe = sa.walk_forward_efficiency(n_splits=5)
    if _ctx_wfe:
        lines.append(f"Walk-Forward Efficiency (Sharpe): {_ctx_wfe['avg_wfe_sharpe']:.3f} — {_ctx_wfe['interpretation']}")
        lines.append(f"Walk-Forward Efficiency (PF): {_ctx_wfe['avg_wfe_pf']:.3f}")
    _ctx_rmc = sa.regime_monte_carlo(n_sims=500, n_regimes=3, block_size=_block_size)
    if _ctx_rmc:
        lines.append(f"Regime MC P(Profit): {_ctx_rmc['prob_profit']:.1f}%")
        for _rname, _rstats in _ctx_rmc["regime_stats"].items():
            lines.append(f"  {_rname}: {_rstats['n_trades']} trades, avg=${_rstats['avg_pnl']:.2f}, WR={_rstats['win_rate']*100:.0f}%, Sharpe={_rstats['sharpe']:.2f}")
    lines.append("")
    lines.append("=== TAIL RISK (VaR / CVaR) ===")
    lines.append(f"VaR 95%: ${tail_risk['var_95']['VaR']:,.2f}")
    lines.append(f"CVaR 95% (Expected Shortfall): ${tail_risk['var_95']['CVaR']:,.2f}")
    lines.append(f"VaR 99%: ${tail_risk['var_99']['VaR']:,.2f}")
    lines.append(f"CVaR 99% (Expected Shortfall): ${tail_risk['var_99']['CVaR']:,.2f}")
    lines.append(f"3-sigma loss estimate: ${tail_risk['sigma3_loss']:,.2f}")
    lines.append(f"5-sigma loss estimate: ${tail_risk['sigma5_loss']:,.2f}")
    lines.append(f"Worst observed loss: ${tail_risk['worst_observed']:,.2f}")
    lines.append("")
    lines.append("=== TRADE AUTOCORRELATION ===")
    if trade_autocorr:
        for lag, val in list(trade_autocorr.items())[:5]:
            _flag = " *** SIGNIFICANT" if abs(val) > 0.1 else ""
            lines.append(f"Lag-{lag}: {val:+.4f}{_flag}")
    else:
        lines.append("Insufficient data for autocorrelation")
    lines.append("")
    lines.append("=== TRADE SAMPLE (first 10 + last 10) ===")
    for i in range(min(10, sa.n_trades)):
        _side = sa.trade_types[i] if sa.trade_types is not None and i < len(sa.trade_types) else "?"
        _pnl = sa.raw_profits[i]
        lines.append(f"  Trade {i+1}: {_side} PnL=${_pnl:+.2f}")
    if sa.n_trades > 20:
        lines.append("  ...")
        for i in range(max(10, sa.n_trades - 10), sa.n_trades):
            _side = sa.trade_types[i] if sa.trade_types is not None and i < len(sa.trade_types) else "?"
            _pnl = sa.raw_profits[i]
            lines.append(f"  Trade {i+1}: {_side} PnL=${_pnl:+.2f}")
    lines.append("")
    lines.append("=== EQUITY CURVE CHECKPOINTS ===")
    _n = sa.n_trades
    for _pct in [10, 25, 50, 75, 90, 100]:
        _idx = min(int(_n * _pct / 100) - 1, _n - 1)
        if _idx >= 0:
            lines.append(f"  At {_pct}% of trades (trade {_idx+1}): equity=${sa.equity[_idx]:,.2f}, DD={sa.drawdown_pct[_idx]*100:.2f}%")

    # ── Market Context (from OHLCV data if available) ────────────
    _mkt_df = st.session_state.get('_pine_cached_data')
    if _mkt_df is not None and len(_mkt_df) > 0:
        lines.append("")
        lines.append("=== MARKET CONTEXT (underlying instrument during backtest) ===")
        # Find close/high/low columns (case-insensitive)
        _cols = {c.lower(): c for c in _mkt_df.columns}
        _close_col = _cols.get('close', _cols.get('Close', None))
        _high_col = _cols.get('high', _cols.get('High', None))
        _low_col = _cols.get('low', _cols.get('Low', None))
        _vol_col = _cols.get('volume', _cols.get('Volume', None))
        _time_col = _cols.get('time', _cols.get('date', _cols.get('datetime', None)))

        if _close_col and _close_col in _mkt_df.columns:
            _closes = pd.to_numeric(_mkt_df[_close_col], errors='coerce').dropna()
            if len(_closes) > 1:
                _mkt_start = float(_closes.iloc[0])
                _mkt_end = float(_closes.iloc[-1])
                _mkt_high = float(_closes.max())
                _mkt_low = float(_closes.min())
                _mkt_return = (_mkt_end / _mkt_start - 1) * 100 if _mkt_start > 0 else 0
                _mkt_range = (_mkt_high - _mkt_low)

                # Buy and hold comparison
                lines.append(f"Bars in dataset: {len(_mkt_df)}")
                lines.append(f"Market start price: ${_mkt_start:,.2f}")
                lines.append(f"Market end price: ${_mkt_end:,.2f}")
                lines.append(f"Market high: ${_mkt_high:,.2f}")
                lines.append(f"Market low: ${_mkt_low:,.2f}")
                lines.append(f"Market range: ${_mkt_range:,.2f}")
                lines.append(f"Buy & Hold return: {_mkt_return:+.2f}%")

                # Strategy vs market comparison
                _strat_ret = sa.total_return_pct
                _alpha = _strat_ret - _mkt_return
                lines.append(f"Strategy return: {_strat_ret:+.2f}%")
                lines.append(f"Alpha vs buy-and-hold: {_alpha:+.2f}%")

                # Market volatility
                _daily_returns = _closes.pct_change().dropna()
                if len(_daily_returns) > 1:
                    _mkt_vol = float(_daily_returns.std() * 100)
                    _mkt_max_daily_drop = float(_daily_returns.min() * 100)
                    _mkt_max_daily_gain = float(_daily_returns.max() * 100)
                    lines.append(f"Market bar-to-bar volatility (std): {_mkt_vol:.4f}%")
                    lines.append(f"Worst single-bar drop: {_mkt_max_daily_drop:.2f}%")
                    lines.append(f"Best single-bar gain: {_mkt_max_daily_gain:+.2f}%")

                # Market drawdown
                _mkt_peak = _closes.cummax()
                _mkt_dd = (_closes - _mkt_peak) / _mkt_peak * 100
                _mkt_max_dd = float(_mkt_dd.min())
                lines.append(f"Market max drawdown: {_mkt_max_dd:.2f}%")

                # Regime detection: count how many 20% swings occurred
                _swing_count = 0
                _last_extreme = _mkt_start
                for _p in _closes:
                    if abs(_p / _last_extreme - 1) > 0.10:
                        _swing_count += 1
                        _last_extreme = _p
                lines.append(f"Regime changes (>10% swings): {_swing_count}")

                # Date range
                if _time_col and _time_col in _mkt_df.columns:
                    try:
                        _times = pd.to_datetime(_mkt_df[_time_col])
                        _date_start = _times.iloc[0].strftime('%Y-%m-%d')
                        _date_end = _times.iloc[-1].strftime('%Y-%m-%d')
                        _span_days = (_times.iloc[-1] - _times.iloc[0]).days
                        lines.append(f"Date range: {_date_start} to {_date_end} ({_span_days} calendar days)")
                    except Exception:
                        pass

        if _vol_col and _vol_col in _mkt_df.columns:
            _vols = pd.to_numeric(_mkt_df[_vol_col], errors='coerce').dropna()
            if len(_vols) > 0:
                lines.append(f"Avg volume per bar: {_vols.mean():,.0f}")
                lines.append(f"Min volume bar: {_vols.min():,.0f}")
                lines.append(f"Max volume bar: {_vols.max():,.0f}")

    # ── Available datasets with rich metadata ────────────────────────
    _cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_cache')
    if os.path.exists(_cache_dir):
        _datasets = sorted([f.replace('.parquet', '') for f in os.listdir(_cache_dir) if f.endswith('.parquet')])
        if _datasets:
            lines.append("")
            lines.append(f"=== AVAILABLE DATASETS FOR BACKTESTING ({len(_datasets)} total) ===")
            lines.append("Use this catalog to recommend the RIGHT dataset for each strategy based on timeframe and duration.")
            lines.append("")
            for _ds in _datasets:
                _ds_path = os.path.join(_cache_dir, _ds + '.parquet')
                _ds_size = os.path.getsize(_ds_path) / (1024*1024)
                _ds_detail = f"  {_ds} ({_ds_size:.1f}MB)"
                try:
                    _ds_df = pd.read_parquet(_ds_path)
                    _ds_bars = len(_ds_df)
                    _ds_detail += f" | {_ds_bars:,} bars"

                    # Find time column
                    _ds_time_col = None
                    if isinstance(_ds_df.index, pd.DatetimeIndex):
                        _ds_time_col = '__index__'
                        _ds_times = _ds_df.index
                    else:
                        for _tc in ['time', 'timestamp', 'date', 'datetime', 'ts_event', 'Date', 'Time']:
                            if _tc in _ds_df.columns:
                                _ds_time_col = _tc
                                break

                    _ds_times = None
                    if _ds_time_col == '__index__':
                        _ds_times = _ds_df.index
                    elif _ds_time_col:
                        _ds_times = pd.to_datetime(_ds_df[_ds_time_col], errors='coerce').dropna()

                    if _ds_time_col and _ds_times is not None and len(_ds_times) > 1:
                        _ds_start = _ds_times.min()
                        _ds_end = _ds_times.max()
                        _ds_span = (_ds_end - _ds_start).days
                        _ds_detail += f" | {_ds_start.strftime('%Y-%m-%d')} to {_ds_end.strftime('%Y-%m-%d')} ({_ds_span} days)"

                        # Infer timeframe from median bar interval
                        _ds_diffs = pd.Series(_ds_times).diff().dropna()
                        _ds_median_sec = _ds_diffs.median().total_seconds()
                        if _ds_median_sec < 90:
                            _ds_tf = "1-min"
                        elif _ds_median_sec < 210:
                            _ds_tf = "3-min"
                        elif _ds_median_sec < 450:
                            _ds_tf = "5-min"
                        elif _ds_median_sec < 1200:
                            _ds_tf = "15-min"
                        elif _ds_median_sec < 2700:
                            _ds_tf = "30-min"
                        elif _ds_median_sec < 5400:
                            _ds_tf = "1-hour"
                        elif _ds_median_sec < 18000:
                            _ds_tf = "4-hour"
                        elif _ds_median_sec < 100000:
                            _ds_tf = "1-day"
                        else:
                            _ds_tf = "1-week"
                        _ds_detail += f" | TIMEFRAME: {_ds_tf}"

                    # Price range from close column
                    _ds_close_col = None
                    for _cc in ['close', 'Close', 'CLOSE']:
                        if _cc in _ds_df.columns:
                            _ds_close_col = _cc
                            break
                    if _ds_close_col:
                        _ds_closes = pd.to_numeric(_ds_df[_ds_close_col], errors='coerce').dropna()
                        if len(_ds_closes) > 0:
                            _ds_detail += f" | Price: {_ds_closes.iloc[0]:,.2f} -> {_ds_closes.iloc[-1]:,.2f} (low {_ds_closes.min():,.2f}, high {_ds_closes.max():,.2f})"
                except Exception:
                    pass
                lines.append(_ds_detail)
            lines.append("")
            lines.append("MATCHING RULE: A strategy's chart timeframe MUST match the dataset timeframe. A 1-min strategy needs 1-min data. A 5-min strategy needs 5-min data. Mismatched timeframes produce meaningless results.")

    # ── Time Analysis (best/worst hours, days, months) ──────────
    try:
        _time_data = sa.time_analysis()
        if _time_data:
            lines.append("")
            lines.append("=== TIME ANALYSIS (identify when the strategy wins/loses) ===")
            for _period_name in ['day_of_week', 'hour', 'month']:
                if _period_name in _time_data and _time_data[_period_name] is not None:
                    _tdf = _time_data[_period_name]
                    lines.append(f"\n  [{_period_name.upper()}]")
                    for _tidx, _trow in _tdf.iterrows():
                        _wr = _trow.get('win_rate', 0)
                        _pf = _trow.get('profit_factor', 0)
                        _nt = int(_trow.get('n_trades', 0))
                        _tp = _trow.get('total_profit', 0)
                        _flag = " *** LOSING PERIOD" if _tp < 0 and _nt >= 5 else ""
                        lines.append(f"    {_tidx}: {_nt} trades, total=${_tp:+.2f}, "
                                     f"WR={_wr*100:.0f}%, PF={_pf:.2f}{_flag}")

            # Worst half-hour brackets (actionable time filters)
            if 'half_hour' in _time_data and _time_data['half_hour'] is not None:
                _hhdf = _time_data['half_hour']
                _losers = _hhdf[_hhdf['total_profit'] < 0].sort_values('total_profit')
                if len(_losers) > 0:
                    lines.append("\n  [WORST HALF-HOUR BRACKETS — consider time filters]")
                    for _hidx, _hrow in _losers.head(5).iterrows():
                        lines.append(f"    {_hidx}: {int(_hrow['n_trades'])} trades, "
                                     f"total=${_hrow['total_profit']:+.2f}, PF={_hrow['profit_factor']:.2f}")
    except Exception:
        pass

    # ── Rolling Stats (detect performance degradation) ─────────
    try:
        _roll_window = min(20, max(5, sa.n_trades // 5))
        _rolling_data = sa.rolling_metrics(window=_roll_window)
        if _rolling_data is not None and 'sharpe' in _rolling_data:
            lines.append("")
            lines.append(f"=== ROLLING STATS (window={_roll_window}, detect degradation/regime drift) ===")
            _r_sharpe = _rolling_data['sharpe']
            _r_wr = _rolling_data.get('win_rate')
            _n_r = len(_r_sharpe)
            for _pct in [10, 25, 50, 75, 100]:
                _idx = min(int(_n_r * _pct / 100) - 1, _n_r - 1)
                if _idx >= 0:
                    _s_val = _r_sharpe[_idx]
                    _wr_val = f", WR={_r_wr[_idx]*100:.0f}%" if _r_wr is not None and _idx < len(_r_wr) else ""
                    lines.append(f"  At {_pct}% of trades: rolling Sharpe={_s_val:.3f}{_wr_val}")

            # Trend detection: is performance improving or degrading?
            if _n_r >= 4:
                _first_q = np.mean(_r_sharpe[:_n_r//4])
                _last_q = np.mean(_r_sharpe[-_n_r//4:])
                if _last_q < _first_q - 0.3:
                    lines.append(f"  *** DEGRADATION DETECTED: first quarter avg Sharpe={_first_q:.3f}, last quarter={_last_q:.3f}")
                elif _last_q > _first_q + 0.3:
                    lines.append(f"  IMPROVING: first quarter avg Sharpe={_first_q:.3f}, last quarter={_last_q:.3f}")
    except Exception:
        pass

    # ── Full Trade Sequence (for pattern detection, <= 300 trades) ─
    if sa.n_trades <= 300 and sa.n_trades > 20:
        lines.append("")
        lines.append("=== FULL TRADE P&L SEQUENCE (for loss clustering / pattern detection) ===")
        _wl_seq = []
        for _ti in range(sa.n_trades):
            _side = sa.trade_types[_ti] if sa.trade_types is not None and _ti < len(sa.trade_types) else "?"
            _pnl = sa.raw_profits[_ti]
            _wl = "W" if _pnl > 0 else ("L" if _pnl < 0 else "B")
            _wl_seq.append(_wl)
            lines.append(f"  T{_ti+1}: {_side} ${_pnl:+.2f}")
        # Show W/L pattern string for quick visual
        lines.append(f"\n  W/L Pattern: {''.join(_wl_seq)}")

    # ── Per-Trade Runup/Drawdown Analysis ──────────────────────
    if sa.runups is not None and sa.drawdowns_per_trade is not None:
        lines.append("")
        lines.append("=== PER-TRADE RUNUP/DRAWDOWN (diagnose SL/TP placement) ===")
        _wins = sa.raw_profits > 0
        _losses = sa.raw_profits < 0
        if _wins.sum() > 0:
            lines.append(f"  Winning trades — Avg runup: ${np.mean(sa.runups[_wins]):.2f}, "
                         f"Avg DD before exit: ${np.mean(sa.drawdowns_per_trade[_wins]):.2f}")
        if _losses.sum() > 0:
            lines.append(f"  Losing trades — Avg runup: ${np.mean(sa.runups[_losses]):.2f}, "
                         f"Avg DD at exit: ${np.mean(sa.drawdowns_per_trade[_losses]):.2f}")
            # How many losers had positive runup (was winning, then reversed)
            _was_winning = (sa.runups[_losses] > 0).sum()
            _loss_count = _losses.sum()
            lines.append(f"  Losers that were winning first: {_was_winning}/{_loss_count} ({_was_winning/_loss_count*100:.0f}%)")
            lines.append(f"  Avg runup before reversal: ${np.mean(sa.runups[_losses][sa.runups[_losses] > 0]):.2f}" if _was_winning > 0 else "")

    # ── Pine Script Source Code ───────────────────────────────────
    _pine_src = st.session_state.get('_pine_source_code')
    if _pine_src:
        lines.append("")
        lines.append("=== PINE SCRIPT SOURCE CODE ===")
        lines.append("This is the ACTUAL strategy code. Reference specific lines, parameters, and logic when suggesting fixes.")
        lines.append("When you suggest a change, show the EXACT code modification (before → after) with line numbers.")
        lines.append("")
        for _ln, _line in enumerate(_pine_src.splitlines(), 1):
            lines.append(f"  {_ln:>4}| {_line}")

    # ── Parsed Parameters (with current values, ranges, groups) ──
    _input_defs = st.session_state.get('_pine_input_defs', [])
    if _input_defs:
        lines.append("")
        lines.append("=== STRATEGY PARAMETERS (parsed from Pine code) ===")
        lines.append("These are the tunable inputs. When suggesting parameter changes, reference exact variable names.")
        _grouped = {}
        for _inp in _input_defs:
            _g = _inp.get('group', '') or 'General'
            if _g not in _grouped:
                _grouped[_g] = []
            _grouped[_g].append(_inp)
        for _grp_name, _grp_inputs in _grouped.items():
            lines.append(f"  [{_grp_name}]")
            for _inp in _grp_inputs:
                _parts = [f"    {_inp.get('key', '?')}: {_inp.get('defval', '?')}"]
                _parts.append(f"(type={_inp.get('type', '?')})")
                if _inp.get('minval') is not None:
                    _parts.append(f"min={_inp['minval']}")
                if _inp.get('maxval') is not None:
                    _parts.append(f"max={_inp['maxval']}")
                if _inp.get('step') is not None:
                    _parts.append(f"step={_inp['step']}")
                if _inp.get('title'):
                    _parts.append(f'"{_inp["title"]}"')
                lines.append(" ".join(_parts))

    return "\n".join(lines)


def _escape_dollars_for_streamlit(text):
    """Escape $ signs so Streamlit doesn't render them as LaTeX math."""
    import re
    # Don't escape $$ (intentional LaTeX) — only solo $ that look like currency
    # Replace $X (currency amounts) with escaped version
    text = re.sub(r'\$(\d)', r'\\$\1', text)
    text = re.sub(r'\$-', r'\\$-', text)
    text = re.sub(r'\$\+', r'\\$+', text)
    return text


def _parse_ai_recommended_params(ai_text):
    """Extract recommended parameter JSON from AI response."""
    import json as _json
    # Look for ```json:recommended_params ... ``` block
    _patterns = [
        r'```json:recommended_params\s*\n(.*?)```',
        r'```json\s*:?\s*recommended_params\s*\n(.*?)```',
        r'```recommended_params\s*\n(.*?)```',
    ]
    for _pat in _patterns:
        _match = re.search(_pat, ai_text, re.DOTALL)
        if _match:
            try:
                return _json.loads(_match.group(1).strip())
            except _json.JSONDecodeError:
                continue
    # Fallback: look for any JSON block near the end of the text
    _all_json = re.findall(r'```json\s*\n(.*?)```', ai_text, re.DOTALL)
    for _block in reversed(_all_json):
        try:
            _parsed = _json.loads(_block.strip())
            if isinstance(_parsed, dict) and all(isinstance(v, (int, float)) for v in _parsed.values()):
                return _parsed
        except _json.JSONDecodeError:
            continue
    return None


def call_ai_analyst(user_msg, chat_key='_ai_chat_history'):
    """Send a message to Claude via CLI stdin pipe (uses Max plan). Falls back to API."""
    import subprocess, shutil

    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    st.session_state[chat_key].append({"role": "user", "content": user_msg})

    # Build the full prompt — system + conversation history
    _full_prompt = AI_SYSTEM_PROMPT
    _full_prompt += "\n\nFORMATTING RULE: When writing dollar amounts, write them as plain numbers with 'USD' prefix instead of '$'. Example: 'USD 500' not '$500'. Write 'negative USD 595' not '-$595'. This is mandatory.\n\n"
    for _msg in st.session_state[chat_key]:
        if _msg["role"] == "user":
            _full_prompt += f"\n[USER]\n{_msg['content']}\n"
        else:
            _full_prompt += f"\n[ASSISTANT]\n{_msg['content']}\n"
    _full_prompt += "\n[ASSISTANT]\n"

    # Method 1: Claude CLI via stdin pipe (uses Max plan — no API key needed)
    _claude_bin = shutil.which('claude') or '/usr/local/bin/claude'
    _cli_err = None
    if os.path.exists(_claude_bin):
        try:
            _result = subprocess.run(
                [_claude_bin, '--output-format', 'text', '--max-turns', '1'],
                input=_full_prompt,
                capture_output=True, text=True, timeout=600,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
            if _result.returncode == 0 and _result.stdout.strip():
                _reply = _escape_dollars_for_streamlit(_result.stdout.strip())
                st.session_state[chat_key].append({"role": "assistant", "content": _reply})
                return _reply
            # CLI returned but no good output
            _cli_err = f"returncode={_result.returncode}"
            if _result.stderr:
                _cli_err += f" | stderr: {_result.stderr.strip()[:200]}"
            if _result.stdout:
                _cli_err += f" | stdout: {_result.stdout.strip()[:200]}"
        except subprocess.TimeoutExpired:
            _cli_err = "CLI timed out after 300s — try a shorter question"
        except Exception as e:
            _cli_err = f"Exception: {e}"
    else:
        _cli_err = f"CLI not found at {_claude_bin}"

    # Method 2: Anthropic API fallback
    _ai_key = os.environ.get('ANTHROPIC_API_KEY', '')
    if _ai_key:
        try:
            import anthropic
            _client = anthropic.Anthropic(api_key=_ai_key)
            _messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state[chat_key]]
            _response = _client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=8192,
                system=AI_SYSTEM_PROMPT,
                messages=_messages
            )
            _reply = _escape_dollars_for_streamlit(_response.content[0].text)
            st.session_state[chat_key].append({"role": "assistant", "content": _reply})
            return _reply
        except Exception as e:
            if st.session_state[chat_key] and st.session_state[chat_key][-1]['role'] == 'user':
                st.session_state[chat_key].pop()
            return f"**CLI error:** {_cli_err}\n\n**API error:** {e}"

    # Neither worked
    if st.session_state[chat_key] and st.session_state[chat_key][-1]['role'] == 'user':
        st.session_state[chat_key].pop()
    return f"**Error:** CLI failed ({_cli_err}) and no API key set as fallback."


# ═══════════════════════════════════════════════════════════════════════════════
# DETECTED COLUMNS SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

detected = {k: v for k, v in col_map.items() if v is not None}
with st.expander(f"🔍 Auto-Detected Columns ({len(detected)} found)", expanded=False):
    for key, val in detected.items():
        st.markdown(f"**{key}** → `{val}`")


# ═══════════════════════════════════════════════════════════════════════════════
# AUTO-AUDIT PIPELINE — ONE-CLICK FULL STRATEGY ANALYSIS + AI FIX + RE-TEST
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
            border: 2px solid rgba(0,200,255,0.4); border-radius: 16px;
            padding: 24px; margin: 16px 0;">
    <div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
        <span style="font-size:2rem;">🏛️</span>
        <div>
            <div style="font-size:1.3rem; font-weight:800; color:#00c8ff;">Auto-Audit Pipeline</div>
            <div style="font-size:0.85rem; opacity:0.6;">One click: Grade → Diagnose → Fix → Re-test → Before/After Monte Carlo</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

_audit_has_pine = '_pine_ast' in st.session_state and '_pine_cached_data' in st.session_state
_audit_has_data = sa is not None and sa.n_trades > 0

_audit_c1, _audit_c2 = st.columns([3, 1])
with _audit_c1:
    if not _audit_has_data:
        st.info("Upload trade data (CSV/XLSX) or a Pine Script above to enable Auto-Audit.")
    elif not _audit_has_pine:
        st.warning("Auto-Audit works best with Pine Script source code (for parameter fixes). "
                   "Upload a .pine file or paste your strategy code in the Pine Script section above. "
                   "Without it, AI will analyze metrics only — no automated re-test.")
def _run_audit_callback():
    """on_click callback — runs BEFORE page rerender so session state is set."""
    _audit_context = build_strategy_context()
    _audit_prompt = (
        "FULL AUTO-AUDIT — analyze this strategy completely:\n"
        "1. Grade (A-F) with composite score\n"
        "2. Top weaknesses with SPECIFIC fixes (line numbers, before/after)\n"
        "3. Expected improvement for each fix\n"
        "4. Risk assessment\n\n"
        "Include the ```json:recommended_params``` block at the end.\n\n"
        "STRATEGY DATA:\n" + _audit_context
    )
    _audit_reply = call_ai_analyst(_audit_prompt, '_auto_audit_chat')

    if not _audit_reply or _audit_reply.startswith("**Error"):
        st.session_state['_auto_audit_results'] = {
            'sa_fixed': None, 'params': None,
            'ai_reply': _audit_reply or "AI call returned empty.",
            'revised_pine': None,
        }
        return

    _audit_params = _parse_ai_recommended_params(_audit_reply)

    # Auto-merge Pine Script if we have source + params
    _revised_pine = None
    _pine_src = st.session_state.get('_pine_source_code', '')
    if _pine_src and _audit_params:
        _rev_lines = _pine_src.splitlines()
        _changes = []
        for _pn, _pv in _audit_params.items():
            for _li in range(len(_rev_lines)):
                _ol = _rev_lines[_li]
                if _pn not in _ol or not re.search(r'input\.(int|float|bool)\s*\(', _ol):
                    continue
                if 'input.bool' in _ol:
                    _nl = re.sub(r'(input\.bool\s*\()\s*(true|false)',
                                 r'\g<1>' + ('true' if _pv else 'false'),
                                 _ol, count=1, flags=re.IGNORECASE)
                else:
                    _nl = re.sub(r'(input\.(?:int|float)\s*\()\s*[-]?[\d.]+',
                                 r'\g<1>' + str(_pv), _ol, count=1)
                if _nl != _ol:
                    _rev_lines[_li] = _nl
                    _changes.append(f"Line {_li+1}: {_pn} -> {_pv}")
        if _changes:
            _hdr = "\n".join([f"// {c}" for c in _changes])
            _revised_pine = (
                f"// AUTO-AUDIT REVISED — {len(_changes)} parameter changes applied\n"
                f"{_hdr}\n\n" + "\n".join(_rev_lines)
            )

    # ── Optimizer: refine AI's parameter suggestions with grid search ──
    _optimizer_results = None
    _optimizer_best = None
    _walk_forward = None
    _sensitivity = None
    _audit_has_pine_cb = ('_pine_ast' in st.session_state and
                          '_pine_cached_data' in st.session_state)

    if _audit_has_pine_cb and _audit_params:
        try:
            from pine_interpreter import PineInterpreter as _AuditInterp
            from pine_optimizer import PineOptimizer, InputParam
            _eff = st.session_state.get('_pine_effective_settings', {})

            # Build optimizer with strategy's effective settings
            _opt = PineOptimizer(
                ast=st.session_state['_pine_ast'],
                df_price=st.session_state['_pine_cached_data'],
                initial_capital=_eff.get('initial_capital', 10000),
                commission_pct=_eff.get('commission_pct', 0.0),
                default_qty=_eff.get('default_qty', 1),
                pyramiding=_eff.get('pyramiding', 1),
                slippage=_eff.get('slippage', 0),
                mintick=_eff.get('mintick', 0.25),
                margin_long=0,
                margin_short=0,
            )

            # Match AI-recommended params to parsed input_defs
            _input_defs = st.session_state.get('_pine_input_defs', [])
            _opt_params = []
            for _idef in _input_defs:
                _title = _idef.get('title', '')
                _key = _idef.get('key', '')
                _name = _title or _key
                if not _name:
                    continue

                # Only enable params the AI flagged
                _is_flagged = _name in _audit_params or _key in _audit_params

                if _is_flagged and _idef.get('type') in ('int', 'float', 'bool'):
                    _ai_val = _audit_params.get(_name, _audit_params.get(_key))
                    _defval = _idef.get('defval')

                    _p = InputParam(
                        name=_name,
                        param_type=_idef.get('type', 'float'),
                        default=_defval,
                        min_val=_idef.get('minval'),
                        max_val=_idef.get('maxval'),
                        step=_idef.get('step'),
                        options=_idef.get('options'),
                        enabled=True,
                    )

                    # Narrow search range around AI's suggestion and current default
                    if _p.param_type in ('int', 'float') and _ai_val is not None and _defval is not None:
                        _center = float(_ai_val)
                        _orig = float(_defval)
                        _range = max(abs(_center - _orig) * 1.5, abs(_center) * 0.3, 1.0)
                        _p.min_val = max(_center - _range, _p.min_val or 0)
                        _p.max_val = _center + _range
                        if _p.max_val <= _p.min_val:
                            _p.max_val = _p.min_val + _range
                        # Cap values per param to keep total combos manageable
                        _p.MAX_VALUES_PER_PARAM = 15

                    _p.generate_values()
                    _opt_params.append(_p)

            # Run optimizer if we have params and combos are reasonable
            _total_combos = 1
            for _p in _opt_params:
                if _p.enabled and _p.values:
                    _total_combos *= len(_p.values)

            if _opt_params and 0 < _total_combos <= 10000:
                _optimizer_results = _opt.optimize(
                    _opt_params, _AuditInterp,
                    sort_by='sharpe_ratio', top_n=10, min_trades=5,
                )
                if _optimizer_results:
                    _optimizer_best = _optimizer_results[0]
                    # Use optimizer's best params instead of AI's guesses
                    _audit_params = dict(_optimizer_best.params)

                    # Walk-forward validation on the best result
                    try:
                        _walk_forward = _opt.walk_forward(
                            _optimizer_best.params, _AuditInterp, n_splits=5
                        )
                    except Exception:
                        _walk_forward = None

                    # Parameter sensitivity: test nearby values for each param
                    _sensitivity = {}
                    for _p in _opt_params:
                        if not _p.enabled or _p.param_type == 'bool':
                            continue
                        _best_val = _optimizer_best.params.get(_p.name)
                        if _best_val is None:
                            continue
                        _nearby = []
                        _step = _p.step or (1 if _p.param_type == 'int' else 0.1)
                        for _delta in [-2, -1, 0, 1, 2]:
                            _test_val = _best_val + _delta * _step
                            _test_overrides = dict(_optimizer_best.params)
                            _test_overrides[_p.name] = int(_test_val) if _p.param_type == 'int' else _test_val
                            try:
                                _nearby_result = _opt._run_single(_test_overrides, _AuditInterp)
                                _nearby.append({
                                    'value': _test_val,
                                    'sharpe': _nearby_result.sharpe_ratio,
                                    'pf': _nearby_result.profit_factor,
                                    'net': _nearby_result.net_profit,
                                })
                            except Exception:
                                pass
                        if _nearby:
                            _sensitivity[_p.name] = _nearby

                    # Re-merge Pine Script with optimizer's best params
                    if _pine_src:
                        _rev_lines = _pine_src.splitlines()
                        _changes = []
                        for _pn, _pv in _audit_params.items():
                            for _li in range(len(_rev_lines)):
                                _ol = _rev_lines[_li]
                                if _pn not in _ol or not re.search(r'input\.(int|float|bool)\s*\(', _ol):
                                    continue
                                if 'input.bool' in _ol:
                                    _nl = re.sub(r'(input\.bool\s*\()\s*(true|false)',
                                                 r'\g<1>' + ('true' if _pv else 'false'),
                                                 _ol, count=1, flags=re.IGNORECASE)
                                else:
                                    _nl = re.sub(r'(input\.(?:int|float)\s*\()\s*[-]?[\d.]+',
                                                 r'\g<1>' + str(_pv), _ol, count=1)
                                if _nl != _ol:
                                    _rev_lines[_li] = _nl
                                    _changes.append(f"Line {_li+1}: {_pn} -> {_pv}")
                        if _changes:
                            _hdr = "\n".join([f"// {c}" for c in _changes])
                            _revised_pine = (
                                f"// AUTO-AUDIT REVISED — {len(_changes)} params optimized (AI + grid search)\n"
                                f"{_hdr}\n\n" + "\n".join(_rev_lines)
                            )
        except Exception:
            pass  # Optimizer failed — fall back to AI params only

    # ── Re-run backtest with best parameters (optimizer or AI) ─────────
    _fix_sa = None
    _fix_mc = None
    _fix_mc_block = None
    _fix_dsr = None
    _fix_wfe = None
    _fix_error = ''
    _too_few = False

    if _audit_has_pine_cb and _audit_params:
        try:
            from pine_interpreter import PineInterpreter as _AuditInterp
            _eff = st.session_state.get('_pine_effective_settings', {})
            _fix_overrides = dict(_audit_params)

            _fix_interp = _AuditInterp(
                st.session_state['_pine_ast'],
                st.session_state['_pine_cached_data'],
                initial_capital=_eff.get('initial_capital', 10000),
                commission_pct=_eff.get('commission_pct', 0.0),
                default_qty=_eff.get('default_qty', 1),
                pyramiding=_eff.get('pyramiding', 1),
                slippage=_eff.get('slippage', 0),
                mintick=_eff.get('mintick', 0.25),
                input_overrides=_fix_overrides,
            )
            _fix_trades = _fix_interp.execute()

            if len(_fix_trades) >= 5:
                _fix_df = _fix_interp.to_dataframe()
                _fix_profits = _fix_df['Profit'].values
                _fix_dates = None
                if 'Date/Time' in _fix_df.columns:
                    _fix_dates = pd.to_datetime(_fix_df['Date/Time'], errors='coerce')
                _fix_types = _fix_df['Type'].values if 'Type' in _fix_df.columns else None
                _fix_contracts = _fix_df['Contracts'].values if 'Contracts' in _fix_df.columns else None
                _fix_entry_p = _fix_df['Price'].values if 'Price' in _fix_df.columns else None
                _fix_exit_p = _fix_df['Exit Price'].values if 'Exit Price' in _fix_df.columns else None
                _fix_runups = _fix_df['Run-up'].values if 'Run-up' in _fix_df.columns else None
                _fix_dd = _fix_df['Drawdown'].values if 'Drawdown' in _fix_df.columns else None
                _fix_comms = _fix_df['Commission'].values if 'Commission' in _fix_df.columns else None

                _fix_sa = StrategyAnalytics(
                    profits=_fix_profits,
                    initial_capital=_eff.get('initial_capital', 10000),
                    dates=_fix_dates,
                    contracts=_fix_contracts,
                    entry_prices=_fix_entry_p,
                    exit_prices=_fix_exit_p,
                    trade_types=_fix_types,
                    runups=_fix_runups,
                    drawdowns_per_trade=_fix_dd,
                    commissions=_fix_comms,
                    risk_free_rate=risk_free_rate,
                )
                _fix_mc = _fix_sa.monte_carlo(n_sims=mc_sims)
                _fix_bs = max(5, min(20, int(np.sqrt(_fix_sa.n_trades))))
                _fix_mc_block = _fix_sa.monte_carlo_block(n_sims=mc_sims, block_size=_fix_bs)
                try:
                    _fix_dsr = _fix_sa.deflated_sharpe_ratio(num_trials=1)
                except Exception:
                    _fix_dsr = {}
                try:
                    _fix_wfe = _fix_sa.walk_forward_efficiency(n_splits=5)
                except Exception:
                    _fix_wfe = None
            else:
                _too_few = True
        except Exception as _fix_e:
            _fix_error = str(_fix_e)

    st.session_state['_auto_audit_results'] = {
        'sa_fixed': _fix_sa,
        'mc_fixed': _fix_mc,
        'mc_block_fixed': _fix_mc_block,
        'dsr_fixed': _fix_dsr or {},
        'wfe_fixed': _fix_wfe,
        'params': _audit_params,
        'ai_reply': _audit_reply,
        'revised_pine': _revised_pine,
        'error': _fix_error,
        'too_few_trades': _too_few,
        'no_pine': not _audit_has_pine_cb,
        # Phase 3+4: optimizer, walk-forward, sensitivity
        'optimizer_results': _optimizer_results,
        'optimizer_best': _optimizer_best,
        'walk_forward': _walk_forward,
        'sensitivity': _sensitivity,
    }

with _audit_c2:
    st.button(
        "🚀 Run Full Auto-Audit",
        type="primary",
        use_container_width=True,
        key="_auto_audit_btn",
        disabled=not _audit_has_data,
        on_click=_run_audit_callback if _audit_has_data else None,
    )

# ── Display Auto-Audit results (persists across reruns) ──────────────
_ar = st.session_state.get('_auto_audit_results')
if _ar:
    st.markdown("---")
    _ar_reply_preview = (_ar.get('ai_reply') or '')[:100]
    st.caption(f"Auto-Audit results loaded — AI reply: {len(_ar.get('ai_reply', '') or '')} chars, "
               f"sa_fixed: {'yes' if _ar.get('sa_fixed') else 'no'}, "
               f"params: {list((_ar.get('params') or {}).keys())[:3]}")

    # ── SECTION A: Before vs After Comparison ────────────────────
    _ar_fixed = _ar.get('sa_fixed')
    if _ar_fixed:
        st.markdown("## Before vs After: AI-Optimized Strategy")

        # Grade comparison
        def _quick_grade(s):
            if s.expectancy < 0 and s.profit_factor < 0.9:
                return "F", "#dc2626"
            if s.expectancy < 0:
                return "D", "#ef4444"
            if s.profit_factor < 1.2:
                return "C", "#f97316"
            if s.profit_factor < 1.5:
                return "B", "#fbbf24"
            if s.profit_factor < 2.0:
                return "B+", "#86efac"
            return "A", "#22c55e"

        _g_before, _gc_before = _quick_grade(sa)
        _g_after, _gc_after = _quick_grade(_ar_fixed)

        st.markdown(f"""
        <div style="display:flex; gap:20px; justify-content:center; margin:20px 0;">
            <div style="text-align:center; padding:20px 40px; background:rgba(239,68,68,0.1);
                        border:2px solid {_gc_before}; border-radius:12px;">
                <div style="font-size:0.9rem; opacity:0.6;">CURRENT</div>
                <div style="font-size:4rem; font-weight:800; color:{_gc_before};">{_g_before}</div>
                <div style="font-size:0.85rem;">PF {sa.profit_factor:.2f} | Exp USD {sa.expectancy:.2f}</div>
            </div>
            <div style="display:flex; align-items:center; font-size:3rem; opacity:0.3;">→</div>
            <div style="text-align:center; padding:20px 40px; background:rgba(34,197,94,0.1);
                        border:2px solid {_gc_after}; border-radius:12px;">
                <div style="font-size:0.9rem; opacity:0.6;">AI-FIXED</div>
                <div style="font-size:4rem; font-weight:800; color:{_gc_after};">{_g_after}</div>
                <div style="font-size:0.85rem;">PF {_ar_fixed.profit_factor:.2f} | Exp USD {_ar_fixed.expectancy:.2f}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Key metrics comparison
        _m_cols = st.columns(6)
        _m_cols[0].metric("Profit Factor", f"{_ar_fixed.profit_factor:.2f}",
                          delta=f"{_ar_fixed.profit_factor - sa.profit_factor:+.2f}")
        _m_cols[1].metric("Expectancy", f"${_ar_fixed.expectancy:.2f}",
                          delta=f"${_ar_fixed.expectancy - sa.expectancy:+.2f}")
        _m_cols[2].metric("Win Rate", f"{_ar_fixed.win_rate*100:.1f}%",
                          delta=f"{(_ar_fixed.win_rate - sa.win_rate)*100:+.1f}%")
        _m_cols[3].metric("Payoff Ratio", f"{_ar_fixed.payoff_ratio:.3f}",
                          delta=f"{_ar_fixed.payoff_ratio - sa.payoff_ratio:+.3f}")
        _m_cols[4].metric("Max DD", f"{_ar_fixed.max_drawdown_pct:.1f}%",
                          delta=f"{_ar_fixed.max_drawdown_pct - sa.max_drawdown_pct:+.1f}%",
                          delta_color="inverse")
        _m_cols[5].metric("Net Profit", f"${_ar_fixed.total_profit:,.0f}",
                          delta=f"${_ar_fixed.total_profit - sa.total_profit:+,.0f}")

        _m2_cols = st.columns(6)
        _m2_cols[0].metric("Sharpe", f"{_ar_fixed.sharpe_ratio:.3f}",
                           delta=f"{_ar_fixed.sharpe_ratio - sa.sharpe_ratio:+.3f}")
        _m2_cols[1].metric("Sortino", f"{_ar_fixed.sortino_ratio:.3f}" if np.isfinite(_ar_fixed.sortino_ratio) else "Inf",
                           delta=f"{_ar_fixed.sortino_ratio - sa.sortino_ratio:+.3f}" if np.isfinite(_ar_fixed.sortino_ratio) and np.isfinite(sa.sortino_ratio) else "")
        _m2_cols[2].metric("SQN", f"{_ar_fixed.sqn:.2f}",
                           delta=f"{_ar_fixed.sqn - sa.sqn:+.2f}")
        _m2_cols[3].metric("Total Trades", f"{_ar_fixed.n_trades}",
                           delta=f"{_ar_fixed.n_trades - sa.n_trades:+d}")
        _m2_cols[4].metric("Recovery Factor", f"{_ar_fixed.recovery_factor:.2f}" if np.isfinite(_ar_fixed.recovery_factor) else "Inf",
                           delta=f"{_ar_fixed.recovery_factor - sa.recovery_factor:+.2f}" if np.isfinite(_ar_fixed.recovery_factor) and np.isfinite(sa.recovery_factor) else "")
        _m2_cols[5].metric("Kelly", f"{_ar_fixed.kelly_criterion:.4f}",
                           delta=f"{_ar_fixed.kelly_criterion - sa.kelly_criterion:+.4f}")

        # ── Parameter Changes Applied ──
        _ar_params = _ar.get('params', {})
        if _ar_params:
            with st.expander(f"📋 {len(_ar_params)} Parameter Changes Applied", expanded=False):
                _pc_cols = st.columns(min(4, len(_ar_params)))
                for _pi, (_pk, _pv) in enumerate(_ar_params.items()):
                    with _pc_cols[_pi % len(_pc_cols)]:
                        _cur = "?"
                        for _idef in st.session_state.get('_pine_input_defs', []):
                            if _idef.get('title') == _pk or _idef.get('key') == _pk:
                                _cur = _idef.get('defval', '?')
                                break
                        st.metric(_pk, f"{_pv}", delta=f"was {_cur}")

        # ── Equity Curves: Before vs After vs Benchmark ──
        st.markdown("### Equity Curves")
        _eq_fig = go.Figure()
        _eq_fig.add_trace(go.Scatter(y=sa.equity, name="Current Strategy",
                                      line=dict(color='#ef4444', width=2)))
        _eq_fig.add_trace(go.Scatter(y=_ar_fixed.equity, name="AI-Optimized Strategy",
                                      line=dict(color='#22c55e', width=2.5)))

        # Buy-and-hold benchmark
        _bench_df = st.session_state.get('_pine_cached_data')
        if _bench_df is not None:
            _bc = {c.lower(): c for c in _bench_df.columns}
            _bclose = _bc.get('close')
            if _bclose and _bclose in _bench_df.columns:
                _bprices = pd.to_numeric(_bench_df[_bclose], errors='coerce').dropna()
                if len(_bprices) > 1:
                    _bh_equity = sa.initial_capital * (_bprices.values / _bprices.values[0])
                    # Resample to trade count (approximate)
                    _bh_indices = np.linspace(0, len(_bh_equity) - 1, len(sa.equity), dtype=int)
                    _bh_resampled = _bh_equity[_bh_indices]
                    _eq_fig.add_trace(go.Scatter(y=_bh_resampled, name="Buy & Hold",
                                                  line=dict(color='#6366f1', width=1.5, dash='dash')))

        _eq_fig.update_layout(**PLOTLY_LAYOUT, height=400,
                              yaxis_title="Equity ($)",
                              legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.5)"))
        st.plotly_chart(_eq_fig, use_container_width=True, key="_audit_eq_chart")

        # ── Monte Carlo: Before vs After ──
        _ar_mc_block = _ar.get('mc_block_fixed') or {}
        if _ar_mc_block and 'paths' in _ar_mc_block:
            st.markdown("### Monte Carlo Projection (Block Bootstrap)")
            _mc_compare_cols = st.columns(2)
            with _mc_compare_cols[0]:
                st.markdown("#### Current Strategy")
                st.metric("P(Profit)", f"{mc_block['prob_profit']:.1f}%")
                st.metric("P(Ruin)", f"{mc_block['prob_ruin']:.2f}%")
                st.metric("Final Equity P50", f"${mc_block['percentiles']['p50']:,.0f}")
                st.metric("Final Equity P5 (worst)", f"${mc_block['percentiles']['p5']:,.0f}")
                st.metric("Max DD P95", f"{mc_block['dd_percentiles']['p95']:.1f}%")
                _cur_mc_fig = go.Figure()
                _n_show = min(200, len(mc_block['paths']))
                for _pi in range(_n_show):
                    _cur_mc_fig.add_trace(go.Scatter(y=mc_block['paths'][_pi], mode='lines',
                                                      line=dict(color='rgba(239,68,68,0.05)', width=0.5),
                                                      showlegend=False))
                _cur_mc_fig.add_trace(go.Scatter(y=sa.equity, name="Actual",
                                                  line=dict(color='white', width=2)))
                _cur_mc_fig.update_layout(**PLOTLY_LAYOUT, height=300, margin=dict(l=40,r=20,t=30,b=30),
                                           title="Current MC Paths", yaxis_title="Equity ($)")
                st.plotly_chart(_cur_mc_fig, use_container_width=True, key="_audit_cur_mc")

            with _mc_compare_cols[1]:
                st.markdown("#### AI-Fixed Strategy")
                st.metric("P(Profit)", f"{_ar_mc_block['prob_profit']:.1f}%",
                          delta=f"{_ar_mc_block['prob_profit'] - mc_block['prob_profit']:+.1f}%")
                st.metric("P(Ruin)", f"{_ar_mc_block['prob_ruin']:.2f}%",
                          delta=f"{_ar_mc_block['prob_ruin'] - mc_block['prob_ruin']:+.2f}%", delta_color="inverse")
                st.metric("Final Equity P50", f"${_ar_mc_block['percentiles']['p50']:,.0f}",
                          delta=f"${_ar_mc_block['percentiles']['p50'] - mc_block['percentiles']['p50']:+,.0f}")
                st.metric("Final Equity P5 (worst)", f"${_ar_mc_block['percentiles']['p5']:,.0f}",
                          delta=f"${_ar_mc_block['percentiles']['p5'] - mc_block['percentiles']['p5']:+,.0f}")
                st.metric("Max DD P95", f"{_ar_mc_block['dd_percentiles']['p95']:.1f}%",
                          delta=f"{_ar_mc_block['dd_percentiles']['p95'] - mc_block['dd_percentiles']['p95']:+.1f}%",
                          delta_color="inverse")
                _fix_mc_fig = go.Figure()
                _n_show_f = min(200, len(_ar_mc_block['paths']))
                for _pi in range(_n_show_f):
                    _fix_mc_fig.add_trace(go.Scatter(y=_ar_mc_block['paths'][_pi], mode='lines',
                                                      line=dict(color='rgba(34,197,94,0.05)', width=0.5),
                                                      showlegend=False))
                _fix_mc_fig.add_trace(go.Scatter(y=_ar_fixed.equity, name="Actual",
                                                  line=dict(color='white', width=2)))
                _fix_mc_fig.update_layout(**PLOTLY_LAYOUT, height=300, margin=dict(l=40,r=20,t=30,b=30),
                                           title="AI-Fixed MC Paths", yaxis_title="Equity ($)")
                st.plotly_chart(_fix_mc_fig, use_container_width=True, key="_audit_fix_mc")

        # ── Drawdown Comparison ──
        st.markdown("### Drawdown Comparison")
        _dd_fig = go.Figure()
        _dd_fig.add_trace(go.Scatter(y=sa.drawdown_pct * 100, name="Current DD", fill='tozeroy',
                                      line=dict(color='#ef4444', width=1), fillcolor='rgba(239,68,68,0.15)'))
        _dd_fig.add_trace(go.Scatter(y=_ar_fixed.drawdown_pct * 100, name="AI-Fixed DD", fill='tozeroy',
                                      line=dict(color='#22c55e', width=1), fillcolor='rgba(34,197,94,0.10)'))
        _dd_fig.update_layout(**PLOTLY_LAYOUT, height=300, yaxis_title="Drawdown %",
                              legend=dict(x=0.01, y=-0.15, orientation="h"))
        st.plotly_chart(_dd_fig, use_container_width=True, key="_audit_dd_chart")

        # ── Institutional Metrics Comparison ──
        st.markdown("### Institutional Metrics: Before vs After")
        _ar_dsr = _ar.get('dsr_fixed', {})
        _ar_wfe = _ar.get('wfe_fixed')
        _cur_dsr = sa.deflated_sharpe_ratio(num_trials=1)
        _cur_wfe = sa.walk_forward_efficiency(n_splits=5)
        _inst_cmp = pd.DataFrame({
            "Metric": ["Sharpe", "Sortino", "Calmar", "SQN", "DSR %", "WFE (Sharpe)",
                       "Profit Factor", "Payoff Ratio", "Max DD %", "P(Profit) MC"],
            "Current": [
                f"{sa.sharpe_ratio:.3f}", f"{sa.sortino_ratio:.3f}" if np.isfinite(sa.sortino_ratio) else "Inf",
                f"{sa.calmar_ratio:.3f}" if np.isfinite(sa.calmar_ratio) else "Inf",
                f"{sa.sqn:.2f}", f"{_cur_dsr.get('dsr_pct', 0):.1f}%",
                f"{_cur_wfe['avg_wfe_sharpe']:.3f}" if _cur_wfe else "N/A",
                f"{sa.profit_factor:.3f}", f"{sa.payoff_ratio:.3f}",
                f"{sa.max_drawdown_pct:.1f}%", f"{mc_block['prob_profit']:.1f}%",
            ],
            "AI-Fixed": [
                f"{_ar_fixed.sharpe_ratio:.3f}", f"{_ar_fixed.sortino_ratio:.3f}" if np.isfinite(_ar_fixed.sortino_ratio) else "Inf",
                f"{_ar_fixed.calmar_ratio:.3f}" if np.isfinite(_ar_fixed.calmar_ratio) else "Inf",
                f"{_ar_fixed.sqn:.2f}", f"{_ar_dsr.get('dsr_pct', 0):.1f}%",
                f"{_ar_wfe['avg_wfe_sharpe']:.3f}" if _ar_wfe else "N/A",
                f"{_ar_fixed.profit_factor:.3f}", f"{_ar_fixed.payoff_ratio:.3f}",
                f"{_ar_fixed.max_drawdown_pct:.1f}%", f"{_ar_mc_block.get('prob_profit', 0):.1f}%",
            ],
            "Target": ["> 1.0", "> 2.0", "> 1.0", "> 2.0", "> 95%", "> 0.5",
                       "> 1.5", "> 1.0", "< -20%", "> 95%"],
        })
        st.dataframe(_inst_cmp, use_container_width=True, hide_index=True)

        # ── Optimizer Results ──
        _ar_opt_results = _ar.get('optimizer_results')
        _ar_opt_best = _ar.get('optimizer_best')
        if _ar_opt_best:
            st.markdown("### Optimizer: Top Parameter Combinations")
            st.caption("The AI identified weak parameters, then the optimizer tested hundreds of combinations to find the true optimal values.")
            _opt_rows = []
            for _oi, _ores in enumerate(_ar_opt_results[:10] if _ar_opt_results else []):
                _opt_rows.append({
                    'Rank': _oi + 1,
                    'Sharpe': f"{_ores.sharpe_ratio:.3f}",
                    'PF': f"{_ores.profit_factor:.3f}",
                    'Net P&L': f"${_ores.net_profit:,.0f}",
                    'Win Rate': f"{_ores.win_rate*100:.1f}%",
                    'Trades': _ores.num_trades,
                    'Max DD': f"{_ores.max_drawdown_pct:.1f}%",
                    'Params': str(_ores.params),
                })
            if _opt_rows:
                st.dataframe(pd.DataFrame(_opt_rows), use_container_width=True, hide_index=True)

        # ── Walk-Forward Validation ──
        _ar_wf = _ar.get('walk_forward')
        if _ar_wf and _ar_wf.get('splits'):
            st.markdown("### Walk-Forward Validation (Overfitting Check)")
            _wf_splits = _ar_wf['splits']
            _profitable_splits = sum(1 for s in _wf_splits if s.get('profitable'))
            _total_splits = len(_wf_splits)
            _consistency = _profitable_splits / _total_splits * 100 if _total_splits > 0 else 0

            _wf_c1, _wf_c2, _wf_c3 = st.columns(3)
            _wf_c1.metric("Profitable Splits", f"{_profitable_splits}/{_total_splits}")
            _wf_c2.metric("Consistency", f"{_consistency:.0f}%",
                          delta="Robust" if _consistency >= 60 else "Possible Overfit",
                          delta_color="normal" if _consistency >= 60 else "inverse")
            _wf_c3.metric("Avg OOS Sharpe", f"{_ar_wf.get('avg_sharpe', 0):.3f}")

            _wf_rows = []
            for _ws in _wf_splits:
                _wf_rows.append({
                    'Split': _ws['split'],
                    'Test Bars': _ws['test_bars'],
                    'Trades': _ws['num_trades'],
                    'Net P&L': f"${_ws['net_profit']:,.2f}",
                    'Sharpe': f"{_ws['sharpe']:.3f}",
                    'Profitable': 'Yes' if _ws['profitable'] else 'No',
                })
            st.dataframe(pd.DataFrame(_wf_rows), use_container_width=True, hide_index=True)

            if _consistency < 60:
                st.warning("Less than 60% of walk-forward splits are profitable. "
                           "The optimized parameters may be overfit to in-sample data. "
                           "Consider using more conservative parameter values.")
            else:
                st.success(f"{_consistency:.0f}% of out-of-sample periods are profitable. "
                           "These parameter improvements appear robust.")

        # ── Parameter Sensitivity ──
        _ar_sens = _ar.get('sensitivity')
        if _ar_sens:
            st.markdown("### Parameter Sensitivity (Stability Check)")
            st.caption("If nearby values also perform well (plateau), the parameter is robust. "
                       "If performance drops sharply (peak), it may be overfit.")
            for _sname, _sdata in _ar_sens.items():
                if len(_sdata) >= 3:
                    _sv = [d['value'] for d in _sdata]
                    _ss = [d['sharpe'] for d in _sdata]
                    _fig_sens = go.Figure()
                    _fig_sens.add_trace(go.Scatter(x=_sv, y=_ss, mode='lines+markers',
                                                    line=dict(color='#00c8ff', width=2),
                                                    marker=dict(size=8)))
                    # Mark the optimal value
                    _best_idx = _ss.index(max(_ss))
                    _fig_sens.add_trace(go.Scatter(x=[_sv[_best_idx]], y=[_ss[_best_idx]],
                                                    mode='markers', marker=dict(size=14, color='#22c55e',
                                                    symbol='star'), name='Optimal'))
                    _fig_sens.update_layout(**PLOTLY_LAYOUT, height=250,
                                            title=f"{_sname} Sensitivity",
                                            xaxis_title=_sname, yaxis_title="Sharpe Ratio")
                    st.plotly_chart(_fig_sens, use_container_width=True,
                                   key=f"_sens_{_sname}")

                    # Stability assessment
                    _sharpe_range = max(_ss) - min(_ss)
                    if _sharpe_range < 0.2:
                        st.caption(f"Stable (Sharpe range: {_sharpe_range:.3f})")
                    elif _sharpe_range < 0.5:
                        st.caption(f"Moderate sensitivity (Sharpe range: {_sharpe_range:.3f})")
                    else:
                        st.warning(f"High sensitivity (Sharpe range: {_sharpe_range:.3f}) — possible overfit")

        # ── Confidence Assessment ──
        _confidence_score = 0
        _confidence_notes = []
        if _ar_wf and _ar_wf.get('splits'):
            _wf_consistency = sum(1 for s in _ar_wf['splits'] if s.get('profitable')) / len(_ar_wf['splits']) * 100
            if _wf_consistency >= 80:
                _confidence_score += 3
                _confidence_notes.append("Walk-forward: 80%+ splits profitable")
            elif _wf_consistency >= 60:
                _confidence_score += 2
                _confidence_notes.append(f"Walk-forward: {_wf_consistency:.0f}% splits profitable")
            else:
                _confidence_notes.append(f"Walk-forward: only {_wf_consistency:.0f}% splits profitable (weak)")

        if _ar_sens:
            _all_stable = all(max(d['sharpe'] for d in v) - min(d['sharpe'] for d in v) < 0.3
                              for v in _ar_sens.values() if len(v) >= 3)
            if _all_stable:
                _confidence_score += 2
                _confidence_notes.append("Parameter sensitivity: all params stable")
            else:
                _confidence_score += 1
                _confidence_notes.append("Parameter sensitivity: some params volatile")

        if _ar_fixed and _ar_fixed.profit_factor > 1.0:
            _confidence_score += 1
            _confidence_notes.append(f"Post-optimization PF: {_ar_fixed.profit_factor:.2f}")

        if _confidence_score >= 5:
            _conf_label = "HIGH CONFIDENCE"
            _conf_color = "#22c55e"
        elif _confidence_score >= 3:
            _conf_label = "MODERATE CONFIDENCE"
            _conf_color = "#fbbf24"
        else:
            _conf_label = "LOW CONFIDENCE — possible overfit"
            _conf_color = "#ef4444"

        st.markdown(f"""
        <div style="border: 2px solid {_conf_color}; border-radius: 12px; padding: 16px; margin: 16px 0;
                    background: rgba({','.join(str(int(_conf_color.lstrip('#')[i:i+2], 16)) for i in (0,2,4))}, 0.1);">
            <div style="font-size: 1.2rem; font-weight: 800; color: {_conf_color};">{_conf_label}</div>
            <div style="font-size: 0.85rem; margin-top: 8px; opacity: 0.8;">
                {'<br>'.join(_confidence_notes)}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── SECTION B: When no automated re-test, offer manual comparison ──
    if not _ar_fixed:
        _ar_params = _ar.get('params', {})
        _ar_error = _ar.get('error', '')
        _ar_too_few = _ar.get('too_few_trades', False)
        _ar_no_pine = _ar.get('no_pine', False)

        if _ar_params:
            st.markdown("## Recommended Parameter Changes")
            st.info("The automated re-test couldn't run (Pine interpreter doesn't support this strategy's complexity). "
                    "**Apply these changes in TradingView, re-export the CSV, and upload below to compare.**")
            _pc_cols = st.columns(min(4, max(1, len(_ar_params))))
            for _pi, (_pk, _pv) in enumerate(_ar_params.items()):
                with _pc_cols[_pi % len(_pc_cols)]:
                    _cur = "?"
                    for _idef in st.session_state.get('_pine_input_defs', []):
                        if _idef.get('title') == _pk or _idef.get('key') == _pk:
                            _cur = _idef.get('defval', '?')
                            break
                    st.metric(_pk, f"{_pv}", delta=f"was {_cur}")

        # ── Upload Fixed CSV for comparison ──
        st.markdown("---")
        st.markdown("### Upload Fixed Results for Comparison")
        st.caption("Apply the AI's recommended parameter changes in TradingView, run the backtest there, "
                   "export the strategy report as CSV, and upload it here to see the before/after comparison.")

        _fixed_csv = st.file_uploader("Upload TradingView CSV (after applying AI fixes)",
                                       type=["csv", "xlsx"], key="_fixed_csv_upload")
        if _fixed_csv:
            try:
                if _fixed_csv.name.endswith('.xlsx'):
                    _fix_raw = pd.read_excel(_fixed_csv)
                else:
                    _fix_raw = pd.read_csv(_fixed_csv)

                _fix_parsed = parse_tradingview_csv(_fix_raw)
                if _fix_parsed is not None and len(_fix_parsed) >= 5:
                    # Extract trade data
                    _fix_col_map = {}
                    _lc = {c.lower().strip(): c for c in _fix_parsed.columns}
                    for _alias in ["net p&l usd", "net p&l", "profit", "pnl", "p&l"]:
                        if _alias in _lc:
                            _fix_col_map['profit'] = _lc[_alias]
                            break
                    _profit_col = _fix_col_map.get('profit')
                    if _profit_col:
                        _fix_profits = pd.to_numeric(_fix_parsed[_profit_col], errors='coerce').fillna(0).values

                        _fix_dates_col = None
                        for _da in ["date and time", "date/time", "datetime", "date"]:
                            if _da in _lc:
                                _fix_dates_col = _lc[_da]
                                break
                        _fix_dates = pd.to_datetime(_fix_parsed[_fix_dates_col], errors='coerce') if _fix_dates_col else None

                        _fix_sa = StrategyAnalytics(
                            profits=_fix_profits,
                            initial_capital=initial_capital,
                            dates=_fix_dates,
                            risk_free_rate=risk_free_rate,
                        )
                        _fix_mc = _fix_sa.monte_carlo(n_sims=mc_sims)
                        _fix_bs = max(5, min(20, int(np.sqrt(_fix_sa.n_trades))))
                        _fix_mc_block = _fix_sa.monte_carlo_block(n_sims=mc_sims, block_size=_fix_bs)
                        _fix_dsr = _fix_sa.deflated_sharpe_ratio(num_trials=1)

                        # Store and re-display as if it were the automated result
                        st.session_state['_auto_audit_results']['sa_fixed'] = _fix_sa
                        st.session_state['_auto_audit_results']['mc_fixed'] = _fix_mc
                        st.session_state['_auto_audit_results']['mc_block_fixed'] = _fix_mc_block
                        st.session_state['_auto_audit_results']['dsr_fixed'] = _fix_dsr
                        st.session_state['_auto_audit_results']['wfe_fixed'] = _fix_sa.walk_forward_efficiency(n_splits=5)
                        st.success(f"Loaded {_fix_sa.n_trades} trades from fixed CSV. Refresh to see comparison.")
                        st.rerun()
                    else:
                        st.error("Could not find a profit/PnL column in the uploaded file.")
                else:
                    st.error("Could not parse the uploaded file as a TradingView strategy report, or too few trades.")
            except Exception as _fix_err:
                st.error(f"Error parsing fixed CSV: {_fix_err}")

    # ── SECTION C: AI Full Analysis ──────────────────────────────
    st.markdown("---")
    st.markdown("## AI Analysis & Code Fixes")
    _ar_reply = _ar.get('ai_reply', '')
    if _ar_reply:
        # Get revised Pine Script — prefer direct key from background process
        _revised_pine = _ar.get('revised_pine')
        if not _revised_pine:
            # Fallback: extract from AI reply text
            _revised_match = re.search(r'```pine:revised_strategy\s*\n(.*?)```', _ar_reply, re.DOTALL)
            if not _revised_match:
                _revised_match = re.search(r'```pinescript\s*\n(.*?)```', _ar_reply, re.DOTALL)
            if not _revised_match:
                _revised_match = re.search(r'```(?:pine|pinescript)?\s*\n(//@version.*?)```', _ar_reply, re.DOTALL)
            if _revised_match:
                _revised_pine = _revised_match.group(1).strip()

        # Show the analysis (without the giant code block cluttering it)
        _display_reply = _ar_reply
        if _revised_pine:
            # Remove the full revised script from the analysis display to keep it clean
            _display_reply = re.sub(
                r'```pine:revised_strategy\s*\n.*?```',
                '\n> **Full revised Pine Script is shown below in the copy-paste section.**\n',
                _display_reply, flags=re.DOTALL
            )
        st.markdown(_display_reply)

        # ── SECTION D: Copy-Paste Revised Pine Script ──────────────
        # Validate revised Pine Script
        if _revised_pine:
            _pine_orig = st.session_state.get('_pine_source_code', '')
            _orig_lines = len(_pine_orig.splitlines()) if _pine_orig else 0
            _rev_lines_count = len(_revised_pine.splitlines())
            _has_version = '//@version' in _revised_pine
            _has_strategy = 'strategy(' in _revised_pine

            _pine_warnings = []
            if _orig_lines > 0 and _rev_lines_count < _orig_lines * 0.5:
                _pine_warnings.append(f"Revised script ({_rev_lines_count} lines) is much shorter "
                                      f"than original ({_orig_lines} lines) — may be truncated")
            if not _has_version:
                _pine_warnings.append("Missing //@version tag")
            if not _has_strategy:
                _pine_warnings.append("Missing strategy() declaration")

        if _revised_pine:
            st.markdown("---")
            st.markdown("""
            <div style="background: linear-gradient(135deg, #0d1117 0%, #1a2332 100%);
                        border: 2px solid #22c55e; border-radius: 12px;
                        padding: 20px; margin: 16px 0;">
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
                    <span style="font-size:1.5rem;">📋</span>
                    <div>
                        <div style="font-size:1.2rem; font-weight:800; color:#22c55e;">
                            Revised Pine Script — Copy & Paste into TradingView
                        </div>
                        <div style="font-size:0.8rem; opacity:0.6;">
                            All AI-recommended fixes are already applied. Paste this into TradingView to verify.
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if _pine_warnings:
                for _pw in _pine_warnings:
                    st.warning(f"Script validation: {_pw}")

            _pine_lines = _revised_pine.count('\n') + 1
            st.text_area(
                f"Revised Pine Script ({_pine_lines} lines)",
                value=_revised_pine,
                height=500,
                key="_revised_pine_output",
                help="Select all (Cmd+A) and copy (Cmd+C), then paste into TradingView Pine Editor"
            )

            # Download button
            st.download_button(
                label="📥 Download Revised Pine Script",
                data=_revised_pine,
                file_name="strategy_revised.pine",
                mime="text/plain",
                key="_download_revised_pine",
                use_container_width=True,
            )
        elif _ar_reply and 'pine:revised_strategy' not in _ar_reply:
            st.warning("The AI didn't include a full revised Pine Script. "
                       "Click 'Run Full Auto-Audit' again or ask in the AI chat: "
                       "'Give me the COMPLETE revised Pine Script with all fixes applied, no truncation.'")

    # ── Clear button ──
    if st.button("🗑️ Clear Auto-Audit Results", key="_clear_audit"):
        if '_auto_audit_results' in st.session_state:
            del st.session_state['_auto_audit_results']
        if '_auto_audit_chat' in st.session_state:
            del st.session_state['_auto_audit_chat']
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

tabs = st.tabs([
    "📊 Overview",
    "📈 Equity & DD",
    "🎯 Trade Analysis",
    "🎲 Monte Carlo",
    "📐 Distribution",
    "📉 Rolling Stats",
    "🕐 Time Analysis",
    "📦 Position Sizing",
    "📋 Full Report",
    "🏆 Strategy Grade",
    "🤖 AI Analyst",
    "🔬 Optimizer",
    "🏛️ Institutional",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 0: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[0]:
    st.subheader("Key Performance Indicators")

    r1 = st.columns(5)
    r1[0].metric("Total Trades", f"{sa.n_trades}")
    r1[1].metric("Net Profit", f"${sa.total_profit:,.2f}",
                  delta=f"{sa.total_return_pct:+.1f}%")
    r1[2].metric("Win Rate", f"{sa.win_rate:.1%}")
    r1[3].metric("Profit Factor", _fmt(sa.profit_factor, ".2f"))
    r1[4].metric("CAGR", f"{sa.cagr:.2f}%")

    r2 = st.columns(5)
    r2[0].metric("Sharpe Ratio", f"{sa.sharpe_ratio:.3f}")
    r2[1].metric("Sortino Ratio", _fmt(sa.sortino_ratio, ".3f"))
    r2[2].metric("Max Drawdown", f"{sa.max_drawdown_pct:.2f}%")
    r2[3].metric("Expectancy", f"${sa.expectancy:,.2f}")
    r2[4].metric("SQN", f"{sa.sqn:.2f}")

    r3 = st.columns(5)
    r3[0].metric("Calmar Ratio", _fmt(sa.calmar_ratio, ".3f"))
    r3[1].metric("Omega Ratio", _fmt(sa.omega_ratio, ".3f"))
    r3[2].metric("Recovery Factor", _fmt(sa.recovery_factor, ".2f"))
    r3[3].metric("Payoff Ratio", _fmt(sa.payoff_ratio, ".2f"))
    r3[4].metric("Kelly Criterion", f"{sa.kelly_criterion:.3f}")

    st.divider()

    # SQN Interpretation
    sqn_val = sa.sqn
    if sqn_val >= 7:
        sqn_grade, sqn_color = "Holy Grail", "🟣"
    elif sqn_val >= 5.1:
        sqn_grade, sqn_color = "Superb", "🟢"
    elif sqn_val >= 3:
        sqn_grade, sqn_color = "Excellent", "🟢"
    elif sqn_val >= 2:
        sqn_grade, sqn_color = "Good", "🟡"
    elif sqn_val >= 1.5:
        sqn_grade, sqn_color = "Below Average", "🟠"
    else:
        sqn_grade, sqn_color = "Poor", "🔴"

    st.markdown(f"**System Quality Number (SQN) Grade:** {sqn_color} **{sqn_grade}** ({sqn_val:.2f})")

    # Statistical significance
    if sa.p_value_profit < 0.01:
        sig = "✅ Highly significant (p < 0.01) — strong evidence of edge"
    elif sa.p_value_profit < 0.05:
        sig = "✅ Significant (p < 0.05) — evidence of edge"
    elif sa.p_value_profit < 0.10:
        sig = "⚠️ Marginally significant (p < 0.10) — weak evidence"
    else:
        sig = "❌ Not significant (p ≥ 0.10) — no statistical evidence of edge"

    st.markdown(f"**Statistical Significance:** {sig} (t={sa.t_stat_profit:.3f}, p={sa.p_value_profit:.6f})")

    st.divider()
    st.plotly_chart(make_equity_chart(sa, mc), use_container_width=True, key="equity_overview")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: EQUITY & DRAWDOWN
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[1]:
    st.subheader("Equity Curve & Drawdown Deep Dive")

    st.plotly_chart(make_equity_chart(sa, mc), use_container_width=True, key="equity_dd_tab")
    st.plotly_chart(make_drawdown_chart(sa), use_container_width=True, key="dd_chart")

    # Drawdown table
    dd_periods = sa._drawdown_periods()
    if dd_periods:
        st.subheader("Top Drawdown Periods")
        dd_df = pd.DataFrame(dd_periods)
        dd_df = dd_df.sort_values("depth").head(10)
        dd_df["depth"] = dd_df["depth"].map(lambda x: f"${x:,.2f}")
        dd_df["depth_pct"] = dd_df["depth_pct"].map(lambda x: f"{x:.2f}%")
        dd_df["recovery"] = dd_df["recovery"].map(lambda x: f"Trade {x}" if x else "Ongoing")
        dd_df["start"] = dd_df["start"].map(lambda x: f"Trade {x}")
        dd_df["trough"] = dd_df["trough"].map(lambda x: f"Trade {x}")
        st.dataframe(dd_df[["start", "trough", "recovery", "depth", "depth_pct", "length"]],
                     use_container_width=True)

    st.subheader("Drawdown Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Max DD ($)", f"${sa.max_drawdown:,.2f}")
    c2.metric("Max DD (%)", f"{sa.max_drawdown_pct:.2f}%")
    c3.metric("Avg DD ($)", f"${sa.avg_drawdown:,.2f}")
    c4.metric("Max DD Duration", f"{sa.max_drawdown_duration} trades")

    st.plotly_chart(make_cumulative_metrics_chart(sa), use_container_width=True, key="cum_metrics")

#JUST TRADES CONFIDENTIAL PROPERTY
# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: TRADE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[2]:
    st.subheader("Individual Trade Analysis")

    st.plotly_chart(make_trade_scatter(sa), use_container_width=True, key="trade_scatter")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Winning Trades**")
        w_metrics = {
            "Count": len(sa.wins),
            "Total": f"${sa.wins.sum():,.2f}",
            "Average": f"${sa.avg_win:,.2f}",
            "Median": f"${sa.median_win:,.2f}",
            "Largest": f"${sa.largest_win:,.2f}",
            "Std Dev": f"${sa.wins.std(ddof=1):,.2f}" if len(sa.wins) > 1 else "N/A",
            "Max Streak": sa.max_consecutive_wins,
            "Avg Streak": f"{sa.avg_consecutive_wins:.1f}",
        }
        st.dataframe(pd.DataFrame(w_metrics, index=["Value"]).T, use_container_width=True)

    with c2:
        st.markdown("**Losing Trades**")
        l_metrics = {
            "Count": len(sa.losses),
            "Total": f"${sa.losses.sum():,.2f}",
            "Average": f"-${sa.avg_loss:,.2f}",
            "Median": f"${sa.median_loss:,.2f}",
            "Largest": f"${sa.largest_loss:,.2f}",
            "Std Dev": f"${sa.losses.std(ddof=1):,.2f}" if len(sa.losses) > 1 else "N/A",
            "Max Streak": sa.max_consecutive_losses,
            "Avg Streak": f"{sa.avg_consecutive_losses:.1f}",
        }
        st.dataframe(pd.DataFrame(l_metrics, index=["Value"]).T, use_container_width=True)

    # Win/Loss by trade type (long/short)
    if trade_types is not None:
        st.divider()
        st.subheader("Performance by Direction")
        type_df = pd.DataFrame({"type": trade_types, "profit": profits})
        type_df["type"] = type_df["type"].str.strip().str.lower()
        type_summary = type_df.groupby("type").agg(
            n_trades=("profit", "count"),
            total_pnl=("profit", "sum"),
            avg_pnl=("profit", "mean"),
            win_rate=("profit", lambda x: (x > 0).mean()),
            max_win=("profit", "max"),
            max_loss=("profit", "min"),
        )
        type_summary["total_pnl"] = type_summary["total_pnl"].map(lambda x: f"${x:,.2f}")
        type_summary["avg_pnl"] = type_summary["avg_pnl"].map(lambda x: f"${x:,.2f}")
        type_summary["win_rate"] = type_summary["win_rate"].map(lambda x: f"{x:.1%}")
        type_summary["max_win"] = type_summary["max_win"].map(lambda x: f"${x:,.2f}")
        type_summary["max_loss"] = type_summary["max_loss"].map(lambda x: f"${x:,.2f}")
        st.dataframe(type_summary, use_container_width=True)

    # Win/Loss by signal
    if signals is not None:
        st.divider()
        st.subheader("Performance by Signal")
        sig_df = pd.DataFrame({"signal": signals, "profit": profits})
        sig_summary = sig_df.groupby("signal").agg(
            n_trades=("profit", "count"),
            total_pnl=("profit", "sum"),
            avg_pnl=("profit", "mean"),
            win_rate=("profit", lambda x: (x > 0).mean()),
        ).sort_values("total_pnl", ascending=False)
        sig_summary["total_pnl"] = sig_summary["total_pnl"].map(lambda x: f"${x:,.2f}")
        sig_summary["avg_pnl"] = sig_summary["avg_pnl"].map(lambda x: f"${x:,.2f}")
        sig_summary["win_rate"] = sig_summary["win_rate"].map(lambda x: f"{x:.1%}")
        st.dataframe(sig_summary, use_container_width=True)

    # MFE / MAE analysis
    if runups is not None and dd_per_trade is not None:
        st.divider()
        st.subheader("MFE / MAE Analysis")
        fig_mfe = go.Figure()
        colors = np.where(profits > 0, COLORS["win"], COLORS["loss"])
        fig_mfe.add_trace(go.Scatter(
            x=dd_per_trade, y=runups, mode="markers",
            marker=dict(color=colors, size=5, opacity=0.6),
            text=[f"P&L: ${p:,.2f}" for p in profits],
            hoverinfo="text",
        ))
        fig_mfe.update_layout(**PLOTLY_LAYOUT, title="MAE vs MFE",
                              xaxis_title="Max Adverse Excursion",
                              yaxis_title="Max Favorable Excursion")
        st.plotly_chart(fig_mfe, use_container_width=True, key="mfe_mae")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: MONTE CARLO
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[3]:
    st.subheader("Monte Carlo Simulation")

    # ── Autocorrelation Warning ──────────────────────────────────────
    _ac_significant = any(abs(v) > 0.1 for v in trade_autocorr.values()) if trade_autocorr else False
    if _ac_significant:
        _max_ac_lag = max(trade_autocorr.keys(), key=lambda k: abs(trade_autocorr[k]))
        _max_ac_val = trade_autocorr[_max_ac_lag]
        st.warning(f"⚠️ **Trade autocorrelation detected** (lag-{_max_ac_lag}: {_max_ac_val:+.3f}). "
                   f"Trades are NOT independent — standard MC underestimates risk. "
                   f"Block Bootstrap (below) accounts for this clustering.")
    else:
        st.success("✅ Low trade autocorrelation — standard and block bootstrap should agree.")

    # ── Simulation selector ──────────────────────────────────────────
    _mc_view = st.radio(
        "Simulation Method",
        ["📊 Standard Bootstrap", "🔒 Block Bootstrap (DCA-safe)", "⚖️ Compare Both"],
        horizontal=True, key="_mc_method"
    )

    _mc_show = mc if _mc_view == "📊 Standard Bootstrap" else mc_block

    st.caption(f"{'Standard' if _mc_view == '📊 Standard Bootstrap' else 'Block'} bootstrap — "
               f"{mc_sims:,} simulations"
               f"{f', block size={_block_size}' if _mc_view != '📊 Standard Bootstrap' else ''}")

    st.plotly_chart(make_equity_chart(sa, _mc_show), use_container_width=True, key="equity_mc_tab")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(make_mc_final_equity_dist(_mc_show, initial_capital), use_container_width=True, key="mc_final_eq")
    with c2:
        st.plotly_chart(make_mc_drawdown_dist(_mc_show), use_container_width=True, key="mc_dd_dist")

    st.divider()

    # ── Probability Table ────────────────────────────────────────────
    if _mc_view == "⚖️ Compare Both":
        st.subheader("Probability Comparison: Standard vs Block Bootstrap")
        _cmp_df = pd.DataFrame({
            "Metric": ["P(Profit)", "P(2× Capital)", "P(Lose 50%+)", "P(Ruin)",
                       "Final Equity P5", "Final Equity P50", "Final Equity P95",
                       "Max DD P50", "Max DD P95", "Max DD P99"],
            "Standard MC": [
                f"{mc['prob_profit']:.1f}%", f"{mc['prob_2x']:.1f}%",
                f"{mc['prob_loss50']:.1f}%", f"{mc['prob_ruin']:.2f}%",
                f"${mc['percentiles']['p5']:,.0f}", f"${mc['percentiles']['p50']:,.0f}",
                f"${mc['percentiles']['p95']:,.0f}",
                f"{mc['dd_percentiles']['p50']:.2f}%", f"{mc['dd_percentiles']['p95']:.2f}%",
                f"{mc['dd_percentiles']['p99']:.2f}%",
            ],
            "Block Bootstrap": [
                f"{mc_block['prob_profit']:.1f}%", f"{mc_block['prob_2x']:.1f}%",
                f"{mc_block['prob_loss50']:.1f}%", f"{mc_block['prob_ruin']:.2f}%",
                f"${mc_block['percentiles']['p5']:,.0f}", f"${mc_block['percentiles']['p50']:,.0f}",
                f"${mc_block['percentiles']['p95']:,.0f}",
                f"{mc_block['dd_percentiles']['p50']:.2f}%", f"{mc_block['dd_percentiles']['p95']:.2f}%",
                f"{mc_block['dd_percentiles']['p99']:.2f}%",
            ],
        })
        # Highlight differences
        st.dataframe(_cmp_df, use_container_width=True, hide_index=True)
        st.caption(f"Block size = {_block_size} trades (adaptive: √n). "
                   "Larger gaps between columns = more trade clustering/autocorrelation.")
    else:
        st.subheader("Probability Table")
        r1 = st.columns(4)
        r1[0].metric("P(Profit)", f"{_mc_show['prob_profit']:.1f}%")
        r1[1].metric("P(2× Capital)", f"{_mc_show['prob_2x']:.1f}%")
        r1[2].metric("P(Lose 50%+)", f"{_mc_show['prob_loss50']:.1f}%")
        r1[3].metric("P(Ruin)", f"{_mc_show['prob_ruin']:.2f}%")

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Final Equity Percentiles**")
        peq = _mc_show["percentiles"]
        eq_df = pd.DataFrame({
            "Percentile": [f"{k.upper()}" for k in peq],
            "Final Equity": [f"${v:,.2f}" for v in peq.values()],
            "Return": [f"{(v / initial_capital - 1) * 100:+.1f}%" for v in peq.values()],
        })
        st.dataframe(eq_df, use_container_width=True, hide_index=True)

    with c2:
        st.markdown("**Max Drawdown Percentiles**")
        pdd = _mc_show["dd_percentiles"]
        dd_df = pd.DataFrame({
            "Percentile": [f"{k.upper()}" for k in pdd],
            "Max Drawdown %": [f"{v:.2f}%" for v in pdd.values()],
        })
        st.dataframe(dd_df, use_container_width=True, hide_index=True)

    # ── Tail Risk Analysis ───────────────────────────────────────────
    st.divider()
    st.subheader("Tail Risk Analysis (VaR / CVaR)")

    _tr = tail_risk
    r2 = st.columns(4)
    r2[0].metric("VaR 95%", f"${_tr['var_95']['VaR']:,.2f}",
                 help="Worst expected loss on 95% of trades")
    r2[1].metric("CVaR 95% (Exp. Shortfall)", f"${_tr['var_95']['CVaR']:,.2f}",
                 help="Average loss in the worst 5% of trades — the TRUE tail risk")
    r2[2].metric("VaR 99%", f"${_tr['var_99']['VaR']:,.2f}")
    r2[3].metric("CVaR 99%", f"${_tr['var_99']['CVaR']:,.2f}",
                 help="Average loss in the worst 1% of trades")

    st.divider()
    st.subheader("Sigma Event Estimates")
    r3 = st.columns(4)
    r3[0].metric("3σ Loss (per trade)", f"${_tr['sigma3_loss']:,.2f}")
    r3[1].metric("5σ Loss (per trade)", f"${_tr['sigma5_loss']:,.2f}")
    r3[2].metric("Worst Observed", f"${_tr['worst_observed']:,.2f}")
    r3[3].metric("Worst Expected (Normal)", f"${_tr['worst_expected_normal']:,.2f}",
                 help="If returns were normal, this is the worst you'd expect given N trades")

    if _tr['worst_observed'] < _tr['sigma3_loss']:
        st.warning(f"⚠️ Worst observed loss ({_tr['worst_observed']:,.2f}) exceeds the 3σ estimate "
                   f"({_tr['sigma3_loss']:,.2f}) — fat tails confirmed. "
                   "Standard risk models underestimate your tail risk.")

    # ── Autocorrelation Chart ────────────────────────────────────────
    if trade_autocorr:
        st.divider()
        st.subheader("Trade Autocorrelation")
        _ac_fig = go.Figure()
        _lags = list(trade_autocorr.keys())
        _ac_vals = list(trade_autocorr.values())
        _ac_colors = ['#ef4444' if abs(v) > 0.1 else '#22c55e' for v in _ac_vals]
        _ac_fig.add_trace(go.Bar(x=[f"Lag {l}" for l in _lags], y=_ac_vals,
                                 marker_color=_ac_colors, opacity=0.85))
        _ac_fig.add_hline(y=0.1, line_dash="dash", line_color="yellow",
                          annotation_text="Significance threshold (+)")
        _ac_fig.add_hline(y=-0.1, line_dash="dash", line_color="yellow",
                          annotation_text="Significance threshold (-)")
        _ac_fig.update_layout(**PLOTLY_LAYOUT, title="PnL Autocorrelation by Lag",
                              yaxis_title="Correlation", height=300)
        st.plotly_chart(_ac_fig, use_container_width=True, key="ac_chart")
        st.caption("Red bars = significant autocorrelation (|r| > 0.1). "
                   "DCA strategies typically show positive lag-1 correlation (wins cluster, losses cluster). "
                   "This means standard MC UNDERESTIMATES drawdown risk.")

    # ── Risk of Ruin ─────────────────────────────────────────────────
    st.divider()
    st.subheader("Risk of Ruin (Analytical)")
    ror_levels = [20, 30, 40, 50, 60, 75, 100]
    ror_vals = {f"Lose {l}%": f"{sa.risk_of_ruin(l):.4f}%" for l in ror_levels}
    st.dataframe(pd.DataFrame(ror_vals, index=["Probability"]).T, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[4]:
    st.subheader("Distribution & Statistical Tests")

    st.plotly_chart(make_distribution_chart(sa), use_container_width=True, key="distribution")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Skewness", f"{sa.skewness:.3f}")
    c2.metric("Excess Kurtosis", f"{sa.excess_kurtosis:.3f}")
    c3.metric("Tail Ratio", f"{sa.tail_ratio:.3f}")
    c4.metric("Common Sense Ratio", f"{sa.common_sense_ratio:.3f}")

    # Interpretation
    sk = sa.skewness
    if sk > 0.5:
        sk_interp = "Positive skew — right tail is longer (favorable)"
    elif sk < -0.5:
        sk_interp = "Negative skew — left tail is longer (unfavorable)"
    else:
        sk_interp = "Approximately symmetric"

    ek = sa.excess_kurtosis
    if ek > 1:
        ek_interp = "Leptokurtic (fat tails) — more extreme events than normal"
    elif ek < -1:
        ek_interp = "Platykurtic (thin tails) — fewer extreme events"
    else:
        ek_interp = "Approximately mesokurtic (normal-like tails)"

    st.markdown(f"**Skewness:** {sk_interp}")
    st.markdown(f"**Kurtosis:** {ek_interp}")

    st.divider()
    st.subheader("Normality Tests")
    norm_tests = sa.normality_tests()
    for name, result in norm_tests.items():
        if "p_value" in result:
            p = result["p_value"]
            reject = "❌ Reject normality" if p < 0.05 else "✅ Cannot reject normality"
            st.markdown(f"**{name}:** statistic={result['statistic']:.4f}, p={p:.6f} → {reject}")
        else:
            st.markdown(f"**{name}:** statistic={result['statistic']:.4f}")
            cvs = result.get("critical_values", {})
            for sl, cv in cvs.items():
                reject = "❌ Reject" if result["statistic"] > cv else "✅ Accept"
                st.markdown(f"  - At {sl}% significance: critical={cv:.4f} → {reject}")

    st.divider()
    st.subheader("Percentile Analysis")
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pct_vals = np.percentile(sa.raw_profits, pcts)
    pct_df = pd.DataFrame({
        "Percentile": [f"P{p}" for p in pcts],
        "P&L ($)": [f"${v:,.2f}" for v in pct_vals],
    })
    st.dataframe(pct_df, use_container_width=True, hide_index=True)

    # VaR / CVaR
    st.divider()
    st.subheader("Value at Risk")
    for conf in [95, 99]:
        var = np.percentile(sa.raw_profits, 100 - conf)
        cvar = sa.raw_profits[sa.raw_profits <= var].mean() if (sa.raw_profits <= var).any() else var
        c1, c2 = st.columns(2)
        c1.metric(f"VaR ({conf}%)", f"${var:,.2f}")
        c2.metric(f"CVaR / Expected Shortfall ({conf}%)", f"${cvar:,.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: ROLLING STATS
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[5]:
    st.subheader(f"Rolling Performance (window = {rolling_window} trades)")

    if rolling is not None:
        st.plotly_chart(make_rolling_chart(rolling, sa), use_container_width=True, key="rolling")
    else:
        st.warning(f"Not enough trades ({sa.n_trades}) for rolling window of {rolling_window}.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6: TIME ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[6]:
    time_data = sa.time_analysis()
    if time_data is None:
        st.info("No date/time column detected — time analysis unavailable.")
    else:
        for period_name, period_df in time_data.items():
            st.subheader(f"Performance by {period_name.replace('_', ' ').title()}")

            fig = go.Figure()
            colors = [COLORS["win"] if v > 0 else COLORS["loss"]
                      for v in period_df["total_profit"]]
            fig.add_trace(go.Bar(
                x=period_df.index, y=period_df["total_profit"],
                marker_color=colors, opacity=0.8,
            ))
            fig.update_layout(**PLOTLY_LAYOUT, title=f"Total P&L by {period_name}",
                              yaxis_title="Total Profit ($)")
            st.plotly_chart(fig, use_container_width=True, key=f"time_{period_name}")

            period_disp = period_df.copy()
            period_disp["total_profit"] = period_disp["total_profit"].map(lambda x: f"${x:,.2f}")
            period_disp["avg_profit"] = period_disp["avg_profit"].map(lambda x: f"${x:,.2f}")
            period_disp["win_rate"] = period_disp["win_rate"].map(lambda x: f"{x:.1%}")
            st.dataframe(period_disp, use_container_width=True)
            st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7: POSITION SIZING
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[7]:
    ps_data = sa.position_size_analysis()
    if ps_data is None:
        st.info("No position size / contracts column detected.")
    else:
        st.subheader("Position Size Analysis")
        st.markdown(f"""
        **Unique sizes:** {ps_data['unique_sizes']} |
        **Range:** {ps_data['min_size']:.2f} – {ps_data['max_size']:.2f} |
        **Average:** {ps_data['avg_size']:.2f} |
        **Size ↔ Profit correlation:** {ps_data['correlation']:.3f}
        """)

        if ps_data["by_bucket"] is not None:
            bucket_disp = ps_data["by_bucket"].copy()
            bucket_disp["total_profit"] = bucket_disp["total_profit"].map(lambda x: f"${x:,.2f}")
            bucket_disp["avg_profit"] = bucket_disp["avg_profit"].map(lambda x: f"${x:,.2f}")
            bucket_disp["win_rate"] = bucket_disp["win_rate"].map(lambda x: f"{x:.1%}")
            bucket_disp["avg_per_contract"] = bucket_disp["avg_per_contract"].map(lambda x: f"${x:,.2f}")
            st.dataframe(bucket_disp, use_container_width=True)
        else:
            st.caption("All trades use the same position size — bucket analysis not applicable.")

        if contracts is not None:
            fig_ps = go.Figure()
            fig_ps.add_trace(go.Scatter(
                x=contracts, y=profits, mode="markers",
                marker=dict(
                    color=np.where(profits > 0, COLORS["win"], COLORS["loss"]),
                    size=5, opacity=0.6,
                ),
            ))
            fig_ps.update_layout(**PLOTLY_LAYOUT,
                                 title="Position Size vs P&L",
                                 xaxis_title="Contracts / Qty",
                                 yaxis_title="Profit ($)")
            st.plotly_chart(fig_ps, use_container_width=True, key="pos_size_scatter")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 8: FULL REPORT
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[8]:
    st.subheader("Complete Metrics Report")

    summary = sa.summary()
    summary_df = pd.DataFrame(summary, index=["Value"]).T
    summary_df.index.name = "Metric"
    st.dataframe(summary_df, use_container_width=True, height=1400)

    # Download CSV (summary only)
    csv_buf = io.StringIO()
    summary_df.to_csv(csv_buf)
    st.download_button(
        label="📥 Download Summary Metrics (CSV)",
        data=csv_buf.getvalue(),
        file_name="just_trades_analytics_report.csv",
        mime="text/csv",
    )

    # Download full Excel with all temporal breakdowns
    st.divider()
    st.subheader("Full Report with Temporal Breakdowns")
    st.caption("Includes performance by hour, half-hour, day of week, week, month, and day+hour cross-analysis")

    xlsx_buf = io.BytesIO()
    with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
        # Sheet 1: Summary metrics
        summary_df.to_excel(writer, sheet_name="Summary Metrics")

        # Sheet 2+: Time breakdowns
        time_export = sa.time_analysis()
        if time_export:
            sheet_names = {
                "hour": "By Hour",
                "half_hour": "By Half Hour",
                "day_of_week": "By Day of Week",
                "week": "By Week",
                "month": "By Month",
                "day_x_hour": "By Day x Hour",
            }
            for key, sheet_name in sheet_names.items():
                if key in time_export:
                    tdf = time_export[key].copy()
                    # Format for readability
                    tdf.index.name = key.replace("_", " ").title()
                    for col in ["total_profit", "avg_profit", "largest_win", "largest_loss"]:
                        if col in tdf.columns:
                            tdf[col] = tdf[col].round(2)
                    if "win_rate" in tdf.columns:
                        tdf["win_rate"] = (tdf["win_rate"] * 100).round(1)
                        tdf = tdf.rename(columns={"win_rate": "win_rate_%"})
                    if "profit_factor" in tdf.columns:
                        tdf["profit_factor"] = tdf["profit_factor"].round(3)
                    tdf = tdf.rename(columns={
                        "total_profit": "total_profit_$",
                        "avg_profit": "avg_profit_$",
                        "largest_win": "largest_win_$",
                        "largest_loss": "largest_loss_$",
                    })
                    tdf.to_excel(writer, sheet_name=sheet_name)

    st.download_button(
        label="📥 Download Full Report with Time Analysis (XLSX)",
        data=xlsx_buf.getvalue(),
        file_name="just_trades_full_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # Commission analysis
    if commissions is not None:
        st.divider()
        st.subheader("Commission Impact")
        total_comm = commissions.sum()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Commissions", f"${total_comm:,.2f}")
        c2.metric("Avg Commission/Trade", f"${commissions.mean():,.2f}")
        c3.metric("Commission % of Gross",
                   f"{total_comm / (sa.total_profit + total_comm) * 100:.2f}%"
                   if abs(sa.total_profit + total_comm) > 0 else "N/A")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 9: STRATEGY GRADE CARD + AI ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[9]:
    st.subheader("🏆 Just Trades Strategy Grade Card")
    st.caption("Composite scoring across profitability, risk management, robustness, and statistical validity")

    def _score(value, thresholds, reverse=False):
        """Score 0-100 based on thresholds. thresholds = [(cutoff, score), ...]"""
        if not np.isfinite(value):
            return 100 if value > 0 else 0
        if reverse:
            for cutoff, sc in thresholds:
                if value >= cutoff:
                    return sc
            return thresholds[-1][1]
        for cutoff, sc in thresholds:
            if value >= cutoff:
                return sc
        return 0

    # Profitability Score (0-100)
    pf_score = _score(sa.profit_factor, [(3.0, 100), (2.0, 85), (1.5, 70), (1.2, 50), (1.0, 30), (0.8, 10)])
    exp_score = _score(sa.expectancy, [(100, 100), (50, 85), (20, 70), (5, 50), (0, 20)])
    wr_score = _score(sa.win_rate, [(0.7, 100), (0.6, 85), (0.5, 70), (0.4, 50), (0.3, 30)])
    ret_score = _score(sa.total_return_pct, [(200, 100), (100, 85), (50, 70), (20, 50), (5, 30)])
    profitability = int(np.mean([pf_score, exp_score, wr_score, ret_score]))

    # Risk Score (0-100, higher = better risk management)
    dd_score = _score(abs(sa.max_drawdown_pct), [(50, 10), (30, 30), (20, 50), (10, 80), (5, 100)], reverse=True)
    sharpe_score = _score(sa.sharpe_ratio, [(3, 100), (2, 85), (1.5, 70), (1, 50), (0.5, 30)])
    sortino_score = _score(sa.sortino_ratio, [(4, 100), (3, 85), (2, 70), (1, 50), (0.5, 30)])
    calmar_score = _score(sa.calmar_ratio, [(3, 100), (2, 85), (1, 70), (0.5, 50), (0.2, 30)])
    risk = int(np.mean([dd_score, sharpe_score, sortino_score, calmar_score]))

    # Robustness Score (0-100)
    mc_profit_score = _score(mc["prob_profit"], [(95, 100), (85, 85), (75, 70), (60, 50), (50, 30)])
    recovery_score = _score(sa.recovery_factor, [(10, 100), (5, 85), (3, 70), (1.5, 50), (1, 30)])
    streak_ratio = sa.max_consecutive_losses / max(sa.max_consecutive_wins, 1)
    streak_score = _score(streak_ratio, [(3, 10), (2, 30), (1.5, 50), (1, 70), (0.5, 100)], reverse=True)
    robustness = int(np.mean([mc_profit_score, recovery_score, streak_score]))

    # Statistical Score (0-100)
    p_score = _score(sa.p_value_profit, [(0.1, 10), (0.05, 50), (0.01, 80), (0.001, 100)], reverse=True)
    sqn_score = _score(sa.sqn, [(7, 100), (5, 85), (3, 70), (2, 50), (1.5, 30)])
    n_score = _score(sa.n_trades, [(500, 100), (200, 85), (100, 70), (50, 50), (30, 30)])
    statistical = int(np.mean([p_score, sqn_score, n_score]))

    # Overall composite
    overall = int(profitability * 0.30 + risk * 0.25 + robustness * 0.25 + statistical * 0.20)

    def _grade_label(score):
        if score >= 90: return "A+", "#22c55e"
        if score >= 80: return "A",  "#34d399"
        if score >= 70: return "B+", "#86efac"
        if score >= 60: return "B",  "#fbbf24"
        if score >= 50: return "C+", "#f59e0b"
        if score >= 40: return "C",  "#f97316"
        if score >= 30: return "D",  "#ef4444"
        return "F", "#dc2626"

    grade, color = _grade_label(overall)

    st.markdown(f"""
    <div style="text-align:center; padding: 30px 0;">
        <div style="font-size: 5rem; font-weight: 800; color: {color};
                    font-family: 'JetBrains Mono', monospace; line-height: 1;">{grade}</div>
        <div style="font-size: 1.2rem; opacity: 0.7; margin-top: 8px;">
            Overall Score: {overall}/100
        </div>
    </div>
    """, unsafe_allow_html=True)

    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Profitability", f"{profitability}/100")
    g2.metric("Risk Management", f"{risk}/100")
    g3.metric("Robustness", f"{robustness}/100")
    g4.metric("Statistical Validity", f"{statistical}/100")

    st.divider()

    # ── Radar chart ───────────────────────────────────────────────────────
    radar_fig = go.Figure()
    cats = ["Profitability", "Risk Mgmt", "Robustness", "Statistics"]
    vals = [profitability, risk, robustness, statistical]
    radar_fig.add_trace(go.Scatterpolar(
        r=vals + [vals[0]], theta=cats + [cats[0]],
        fill="toself", fillcolor="rgba(99,102,241,0.2)",
        line=dict(color=COLORS["equity"], width=2),
        name="Strategy",
    ))
    radar_fig.update_layout(
        **PLOTLY_LAYOUT,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100],
                            gridcolor="rgba(255,255,255,0.08)"),
            bgcolor="rgba(0,0,0,0)",
            angularaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        ),
        showlegend=False, height=350,
    )
    st.plotly_chart(radar_fig, use_container_width=True, key="grade_radar")

    # Breakdown details
    with st.expander("📐 Score Breakdown Details", expanded=True):
        breakdown = pd.DataFrame({
            "Category": [
                "Profitability", "", "", "",
                "Risk Management", "", "", "",
                "Robustness", "", "",
                "Statistical Validity", "", "",
            ],
            "Metric": [
                "Profit Factor", "Expectancy ($)", "Win Rate", "Total Return %",
                "Max Drawdown %", "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio",
                "MC P(Profit)", "Recovery Factor", "Loss/Win Streak Ratio",
                "p-Value", "SQN", "Sample Size (N)",
            ],
            "Value": [
                _fmt(sa.profit_factor, ".3f"), f"${sa.expectancy:,.2f}",
                f"{sa.win_rate:.1%}", f"{sa.total_return_pct:.1f}%",
                f"{abs(sa.max_drawdown_pct):.2f}%", f"{sa.sharpe_ratio:.3f}",
                _fmt(sa.sortino_ratio, ".3f"), _fmt(sa.calmar_ratio, ".3f"),
                f"{mc['prob_profit']:.1f}%", _fmt(sa.recovery_factor, ".2f"),
                f"{streak_ratio:.2f}",
                f"{sa.p_value_profit:.6f}", f"{sa.sqn:.2f}", f"{sa.n_trades}",
            ],
            "Score": [
                pf_score, exp_score, wr_score, ret_score,
                dd_score, sharpe_score, sortino_score, calmar_score,
                mc_profit_score, recovery_score, streak_score,
                p_score, sqn_score, n_score,
            ],
        })
        st.dataframe(breakdown, use_container_width=True, hide_index=True)

    st.divider()

    # ── Warnings & recommendations ────────────────────────────────────────
    st.subheader("⚠️ Alerts & Recommendations")
    alerts = []
    if sa.n_trades < 50:
        alerts.append(("🔴", "Very small sample size", "Fewer than 50 trades makes all statistics unreliable. Run the strategy longer or on more instruments before drawing conclusions."))
    elif sa.n_trades < 100:
        alerts.append(("🟡", "Small sample size", "Under 100 trades. Results may shift significantly with more data. Consider extending the backtest period."))
    if sa.p_value_profit > 0.05:
        alerts.append(("🔴", "No statistical edge detected", f"p-value is {sa.p_value_profit:.4f} — cannot reject the hypothesis that mean profit is zero. The apparent profit may be random noise."))
    if abs(sa.max_drawdown_pct) > 30:
        alerts.append(("🟠", "Severe max drawdown", f"Max DD is {sa.max_drawdown_pct:.1f}%. Consider reducing position sizes by {int(abs(sa.max_drawdown_pct)/15)}x or adding tighter stop losses."))
    elif abs(sa.max_drawdown_pct) > 15:
        alerts.append(("🟡", "Moderate drawdown", f"Max DD is {sa.max_drawdown_pct:.1f}%. Acceptable but monitor closely during live trading."))
    if sa.profit_factor < 1.0:
        alerts.append(("🔴", "Negative expectancy", "Profit factor below 1.0 means the strategy loses money. Do not trade this live."))
    elif sa.profit_factor < 1.2:
        alerts.append(("🟡", "Marginal profit factor", "Very thin edge at {:.2f}. Slippage and commissions in live trading may erase profitability entirely.".format(sa.profit_factor)))
    if sa.skewness < -1:
        alerts.append(("🟠", "Heavy negative skew", f"Skewness is {sa.skewness:.2f}. The strategy has a fat left tail — rare large losses can wipe out many small wins. Consider adding max-loss circuit breakers."))
    if sa.excess_kurtosis > 3:
        alerts.append(("🟠", "Extreme kurtosis", f"Excess kurtosis is {sa.excess_kurtosis:.2f}. Expect more extreme outcomes (both directions) than normal. VaR estimates will understate true risk."))
    if sa.kelly_criterion > 0.5:
        alerts.append(("🟡", "High Kelly fraction", f"Full Kelly is {sa.kelly_criterion:.3f}. This is extremely aggressive — use ¼ Kelly ({sa.kelly_criterion/4:.3f}) in practice to survive bad runs."))
    if sa.kelly_criterion < 0:
        alerts.append(("🔴", "Negative Kelly", "The Kelly criterion is negative — mathematically recommends not trading this strategy at any size."))
    if mc["prob_profit"] < 60:
        alerts.append(("🔴", "Low Monte Carlo confidence", f"Only {mc['prob_profit']:.1f}% of {mc_sims} simulations ended profitable. The edge is fragile."))
    if sa.max_consecutive_losses >= 10:
        alerts.append(("🟠", "Long losing streak", f"{sa.max_consecutive_losses} consecutive losses observed. At your position size, this equals ~${sa.avg_loss * sa.max_consecutive_losses:,.0f} peak pain. Ensure you can withstand this psychologically."))
    if commissions is not None:
        comm_pct = commissions.sum() / (abs(sa.total_profit) + commissions.sum()) * 100 if (abs(sa.total_profit) + commissions.sum()) > 0 else 0
        if comm_pct > 30:
            alerts.append(("🟠", "High commission drag", f"Commissions consume {comm_pct:.0f}% of gross profits. Consider reducing trade frequency or negotiating lower fees."))
    if sa.win_rate > 0.75 and sa.payoff_ratio < 0.5:
        alerts.append(("🟡", "High WR / low payoff", "You win often but wins are much smaller than losses. One bad streak could erase weeks of gains. Tighten stops or let winners run longer."))
    if sa.win_rate < 0.35 and sa.payoff_ratio < 2.0:
        alerts.append(("🔴", "Low WR + low payoff", "Both win rate and payoff ratio are poor. The strategy needs either more frequent wins or bigger wins per trade."))

    # Signal-level alerts
    if signals is not None:
        sig_df_alert = pd.DataFrame({"signal": signals, "profit": profits})
        sig_perf = sig_df_alert.groupby("signal")["profit"].agg(["sum", "count", "mean"])
        bad_signals = sig_perf[sig_perf["sum"] < 0]
        if len(bad_signals) > 0:
            sig_names = ", ".join(bad_signals.index.tolist())
            alerts.append(("🟡", "Losing signals detected", f"These signals have negative total P&L: **{sig_names}**. Consider removing or re-tuning them."))

    if not alerts:
        st.success("✅ No major alerts. Strategy looks solid across all dimensions.")
    else:
        for icon, title, detail in alerts:
            st.markdown(f"{icon} **{title}** — {detail}")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # AI ANALYSIS PROMPT BUILDER
    # ══════════════════════════════════════════════════════════════════════
    st.subheader("🤖 AI Strategy Analysis")
    st.caption("AI-powered strategy analysis — uses Gemini to give you a professional hedge fund–grade report")

    # Build the comprehensive prompt
    alert_text = "\n".join([f"- [{icon}] {title}: {detail}" for icon, title, detail in alerts]) if alerts else "No alerts — strategy looks clean."

    # Signal breakdown for prompt
    signal_breakdown = ""
    if signals is not None:
        sig_df_prompt = pd.DataFrame({"signal": signals, "profit": profits})
        sig_stats = sig_df_prompt.groupby("signal")["profit"].agg(["count", "sum", "mean"])
        sig_stats.columns = ["trades", "total_pnl", "avg_pnl"]
        sig_lines = []
        for sig_name, row in sig_stats.iterrows():
            wr = (sig_df_prompt[sig_df_prompt["signal"] == sig_name]["profit"] > 0).mean()
            sig_lines.append(f"  {sig_name}: {int(row['trades'])} trades, ${row['total_pnl']:,.2f} total, ${row['avg_pnl']:,.2f} avg, {wr:.0%} WR")
        signal_breakdown = "\n".join(sig_lines)

    # Direction breakdown
    direction_breakdown = ""
    if trade_types is not None:
        dir_df = pd.DataFrame({"type": trade_types, "profit": profits})
        dir_df["type"] = dir_df["type"].str.strip().str.lower()
        for d in dir_df["type"].unique():
            subset = dir_df[dir_df["type"] == d]["profit"]
            wr = (subset > 0).mean()
            direction_breakdown += f"  {d.title()}: {len(subset)} trades, ${subset.sum():,.2f} total, {wr:.0%} WR\n"

    ai_prompt = f"""You are a senior quantitative analyst at a hedge fund. I'm sharing my complete backtest results for a trading strategy. Analyze every metric critically and give me:

1. **OVERALL ASSESSMENT** — Is this strategy viable for live trading? Grade it honestly.
2. **STRENGTHS** — What's working well? Which metrics stand out positively?
3. **WEAKNESSES** — What are the biggest risks? Where could this blow up?
4. **SPECIFIC IMPROVEMENTS** — Concrete, actionable suggestions to improve the strategy. Be specific about what parameters to adjust.
5. **POSITION SIZING RECOMMENDATION** — Based on Kelly, drawdown, and risk metrics, what position size should I use?
6. **LIVE TRADING READINESS** — What additional testing/validation do I need before going live?

═══ STRATEGY METRICS ═══

Overview:
  Total Trades: {sa.n_trades}
  Date Range: {dates.iloc[0] if dates is not None else 'N/A'} → {dates.iloc[-1] if dates is not None else 'N/A'}
  Initial Capital: ${initial_capital:,}
  Net Profit: ${sa.total_profit:,.2f}
  Total Return: {sa.total_return_pct:.2f}%
  CAGR: {sa.cagr:.2f}%

Win/Loss Profile:
  Win Rate: {sa.win_rate:.1%} | Loss Rate: {sa.loss_rate:.1%}
  Avg Win: ${sa.avg_win:,.2f} | Avg Loss: -${sa.avg_loss:,.2f}
  Largest Win: ${sa.largest_win:,.2f} | Largest Loss: ${sa.largest_loss:,.2f}
  Median Win: ${sa.median_win:,.2f} | Median Loss: ${sa.median_loss:,.2f}
  Profit Factor: {_fmt(sa.profit_factor, '.3f')}
  Payoff Ratio: {_fmt(sa.payoff_ratio, '.3f')}
  Expectancy: ${sa.expectancy:,.2f} per trade

Risk-Adjusted Returns:
  Sharpe Ratio: {sa.sharpe_ratio:.3f}
  Sortino Ratio: {_fmt(sa.sortino_ratio, '.3f')}
  Calmar Ratio: {_fmt(sa.calmar_ratio, '.3f')}
  Omega Ratio: {_fmt(sa.omega_ratio, '.3f')}
  Ulcer Index: {sa.ulcer_index:.3f}

Drawdown:
  Max Drawdown: ${sa.max_drawdown:,.2f} ({sa.max_drawdown_pct:.2f}%)
  Avg Drawdown: ${sa.avg_drawdown:,.2f} ({sa.avg_drawdown_pct:.2f}%)
  Max DD Duration: {sa.max_drawdown_duration} trades
  Recovery Factor: {_fmt(sa.recovery_factor, '.2f')}

Statistical Validity:
  t-Statistic: {sa.t_stat_profit:.3f}
  p-Value: {sa.p_value_profit:.6f}
  SQN (System Quality Number): {sa.sqn:.2f}
  Skewness: {sa.skewness:.3f}
  Excess Kurtosis: {sa.excess_kurtosis:.3f}

Position Sizing:
  Kelly Criterion: {sa.kelly_criterion:.4f}
  Half Kelly: {sa.half_kelly:.4f}

Streaks:
  Max Consecutive Wins: {sa.max_consecutive_wins}
  Max Consecutive Losses: {sa.max_consecutive_losses}
  Avg Consecutive Wins: {sa.avg_consecutive_wins:.1f}
  Avg Consecutive Losses: {sa.avg_consecutive_losses:.1f}

Monte Carlo ({mc_sims:,} simulations):
  P(Profitable): {mc['prob_profit']:.1f}%
  P(2x Capital): {mc['prob_2x']:.1f}%
  P(Lose 50%+): {mc['prob_loss50']:.1f}%
  P(Ruin): {mc['prob_ruin']:.2f}%
  Median Final Equity: ${mc['percentiles']['p50']:,.2f}
  5th Percentile: ${mc['percentiles']['p5']:,.2f}
  95th Percentile: ${mc['percentiles']['p95']:,.2f}

Strategy Grade: {grade} ({overall}/100)
  Profitability: {profitability}/100
  Risk Management: {risk}/100
  Robustness: {robustness}/100
  Statistical Validity: {statistical}/100

{f"Signal Performance:" + chr(10) + signal_breakdown if signal_breakdown else ""}
{f"Direction Performance:" + chr(10) + direction_breakdown if direction_breakdown else ""}

Alerts Detected:
{alert_text}

Based on all of the above, give me your complete professional analysis. Be brutally honest — I want to know if this is ready for real money or not, and exactly what I need to fix."""

    # Display prompt and run Claude analysis
    with st.expander("📋 View Full Metrics Prompt", expanded=False):
        st.code(ai_prompt, language="text")

    if st.button("🤖 Run AI Analysis (Claude)", key="run_claude_grade_analysis", type="primary"):
        with st.spinner("Claude is analyzing your strategy... (15-30s)"):
            _grade_reply = call_ai_analyst(ai_prompt, '_ai_grade_chat')
            if _grade_reply:
                st.markdown("---")
                st.markdown("### 🤖 Claude Strategy Analysis")
                st.markdown(_grade_reply)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # TRADE P&L HEATMAP (Calendar Style)
    # ══════════════════════════════════════════════════════════════════════
    st.subheader("🗓️ P&L Calendar Heatmap")

    if dates is not None:
        heatmap_df = pd.DataFrame({"date": pd.to_datetime(dates, errors="coerce"), "profit": profits})
        heatmap_df = heatmap_df.dropna(subset=["date"])
        if not heatmap_df.empty:
            heatmap_df["week"] = heatmap_df["date"].dt.isocalendar().week.astype(int)
            heatmap_df["dow"] = heatmap_df["date"].dt.dayofweek
            hm_pivot = heatmap_df.groupby(["week", "dow"])["profit"].sum().reset_index()
            hm_pivot = hm_pivot.pivot(index="dow", columns="week", values="profit").fillna(0)

            dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            hm_fig = go.Figure(data=go.Heatmap(
                z=hm_pivot.values,
                x=[f"W{c}" for c in hm_pivot.columns],
                y=[dow_labels[i] for i in hm_pivot.index],
                colorscale=[
                    [0, "#ef4444"],
                    [0.35, "#fca5a5"],
                    [0.5, "#1e1e2e"],
                    [0.65, "#86efac"],
                    [1, "#22c55e"],
                ],
                zmid=0,
                text=np.round(hm_pivot.values, 0),
                texttemplate="$%{text:.0f}",
                textfont=dict(size=9),
                hovertemplate="Week %{x}<br>%{y}<br>P&L: $%{z:,.2f}<extra></extra>",
                colorbar=dict(title="P&L ($)", thickness=15),
            ))
            hm_fig.update_layout(
                **PLOTLY_LAYOUT,
                title="Weekly P&L Heatmap",
                height=280,
            )
            hm_fig.update_yaxes(autorange="reversed")
            st.plotly_chart(hm_fig, use_container_width=True, key="pnl_heatmap")
    else:
        st.info("No date column detected — heatmap unavailable.")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # WIN / LOSS DISTRIBUTION COMPARISON
    # ══════════════════════════════════════════════════════════════════════
    st.subheader("📊 Win vs Loss Distribution Overlay")

    wl_fig = go.Figure()
    if len(sa.wins) > 1:
        wl_fig.add_trace(go.Histogram(
            x=sa.wins, nbinsx=30, name="Wins",
            marker_color=COLORS["win"], opacity=0.6,
        ))
    if len(sa.losses) > 1:
        wl_fig.add_trace(go.Histogram(
            x=np.abs(sa.losses), nbinsx=30, name="Losses (abs)",
            marker_color=COLORS["loss"], opacity=0.6,
        ))
    wl_fig.update_layout(
        **PLOTLY_LAYOUT, barmode="overlay",
        title="Win Size vs Loss Size Distribution",
        xaxis_title="Trade Magnitude ($)", yaxis_title="Frequency",
    )
    st.plotly_chart(wl_fig, use_container_width=True, key="win_loss_overlay")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # WHAT-IF SCENARIOS
    # ══════════════════════════════════════════════════════════════════════
    st.subheader("🔧 What-If Position Size Simulator")
    st.caption("See how different position sizes would have changed your results")

    whatif_multiplier = st.slider(
        "Position Size Multiplier",
        min_value=0.25, max_value=3.0, value=1.0, step=0.25,
        key="whatif_slider",
    )

    if whatif_multiplier != 1.0:
        wi_profits = sa.raw_profits * whatif_multiplier
        wi_eq = np.cumsum(wi_profits) + initial_capital
        wi_peak = np.maximum.accumulate(wi_eq)
        wi_dd = wi_eq - wi_peak
        wi_dd_pct = np.where(wi_peak > 0, wi_dd / wi_peak, 0).min() * 100
        wi_total = wi_profits.sum()
        wi_wins = wi_profits[wi_profits > 0]
        wi_losses = wi_profits[wi_profits < 0]
        wi_pf = wi_wins.sum() / abs(wi_losses.sum()) if len(wi_losses) > 0 else np.inf

        wc1, wc2, wc3, wc4 = st.columns(4)
        wc1.metric("Net Profit", f"${wi_total:,.2f}",
                    delta=f"{(wi_total - sa.total_profit):+,.2f} vs actual")
        wc2.metric("Max Drawdown %", f"{wi_dd_pct:.2f}%",
                    delta=f"{(wi_dd_pct - sa.max_drawdown_pct):+.2f}%")
        wc3.metric("Profit Factor", _fmt(wi_pf, ".2f"))
        wc4.metric("Final Equity", f"${wi_eq[-1]:,.2f}")

        wi_fig = go.Figure()
        wi_fig.add_trace(go.Scatter(x=list(range(sa.n_trades)), y=sa.equity,
                                     mode="lines", name="Actual",
                                     line=dict(color=COLORS["equity"], width=2)))
        wi_fig.add_trace(go.Scatter(x=list(range(sa.n_trades)), y=wi_eq,
                                     mode="lines", name=f"{whatif_multiplier}x Size",
                                     line=dict(color=COLORS["accent"], width=2, dash="dash")))
        wi_fig.update_layout(**PLOTLY_LAYOUT, title="Equity: Actual vs What-If",
                             yaxis_title="Equity ($)", hovermode="x unified")
        st.plotly_chart(wi_fig, use_container_width=True, key="whatif_chart")
    else:
        st.info("Move the slider to simulate different position sizes.")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════
    # TRADE EFFICIENCY SCORE
    # ══════════════════════════════════════════════════════════════════════
    if runups is not None:
        st.subheader("⚡ Trade Efficiency")
        st.caption("How much of each trade's maximum potential (MFE) did you actually capture?")

        # Efficiency = profit / runup for winners, 0 for losers
        safe_runups = np.where(runups > 0, runups, np.nan)
        efficiency = np.where(profits > 0, profits / safe_runups, 0)
        efficiency = np.nan_to_num(efficiency, nan=0)
        efficiency = np.clip(efficiency, 0, 1)

        avg_eff = efficiency[profits > 0].mean() * 100 if (profits > 0).any() else 0

        ec1, ec2, ec3 = st.columns(3)
        ec1.metric("Avg Win Efficiency", f"{avg_eff:.1f}%")
        ec2.metric("Best Capture", f"{efficiency.max() * 100:.1f}%")
        ec3.metric("Trades >80% Efficient", f"{(efficiency > 0.8).sum()}")

        eff_fig = go.Figure()
        eff_fig.add_trace(go.Histogram(
            x=efficiency[profits > 0] * 100, nbinsx=20,
            marker_color=COLORS["equity"], opacity=0.7,
            name="Win Efficiency %",
        ))
        eff_fig.update_layout(**PLOTLY_LAYOUT, title="Winning Trade Efficiency Distribution",
                              xaxis_title="% of MFE Captured", yaxis_title="Count")
        st.plotly_chart(eff_fig, use_container_width=True, key="efficiency_hist")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 10: AI QUANT ANALYST — Deep Analysis Presets
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[10]:
    st.subheader("🤖 AI Quant Analyst")
    st.caption("Institutional-grade strategy analysis powered by Claude. All 70+ backtest metrics + Monte Carlo + equity curve fed to an AI quant for deep risk review.")

    if True:
        # ── Analysis presets ─────────────────────────────────────────────
        _ai_mode = st.radio(
            "Analysis Mode",
            ["🛠️ Improve Strategy", "🔍 Full Audit + Fix", "💰 Deploy Plan ($1B)",
             "🧬 Reverse Engineer", "⚠️ Stress Test + Fixes", "💬 Ask Anything"],
            horizontal=True,
            key="_ai_analysis_mode"
        )

        _ai_preset_prompts = {
            "🛠️ Improve Strategy": """You are my quant engineer. Analyze this strategy's data and give me a COMPLETE IMPROVEMENT PLAN. Don't just tell me what's wrong — tell me exactly how to fix every weakness.

For each issue you find, use this format:
**PROBLEM:** [what's wrong, with the specific metric]
**WHY IT MATTERS:** [consequence if unfixed]
**FIX:** [exact change — specific parameter values, logic changes, or additions]
**EXPECTED RESULT:** [what the improved metric should look like]

Cover ALL of these areas:

1. **Entry Logic Improvements** — Based on the win rate, trade frequency, and PnL distribution, is the entry too aggressive or too conservative? What filter would eliminate the worst 10-20% of trades? Look at the trade sample for patterns.

2. **Exit Logic Improvements** — Analyze TP vs SL balance. The payoff ratio and avg win vs avg loss tell the story. Give me EXACT tick values for TP and SL adjustments. If TP is too tight, say "widen TP from X ticks to Y ticks." If SL is too wide, say "tighten SL from X to Y."

3. **Risk Management Additions** — Based on max consecutive losses, max drawdown, and kurtosis, design a complete risk overlay:
   - Max daily loss limit (exact dollar amount)
   - Max consecutive loss pause (how many losses, how long to pause)
   - Drawdown circuit breaker (at what % to halt)
   - Position size reduction rules during drawdowns

4. **Regime Filter** — Based on the equity curve checkpoints, identify WHERE the strategy struggles. Propose a specific regime filter (ADX threshold, VIX level, volatility percentile) that would have avoided the worst drawdown periods.

5. **Position Sizing Optimization** — Given the Kelly criterion, Sharpe, and max DD, calculate:
   - Optimal fixed fractional size
   - Vol-adjusted position sizing formula
   - Exact contract counts for different account sizes (USD 50K, USD 100K, USD 500K, USD 1M, USD 10M)

6. **Parameter Tuning Priorities** — Rank which parameters matter most for improving the strategy. Which ones should I optimize first? Which ones should I leave alone?

7. **Parameter Tuning Priorities** — Using the parsed parameters table, rank which ones matter most. For each, give: current value, recommended value, and optimizer range (min, max, step) to test.

8. **PINE CODE CHANGES** — For EVERY fix above, write the EXACT Pine Script code modification. Show the line number, the current code, and the replacement code. If adding new logic (circuit breakers, regime filters, trailing stops), write the COMPLETE code block I can paste directly into my strategy. Mark where each block goes (e.g., "Insert after line 68"). Match my existing code style.

9. **Action Plan** — Give me a numbered checklist of changes to make, in priority order. Each item should be specific enough that I can implement it immediately.

Be specific. Be actionable. Every suggestion must have a number AND code attached to it. If you don't have the Pine source code, tell me to re-run the backtest so the code gets loaded.""",

            "🔍 Full Audit + Fix": """Perform a comprehensive audit AND improvement plan. For every weakness, provide the fix.

**PART 1: DIAGNOSIS** (be brutally honest)
1. Executive Summary — deployable YES/NO/CONDITIONAL
2. Edge Analysis — is the statistical edge real or noise? (cite p-value, SQN, profit factor)
3. Risk Profile — max DD, tail risk, worst-case at USD 1B allocation
4. Monte Carlo Reality Check — are the MC results trustworthy given the trade distribution?
5. Sample Size Assessment — enough data to be confident?
6. Equity Curve Health — regime dependency, flat periods, concerning patterns

**PART 2: PRESCRIPTION** (fix every weakness you found)
For EACH problem identified in Part 1:
- **Current:** [metric value]
- **Target:** [what it should be]
- **Fix:** [specific parameter change, logic addition, or structural modification]
- **Priority:** CRITICAL / HIGH / MEDIUM / LOW

**PART 3: PINE CODE FIXES**
For EVERY prescription in Part 2, write the exact Pine Script code change. Show line numbers, before/after code, and where to insert new blocks. Write complete, copy-pasteable code. If adding a new section (e.g., circuit breaker, regime filter), write the entire block with comments.

**PART 4: IMPLEMENTATION ROADMAP**
Numbered list of changes in priority order, with expected impact of each change.

**PART 5: MONITORING PLAN**
What to watch after implementing changes. Specific metrics, thresholds, and review schedule.""",

            "💰 Deploy Plan ($1B)": """I'm deploying this strategy on a USD 1B account. Build me a complete deployment plan that MAXIMIZES safety while capturing the edge.

1. **GO/NO-GO** — Confidence level 0-100%. If conditional, state EXACT conditions.

2. **Phased Rollout Plan:**
   - Phase 0: What to fix BEFORE any live capital (specific changes)
   - Phase 1: Paper trading — duration, success criteria, metrics to track
   - Phase 2: Pilot (micro allocation) — exact USD amount, contract count, duration
   - Phase 3: Scale-up — criteria to increase allocation, step sizes, timeline
   - Phase 4: Full allocation — target % of AUM, conditions to maintain

3. **Risk Architecture** — Design the complete risk overlay:
   - Per-trade max loss (USD and % of allocated capital)
   - Daily max loss with auto-halt
   - Weekly max loss
   - Max drawdown circuit breaker (% and USD)
   - Max consecutive losses → pause duration
   - Correlation limit with existing strategies
   - Vol regime throttle (reduce size when VIX > X)

4. **Position Sizing Matrix:**
   | Account Size | Contracts (NQ) | Contracts (MNQ) | Max Risk/Trade |
   Give me this for USD 100K, 500K, 1M, 5M, 10M, 50M, 100M allocations.

5. **Slippage Budget** — Expected degradation. Recalculate expectancy at 2x and 3x slippage.

6. **Week 1 Playbook** — Exactly what to do on Monday, Tuesday, etc. Daily checklist.

7. **Emergency Procedures** — If max DD hits X, do Y. If consecutive losses hit N, do Z.""",

            "🧬 Reverse Engineer": """Reverse-engineer this strategy from the trade data and suggest structural improvements.

1. **Strategy DNA** — Based on the trade sample, PnL distribution, and metrics, deduce:
   - What type of strategy is this? (mean reversion, momentum, DCA, scalping, etc.)
   - What's the likely entry logic? (support/resistance bounce, breakout, indicator cross?)
   - What's the exit logic? (fixed TP, trailing stop, time-based, indicator-based?)
   - Is this a DCA/averaging strategy? (look at position sizing patterns)

2. **The Repeating Pattern** — If you see repeating PnL values in the trade sample, explain what they mean. What TP level produces that PnL? What SL level produces the losses?

3. **Structural Weaknesses** — Based on the deduced strategy type:
   - What market conditions will break this strategy?
   - What's the theoretical max loss scenario?
   - Where is the strategy leaving money on the table?

4. **Architectural Improvements** — Suggest modifications to the strategy STRUCTURE (not just parameters):
   - Should it add a trend filter?
   - Should it switch between modes (range vs trend)?
   - Should it have adaptive TP/SL based on volatility?
   - Should it have time-of-day filters?
   - Should it scale in/out differently?

5. **Next-Generation Design** — If you were building this strategy from scratch with the same edge, what would you design differently? Give me a blueprint.""",

            "⚠️ Stress Test + Fixes": """Stress test this strategy AND for each failure point, give me the fix.

For each scenario, use this format:
**SCENARIO:** [description]
**IMPACT:** [estimated loss in USD at current sizing]
**FIX:** [specific change to survive this scenario]

Scenarios to test:

1. **3-sigma event** — Given kurtosis and largest loss, what does a 3-sigma day look like? What about 5-sigma? Calculate USD loss at various allocations.
   → FIX: What stop-loss or circuit breaker survives this?

2. **Double consecutive losses** — If max consecutive losses doubles, what's the drawdown?
   → FIX: What pause rule prevents this from becoming catastrophic?

3. **Slippage degradation** — At 2x, 3x, 5x slippage, recalculate profit factor and expectancy. At what slippage does the edge disappear?
   → FIX: What minimum tick profit makes the strategy robust to slippage?

4. **March 2020 scenario** — VIX at 80, gaps, limit downs. How does this strategy behave?
   → FIX: What VIX filter or vol throttle protects against this?

5. **Liquidity vacuum** — Spreads widen 3x, partial fills at 50%.
   → FIX: What position size cap prevents this from mattering?

6. **Regime change** — Strategy enters a prolonged trending market (6+ months of one-direction move).
   → FIX: What regime detector pauses the strategy?

7. **COMPLETE RISK OVERLAY** — Based on all scenarios above, design the full risk system:
   - Daily max loss: USD ___
   - Weekly max loss: USD ___
   - Max DD halt: ___%
   - Consecutive loss pause: ___ losses → ___ bars
   - VIX filter: pause when VIX > ___
   - Position reduction: at ___% DD, cut size by ___% """,

            "💬 Ask Anything": None,
        }

        # ── Context display ──────────────────────────────────────────────
        with st.expander("📋 Strategy Data Being Sent to AI", expanded=False):
            st.code(build_strategy_context(), language="text")

        # ── Custom question input ────────────────────────────────────────
        _custom_q = ""
        if _ai_mode == "💬 Ask Anything":
            _custom_q = st.text_area(
                "Your question for the AI Quant Analyst:",
                height=100,
                placeholder="e.g., What's the optimal position size for NQ vs MNQ? Should I worry about the kurtosis? Compare this to typical CTA performance.",
                key="_ai_custom_question"
            )

        # ── Run + Clear ──────────────────────────────────────────────────
        _col_r1, _col_r2 = st.columns([1, 3])
        with _col_r1:
            _run_ai = st.button("🧠 Analyze Strategy", key="_run_ai_analysis", use_container_width=True)
        with _col_r2:
            if st.button("🗑️ Clear Chat", key="_clear_ai_tab_chat"):
                st.session_state['_ai_chat_history'] = []
                st.rerun()

        if _run_ai:
            _user_prompt = _ai_preset_prompts.get(_ai_mode)
            if _ai_mode == "💬 Ask Anything":
                if not _custom_q.strip():
                    st.warning("Please enter a question.")
                    _user_prompt = None
                else:
                    _user_prompt = _custom_q.strip()

            if _user_prompt:
                _ctx = build_strategy_context()
                _full_msg = f"Here is the complete backtest data for the strategy:\n\n```\n{_ctx}\n```\n\n{_user_prompt}"
                with st.spinner("🧠 AI Quant Analyst thinking... (30-90 seconds for detailed analysis)"):
                    _reply = call_ai_analyst(_full_msg, '_ai_chat_history')
                if _reply:
                    st.divider()
                    st.markdown(_reply)
                else:
                    st.error("No response received. Check the error details above.")

        # ── Display conversation ─────────────────────────────────────────
        if st.session_state.get('_ai_chat_history'):
            st.divider()
            for _msg in st.session_state['_ai_chat_history']:
                if _msg['role'] == 'user':
                    with st.expander("📤 Your Query", expanded=False):
                        st.markdown(_msg['content'][:500] + ("..." if len(_msg['content']) > 500 else ""))
                else:
                    st.markdown(_msg['content'])
                    st.divider()

            # ── Follow-up ────────────────────────────────────────────────
            _followup = st.text_input(
                "Ask a follow-up question:",
                placeholder="e.g., What if I reduce position size by 50%? How does this compare to SPX buy-and-hold?",
                key="_ai_followup_tab"
            )
            if st.button("📨 Send", key="_send_followup_tab"):
                if _followup.strip():
                    with st.spinner("🧠 Thinking..."):
                        _fu_reply = call_ai_analyst(_followup.strip(), '_ai_chat_history')
                    if _fu_reply:
                        st.divider()
                        st.markdown(_fu_reply)


    # ══════════════════════════════════════════════════════════════════════
    # "TEST AI FIXES" — Re-run backtest with AI-recommended params, show before/after
    # ══════════════════════════════════════════════════════════════════════
    _has_ast = '_pine_ast' in st.session_state and '_pine_cached_data' in st.session_state
    _last_ai_msg = None
    for _m in reversed(st.session_state.get('_ai_chat_history', [])):
        if _m['role'] == 'assistant':
            _last_ai_msg = _m['content']
            break

    if _last_ai_msg and _has_ast:
        _rec_params = _parse_ai_recommended_params(_last_ai_msg)
        if _rec_params:
            st.divider()
            st.subheader("🧪 Test AI Recommendations")
            st.caption("The AI suggested specific parameter changes. Run a side-by-side comparison to see the projected impact.")

            with st.expander("📋 Recommended Changes", expanded=True):
                _param_cols = st.columns(min(4, len(_rec_params)))
                for _i, (_pname, _pval) in enumerate(_rec_params.items()):
                    with _param_cols[_i % len(_param_cols)]:
                        # Find current value
                        _cur_val = "?"
                        for _idef in st.session_state.get('_pine_input_defs', []):
                            if _idef.get('title') == _pname or _idef.get('key') == _pname:
                                _cur_val = _idef.get('defval', '?')
                                break
                        st.metric(_pname, f"{_pval}", delta=f"was {_cur_val}")

            if st.button("🚀 Run Before vs After Comparison", type="primary", key="_test_ai_fixes", use_container_width=True):
                with st.status("Running backtest with AI-recommended parameters...", expanded=True) as _fix_status:
                    try:
                        st.write("⚙️ Applying AI parameter overrides...")
                        _fix_overrides = {}
                        for _pname, _pval in _rec_params.items():
                            _fix_overrides[_pname] = _pval

                        st.write(f"📊 Re-running on {len(st.session_state['_pine_cached_data']):,} bars...")

                        from pine_interpreter import PineInterpreter as _FixInterp
                        # Use effective settings from strategy() if available, else sidebar defaults
                        _eff_fix = st.session_state.get('_pine_effective_settings', {})
                        _pc = _eff_fix.get('commission_pct', pine_commission if 'pine_commission' in dir() else 0.0)
                        _pq = _eff_fix.get('default_qty', pine_qty if 'pine_qty' in dir() else 1)
                        _pp = _eff_fix.get('pyramiding', pine_pyramiding if 'pine_pyramiding' in dir() else 1)
                        _ps = _eff_fix.get('slippage', pine_slippage if 'pine_slippage' in dir() else 0)
                        _pm = _eff_fix.get('mintick', pine_mintick if 'pine_mintick' in dir() else 0.25)
                        _pml = pine_margin_long if 'pine_margin_long' in dir() else 0
                        _pms = pine_margin_short if 'pine_margin_short' in dir() else 0
                        _fix_cap = _eff_fix.get('initial_capital', initial_capital)
                        _fix_interp = _FixInterp(
                            st.session_state['_pine_ast'],
                            st.session_state['_pine_cached_data'],
                            initial_capital=_fix_cap,
                            commission_pct=_pc,
                            default_qty=_pq,
                            pyramiding=_pp,
                            slippage=_ps,
                            mintick=_pm,
                            margin_long=_pml,
                            margin_short=_pms,
                            input_overrides=_fix_overrides,
                        )
                        _fix_trades = _fix_interp.execute()
                        st.write(f"✅ Got **{len(_fix_trades)}** trades with AI params")

                        if len(_fix_trades) < 5:
                            st.warning("Too few trades with AI parameters. The recommended changes may be too restrictive.")
                            _fix_status.update(label="⚠️ Too few trades", state="error")
                        else:
                            # Build trade arrays
                            _fix_profits = [t.get('profit', 0) for t in _fix_trades]
                            _fix_types = [t.get('type', 'long') for t in _fix_trades]
                            _fix_dates = [t.get('exit_time') or t.get('entry_time') for t in _fix_trades]
                            _fix_contracts = [t.get('qty', 1) for t in _fix_trades]
                            _fix_entry_prices = [t.get('entry_price', 0) for t in _fix_trades]
                            _fix_exit_prices = [t.get('exit_price', 0) for t in _fix_trades]
                            _fix_commissions = [t.get('commission', 0) for t in _fix_trades]

                            _fix_sa = StrategyAnalytics(
                                profits=_fix_profits,
                                initial_capital=initial_capital,
                                dates=_fix_dates,
                                contracts=_fix_contracts,
                                entry_prices=_fix_entry_prices,
                                exit_prices=_fix_exit_prices,
                                trade_types=_fix_types,
                                commissions=_fix_commissions,
                                risk_free_rate=risk_free_rate,
                            )

                            # Monte Carlo
                            _fix_mc = _fix_sa.monte_carlo(n_sims=mc_sims)
                            _fix_block_size = max(5, min(20, int(np.sqrt(_fix_sa.n_trades))))
                            _fix_mc_block = _fix_sa.monte_carlo_block(n_sims=mc_sims, block_size=_fix_block_size)
                            _fix_tail = _fix_sa.tail_risk_metrics()

                            st.session_state['_ai_fix_results'] = {
                                'sa': _fix_sa, 'mc': _fix_mc, 'mc_block': _fix_mc_block,
                                'tail': _fix_tail, 'params': _rec_params, 'n_trades': len(_fix_trades),
                            }
                            _fix_status.update(label="✅ Comparison ready!", state="complete")
                    except Exception as _fix_err:
                        st.error(f"Backtest failed: {_fix_err}")
                        _fix_status.update(label="❌ Failed", state="error")

            # ── Display comparison if available ──
            _fix_res = st.session_state.get('_ai_fix_results')
            if _fix_res:
                _fix_sa = _fix_res['sa']
                _fix_mc = _fix_res['mc']
                _fix_mc_block = _fix_res['mc_block']
                _fix_tail = _fix_res['tail']

                st.divider()
                st.subheader("📊 Before vs After — AI Recommended Fixes")

                # Key metrics comparison
                _cmp_cols = st.columns(4)
                with _cmp_cols[0]:
                    _grade_before = "F" if sa.expectancy < 0 else "D" if sa.profit_factor < 1.2 else "C" if sa.profit_factor < 1.5 else "B" if sa.profit_factor < 2.0 else "A"
                    _grade_after = "F" if _fix_sa.expectancy < 0 else "D" if _fix_sa.profit_factor < 1.2 else "C" if _fix_sa.profit_factor < 1.5 else "B" if _fix_sa.profit_factor < 2.0 else "A"
                    st.metric("Grade", _grade_after, delta=f"was {_grade_before}")
                with _cmp_cols[1]:
                    st.metric("Profit Factor", f"{_fix_sa.profit_factor:.2f}", delta=f"{_fix_sa.profit_factor - sa.profit_factor:+.2f}")
                with _cmp_cols[2]:
                    st.metric("Expectancy", f"${_fix_sa.expectancy:.2f}", delta=f"${_fix_sa.expectancy - sa.expectancy:+.2f}")
                with _cmp_cols[3]:
                    st.metric("Max DD", f"{_fix_sa.max_dd_pct*100:.1f}%", delta=f"{(_fix_sa.max_dd_pct - sa.max_dd_pct)*100:+.1f}%", delta_color="inverse")

                _cmp_cols2 = st.columns(4)
                with _cmp_cols2[0]:
                    st.metric("Win Rate", f"{_fix_sa.win_rate*100:.1f}%", delta=f"{(_fix_sa.win_rate - sa.win_rate)*100:+.1f}%")
                with _cmp_cols2[1]:
                    st.metric("Payoff Ratio", f"{_fix_sa.payoff_ratio:.3f}", delta=f"{_fix_sa.payoff_ratio - sa.payoff_ratio:+.3f}")
                with _cmp_cols2[2]:
                    st.metric("Total Trades", f"{_fix_sa.n_trades}", delta=f"{_fix_sa.n_trades - sa.n_trades:+d}")
                with _cmp_cols2[3]:
                    st.metric("Net Profit", f"${_fix_sa.total_profit:,.0f}", delta=f"${_fix_sa.total_profit - sa.total_profit:+,.0f}")

                # Equity curves comparison
                st.markdown("### Equity Curves")
                _eq_fig = go.Figure()
                _eq_fig.add_trace(go.Scatter(y=sa.equity, name="Current", line=dict(color='red', width=1.5)))
                _eq_fig.add_trace(go.Scatter(y=_fix_sa.equity, name="AI Fixed", line=dict(color='lime', width=2)))
                _eq_fig.update_layout(template='plotly_dark', height=350, margin=dict(l=40,r=20,t=30,b=30),
                                     yaxis_title="Equity ($)", legend=dict(x=0.01, y=0.99))
                st.plotly_chart(_eq_fig, use_container_width=True, key="_eq_compare_chart")

                # Monte Carlo comparison
                st.markdown("### Monte Carlo Comparison (Block Bootstrap)")
                _mc_cmp = st.columns(2)
                with _mc_cmp[0]:
                    st.markdown("**Current Strategy**")
                    st.metric("Final Equity (P50)", f"${mc_block['p50_final']:,.0f}")
                    st.metric("Max DD (P50)", f"{mc_block['p50_dd']:.1f}%")
                    st.metric("P(Profit)", f"{mc_block['prob_profit']:.0f}%")
                    if tail_risk:
                        st.metric("CVaR 99%", f"${tail_risk['cvar_99']:,.2f}")
                with _mc_cmp[1]:
                    st.markdown("**AI Fixed Strategy**")
                    st.metric("Final Equity (P50)", f"${_fix_mc_block['p50_final']:,.0f}",
                             delta=f"${_fix_mc_block['p50_final'] - mc_block['p50_final']:+,.0f}")
                    st.metric("Max DD (P50)", f"{_fix_mc_block['p50_dd']:.1f}%",
                             delta=f"{_fix_mc_block['p50_dd'] - mc_block['p50_dd']:+.1f}%", delta_color="inverse")
                    st.metric("P(Profit)", f"{_fix_mc_block['prob_profit']:.0f}%",
                             delta=f"{_fix_mc_block['prob_profit'] - mc_block['prob_profit']:+.0f}%")
                    if _fix_tail:
                        st.metric("CVaR 99%", f"${_fix_tail['cvar_99']:,.2f}",
                                 delta=f"${_fix_tail['cvar_99'] - tail_risk['cvar_99']:+,.2f}", delta_color="inverse")

                # Drawdown comparison
                st.markdown("### Drawdown Comparison")
                _dd_fig = go.Figure()
                _dd_fig.add_trace(go.Scatter(y=[d*100 for d in sa.drawdown_pct], name="Current DD", fill='tozeroy',
                                            line=dict(color='red', width=1), fillcolor='rgba(255,0,0,0.15)'))
                _dd_fig.add_trace(go.Scatter(y=[d*100 for d in _fix_sa.drawdown_pct], name="AI Fixed DD", fill='tozeroy',
                                            line=dict(color='lime', width=1), fillcolor='rgba(0,255,0,0.10)'))
                _dd_fig.update_layout(template='plotly_dark', height=250, margin=dict(l=40,r=20,t=30,b=30),
                                     yaxis_title="Drawdown %", legend=dict(x=0.01, y=0.01))
                st.plotly_chart(_dd_fig, use_container_width=True, key="_dd_compare_chart")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 11: OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[11]:
    st.subheader("🔬 Strategy Input Optimizer")
    st.caption("Upload a Pine Script, configure parameter ranges, and find the top-performing input combinations via grid search.")

    # ── Data source: local datasets or backtest cache ────────────────────
    _opt_has_backtest = ('_pine_ast' in st.session_state and '_pine_cached_data' in st.session_state)

    # Scan data_cache for available datasets
    _opt_cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_cache')
    _opt_datasets = {}
    if os.path.exists(_opt_cache_dir):
        for f in sorted(os.listdir(_opt_cache_dir)):
            if f.endswith('.parquet'):
                _opt_datasets[f.replace('.parquet', '')] = os.path.join(_opt_cache_dir, f)
            elif f.endswith('.csv'):
                _opt_datasets[f.replace('.csv', '') + ' (CSV)'] = os.path.join(_opt_cache_dir, f)

    # Strategy source selector
    _opt_source = st.radio(
        "Strategy Source",
        ["📝 Paste Pine Script", "♻️ Use Current Backtest"] if _opt_has_backtest else ["📝 Paste Pine Script"],
        horizontal=True, key="_opt_source", label_visibility="collapsed",
    )

    _opt_ast = None
    _opt_data = None
    _opt_strat_name = "Strategy"
    _opt_ready = False

    if _opt_source == "♻️ Use Current Backtest" and _opt_has_backtest:
        _opt_ast = st.session_state['_pine_ast']
        _opt_data = st.session_state['_pine_cached_data']
        _opt_strat_name = st.session_state.get('_pine_strategy_name', 'Strategy')
        _opt_ready = True
        st.markdown(f"**Strategy:** `{_opt_strat_name}` &nbsp;|&nbsp; "
                    f"**Data:** `{len(_opt_data):,}` bars")
    else:
        # Standalone Pine input + dataset picker
        oc1, oc2 = st.columns([3, 1])
        with oc2:
            # Dataset selector
            if _opt_datasets:
                _opt_dataset_choice = st.selectbox(
                    "Dataset", list(_opt_datasets.keys()), key="_opt_dataset",
                    help="Pre-loaded Databento/TV data from data_cache/")
            else:
                st.warning("No datasets in data_cache/")
                _opt_dataset_choice = None

            _opt_mintick = st.number_input("Tick Size", value=0.25, min_value=0.0001, step=0.01,
                                           format="%.4f", key="_opt_mintick")
            _opt_commission = st.number_input("Commission %", value=0.0, min_value=0.0,
                                              max_value=5.0, step=0.01, key="_opt_commission") / 100
            _opt_qty = st.number_input("Default Qty", value=1, min_value=1, key="_opt_qty")
            _opt_pyramiding = st.number_input("Pyramiding", value=1, min_value=1, max_value=50, key="_opt_pyramiding")
            _opt_slippage = st.number_input("Slippage (ticks)", value=0, min_value=0, key="_opt_slippage")

        with oc1:
            _opt_pine_code = st.text_area(
                "Pine Script Code", height=250, key="_opt_pine_code",
                placeholder="Paste your Pine Script v6 strategy here...")

            _opt_parse_btn = st.button("⚡ Parse & Load", type="primary", key="_opt_parse_btn")

        if _opt_parse_btn and _opt_pine_code and _opt_pine_code.strip() and _opt_dataset_choice:
            with st.status("Loading strategy + data...", expanded=True) as _opt_load_status:
                # Parse Pine
                try:
                    st.write("📝 Parsing Pine Script...")
                    _opt_tokens = lexer(_opt_pine_code)
                    _opt_parsed_ast = Parser(_opt_tokens).parse()
                    st.write(f"✅ Parsed **{len(_opt_parsed_ast.statements)}** statements")
                except Exception as e:
                    st.error(f"Parse error: {e}")
                    st.stop()

                # Load dataset
                try:
                    _opt_file_path = _opt_datasets[_opt_dataset_choice]
                    st.write(f"📊 Loading `{_opt_dataset_choice}`...")
                    if _opt_file_path.endswith('.parquet'):
                        _opt_loaded_df = pd.read_parquet(_opt_file_path)
                    else:
                        _opt_loaded_df = pd.read_csv(_opt_file_path)
                        # Try to parse time column
                        if 'time' in _opt_loaded_df.columns:
                            _opt_loaded_df['time'] = pd.to_datetime(_opt_loaded_df['time'], unit='s')
                            _opt_loaded_df.set_index('time', inplace=True)
                        # Standardize columns
                        col_remap = {'open': 'Open', 'high': 'High', 'low': 'Low',
                                     'close': 'Close', 'volume': 'Volume'}
                        _opt_loaded_df.rename(columns=col_remap, inplace=True)

                    # Ensure OHLCV columns exist
                    for col in ['Open', 'High', 'Low', 'Close']:
                        if col not in _opt_loaded_df.columns:
                            st.error(f"Missing column: {col}")
                            st.stop()
                    if 'Volume' not in _opt_loaded_df.columns:
                        _opt_loaded_df['Volume'] = 0.0

                    _opt_loaded_df = _opt_loaded_df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
                    st.write(f"✅ **{len(_opt_loaded_df):,}** bars loaded")

                    # Cache for optimizer use
                    st.session_state['_opt_standalone_ast'] = _opt_parsed_ast
                    st.session_state['_opt_standalone_data'] = _opt_loaded_df
                    st.session_state['_opt_standalone_settings'] = {
                        'mintick': _opt_mintick, 'commission': _opt_commission,
                        'qty': _opt_qty, 'pyramiding': _opt_pyramiding,
                        'slippage': _opt_slippage,
                    }
                    # Clear previous optimization params so they get re-extracted
                    st.session_state.pop('_opt_params', None)
                    st.session_state.pop('_opt_results', None)
                    st.session_state.pop('_wf_result', None)

                    _opt_load_status.update(label=f"✅ Ready — {len(_opt_loaded_df):,} bars", state="complete")
                except Exception as e:
                    st.error(f"Data load error: {e}")
                    st.stop()

        # Check if standalone data is cached
        if '_opt_standalone_ast' in st.session_state and '_opt_standalone_data' in st.session_state:
            _opt_ast = st.session_state['_opt_standalone_ast']
            _opt_data = st.session_state['_opt_standalone_data']
            _opt_ready = True
            _s = st.session_state.get('_opt_standalone_settings', {})
            st.markdown(f"**Data:** `{len(_opt_data):,}` bars &nbsp;|&nbsp; "
                        f"**Tick:** `{_s.get('mintick', 0.25)}`")

    if _opt_ready and _opt_ast is not None and _opt_data is not None:

        # Resolve execution settings — standalone overrides backtest sidebar
        _os = st.session_state.get('_opt_standalone_settings', {})
        _oc = _os.get('commission', pine_commission if input_mode == '🌲 Pine Script' else 0.0)
        _oq = _os.get('qty', pine_qty if input_mode == '🌲 Pine Script' else 1)
        _op = _os.get('pyramiding', pine_pyramiding if input_mode == '🌲 Pine Script' else 1)
        _osl = _os.get('slippage', pine_slippage if input_mode == '🌲 Pine Script' else 0)
        _omt = _os.get('mintick', pine_mintick if input_mode == '🌲 Pine Script' else 0.01)
        _oml = pine_margin_long if input_mode == '🌲 Pine Script' else 0
        _oms = pine_margin_short if input_mode == '🌲 Pine Script' else 0

        # ── Step 1: Extract inputs and let user configure ranges ──────────
        if '_opt_params' not in st.session_state:
            # First time — extract inputs via dry run
            with st.spinner("Extracting strategy inputs..."):
                try:
                    _opt_optimizer = PineOptimizer(
                        ast=_opt_ast,
                        df_price=_opt_data,
                        initial_capital=initial_capital,
                        commission_pct=_oc,
                        default_qty=_oq,
                        pyramiding=_op,
                        slippage=_osl,
                        mintick=_omt,
                        margin_long=_oml,
                        margin_short=_oms,
                    )
                    _extracted = _opt_optimizer.extract_inputs(PineInterpreter)
                    st.session_state['_opt_params'] = _extracted
                except Exception as e:
                    st.error(f"Failed to extract inputs: {e}")
                    _extracted = []

        _opt_params = st.session_state.get('_opt_params', [])

        if not _opt_params:
            st.warning("No `input.*()` parameters found in this strategy. Nothing to optimize.")
        else:
            st.markdown("---")
            st.markdown("### Parameter Configuration")
            st.caption("Enable the inputs you want to optimize and set their ranges. Disabled inputs use their default value.")

            # Group inputs
            _opt_grouped = {}
            for idx, p in enumerate(_opt_params):
                # Try to get the group from the original input_defs
                _input_defs = st.session_state.get('_pine_input_defs', [])
                group = 'General'
                if idx < len(_input_defs):
                    group = _input_defs[idx].get('group', '') or 'General'
                if group not in _opt_grouped:
                    _opt_grouped[group] = []
                _opt_grouped[group].append((idx, p))

            for group_name, group_params in _opt_grouped.items():
                st.markdown(f"**{group_name}**")
                for idx, p in group_params:
                    col_enable, col_name, col_range, col_vals = st.columns([0.8, 2.5, 4, 1.5])

                    with col_enable:
                        enabled = st.checkbox("", value=p.enabled, key=f"_opt_en_{idx}",
                                              label_visibility="collapsed")
                        p.enabled = enabled

                    with col_name:
                        st.markdown(f"`{p.name}`")
                        st.caption(f"Type: {p.param_type} | Default: {p.default}")

                    with col_range:
                        if p.param_type == 'bool':
                            st.caption("Values: True, False")
                        elif p.param_type in ('string', 'source') and p.options:
                            st.caption(f"Options: {', '.join(str(o) for o in p.options)}")
                        elif p.param_type in ('int', 'float'):
                            rc1, rc2, rc3 = st.columns(3)
                            with rc1:
                                new_min = st.number_input(
                                    "Min", value=float(p.min_val) if p.min_val is not None else float(p.default or 1) * 0.5,
                                    key=f"_opt_min_{idx}", format="%.4f" if p.param_type == 'float' else "%.0f",
                                )
                                p.min_val = int(new_min) if p.param_type == 'int' else new_min
                            with rc2:
                                new_max = st.number_input(
                                    "Max", value=float(p.max_val) if p.max_val is not None else float(p.default or 1) * 2.0,
                                    key=f"_opt_max_{idx}", format="%.4f" if p.param_type == 'float' else "%.0f",
                                )
                                p.max_val = int(new_max) if p.param_type == 'int' else new_max
                            with rc3:
                                default_step = p.step if p.step is not None else (1 if p.param_type == 'int' else 0.1)
                                new_step = st.number_input(
                                    "Step", value=float(default_step), min_value=0.0001,
                                    key=f"_opt_step_{idx}", format="%.4f" if p.param_type == 'float' else "%.0f",
                                )
                                p.step = int(new_step) if p.param_type == 'int' and new_step >= 1 else new_step
                            # Regenerate values after user changes
                            p.generate_values()
                        else:
                            st.caption(f"Default: {p.default}")

                    with col_vals:
                        if enabled and p.values:
                            st.metric("Values", len(p.values))
                        else:
                            st.caption("—")

            # ── Combination count and controls ────────────────────────────
            st.markdown("---")
            _total_combos = PineOptimizer.estimate_combinations(_opt_params)

            ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns([2, 2, 2, 2])
            with ctrl_c1:
                st.metric("Total Combinations", f"{_total_combos:,}")
            with ctrl_c2:
                _opt_sort_by = st.selectbox("Rank By", [
                    'net_profit', 'sharpe_ratio', 'profit_factor', 'sortino_ratio',
                    'win_rate', 'sqn', 'calmar_ratio', 'expectancy', 'max_drawdown_pct',
                ], index=0, key="_opt_sort_by")
            with ctrl_c3:
                _opt_top_n = st.number_input("Top N Results", value=10, min_value=1, max_value=100, key="_opt_top_n")
            with ctrl_c4:
                _opt_min_trades = st.number_input("Min Trades", value=5, min_value=1, max_value=100, key="_opt_min_trades")

            if _total_combos > 50_000:
                st.error(f"**{_total_combos:,} combinations** exceeds the 50,000 limit. "
                         f"Narrow your ranges, increase step sizes, or disable some parameters.")

            # Time estimate
            if _total_combos > 0 and _total_combos <= 50_000:
                # Rough estimate: ~0.05s per run for small data, more for large
                _bars = len(_opt_data)
                _est_per_run = max(0.01, _bars / 50000)  # rough scaling
                _est_total = _total_combos * _est_per_run
                if _est_total < 60:
                    _est_label = f"~{_est_total:.0f}s"
                elif _est_total < 3600:
                    _est_label = f"~{_est_total/60:.1f}min"
                else:
                    _est_label = f"~{_est_total/3600:.1f}hr"
                st.caption(f"Estimated time: {_est_label} ({_bars:,} bars x {_total_combos:,} combos)")

            # ── Run Optimization ──────────────────────────────────────────
            _run_opt = st.button("🚀 Run Optimization", type="primary", key="_run_opt_btn",
                                 disabled=(_total_combos > 50_000 or _total_combos == 0))

            if _run_opt:
                _opt_optimizer = PineOptimizer(
                    ast=_opt_ast,
                    df_price=_opt_data,
                    initial_capital=initial_capital,
                    commission_pct=_oc, default_qty=_oq, pyramiding=_op,
                    slippage=_osl, mintick=_omt,
                    margin_long=_oml, margin_short=_oms,
                )

                _opt_progress = st.progress(0, text="Starting optimization...")
                _opt_status_text = st.empty()
                _opt_t0 = time.time()

                def _opt_progress_cb(current, total):
                    pct = current / total
                    elapsed = time.time() - _opt_t0
                    rate = current / elapsed if elapsed > 0 else 0
                    eta = (total - current) / rate if rate > 0 else 0
                    _opt_progress.progress(
                        pct,
                        text=f"Testing combination {current:,}/{total:,} — "
                             f"{rate:.1f} combos/sec — ETA {eta:.0f}s"
                    )

                try:
                    _opt_results = _opt_optimizer.optimize(
                        params=_opt_params,
                        PineInterpreterClass=PineInterpreter,
                        sort_by=_opt_sort_by,
                        top_n=_opt_top_n,
                        min_trades=_opt_min_trades,
                        progress_callback=_opt_progress_cb,
                    )

                    _opt_elapsed = time.time() - _opt_t0
                    _opt_progress.progress(1.0, text=f"Complete — {_total_combos:,} combos in {_opt_elapsed:.1f}s")

                    st.session_state['_opt_results'] = _opt_results
                    st.session_state['_opt_elapsed'] = _opt_elapsed

                except ValueError as ve:
                    st.error(str(ve))
                except Exception as e:
                    st.error(f"Optimization failed: {e}")

            # ── Display Results ───────────────────────────────────────────
            if '_opt_results' in st.session_state:
                _opt_results = st.session_state['_opt_results']
                _opt_elapsed = st.session_state.get('_opt_elapsed', 0)

                if not _opt_results:
                    st.warning("No combinations met the minimum trade count. Try lowering the min trades threshold.")
                else:
                    st.markdown("---")
                    st.markdown(f"### Top {len(_opt_results)} Results")
                    st.caption(f"Ranked by **{_opt_sort_by}** | "
                               f"Completed in **{_opt_elapsed:.1f}s** | "
                               f"Min trades: **{_opt_min_trades}**")

                    # Results table
                    _opt_df = PineOptimizer.to_dataframe(_opt_results)

                    # Format for display
                    _opt_display = _opt_df.copy()
                    if 'Win Rate' in _opt_display.columns:
                        _opt_display['Win Rate'] = (_opt_display['Win Rate'] * 100).round(1).astype(str) + '%'
                    if 'Max DD%' in _opt_display.columns:
                        _opt_display['Max DD%'] = _opt_display['Max DD%'].round(2).astype(str) + '%'
                    if 'Return %' in _opt_display.columns:
                        _opt_display['Return %'] = _opt_display['Return %'].round(2).astype(str) + '%'

                    # Replace inf with symbol
                    _opt_display = _opt_display.replace([float('inf'), float('-inf')], '∞')
                    _opt_display = _opt_display.fillna('—')

                    st.dataframe(_safe_df(_opt_df), use_container_width=True, hide_index=True)

                    # ── Visual comparison of top results ──────────────────
                    st.markdown("### Performance Comparison")

                    # Net Profit bar chart
                    _bar_fig = go.Figure()
                    _labels = [f"#{r.params}" for r in _opt_results[:10]]
                    _short_labels = [f"#{i+1}" for i in range(min(10, len(_opt_results)))]
                    _profits = [r.net_profit for r in _opt_results[:10]]
                    _colors = [COLORS['win'] if p > 0 else COLORS['loss'] for p in _profits]

                    _bar_fig.add_trace(go.Bar(
                        x=_short_labels, y=_profits,
                        marker_color=_colors, opacity=0.85,
                        text=[f"${p:,.0f}" for p in _profits],
                        textposition='outside',
                    ))
                    _bar_fig.update_layout(
                        **PLOTLY_LAYOUT,
                        title="Net Profit by Rank",
                        yaxis_title="Net Profit ($)",
                        xaxis_title="Rank",
                        showlegend=False,
                    )
                    st.plotly_chart(_bar_fig, use_container_width=True, key="opt_profit_bar")

                    # Sharpe vs Profit scatter
                    if len(_opt_results) >= 2:
                        _scatter_fig = go.Figure()
                        _sharpes = [r.sharpe_ratio if r.sharpe_ratio != float('inf') else 5.0 for r in _opt_results]
                        _net_profits = [r.net_profit for r in _opt_results]
                        _win_rates = [r.win_rate * 100 for r in _opt_results]
                        _hover_texts = [
                            f"Rank #{i+1}<br>Profit: ${r.net_profit:,.0f}<br>"
                            f"Sharpe: {r.sharpe_ratio:.2f}<br>Win Rate: {r.win_rate:.1%}<br>"
                            f"Trades: {r.num_trades}<br>Max DD: {r.max_drawdown_pct:.1f}%<br>"
                            + "<br>".join(f"{k}: {v}" for k, v in r.params.items())
                            for i, r in enumerate(_opt_results)
                        ]

                        _scatter_fig.add_trace(go.Scatter(
                            x=_sharpes, y=_net_profits,
                            mode='markers+text',
                            marker=dict(
                                size=[max(8, wr / 3) for wr in _win_rates],
                                color=_win_rates,
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(title="Win %"),
                            ),
                            text=_short_labels,
                            textposition='top center',
                            hovertext=_hover_texts,
                            hoverinfo='text',
                        ))
                        _scatter_fig.update_layout(
                            **PLOTLY_LAYOUT,
                            title="Sharpe Ratio vs Net Profit (bubble size = win rate)",
                            xaxis_title="Sharpe Ratio",
                            yaxis_title="Net Profit ($)",
                            showlegend=False,
                        )
                        st.plotly_chart(_scatter_fig, use_container_width=True, key="opt_scatter")

                    # ── Detailed view of selected result ──────────────────
                    st.markdown("### Inspect Result")
                    _inspect_rank = st.selectbox(
                        "Select a result to inspect",
                        options=list(range(len(_opt_results))),
                        format_func=lambda i: f"Rank #{i+1} — ${_opt_results[i].net_profit:,.2f} profit, "
                                              f"{_opt_results[i].num_trades} trades, "
                                              f"Sharpe {_opt_results[i].sharpe_ratio:.2f}",
                        key="_opt_inspect_rank",
                    )

                    _selected = _opt_results[_inspect_rank]

                    # Show params
                    st.markdown("**Input Settings:**")
                    _param_cols = st.columns(min(4, len(_selected.params)))
                    for i, (k, v) in enumerate(_selected.params.items()):
                        _param_cols[i % len(_param_cols)].code(f"{k} = {v}")

                    # Show metrics
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Net Profit", f"${_selected.net_profit:,.2f}")
                    m2.metric("Sharpe", f"{_selected.sharpe_ratio:.3f}")
                    m3.metric("Win Rate", f"{_selected.win_rate:.1%}")
                    m4.metric("Profit Factor", f"{_selected.profit_factor:.2f}" if _selected.profit_factor != float('inf') else "∞")
                    m5.metric("Max Drawdown", f"{_selected.max_drawdown_pct:.2f}%")

                    m6, m7, m8, m9, m10 = st.columns(5)
                    m6.metric("Trades", _selected.num_trades)
                    m7.metric("Sortino", f"{_selected.sortino_ratio:.3f}" if _selected.sortino_ratio != float('inf') else "∞")
                    m8.metric("Expectancy", f"${_selected.expectancy:,.2f}")
                    m9.metric("SQN", f"{_selected.sqn:.3f}")
                    m10.metric("Avg Trade", f"${_selected.avg_trade:,.2f}")

                    # ── Walk-Forward Validation ───────────────────────────
                    st.markdown("---")
                    st.markdown("### Walk-Forward Validation")
                    st.caption("Tests the selected parameter set across multiple time windows to check robustness.")

                    wf_c1, wf_c2 = st.columns([1, 3])
                    with wf_c1:
                        _wf_splits = st.number_input("Splits", value=5, min_value=2, max_value=20, key="_wf_splits")

                    _run_wf = st.button("Run Walk-Forward", key="_run_wf_btn")

                    if _run_wf:
                        _wf_optimizer = PineOptimizer(
                            ast=_opt_ast,
                            df_price=_opt_data,
                            initial_capital=initial_capital,
                            commission_pct=_oc, default_qty=_oq, pyramiding=_op,
                            slippage=_osl, mintick=_omt,
                            margin_long=_oml, margin_short=_oms,
                        )

                        with st.spinner("Running walk-forward analysis..."):
                            try:
                                _wf_result = _wf_optimizer.walk_forward(
                                    params=_selected.params,
                                    PineInterpreterClass=PineInterpreter,
                                    n_splits=_wf_splits,
                                )
                                st.session_state['_wf_result'] = _wf_result
                            except Exception as e:
                                st.error(f"Walk-forward failed: {e}")

                    if '_wf_result' in st.session_state:
                        _wf = st.session_state['_wf_result']

                        if _wf['splits']:
                            wf_m1, wf_m2, wf_m3 = st.columns(3)
                            wf_m1.metric("Avg Profit/Split", f"${_wf['avg_profit']:,.2f}")
                            wf_m2.metric("Avg Sharpe/Split", f"{_wf['avg_sharpe']:.3f}")
                            _cons = _wf['consistency_score']
                            _cons_color = "normal" if _cons >= 60 else "off"
                            wf_m3.metric("Consistency", f"{_cons:.0f}%", delta="Robust" if _cons >= 60 else "Fragile",
                                         delta_color=_cons_color)

                            # Split results table
                            _wf_df = pd.DataFrame(_wf['splits'])
                            _wf_df['profitable'] = _wf_df['profitable'].map({True: '✅', False: '❌'})
                            st.dataframe(_wf_df, use_container_width=True, hide_index=True)

                            # Per-split profit bar chart
                            _wf_bar = go.Figure()
                            _wf_profits = [s['net_profit'] for s in _wf['splits']]
                            _wf_colors = [COLORS['win'] if p > 0 else COLORS['loss'] for p in _wf_profits]
                            _wf_bar.add_trace(go.Bar(
                                x=[f"Split {s['split']}" for s in _wf['splits']],
                                y=_wf_profits,
                                marker_color=_wf_colors,
                                text=[f"${p:,.0f}" for p in _wf_profits],
                                textposition='outside',
                            ))
                            _wf_bar.update_layout(
                                **PLOTLY_LAYOUT,
                                title="Profit by Walk-Forward Split",
                                yaxis_title="Net Profit ($)",
                                showlegend=False,
                            )
                            st.plotly_chart(_wf_bar, use_container_width=True, key="wf_profit_bar")
                        else:
                            st.warning("Not enough data for walk-forward analysis. Try fewer splits.")

                    # ── Apply to Backtest ─────────────────────────────────
                    st.markdown("---")
                    _apply_opt = st.button("📋 Apply These Settings to Backtest", key="_apply_opt_btn",
                                           help="Copies the selected result's input values to the main backtest inputs")

                    if _apply_opt:
                        for k, v in _selected.params.items():
                            st.session_state[f"pine_input_{k}"] = v
                        st.success(f"Applied Rank #{_inspect_rank + 1} settings to backtest inputs. "
                                   f"Click **🔄 Apply** in the Pine Script section to re-run.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 12: INSTITUTIONAL ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

with tabs[12]:
    st.subheader("🏛️ Institutional-Grade Analytics")
    st.caption("Deflated Sharpe Ratio, Walk-Forward Efficiency, Regime Analysis, Monthly Heatmap, QuantStats Tearsheet")

    # ── Deflated Sharpe Ratio ────────────────────────────────────────
    st.markdown("### Deflated Sharpe Ratio (Bailey & Lopez de Prado)")
    st.caption("Corrects Sharpe for overfitting when you've tested multiple parameter sets or strategies.")

    _dsr_trials = st.number_input(
        "Number of strategies/parameter sets tested",
        min_value=1, max_value=10000, value=1, step=1, key="_dsr_trials",
        help="How many different configurations did you test before arriving at this one? More trials = higher bar for significance."
    )
    _dsr = sa.deflated_sharpe_ratio(num_trials=_dsr_trials)

    _dsr_c1, _dsr_c2, _dsr_c3, _dsr_c4 = st.columns(4)
    with _dsr_c1:
        _dsr_color = "normal" if _dsr["dsr_pct"] > 50 else "off"
        st.metric("DSR Probability", f"{_dsr['dsr_pct']:.1f}%",
                  delta="Significant" if _dsr["dsr_pct"] > 95 else "Marginal" if _dsr["dsr_pct"] > 50 else "Not Significant",
                  delta_color=_dsr_color)
    with _dsr_c2:
        st.metric("Observed Sharpe", f"{_dsr['sr_observed']:.3f}")
    with _dsr_c3:
        st.metric("Expected Max SR (null)", f"{_dsr['sr_star']:.3f}",
                  help="Expected best Sharpe from random noise given N trials")
    with _dsr_c4:
        st.metric("Overfitting Haircut", f"{_dsr['haircut_pct']:.1f}%",
                  delta="Low" if _dsr["haircut_pct"] < 20 else "High",
                  delta_color="normal" if _dsr["haircut_pct"] < 20 else "off")

    if _dsr["dsr_pct"] >= 95:
        st.success("**DSR > 95%** — The Sharpe ratio is statistically significant even after correcting for multiple testing.")
    elif _dsr["dsr_pct"] >= 50:
        st.warning("**DSR 50-95%** — Marginal significance. The Sharpe may partially be explained by overfitting / luck.")
    else:
        st.error("**DSR < 50%** — The Sharpe ratio is likely explained by overfitting. The strategy would not pass institutional review.")

    # DSR sensitivity chart: sweep num_trials from 1 to 100
    _dsr_sweep_trials = list(range(1, min(201, max(10, _dsr_trials * 3))))
    _dsr_sweep_vals = [sa.deflated_sharpe_ratio(t)["dsr_pct"] for t in _dsr_sweep_trials]
    _dsr_fig = go.Figure()
    _dsr_fig.add_trace(go.Scatter(x=_dsr_sweep_trials, y=_dsr_sweep_vals,
                                   mode="lines", line=dict(color=BRAND["teal"], width=2),
                                   name="DSR %"))
    _dsr_fig.add_hline(y=95, line_dash="dash", line_color="lime",
                        annotation_text="95% Significance", annotation_position="top left")
    _dsr_fig.add_hline(y=50, line_dash="dot", line_color="orange",
                        annotation_text="50% Threshold", annotation_position="bottom left")
    _dsr_fig.add_vline(x=_dsr_trials, line_dash="dash", line_color="white",
                        annotation_text=f"Your trials ({_dsr_trials})", annotation_position="top right")
    _dsr_fig.update_layout(**PLOTLY_LAYOUT, title="DSR vs Number of Trials Tested",
                           xaxis_title="Number of Strategies/Params Tested",
                           yaxis_title="DSR Probability (%)", height=350)
    st.plotly_chart(_dsr_fig, use_container_width=True, key="_dsr_sweep_chart")

    st.divider()

    # ── Walk-Forward Efficiency ──────────────────────────────────────
    st.markdown("### Walk-Forward Efficiency")
    st.caption("Measures how well in-sample performance predicts out-of-sample results. WFE > 0.5 = good. > 0.8 = excellent.")

    _wfe_splits = st.slider("Number of WF splits", 3, 10, 5, key="_inst_wfe_splits")
    _wfe = sa.walk_forward_efficiency(n_splits=_wfe_splits)

    if _wfe:
        _wfe_c1, _wfe_c2, _wfe_c3 = st.columns(3)
        with _wfe_c1:
            _wfe_color = "normal" if _wfe["avg_wfe_sharpe"] > 0.5 else "off"
            st.metric("WFE (Sharpe)", f"{_wfe['avg_wfe_sharpe']:.3f}",
                      delta=_wfe["interpretation"], delta_color=_wfe_color)
        with _wfe_c2:
            st.metric("WFE (Profit Factor)", f"{_wfe['avg_wfe_pf']:.3f}")
        with _wfe_c3:
            st.metric("Assessment", _wfe["interpretation"])

        # WFE per-fold chart
        _wfe_df = pd.DataFrame(_wfe["folds"])
        _wfe_fig = make_subplots(rows=1, cols=2, subplot_titles=["Sharpe: IS vs OOS", "Profit Factor: IS vs OOS"])
        _wfe_fig.add_trace(go.Bar(name="In-Sample", x=[f"Fold {r['fold']}" for r in _wfe["folds"]],
                                   y=_wfe_df["is_sharpe"], marker_color="rgba(0,200,255,0.6)"), row=1, col=1)
        _wfe_fig.add_trace(go.Bar(name="Out-of-Sample", x=[f"Fold {r['fold']}" for r in _wfe["folds"]],
                                   y=_wfe_df["oos_sharpe"], marker_color="rgba(0,255,100,0.6)"), row=1, col=1)
        _wfe_fig.add_trace(go.Bar(name="IS PF", x=[f"Fold {r['fold']}" for r in _wfe["folds"]],
                                   y=_wfe_df["is_pf"], marker_color="rgba(0,200,255,0.6)", showlegend=False), row=1, col=2)
        _wfe_fig.add_trace(go.Bar(name="OOS PF", x=[f"Fold {r['fold']}" for r in _wfe["folds"]],
                                   y=_wfe_df["oos_pf"], marker_color="rgba(0,255,100,0.6)", showlegend=False), row=1, col=2)
        _wfe_fig.update_layout(**PLOTLY_LAYOUT, height=350, barmode="group")
        st.plotly_chart(_wfe_fig, use_container_width=True, key="_wfe_fold_chart")
    else:
        st.info("Not enough trades for walk-forward analysis (need 50+).")

    st.divider()

    # ── Regime-Conditional Monte Carlo ───────────────────────────────
    st.markdown("### Regime-Conditional Monte Carlo")
    st.caption("Stratifies trades by volatility regime — reveals if your strategy only works in calm markets.")

    _regime_mc = sa.regime_monte_carlo(n_sims=mc_sims, n_regimes=3, block_size=_block_size)
    if _regime_mc:
        # Per-regime stat cards
        _reg_cols = st.columns(len(_regime_mc["regime_stats"]))
        for i, (regime_name, rstats) in enumerate(_regime_mc["regime_stats"].items()):
            with _reg_cols[i]:
                _rcolor = "#22c55e" if rstats["avg_pnl"] > 0 else "#ef4444"
                st.markdown(f"""
                <div style="background: rgba({'34,197,94' if rstats['avg_pnl'] > 0 else '239,68,68'},0.1);
                            border: 1px solid {_rcolor}33; border-radius: 8px; padding: 12px; text-align: center;">
                    <div style="font-weight:700; color:{_rcolor}; font-size:1.1rem;">{regime_name}</div>
                    <div style="font-size:0.8rem; opacity:0.6;">{rstats['n_trades']} trades ({rstats['pct_of_total']:.0f}%)</div>
                </div>
                """, unsafe_allow_html=True)
                st.metric("Avg PnL", f"${rstats['avg_pnl']:,.2f}")
                st.metric("Win Rate", f"{rstats['win_rate']*100:.1f}%")
                st.metric("Sharpe", f"{rstats['sharpe']:.2f}")
                st.metric("Max DD", f"${rstats['max_dd']:,.0f}")

        # Regime MC probability comparison
        st.markdown("#### Regime-Aware Monte Carlo vs Standard")
        _rmc_c1, _rmc_c2 = st.columns(2)
        with _rmc_c1:
            st.metric("Standard MC P(Profit)", f"{mc_block['prob_profit']:.1f}%")
            st.metric("Standard MC Final Equity P50", f"${mc_block['percentiles']['p50']:,.0f}")
        with _rmc_c2:
            st.metric("Regime MC P(Profit)", f"{_regime_mc['prob_profit']:.1f}%",
                      delta=f"{_regime_mc['prob_profit'] - mc_block['prob_profit']:+.1f}%")
            st.metric("Regime MC Final Equity P50", f"${_regime_mc['percentiles']['p50']:,.0f}",
                      delta=f"${_regime_mc['percentiles']['p50'] - mc_block['percentiles']['p50']:+,.0f}")

        # Regime trade PnL distribution
        _regime_pnl_fig = go.Figure()
        for regime_name, rstats in _regime_mc["regime_stats"].items():
            _rmask = _regime_mc["regimes"] == list(_regime_mc["regime_stats"].keys()).index(regime_name)
            _regime_pnls = sa.raw_profits[_rmask]
            _regime_pnl_fig.add_trace(go.Histogram(x=_regime_pnls, name=regime_name, opacity=0.6, nbinsx=40))
        _regime_pnl_fig.update_layout(**PLOTLY_LAYOUT, title="PnL Distribution by Regime",
                                       xaxis_title="Trade PnL ($)", yaxis_title="Count", barmode="overlay", height=350)
        st.plotly_chart(_regime_pnl_fig, use_container_width=True, key="_regime_pnl_dist")

    else:
        st.info("Not enough trades for regime analysis.")

    st.divider()

    # ── Monthly Returns Heatmap ──────────────────────────────────────
    st.markdown("### Monthly Returns Heatmap")
    _monthly = sa.monthly_returns()
    if _monthly is not None and not _monthly.empty:
        # Add YTD column
        _monthly["YTD"] = _monthly.sum(axis=1)

        _heat_fig = go.Figure(data=go.Heatmap(
            z=_monthly.values,
            x=_monthly.columns.tolist(),
            y=[str(y) for y in _monthly.index.tolist()],
            colorscale=[[0, "#ef4444"], [0.5, "#1a1a2e"], [1, "#22c55e"]],
            zmid=0,
            text=[[f"${v:,.0f}" for v in row] for row in _monthly.values],
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False,
        ))
        _heat_fig.update_layout(**PLOTLY_LAYOUT, title="Monthly PnL Heatmap",
                                height=max(200, len(_monthly) * 45 + 80))
        st.plotly_chart(_heat_fig, use_container_width=True, key="_monthly_heatmap")
    else:
        st.info("Date information not available for monthly heatmap.")

    st.divider()

    # ── QuantStats Tearsheet ─────────────────────────────────────────
    st.markdown("### QuantStats Tearsheet")
    st.caption("Full institutional tearsheet — downloadable HTML report with 30+ metrics.")

    _qs_metrics = sa.quantstats_metrics()
    if _qs_metrics:
        _qs_c1, _qs_c2, _qs_c3, _qs_c4 = st.columns(4)
        with _qs_c1:
            st.metric("QS CAGR", f"{_qs_metrics['cagr']:.2f}%")
            st.metric("QS Volatility", f"{_qs_metrics['volatility']:.2f}%")
        with _qs_c2:
            st.metric("QS Sharpe", f"{_qs_metrics['sharpe']:.3f}")
            st.metric("QS Win Rate", f"{_qs_metrics['win_rate']:.1f}%")
        with _qs_c3:
            st.metric("QS Sortino", f"{_qs_metrics['sortino']:.3f}")
            st.metric("QS Best Day", f"{_qs_metrics['best_day']:.2f}%")
        with _qs_c4:
            st.metric("QS Max DD", f"{_qs_metrics['max_drawdown']:.2f}%")
            st.metric("QS Worst Day", f"{_qs_metrics['worst_day']:.2f}%")

    _gen_tearsheet = st.button("📄 Generate Full QuantStats HTML Tearsheet", key="_gen_qs_tearsheet")
    if _gen_tearsheet:
        with st.spinner("Generating tearsheet..."):
            _qs_html = sa.generate_quantstats_html()
            if _qs_html:
                _b64 = base64.b64encode(_qs_html.encode()).decode()
                st.markdown(
                    f'<a href="data:text/html;base64,{_b64}" download="quantstats_tearsheet.html">'
                    f'<button style="background:#19b8ff; color:white; border:none; padding:10px 20px; '
                    f'border-radius:6px; cursor:pointer; font-weight:700;">Download Tearsheet (HTML)</button></a>',
                    unsafe_allow_html=True,
                )
                st.success("Tearsheet generated. Click above to download.")
            else:
                st.error("Could not generate tearsheet. Ensure trades have date information.")

    st.divider()

    # ── Institutional Summary Table ──────────────────────────────────
    st.markdown("### Institutional Metrics Summary")
    _inst_data = {
        "Metric": [
            "Sharpe Ratio (annualized)", "Deflated Sharpe Ratio",
            "Sortino Ratio", "Calmar Ratio", "Omega Ratio",
            "SQN", "Walk-Forward Efficiency",
            "Profit Factor", "Payoff Ratio", "Kelly Criterion",
            "Max Drawdown %", "Recovery Factor", "Ulcer Performance Index",
            "Risk of Ruin (50%)", "P(Profit) — Block MC",
            "Skewness", "Excess Kurtosis", "p-Value",
        ],
        "Value": [
            f"{sa.sharpe_ratio:.3f}", f"{_dsr['dsr_pct']:.1f}%",
            f"{sa.sortino_ratio:.3f}", f"{sa.calmar_ratio:.3f}", f"{sa.omega_ratio:.3f}",
            f"{sa.sqn:.2f}", f"{_wfe['avg_wfe_sharpe']:.3f}" if _wfe else "N/A",
            f"{sa.profit_factor:.3f}", f"{sa.payoff_ratio:.3f}", f"{sa.kelly_criterion:.4f}",
            f"{sa.max_drawdown_pct:.2f}%", f"{sa.recovery_factor:.2f}", f"{sa.ulcer_performance_index:.3f}",
            f"{sa.risk_of_ruin(50):.2f}%", f"{mc_block['prob_profit']:.1f}%",
            f"{sa.skewness:.3f}", f"{sa.excess_kurtosis:.3f}", f"{sa.p_value_profit:.6f}",
        ],
        "Institutional Threshold": [
            "> 1.0 (good), > 2.0 (excellent)", "> 95% (significant)",
            "> 2.0", "> 1.0", "> 1.5",
            "> 2.0 (good), > 5.0 (excellent)", "> 0.5 (good), > 0.8 (excellent)",
            "> 1.5", "> 1.0", "> 0 (positive edge)",
            "< -20%", "> 3.0", "> 1.0",
            "< 1%", "> 95%",
            "Positive preferred", "< 3 (no fat tails)", "< 0.05 (significant)",
        ],
    }
    st.dataframe(pd.DataFrame(_inst_data), use_container_width=True, hide_index=True)


#JUST TRADES CONFIDENTIAL PROPERTY
# ═══════════════════════════════════════════════════════════════════════════════
# PERSISTENT AI QUANT CHAT — Always visible at bottom of page
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border: 1px solid rgba(0,200,255,0.3); border-radius: 12px;
            padding: 16px 20px 8px 20px; margin: 8px 0;">
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
        <span style="font-size:1.4rem;">🤖</span>
        <span style="font-size:1.1rem; font-weight:700; color:#00c8ff;">AI Quant Analyst</span>
        <span style="font-size:0.75rem; opacity:0.5; margin-left:auto;">Powered by Claude — Always Available</span>
    </div>
</div>
""", unsafe_allow_html=True)

if True:
    # Quick action buttons
    _qa1, _qa2, _qa3, _qa4, _qa5 = st.columns(5)
    with _qa1:
        _btn_quick = st.button("⚡ Grade + Fix", key="_global_ai_quick", use_container_width=True)
    with _qa2:
        _btn_risk = st.button("🛠️ Top 3 Fixes", key="_global_ai_risk", use_container_width=True)
    with _qa3:
        _btn_size = st.button("📐 Position Size", key="_global_ai_size", use_container_width=True)
    with _qa4:
        _btn_deploy = st.button("🚀 Deploy Plan", key="_global_ai_deploy", use_container_width=True)
    with _qa5:
        if st.button("🗑️ Clear", key="_global_ai_clear", use_container_width=True):
            st.session_state['_global_ai_chat'] = []
            st.rerun()

    # Quick-fire prompts
    _quick_prompts = {
        '_global_ai_quick': "Grade this strategy A-F. Then for each weakness behind the grade, give me the SPECIFIC FIX with exact Pine Script code changes (line numbers, before/after). Format: Grade → Weakness → Fix (with code). Keep it concise but actionable. If you have the Pine source code, reference exact lines. At the END, include the ```json:recommended_params``` block AND the COMPLETE ```pine:revised_strategy``` block with ALL fixes applied — every line, no truncation.",
        '_global_ai_risk': "Identify the TOP 3 weaknesses in this strategy. For EACH one, give me: the problem (with the metric), the fix (with exact Pine Script code to change or add — show line numbers and before/after), and what the improved metric should look like. At the END, include the ```json:recommended_params``` block AND the COMPLETE ```pine:revised_strategy``` block with ALL fixes applied — every single line of the original script, modified where needed. No '...' placeholders.",
        '_global_ai_size': "Build me a position sizing table for this strategy. Include columns for: Account Size, Max Contracts (NQ), Max Contracts (MNQ), Risk Per Trade, Daily Max Loss. Cover these account sizes: USD 50K, 100K, 500K, 1M, 5M, 10M, 100M. Factor in Kelly, max DD, and worst-case tail risk. Give exact numbers.",
        '_global_ai_deploy': "Build me a 5-day deployment plan for this strategy starting Monday. For each day, tell me: what to do, what to monitor, what thresholds to set, and when to pull the plug. Include exact contract counts, loss limits, and success criteria for scaling up. If there are blockers, tell me EXACTLY how to fix them before Monday with Pine Script code changes."
    }

    _triggered_prompt = None
    if _btn_quick:
        _triggered_prompt = _quick_prompts['_global_ai_quick']
    elif _btn_risk:
        _triggered_prompt = _quick_prompts['_global_ai_risk']
    elif _btn_size:
        _triggered_prompt = _quick_prompts['_global_ai_size']
    elif _btn_deploy:
        _triggered_prompt = _quick_prompts['_global_ai_deploy']

    if _triggered_prompt:
        _ctx = build_strategy_context()
        _full_msg = f"Strategy data:\n```\n{_ctx}\n```\n\n{_triggered_prompt}"
        with st.spinner("🧠 Analyzing..."):
            call_ai_analyst(_full_msg, '_global_ai_chat')
        st.rerun()

    # Display global chat history
    if st.session_state.get('_global_ai_chat'):
        for _msg in st.session_state['_global_ai_chat']:
            if _msg['role'] == 'user':
                with st.chat_message("user"):
                    # Show abbreviated version of user message
                    _display = _msg['content']
                    if "Strategy data:" in _display:
                        _display = _display.split("```")[-1].strip() if "```" in _display else _display[-200:]
                    st.markdown(_display[:300])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(_msg['content'])

    # Free-form chat input — always at the bottom
    _chat_input = st.chat_input(
        "Ask the AI Quant Analyst anything about your strategy...",
        key="_global_ai_input"
    )
    if _chat_input:
        _ctx = build_strategy_context()
        _full_msg = f"Strategy data:\n```\n{_ctx}\n```\n\n{_chat_input}"
        with st.spinner("🧠 Thinking..."):
            call_ai_analyst(_full_msg, '_global_ai_chat')
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="jt-footer">
    JUST TRADES — STRATEGY ANALYTICS ENGINE<br>
    Built for traders who demand precision.
</div>
""", unsafe_allow_html=True)