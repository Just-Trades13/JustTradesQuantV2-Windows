"""
Pine Script Strategy Optimizer
Runs parameter grid search to find top-performing input combinations.
Designed to work with the PineInterpreter from app.py without circular imports.
"""
import itertools
import time
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Callable


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class InputParam:
    """One optimizable input parameter extracted from a Pine strategy."""
    name: str
    param_type: str  # int, float, bool, string, source
    default: Any = None
    min_val: Any = None
    max_val: Any = None
    step: Any = None
    options: Optional[list] = None
    enabled: bool = True
    values: list = field(default_factory=list)

    MAX_VALUES_PER_PARAM = 50

    def generate_values(self) -> list:
        """Build the discrete list of values to test for this parameter."""
        if self.param_type == 'bool':
            self.values = [True, False]
            return self.values

        if self.param_type in ('string', 'source') and self.options:
            self.values = list(self.options)
            return self.values

        # Numeric types (int / float)
        if self.param_type in ('int', 'float'):
            mn = self.min_val
            mx = self.max_val
            st = self.step

            # Smart defaults when Pine code didn't specify range
            if mn is None or mx is None:
                defval = self.default if self.default is not None else 1
                defval = float(defval) if defval != 0 else 1.0
                abs_def = abs(defval) if defval != 0 else 1.0
                if mn is None:
                    mn = max(defval * 0.5, 1) if defval > 0 else defval * 2.0
                if mx is None:
                    mx = defval * 2.0 if defval > 0 else max(defval * 0.5, 1)
                # Ensure min < max
                if mn > mx:
                    mn, mx = mx, mn

            if st is None:
                if self.param_type == 'int':
                    st = max(1, int((mx - mn) / 20)) if mx != mn else 1
                else:
                    st = (mx - mn) / 20 if mx != mn else abs(mn) * 0.1 or 0.1

            mn = float(mn)
            mx = float(mx)
            st = float(st)

            if st <= 0:
                st = 1.0 if self.param_type == 'int' else 0.1

            # Generate values with numpy arange, add half-step to include max
            vals = np.arange(mn, mx + st / 2, st)

            if self.param_type == 'int':
                vals = [int(round(v)) for v in vals]
                # Deduplicate while preserving order
                seen = set()
                deduped = []
                for v in vals:
                    if v not in seen:
                        seen.add(v)
                        deduped.append(v)
                vals = deduped
            else:
                # Round floats to avoid artifacts — use enough decimals
                decimals = max(len(str(st).rstrip('0').split('.')[-1]) if '.' in str(st) else 0, 2)
                vals = [round(float(v), decimals) for v in vals]
                seen = set()
                deduped = []
                for v in vals:
                    if v not in seen:
                        seen.add(v)
                        deduped.append(v)
                vals = deduped

            # Cap to prevent explosion
            if len(vals) > self.MAX_VALUES_PER_PARAM:
                indices = np.linspace(0, len(vals) - 1, self.MAX_VALUES_PER_PARAM, dtype=int)
                vals = [vals[i] for i in indices]

            self.values = vals
            return self.values

        # Fallback: string/source without options — just use default
        if self.default is not None:
            self.values = [self.default]
        else:
            self.values = []
        return self.values


@dataclass
class OptimizationResult:
    """Metrics for a single parameter combination run."""
    params: dict
    net_profit: float = 0.0
    total_return_pct: float = 0.0
    num_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    expectancy: float = 0.0
    calmar_ratio: float = 0.0
    sqn: float = 0.0
    avg_trade: float = 0.0
    trades: Optional[list] = None


# ── Optimizer ─────────────────────────────────────────────────────────────────

class PineOptimizer:
    """
    Grid-search optimizer for PineScript strategies.
    Takes AST + price data, sweeps input parameters, ranks results.
    """

    def __init__(self, ast, df_price: pd.DataFrame, initial_capital: float = 10000,
                 commission_pct: float = 0.0, default_qty: int = 1,
                 pyramiding: int = 1, slippage: int = 0, mintick: float = 0.01,
                 margin_long: int = 0, margin_short: int = 0):
        self.ast = ast
        self.df_price = df_price
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.default_qty = default_qty
        self.pyramiding = pyramiding
        self.slippage = slippage
        self.mintick = mintick
        self.margin_long = margin_long
        self.margin_short = margin_short

    def extract_inputs(self, PineInterpreterClass) -> List[InputParam]:
        """
        Do a dry run with default inputs to capture all input_defs,
        then convert each to an InputParam with smart defaults.
        """
        interp = PineInterpreterClass(
            ast=self.ast,
            df=self.df_price,
            initial_capital=self.initial_capital,
            commission_pct=self.commission_pct,
            default_qty=self.default_qty,
            pyramiding=self.pyramiding,
            slippage=self.slippage,
            mintick=self.mintick,
            margin_long=self.margin_long,
            margin_short=self.margin_short,
            input_overrides={},
        )
        interp.execute()

        params = []
        for idef in interp.input_defs:
            p = InputParam(
                name=idef.get('key') or idef.get('title', ''),
                param_type=idef.get('type', 'float'),
                default=idef.get('defval'),
                min_val=idef.get('minval'),
                max_val=idef.get('maxval'),
                step=idef.get('step'),
                options=idef.get('options'),
                enabled=False,  # Default OFF — user selects which to optimize
            )
            p.generate_values()
            params.append(p)

        return params

    @staticmethod
    def estimate_combinations(params: List[InputParam]) -> int:
        """Count total parameter combinations for enabled params."""
        total = 1
        for p in params:
            if p.enabled and p.values:
                total *= len(p.values)
        return total

    def _run_single(self, input_overrides: dict,
                    PineInterpreterClass) -> OptimizationResult:
        """Execute strategy with given overrides and compute metrics."""
        interp = PineInterpreterClass(
            ast=self.ast,
            df=self.df_price,
            initial_capital=self.initial_capital,
            commission_pct=self.commission_pct,
            default_qty=self.default_qty,
            pyramiding=self.pyramiding,
            slippage=self.slippage,
            mintick=self.mintick,
            margin_long=self.margin_long,
            margin_short=self.margin_short,
            input_overrides=input_overrides,
        )
        trades = interp.execute()
        return self._compute_metrics(input_overrides, trades)

    def _compute_metrics(self, params: dict, trades: list) -> OptimizationResult:
        """Fast metric computation — avoids StrategyAnalytics overhead."""
        n = len(trades)
        if n == 0:
            return OptimizationResult(params=dict(params), num_trades=0)

        profits = np.array([t['Profit'] for t in trades], dtype=float)
        net_profit = float(profits.sum())
        equity = np.cumsum(profits) + self.initial_capital
        total_return_pct = (equity[-1] / self.initial_capital - 1) * 100

        # Win / loss splits
        wins = profits[profits > 0]
        losses = profits[profits < 0]
        win_rate = len(wins) / n if n > 0 else 0.0

        # Profit factor
        gross_profit = wins.sum() if len(wins) > 0 else 0.0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (
            float('inf') if gross_profit > 0 else 0.0
        )

        # Per-trade returns (relative to equity before each trade)
        equity_before = np.concatenate([[self.initial_capital], equity[:-1]])
        safe_eb = np.where(equity_before != 0, equity_before, 1.0)
        returns = profits / safe_eb

        # Sharpe ratio (annualized, approximate with 252 trading days)
        mean_r = returns.mean()
        std_r = returns.std(ddof=1) if n > 1 else 0.0
        sharpe_ratio = (mean_r / std_r * math.sqrt(252)) if std_r > 1e-12 else 0.0

        # Sortino ratio
        downside = returns[returns < 0]
        if len(downside) > 0:
            downside_std = math.sqrt(np.mean(downside ** 2))
            sortino_ratio = (mean_r / downside_std * math.sqrt(252)) if downside_std > 1e-12 else float('inf')
        else:
            sortino_ratio = float('inf') if mean_r > 0 else 0.0

        # Max drawdown %
        peak = np.maximum.accumulate(equity)
        dd_pct = np.where(peak > 0, (equity - peak) / peak, 0.0)
        max_drawdown_pct = float(dd_pct.min()) * 100  # negative number

        # Expectancy and avg trade
        avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
        avg_loss = float(abs(losses.mean())) if len(losses) > 0 else 0.0
        loss_rate = len(losses) / n if n > 0 else 0.0
        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        avg_trade = float(profits.mean())

        # SQN = sqrt(N) * mean / std
        std_profit = float(profits.std(ddof=1)) if n > 1 else 0.0
        sqn = (math.sqrt(n) * avg_trade / std_profit) if std_profit > 1e-12 else 0.0

        # CAGR for Calmar
        total_ret = equity[-1] / self.initial_capital
        # Estimate years from bar count (approximate)
        n_bars = len(self.df_price)
        years = n_bars / 252 if n_bars > 0 else 1.0
        if total_ret > 0 and years > 0:
            cagr = (total_ret ** (1 / years) - 1) * 100
        else:
            cagr = 0.0
        calmar_ratio = (cagr / abs(max_drawdown_pct)) if abs(max_drawdown_pct) > 1e-12 else (
            float('inf') if cagr > 0 else 0.0
        )

        return OptimizationResult(
            params=dict(params),
            net_profit=round(net_profit, 2),
            total_return_pct=round(total_return_pct, 2),
            num_trades=n,
            win_rate=round(win_rate, 4),
            profit_factor=round(profit_factor, 4) if profit_factor != float('inf') else float('inf'),
            sharpe_ratio=round(sharpe_ratio, 4),
            sortino_ratio=round(sortino_ratio, 4) if sortino_ratio != float('inf') else float('inf'),
            max_drawdown_pct=round(max_drawdown_pct, 2),
            expectancy=round(expectancy, 2),
            calmar_ratio=round(calmar_ratio, 4) if calmar_ratio != float('inf') else float('inf'),
            sqn=round(sqn, 4),
            avg_trade=round(avg_trade, 2),
            trades=None,  # Don't store trades by default to save memory
        )

    def optimize(self, params: List[InputParam], PineInterpreterClass,
                 sort_by: str = 'net_profit', top_n: int = 10,
                 min_trades: int = 5,
                 progress_callback: Optional[Callable[[int, int], None]] = None
                 ) -> List[OptimizationResult]:
        """
        Grid search over all enabled parameter combinations.
        Returns top_n results sorted by sort_by metric.
        """
        # Build lists of (name, values) for enabled params only
        active_params = [(p.name, p.values) for p in params if p.enabled and p.values]
        if not active_params:
            raise ValueError("No enabled parameters with values to optimize.")

        names = [name for name, _ in active_params]
        value_lists = [vals for _, vals in active_params]

        total = 1
        for vals in value_lists:
            total *= len(vals)

        if total > 50_000:
            raise ValueError(
                f"Too many combinations ({total:,}). Maximum is 50,000. "
                f"Narrow parameter ranges or increase step sizes. "
                f"Current params: {', '.join(f'{n}={len(v)} values' for n, v in active_params)}"
            )

        results: List[OptimizationResult] = []
        t0 = time.time()

        for i, combo in enumerate(itertools.product(*value_lists)):
            overrides = dict(zip(names, combo))

            try:
                result = self._run_single(overrides, PineInterpreterClass)
                if result.num_trades >= min_trades:
                    results.append(result)
            except Exception:
                # Skip combinations that cause interpreter errors
                pass

            if progress_callback is not None:
                progress_callback(i + 1, total)

        # Sort results
        # For max_drawdown_pct, less negative is better, so sort ascending (reverse=False)
        # For all other metrics, higher is better, so sort descending (reverse=True)
        if sort_by == 'max_drawdown_pct':
            # Less negative = better, so sort descending (closest to 0 first)
            results.sort(key=lambda r: getattr(r, sort_by, 0), reverse=True)
        else:
            results.sort(
                key=lambda r: (
                    getattr(r, sort_by, 0)
                    if getattr(r, sort_by, 0) != float('inf')
                    else 1e18
                ),
                reverse=True,
            )

        return results[:top_n]

    def walk_forward(self, params: dict, PineInterpreterClass,
                     n_splits: int = 5) -> dict:
        """
        Walk-forward validation for a single parameter set.
        Splits data into n_splits windows, trains on 70%, tests on 30%.
        Returns per-split results and summary stats.
        """
        n_bars = len(self.df_price)
        if n_bars < n_splits * 10:
            raise ValueError(
                f"Not enough bars ({n_bars}) for {n_splits} walk-forward splits. "
                f"Need at least {n_splits * 10} bars."
            )

        split_size = n_bars // n_splits
        splits = []

        for i in range(n_splits):
            start = i * split_size
            end = start + split_size if i < n_splits - 1 else n_bars

            window = self.df_price.iloc[start:end].copy().reset_index(drop=True)
            train_end = int(len(window) * 0.7)

            # Test on the last 30%
            test_df = window.iloc[train_end:].copy().reset_index(drop=True)

            if len(test_df) < 5:
                continue

            # Run strategy on test period with the given params
            interp = PineInterpreterClass(
                ast=self.ast,
                df=test_df,
                initial_capital=self.initial_capital,
                commission_pct=self.commission_pct,
                default_qty=self.default_qty,
                pyramiding=self.pyramiding,
                slippage=self.slippage,
                mintick=self.mintick,
                margin_long=self.margin_long,
                margin_short=self.margin_short,
                input_overrides=params,
            )
            trades = interp.execute()

            profits = [t['Profit'] for t in trades]
            split_profit = sum(profits)
            split_trades = len(trades)

            # Per-split Sharpe
            if split_trades > 1:
                p_arr = np.array(profits)
                eq = np.cumsum(p_arr) + self.initial_capital
                eq_before = np.concatenate([[self.initial_capital], eq[:-1]])
                safe_eb = np.where(eq_before != 0, eq_before, 1.0)
                rets = p_arr / safe_eb
                std_r = rets.std(ddof=1)
                split_sharpe = float(rets.mean() / std_r * math.sqrt(252)) if std_r > 1e-12 else 0.0
            else:
                split_sharpe = 0.0

            splits.append({
                'split': i + 1,
                'test_bars': len(test_df),
                'num_trades': split_trades,
                'net_profit': round(split_profit, 2),
                'sharpe': round(split_sharpe, 4),
                'profitable': split_profit > 0,
            })

        if not splits:
            return {
                'splits': [],
                'avg_profit': 0.0,
                'avg_sharpe': 0.0,
                'consistency_score': 0.0,
            }

        avg_profit = round(sum(s['net_profit'] for s in splits) / len(splits), 2)
        avg_sharpe = round(sum(s['sharpe'] for s in splits) / len(splits), 4)
        profitable_splits = sum(1 for s in splits if s['profitable'])
        consistency_score = round(profitable_splits / len(splits) * 100, 1)

        return {
            'splits': splits,
            'avg_profit': avg_profit,
            'avg_sharpe': avg_sharpe,
            'consistency_score': consistency_score,
        }

    @staticmethod
    def to_dataframe(results: List[OptimizationResult]) -> pd.DataFrame:
        """Convert results list to a ranked DataFrame."""
        if not results:
            return pd.DataFrame()

        rows = []
        # Collect all param names across results
        all_param_names = []
        seen_names = set()
        for r in results:
            for k in r.params:
                if k not in seen_names:
                    seen_names.add(k)
                    all_param_names.append(k)

        for rank, r in enumerate(results, 1):
            row = {
                'Rank': rank,
                'Net Profit': r.net_profit,
                'Return %': r.total_return_pct,
                'Trades': r.num_trades,
                'Win Rate': r.win_rate,
                'Profit Factor': r.profit_factor,
                'Sharpe': r.sharpe_ratio,
                'Sortino': r.sortino_ratio,
                'Max DD%': r.max_drawdown_pct,
                'Expectancy': r.expectancy,
                'SQN': r.sqn,
                'Calmar': r.calmar_ratio,
                'Avg Trade': r.avg_trade,
            }
            for pname in all_param_names:
                row[pname] = r.params.get(pname, '')
            rows.append(row)

        return pd.DataFrame(rows)
