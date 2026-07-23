"""
==============================================================================
BorsaNeuron - Advanced Risk & Portfolio Performance Analytics Engine
Author: İbrahim Tatar
Description: Calculates institutional risk metrics (Sharpe, Sortino, Max Drawdown,
             Value at Risk (VaR), Calmar Ratio) and Monte Carlo simulations.
==============================================================================
"""

import numpy as np
import pandas as pd

class BorsaNeuronRiskEngine:
    def __init__(self, risk_free_rate=0.45):
        """
        Default risk-free rate set to 45% (annualized rate baseline).
        """
        self.rf_daily = (1 + risk_free_rate) ** (1 / 252) - 1

    def calculate_portfolio_metrics(self, equity_curve):
        """
        Computes comprehensive performance and risk metrics for a backtested equity curve.
        """
        if isinstance(equity_curve, (list, np.ndarray)):
            equity = pd.Series(equity_curve)
        else:
            equity = equity_curve.copy()

        daily_returns = equity.pct_change().dropna()
        
        # 1. Cumulative & Annualized Return
        total_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]
        n_days = len(equity)
        annualized_return = ((1 + total_return) ** (252 / max(n_days, 1))) - 1

        # 2. Annualized Volatility
        daily_std = daily_returns.std()
        annualized_volatility = daily_std * np.sqrt(252)

        # 3. Sharpe Ratio
        excess_returns = daily_returns - self.rf_daily
        sharpe_ratio = (excess_returns.mean() / (daily_std + 1e-8)) * np.sqrt(252)

        # 4. Sortino Ratio (Downside Risk)
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = downside_returns.std()
        sortino_ratio = (excess_returns.mean() / (downside_std + 1e-8)) * np.sqrt(252)

        # 5. Maximum Drawdown (MDD)
        rolling_max = equity.cummax()
        drawdowns = (equity - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()

        # 6. Calmar Ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

        # 7. Value at Risk (VaR 95% and 99%)
        var_95 = np.percentile(daily_returns, 5)
        var_99 = np.percentile(daily_returns, 1)

        metrics = {
            'initial_capital': float(equity.iloc[0]),
            'final_capital': float(equity.iloc[-1]),
            'total_return_pct': float(total_return * 100),
            'annualized_return_pct': float(annualized_return * 100),
            'annualized_volatility_pct': float(annualized_volatility * 100),
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'max_drawdown_pct': float(max_drawdown * 100),
            'calmar_ratio': float(calmar_ratio),
            'var_95_daily_pct': float(var_95 * 100),
            'var_99_daily_pct': float(var_99 * 100),
            'win_rate_pct': float((daily_returns > 0).mean() * 100)
        }

        return metrics, drawdowns

    def run_monte_carlo_simulation(self, current_capital, mean_daily_ret, std_daily_ret, days=30, simulations=500):
        """
        Runs Monte Carlo simulation for future portfolio path projections.
        """
        simulation_matrix = np.zeros((days, simulations))
        simulation_matrix[0] = current_capital
        
        for t in range(1, days):
            random_shocks = np.random.normal(mean_daily_ret, std_daily_ret, simulations)
            simulation_matrix[t] = simulation_matrix[t-1] * (1 + random_shocks)

        p10 = np.percentile(simulation_matrix[-1], 10)
        p50 = np.percentile(simulation_matrix[-1], 50)
        p90 = np.percentile(simulation_matrix[-1], 90)

        return {
            'matrix': simulation_matrix,
            'p10_pessimistic': float(p10),
            'p50_median': float(p50),
            'p90_optimistic': float(p90)
        }

if __name__ == "__main__":
    # Quick self-test
    curve = 100000 * (1 + np.random.normal(0.001, 0.015, 252)).cumprod()
    engine = BorsaNeuronRiskEngine()
    results, _ = engine.calculate_portfolio_metrics(curve)
    print("Risk Engine Initialized & Validated:", results)
