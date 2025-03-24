"""
backtest.py (modes)
===================
Launcher module for backtest mode.
Creates an instance of AdvancedTradingBot, runs the backtest simulation in a separate thread,
and launches the backtest dashboard to display updated metrics dynamically.
"""

"""
backtest.py

Mode launcher for backtesting.
"""

from src.backtest.backtest import run_backtest

if __name__ == '__main__':
    run_backtest()
