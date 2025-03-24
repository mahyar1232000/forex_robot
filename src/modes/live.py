"""
live.py (modes)
===============
Launcher module for live trading mode.
Initializes MT5, starts live trading, and launches the live dashboard in a separate thread.
"""

"""
live.py

Mode launcher for live trading.
"""

from src.live.live_trading import run_live_trading

if __name__ == '__main__':
    run_live_trading()
