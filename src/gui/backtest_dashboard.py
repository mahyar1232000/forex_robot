"""
backtest_dashboard.py
=====================
Provides a graphical dashboard for backtest/optimization mode using Tkinter.
Displays metrics such as initial/final balance, total profit, number of trades, win rate, etc.
"""

import matplotlib.pyplot as plt


def display_backtest_results(results):
    trades = results.get("trades", [])
    balances = [trade["balance"] for trade in trades]
    plt.plot(balances)
    plt.title("Backtest Performance")
    plt.xlabel("Trade Number")
    plt.ylabel("Balance")
    plt.show()


def main():
    results = {"trades": [{"balance": 10000}, {"balance": 10010}, {"balance": 10000}]}
    display_backtest_results(results)


if __name__ == '__main__':
    main()
