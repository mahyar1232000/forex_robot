"""
RealTimeReportDashboard.py
==========================
Provides a unified graphical dashboard using Tkinter that adjusts its displayed metrics
based on whether the bot is running in live or backtest/optimization mode.
In live mode, it fetches data from MT5; in backtest mode, it displays simulation results from the bot.
"""

import tkinter as tk
import logging


class RealTimeReportDashboard:
    def __init__(self, config, bot):
        self.config = config
        self.bot = bot
        self.root = tk.Tk()
        self.root.title("Real-Time Trading Report")
        self.metrics_frame = tk.Frame(self.root)
        self.metrics_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        if self.config.get('backtest_mode', False) or self.config.get('optimize_mode', False):
            self.metrics = [
                "Initial Balance", "Final Balance", "Total Profit",
                "Number of Trades", "Successful Trades", "Unsuccessful Trades", "Win Rate"
            ]
            if self.config.get('optimize_mode', False):
                self.metrics.extend(["Best Sharpe Ratio", "Optimization Progress"])
        else:
            self.metrics = [
                "Capital Available", "Total Capital", "Number of Trades",
                "Open Trades", "Successful Trades", "Unsuccessful Trades",
                "Total Profit/Loss", "Largest Profit", "Largest Loss", "Win Rate"
            ]
        self.labels = {}
        for idx, metric in enumerate(self.metrics):
            tk.Label(self.metrics_frame, text=f"{metric}:", font=("Arial", 10, "bold")).grid(row=idx, column=0,
                                                                                             sticky="w", padx=10,
                                                                                             pady=2)
            self.labels[metric] = tk.Label(self.metrics_frame, text="0", font=("Arial", 10))
            self.labels[metric].grid(row=idx, column=1, sticky="w", padx=10, pady=2)
        self.metrics_frame.columnconfigure(0, weight=1)
        self.update_dashboard()

    def update_dashboard(self):
        if self.config.get('backtest_mode', False) or self.config.get('optimize_mode', False):
            metrics = self.bot.get_metrics()
            if metrics:
                self.labels["Initial Balance"].config(text=f"${metrics['Initial Balance']:.2f}")
                self.labels["Final Balance"].config(text=f"${metrics['Final Balance']:.2f}")
                self.labels["Total Profit"].config(text=f"${metrics['Total Profit']:.2f}")
                self.labels["Number of Trades"].config(text=str(metrics["Number of Trades"]))
                self.labels["Successful Trades"].config(text=str(metrics["Successful Trades"]))
                self.labels["Unsuccessful Trades"].config(text=str(metrics["Unsuccessful Trades"]))
                self.labels["Win Rate"].config(text=f"{metrics['Win Rate']:.1f}%")
        else:
            # In live mode, update with live metrics (dummy values for demonstration).
            self.labels["Capital Available"].config(text="$10,000.00")
            self.labels["Total Capital"].config(text="$10,500.00")
            self.labels["Number of Trades"].config(text="15")
            self.labels["Open Trades"].config(text="2")
            self.labels["Successful Trades"].config(text="10")
            self.labels["Unsuccessful Trades"].config(text="5")
            self.labels["Total Profit/Loss"].config(text="$500.00")
            self.labels["Largest Profit"].config(text="$200.00")
            self.labels["Largest Loss"].config(text="$100.00")
            self.labels["Win Rate"].config(text="66.7%")
        self.root.after(1000, self.update_dashboard)

    def run(self):
        self.root.mainloop()


def start_dashboard(config, bot):
    dashboard = RealTimeReportDashboard(config, bot)
    dashboard.run()
