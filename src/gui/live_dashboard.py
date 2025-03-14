import tkinter as tk
from tkinter import ttk
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class LiveDashboard(tk.Tk):
    def __init__(self, trading_bot):
        super().__init__()
        self.bot = trading_bot
        self.title("Live Trading Dashboard")
        self.geometry("1200x800")
        self._create_widgets()
        self._schedule_updates()

    def _create_widgets(self):
        self.metrics_frame = ttk.LabelFrame(self, text="Live Metrics")
        self.metrics_frame.pack(fill=tk.X, padx=10, pady=10)
        metrics = ['balance', 'equity', 'trades', 'win_rate']
        self.labels = {}
        for metric in metrics:
            frame = ttk.Frame(self.metrics_frame)
            frame.pack(fill=tk.X, padx=5, pady=2)
            label = ttk.Label(frame, text=metric.replace('_', ' ').title(), width=15)
            label.pack(side=tk.LEFT)
            value = ttk.Label(frame, text="0.00", width=10)
            value.pack(side=tk.LEFT)
            self.labels[metric] = value

        self.figure = plt.Figure(figsize=(10, 4))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _schedule_updates(self):
        self._update_metrics()
        self.after(5000, self._schedule_updates)

    def _update_metrics(self):
        self.bot.run_strategy_cycle()
        acct = self.bot.get_account_info()
        balance = acct.get('balance', 0.0)
        equity = acct.get('equity', balance)
        self.labels['balance'].config(text=f"{balance:.2f}")
        self.labels['equity'].config(text=f"{equity:.2f}")
        trades = self.bot.performance_monitor.total_trades
        win_rate = self.bot.performance_monitor.win_rate
        self.labels['trades'].config(text=str(trades))
        self.labels['win_rate'].config(text=f"{win_rate:.2f}")

        self.ax.clear()
        self.ax.plot(self.bot.equity_curve, label='Equity Curve')
        self.ax.set_title("Equity Curve")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Balance")
        self.ax.legend()
        self.canvas.draw()
