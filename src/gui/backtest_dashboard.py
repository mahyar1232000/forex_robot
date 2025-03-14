import tkinter as tk
from tkinter import ttk
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class BacktestDashboard(tk.Tk):
    def __init__(self, results):
        super().__init__()
        self.title("Backtest Results")
        self.geometry("1400x900")
        self.results = results
        self.create_widgets()

    def create_widgets(self):
        ttk.Label(self, text="Optimization History").grid(row=0, column=0)
        self.optimization_table = ttk.Treeview(self, columns=('Parameters', 'Performance'))
        self.optimization_table.heading('#0', text='Run')
        self.optimization_table.heading('Parameters', text='Parameters')
        self.optimization_table.heading('Performance', text='Performance')
        self.optimization_table.grid(row=1, column=0, padx=10, pady=10)
        self.populate_optimization_table()

        self.figure = plt.Figure(figsize=(10, 4))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, self)
        self.canvas.get_tk_widget().grid(row=0, column=1, rowspan=3, padx=10, pady=10)
        self.draw_equity_curve()

        ttk.Label(self, text="Key Metrics").grid(row=2, column=0)
        self.metrics_frame = ttk.Frame(self)
        self.metrics_frame.grid(row=3, column=0, padx=10, pady=10)
        self.populate_metrics()

    def populate_optimization_table(self):
        history = self.results.get('optimization_history', [])
        for idx, (params, perf) in enumerate(history):
            self.optimization_table.insert('', 'end', text=f"Run {idx + 1}", values=(str(params), f"{perf:.2f}"))

    def draw_equity_curve(self):
        eq = self.results.get('equity_curve', [])
        self.ax.clear()
        self.ax.plot(eq, label='Equity Curve')
        self.ax.set_title("Equity Curve")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Balance")
        self.ax.legend()
        self.canvas.draw()

    def populate_metrics(self):
        metrics = [
            ('Final Balance', self.results.get('final_balance', 0.0)),
            ('Sharpe Ratio', self.results.get('sharpe_ratio', 0.0)),
            ('Max Drawdown', self.results.get('max_drawdown', 0.0)),
            ('Win Rate', self.results.get('win_rate', 0.0))
        ]
        for idx, (label, value) in enumerate(metrics):
            ttk.Label(self.metrics_frame, text=label).grid(row=idx, column=0, sticky='w')
            ttk.Label(self.metrics_frame, text=str(value)).grid(row=idx, column=1, sticky='e')
