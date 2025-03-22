"""
live_dashboard.py
=================
Provides a graphical dashboard for live trading using Tkinter.
Displays real-time metrics fetched from MT5 via MT5Broker.
"""

import time
from datetime import datetime
from typing import Dict, List
from src.core.broker import MT5Broker
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LiveDashboard:
    def __init__(self, broker: MT5Broker, refresh_interval: int = 5):
        self.broker = broker
        self.refresh_interval = refresh_interval
        self.metrics: Dict[str, Dict] = {}
        self.running = False
        self.start_time = datetime.now()

    def start(self):
        self.running = True
        logger.info("Starting live dashboard...")
        try:
            while self.running:
                self.update()
                self.display()
                time.sleep(self.refresh_interval)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.running = False
        logger.info("Dashboard stopped")

    def update(self):
        try:
            account_info = self.broker.get_account_info()
            positions = self.broker.get_open_positions()
            trade_history = self.broker.get_trade_history()
            self.metrics = {
                'account': {
                    'balance': account_info.get('balance', 0),
                    'equity': account_info.get('equity', 0),
                    'margin': account_info.get('margin', 0)
                },
                'positions': {
                    'count': len(positions),
                    'exposure': sum(p.get('volume', 0) for p in positions)
                },
                'performance': {
                    'total_trades': len(trade_history),
                    'profit_loss': sum(t.get('profit', 0) for t in trade_history),
                    'win_rate': self._calculate_win_rate(trade_history)
                },
                'system': {
                    'uptime': datetime.now() - self.start_time,
                    'refresh_rate': self.refresh_interval
                }
            }
        except Exception as e:
            logger.error(f"Dashboard update failed: {str(e)}")

    def display(self):
        print("\n" + "=" * 60)
        print(f"{' LIVE TRADING DASHBOARD ':=^60}")
        print(f"{'Uptime:':<15} {self.metrics.get('system', {}).get('uptime', 'N/A')}")
        print(f"{'Refresh Rate:':<15} {self.metrics.get('system', {}).get('refresh_rate', 'N/A')}s\n")
        print(f"{'Account Balance:':<25} ${self.metrics.get('account', {}).get('balance', 0):,.2f}")
        print(f"{'Account Equity:':<25} ${self.metrics.get('account', {}).get('equity', 0):,.2f}")
        print(f"{'Used Margin:':<25} ${self.metrics.get('account', {}).get('margin', 0):,.2f}\n")
        print(f"{'Open Positions:':<25} {self.metrics.get('positions', {}).get('count', 0)}")
        print(f"{'Market Exposure:':<25} {self.metrics.get('positions', {}).get('exposure', 0):.2f} lots\n")
        print(f"{'Total Trades:':<25} {self.metrics.get('performance', {}).get('total_trades', 0)}")
        print(f"{'Total P/L:':<25} ${self.metrics.get('performance', {}).get('profit_loss', 0):,.2f}")
        print(f"{'Win Rate:':<25} {self.metrics.get('performance', {}).get('win_rate', 0):.2f}%")
        print("=" * 60 + "\n")

    def _calculate_win_rate(self, trades: List[Dict]) -> float:
        if not trades:
            return 0.0
        wins = len([t for t in trades if t.get('profit', 0) > 0])
        return (wins / len(trades)) * 100


def start_live_dashboard(broker: MT5Broker, refresh_interval: int = 5):
    dashboard = LiveDashboard(broker, refresh_interval)
    dashboard.start()


if __name__ == "__main__":
    from src.core.broker import initialize_broker

    broker = initialize_broker()
    start_live_dashboard(broker)
