import logging


class PerformanceMonitor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.total_trades = 0
        self.winning_trades = 0

    def update(self, trade_result):
        self.total_trades += 1
        if trade_result.get('profit', 0.0) > 0:
            self.winning_trades += 1

    @property
    def win_rate(self):
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    def check_performance(self):
        if self.total_trades > 10 and self.win_rate < 0.4:
            return True
        return False
