class TradingCircuitBreaker:
    def __init__(self, max_drawdown=0.15, max_loss_streak=5):
        self.max_drawdown = max_drawdown
        self.max_loss_streak = max_loss_streak
        self.reset()

    def reset(self):
        self.drawdown = 0.0
        self.loss_streak = 0
        self.peak_balance = 0.0

    def check_balance(self, current_balance):
        self.peak_balance = max(self.peak_balance, current_balance)
        self.drawdown = (self.peak_balance - current_balance) / self.peak_balance

        if self.drawdown > self.max_drawdown:
            raise TradingHaltedError(f"Max drawdown {self.max_drawdown * 100}% exceeded")

    def update_trade_result(self, profit):
        if profit < 0:
            self.loss_streak += 1
            if self.loss_streak >= self.max_loss_streak:
                raise TradingHaltedError(f"{self.max_loss_streak} consecutive losses")
        else:
            self.loss_streak = 0
