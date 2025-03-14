import numpy as np


class AdvancedRiskManager:
    def calculate_stop_loss(self, data, current_price):
        atr = self.calculate_atr(data, period=14)
        return current_price - 2 * atr

    def calculate_position_size(self, balance, risk_pct, stop_loss_pips):
        risk_amount = balance * risk_pct
        return risk_amount / (stop_loss_pips * 10)  # PIP value for FX

    def calculate_atr(self, data, period=14):
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        tr = np.max(np.stack([high_low, high_close, low_close]), axis=0)
        return tr.rolling(period).mean().iloc[-1]
