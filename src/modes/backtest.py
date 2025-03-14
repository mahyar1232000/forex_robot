import MetaTrader5 as mt5
import datetime
import numpy as np
import logging
from src.utils.config import ConfigManager
from src.core.LSTMTradingStrategy import LSTMTradingStrategy
from src.core.AdvancedRiskManager import AdvancedRiskManager


class Backtester:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        self.config = ConfigManager().load_config('backtest')
        self.initialize_mt5()

    def initialize_mt5(self):
        if not mt5.initialize():
            raise RuntimeError("Failed to initialize MT5")
        symbol = self.config['symbol']
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"Failed to select symbol: {symbol}")

    def optimize_parameters(self):
        settings = {
            'symbol': self.config['symbol'],
            'timeframe': self._get_timeframe(),
            'start_date': self.config['start_date'],
            'end_date': self.config['end_date'],
        }
        results = self.run_backtest(settings)
        return results

    def run_backtest(self, settings):
        start_dt = datetime.datetime.strptime(settings['start_date'], '%Y-%m-%d')
        end_dt = datetime.datetime.strptime(settings['end_date'], '%Y-%m-%d')
        rates = mt5.copy_rates_range(settings['symbol'], settings['timeframe'], start_dt, end_dt)
        if rates is None or len(rates) == 0:
            raise RuntimeError("No historical data retrieved")
        balance = 10000.0
        equity_curve = [balance]
        open_position = None
        risk_manager = AdvancedRiskManager()
        strategy = LSTMTradingStrategy(self.config['strategy']['parameters'])
        total_trades = 0
        winning_trades = 0

        for i in range(len(rates)):
            bar = rates[i]
            current_price = bar['close']

            if open_position is not None:
                if (open_position['signal'] == 'buy' and current_price <= open_position['stop_loss']) or \
                        (open_position['signal'] == 'sell' and current_price >= open_position['stop_loss']):
                    pnl = (open_position['signal_mult'] * (open_position['stop_loss'] - open_position['entry'])) * \
                          open_position['size']
                    balance += pnl
                    open_position = None
                    total_trades += 1
                elif (open_position['signal'] == 'buy' and current_price >= open_position['take_profit']) or \
                        (open_position['signal'] == 'sell' and current_price <= open_position['take_profit']):
                    pnl = (open_position['signal_mult'] * (open_position['take_profit'] - open_position['entry'])) * \
                          open_position['size']
                    balance += pnl
                    open_position = None
                    total_trades += 1
                    if pnl > 0:
                        winning_trades += 1

            if open_position is None:
                signal = strategy.generate_signal(current_price)
                if signal in ['buy', 'sell']:
                    stop_loss = risk_manager.calculate_stop_loss(current_price)
                    take_profit = risk_manager.calculate_take_profit(current_price, stop_loss)
                    signal_mult = 1 if signal == 'buy' else -1
                    size = risk_manager.calculate_position_size(balance, self.config['risk_per_trade'],
                                                                abs(current_price - stop_loss))
                    open_position = {
                        'signal': signal,
                        'signal_mult': signal_mult,
                        'entry': current_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'size': size
                    }
            equity_curve.append(balance)

        sharpe_ratio = self._calculate_sharpe(equity_curve)
        max_drawdown = self._calculate_drawdown(equity_curve)
        win_rate = (winning_trades / total_trades) if total_trades > 0 else 0.0

        results = {
            'optimization_history': [(self.config['strategy']['parameters'], balance)],
            'equity_curve': equity_curve,
            'final_balance': balance,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate
        }
        return results

    def _get_timeframe(self):
        return getattr(mt5, f"TIMEFRAME_{self.config['timeframe']}")

    def _calculate_sharpe(self, equity_curve):
        returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
        if np.std(returns) == 0:
            return 0.0
        return np.mean(returns) / np.std(returns)

    def _calculate_drawdown(self, equity_curve):
        equity = np.array(equity_curve)
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak)
        return float(np.min(drawdown))
