import time
import logging
from threading import Thread
from src.core.DataManager import DataManager
from src.core.LSTMTradingStrategy import LSTMTradingStrategy
from src.core.AdvancedRiskManager import AdvancedRiskManager
from src.core.PerformanceMonitor import PerformanceMonitor
from src.core.Broker import Broker
from src.core.WalkForwardOptimizer import WalkForwardOptimizer
from src.utils.config import ConfigManager


class LiveTrader:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        # Initialize components
        self.data_manager = DataManager(mode='live')
        self.risk_manager = AdvancedRiskManager()
        self.strategy = LSTMTradingStrategy(config['strategy']['parameters'])
        self.optimizer = WalkForwardOptimizer()
        self.performance_monitor = PerformanceMonitor()
        self.broker = Broker()

        self.running = True
        self.optimization_thread = Thread(target=self._periodic_optimization, daemon=True)
        self.optimization_thread.start()

        acct_info = self.get_account_info()
        self.balance = acct_info.get('balance', 10000.0)
        self.equity_curve = [self.balance]

    def get_account_info(self):
        return self.broker.get_account_info()

    def _periodic_optimization(self):
        while self.running:
            time.sleep(86400)  # every 24 hours
            recent_data = self.data_manager.get_historical_data()
            best_params = self.optimizer.optimize(recent_data)
            self.strategy = LSTMTradingStrategy(best_params)
            ConfigManager().update_live_config(best_params)
            self.logger.info("Periodic optimization completed.")

    def get_market_price(self):
        price = self.data_manager.get_latest_price()
        if price is None:
            raise ValueError("No market price available")
        return price

    def calculate_take_profit(self, current_price, stop_loss):
        risk = abs(current_price - stop_loss)
        risk_reward_ratio = self.config.get('risk_reward_ratio', 2)
        return current_price + risk * risk_reward_ratio

    def log_trade(self, result):
        self.logger.info(f"Trade executed: {result}")

    def run_strategy_cycle(self):
        current_price = self.get_market_price()
        signal = self.strategy.generate_signal(current_price)
        if signal is not None:
            self.execute_trade(signal)

    def execute_trade(self, signal):
        try:
            current_price = self.get_market_price()
            stop_loss = self.risk_manager.calculate_stop_loss(current_price)
            take_profit = self.calculate_take_profit(current_price, stop_loss)
            trade_details = {
                'signal': signal,
                'price': current_price,
                'size': self.risk_manager.calculate_position_size(self.balance,
                                                                  self.config['risk_per_trade'],
                                                                  abs(current_price - stop_loss)),
                'stop_loss': stop_loss,
                'take_profit': take_profit
            }
            result = self.broker.execute_order(trade_details)
            self.performance_monitor.update(result)
            self.log_trade(result)
            acct_info = self.get_account_info()
            self.balance = acct_info.get('balance', self.balance)
            self.equity_curve.append(self.balance)
            if self.performance_monitor.check_performance():
                self._trigger_optimization()
        except Exception as e:
            self.logger.error(f"Trade execution failed: {str(e)}")

    def _trigger_optimization(self):
        recent_data = self.data_manager.get_historical_data()
        best_params = self.optimizer.optimize(recent_data)
        self.strategy = LSTMTradingStrategy(best_params)
        ConfigManager().update_live_config(best_params)
        self.logger.info("Manual optimization triggered.")

    def shutdown(self):
        self.running = False
        self.optimization_thread.join()
        self.logger.info("Live trader shutdown.")
