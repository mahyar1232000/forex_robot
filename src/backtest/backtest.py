import pandas as pd
from src.core.DataManager import DataManager
from src.core.strategy import Strategy
from src.core.PerformanceMonitor import PerformanceMonitor
from config.settings import INITIAL_BALANCE
from src.utils.logger import setup_logger
from src.core.AdvancedTradingBot import AdvancedTradingBot

logger = setup_logger()


def run_backtest():
    logger.info("Starting backtest mode.")
    bot = AdvancedTradingBot()
    bot.config['backtest_mode'] = True
    bot.reset_backtest()
    # Load historical data from DataManager
    dm = DataManager()
    bot.historical_data = dm.load_historical_data()
    if bot.historical_data.empty:
        logger.error("No historical data available for backtesting.")
        return
    bot.run_backtest()
    results = {
        "final_balance": bot.account_balance,
        "trade_history": bot.trade_history
    }
    logger.info("Backtest completed.")
    return results


def main():
    results = run_backtest()
    print("Backtest Results:")
    print(results)


if __name__ == '__main__':
    main()
