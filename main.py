"""
Entry point for the AI Forex Trading Robot.
Usage: python main.py --mode [live|backtest|train|optimize]
"""

import argparse
import time
from src.utils.logger import setup_logger
import logging

setup_logger()
logger = logging.getLogger(__name__)

from src.core.AdvancedTradingBot import AdvancedTradingBot
from src.optimizer.optimizer import optimize_trading_parameters
from src.core.LSTMTradingStrategy import LSTMTradingStrategy
from config.settings import TIME_SLEEP


def main():
    parser = argparse.ArgumentParser(description="AI Forex Trading Robot")
    parser.add_argument("--mode", choices=["live", "backtest", "train", "optimize"], required=True,
                        help="Select mode: live, backtest, train, or optimize")
    parser.add_argument("--data", type=str, default="data/historical.csv",
                        help="Path to historical data (used in backtest and train)")
    parser.add_argument("--model", type=str, default="models/lstm_model.h5",
                        help="Path to save/load the LSTM model")
    args = parser.parse_args()

    if args.mode == "train":
        strategy = LSTMTradingStrategy(args.data, args.model)
        summary = strategy.execute_strategy()
        logger.info("Model training completed. Model summary: %s", summary)
        print("Model training completed. Model summary:")
        print(summary)
        print("Now entering live trading mode...")
        args.mode = "live"

    if args.mode == "optimize":
        best_params = optimize_trading_parameters()
        print("Optimization completed. Best parameters:")
        print(best_params)
        return

    if args.mode == "live":
        bot = AdvancedTradingBot()
        logger.info("Starting live trading. Press Ctrl+C to stop.")
        while True:
            bot.execute_trade(bot.analyze_market(bot.get_market_data()))
            time.sleep(TIME_SLEEP * 60)

    elif args.mode == "backtest":
        bot = AdvancedTradingBot({"backtest_mode": True})
        bot.run_backtest()


if __name__ == "__main__":
    main()
