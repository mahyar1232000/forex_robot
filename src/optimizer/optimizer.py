"""
optimizer.py

Runs parameter optimization for AdvancedTradingBot based on the grid defined in config/settings.py.
"""

from config import settings
from src.core.AdvancedTradingBot import AdvancedTradingBot
from src.utils.logger import get_logger

logger = get_logger(__name__)


def optimize_trading_parameters():
    bot = AdvancedTradingBot({"backtest_mode": True})
    from datetime import datetime, timedelta
    bot.historical_data = bot.get_historical_data(
        datetime.now() - timedelta(days=settings.backtest_days),
        datetime.now()
    )
    best_params = bot.optimize_parameters(settings.optimization_grid)
    logger.info(f"Optimized parameters: {best_params}")
    return best_params


if __name__ == '__main__':
    optimize_trading_parameters()
