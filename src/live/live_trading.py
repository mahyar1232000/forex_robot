import time

from config.settings import TIME_SLEEP
from src.core.broker import initialize_broker
from src.core.strategy import Strategy
from src.utils.logger import setup_logger

logger = setup_logger()


def run_live_trading():
    broker = initialize_broker()
    strategy = Strategy()
    try:
        while True:
            market_data = {"close": 1.1050}  # Replace with real-time data fetching logic
            signal = strategy.generate_signal(market_data)
            logger.info(f"Generated signal: {signal}")
            broker.execute_trade(
                order_type=1 if signal == "BUY" else 2,
                lot_size=1,
                stop_loss=0.0010,
                take_profit=0.0020
            )
            time.sleep(TIME_SLEEP * 60)  # Use sleep interval from settings
    except KeyboardInterrupt:
        logger.info("Live trading interrupted by user.")


def main():
    run_live_trading()


if __name__ == '__main__':
    main()
