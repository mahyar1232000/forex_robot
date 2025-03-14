import pandas as pd
from src.core.Broker import Broker
from src.core.LSTMTradingStrategy import LSTMTradingStrategy


def main(mode):
    if mode == "live":
        # Initialize broker
        broker = Broker()
        broker.initialize_broker()

        # Execute trading strategy
        strategy = LSTMTradingStrategy()
        data = pd.DataFrame()  # Replace with actual data
        strategy.execute_strategy(data)
    else:
        print("Invalid mode. Use 'live' for live trading.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, help="Mode to run the script (e.g., 'live')")
    args = parser.parse_args()
    main(args.mode)
