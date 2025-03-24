# src/admin/admin.py

import json
from src.utils.logger import get_logger
from config import settings

logger = get_logger(__name__)

class AdminInterface:
    def __init__(self, trading_bot):
        self.trading_bot = trading_bot  # Instance of AdvancedTradingBot
        self.settings = settings.__dict__.copy()

    def view_settings(self):
        print("\nCurrent Settings:")
        for key, value in self.settings.items():
            print(f"{key}: {value}")

    def update_setting(self):
        key = input("\nEnter the setting key to update: ")
        if key not in self.settings:
            print("Setting not found.")
            return
        value = input("Enter new value (in JSON format): ")
        try:
            parsed_value = json.loads(value)
        except Exception:
            parsed_value = value
        self.settings[key] = parsed_value
        self.trading_bot.config[key] = parsed_value
        print(f"Updated {key} to {parsed_value}")
        logger.info(f"Admin updated setting {key} to {parsed_value}")

    def view_trade_history(self):
        history = self.trading_bot.trade_history
        if not history:
            print("\nNo trades executed yet.")
            return
        print("\nTrade History:")
        for trade in history:
            print(trade)

    def trigger_optimizer(self):
        if hasattr(self.trading_bot, 'optimize_parameters'):
            best_params = self.trading_bot.optimize_parameters(self.trading_bot.config.get("optimization_grid"))
            print("\nOptimizer completed. Best parameters:")
            print(best_params)
            logger.info(f"Admin triggered optimizer, best parameters: {best_params}")
        else:
            print("\nOptimizer not available.")

    def debug_mode(self):
        # Set logger level to DEBUG.
        logger.setLevel("DEBUG")
        print("\nDebug mode activated. Logger level set to DEBUG.")
        logger.debug("Debug mode is active.")

    def run_once(self):
        """
        Run the admin interface for a single session, then return.
        This can be called periodically during live/backtest modes.
        """
        print("\n--- Admin Interface (One-Time Check) ---")
        print("1. View Settings")
        print("2. Update Setting")
        print("3. View Trade History")
        print("4. Trigger Optimizer")
        print("5. Enable Debug Mode")
        print("6. Exit Admin Check")
        choice = input("Select an option (1-6): ")
        if choice == "1":
            self.view_settings()
        elif choice == "2":
            self.update_setting()
        elif choice == "3":
            self.view_trade_history()
        elif choice == "4":
            self.trigger_optimizer()
        elif choice == "5":
            self.debug_mode()
        elif choice == "6":
            print("Exiting Admin Check.")
        else:
            print("Invalid choice.")

    def run(self):
        """
        Run the admin interface interactively (if desired).
        """
        while True:
            print("\n--- Admin Interface ---")
            print("1. View Settings")
            print("2. Update Setting")
            print("3. View Trade History")
            print("4. Trigger Optimizer")
            print("5. Enable Debug Mode")
            print("6. Exit Admin Interface")
            choice = input("Select an option (1-6): ")
            if choice == "1":
                self.view_settings()
            elif choice == "2":
                self.update_setting()
            elif choice == "3":
                self.view_trade_history()
            elif choice == "4":
                self.trigger_optimizer()
            elif choice == "5":
                self.debug_mode()
            elif choice == "6":
                print("Exiting Admin Interface.")
                break
            else:
                print("Invalid choice. Please try again.")
