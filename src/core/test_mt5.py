import MetaTrader5 as mt5

# Initialize the MetaTrader 5 terminal
if not mt5.initialize():
    print("Initialization failed")
    quit()

# Display account information
account_info = mt5.account_info()
if account_info is not None:
    print(account_info)

# Retrieve the latest tick data for EURUSD
symbol = "EURUSD"
tick = mt5.symbol_info_tick(symbol)
if tick is not None:
    print(tick)

# Shut down the connection to the MetaTrader 5 terminal
mt5.shutdown()
