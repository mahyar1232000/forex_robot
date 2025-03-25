# config/settings.py

# ----------------------------
# Trading & Broker Settings
# ----------------------------
SYMBOL = "EURUSD_o"  # Default symbol.
MAGIC_NUMBER = 202308  # Unique trade identifier.
INITIAL_BALANCE = 200.0  # Starting balance for backtesting.
TIME_SLEEP = 15  # Minutes between trade iterations.
REPEAT_BACKTEST = True

# ----------------------------
# Technical Indicator Parameters
# ----------------------------
timeframe = 15  # Candle timeframe in minutes.
ema_fast = 10  # Fast EMA period.
ema_slow = 24  # Slow EMA period.
rsi_period = 12  # RSI period.

# ----------------------------
# Backtesting Parameters
# ----------------------------
backtest_days = 30  # Days of historical data for backtesting.
analysis_period = 500  # Number of bars to retrieve.
historical_data_path = "data/historical.csv"

# ----------------------------
# Risk Management Settings
# ----------------------------
RISK_PERCENT = 1.5  # Percentage of balance risked per trade.
RISK_REWARD_RATIO = 2.5  # Reward-to-risk ratio.
SPREAD_THRESHOLD = 20  # Maximum acceptable spread threshold.

# ----------------------------
# Market Regime Detection Parameters
# ----------------------------
adx_trending_threshold = 25  # ADX threshold.
atr_volatility_multiplier = 1.2  # ATR multiplier for volatility.
atr_multiplier_trending = 2.0  # ATR multiplier in trending markets.
atr_multiplier_volatile = 1.5  # ATR multiplier in volatile markets.
atr_multiplier_sideways = 2.0  # ATR multiplier in sideways markets.

# ----------------------------
# Trade Execution Settings
# ----------------------------
min_confidence = 0.5  # Minimum confidence (e.g., 0.5 means 50%) required to trigger a trade.

# ----------------------------
# LSTM Strategy Settings
# ----------------------------
lstm_sequence_length = 50  # LSTM sequence length.
lstm_epochs = 120  # Training epochs.
lstm_batch_size = 128  # Batch size.
lstm_patience = 15  # Patience for early stopping.

# ----------------------------
# Optimization Settings
# ----------------------------
optimization_grid = {
    'ema_fast': [10, 12, 14],
    'ema_slow': [24, 26, 28],
    'rsi_period': [12, 14, 16],
    'RISK_PERCENT': [0.5, 1.0, 1.5],
    'RISK_REWARD_RATIO': [1.5, 2.0, 2.5]
}
