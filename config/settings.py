# config/settings.py

# ----------------------------
# Trading & Broker Settings
# ----------------------------
SYMBOL = "EURUSD_o"  # Default symbol EURUSD_o.
MAGIC_NUMBER = 202308  # Unique trade identifier.
INITIAL_BALANCE = 200.0  # Starting balance for backtesting.
TIME_SLEEP = 15  # Minutes to wait between trade iterations.
REPEAT_BACKTEST = True  # Whether to run repeated backtests.

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
analysis_period = 1000  # Number of candles used in live analysis.
historical_data_path = "data/historical.csv"  # Path to historical data.

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
min_confidence = 0.5  # Minimum confidence (%) required to trigger a trade. confidence is in the range [0, 1] (for example, 0.62 means 62%)

# ----------------------------
# LSTM Strategy Settings
# ----------------------------
lstm_sequence_length = 50  # LSTM sequence length.
lstm_epochs = 120  # Training epochs.
lstm_batch_size = 128  # Batch size.
lstm_patience = 15  # Patience for early stopping.

# ----------------------------
# Additional Admin Settings
# ----------------------------
admin_enabled = True  # Set to True to enable admin monitoring.
admin_check_interval = 1  # Admin check interval (in minutes) during live/backtest runs.

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
