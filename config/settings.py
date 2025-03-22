# config/settings.py

# ----------------------------
# Trading & Broker Settings
# ----------------------------
SYMBOL = "EURUSD_o"  # Default symbol (can be changed dynamically).
MAGIC_NUMBER = 202308  # Unique identifier for trades.
INITIAL_BALANCE = 200.0  # Starting balance for backtesting.
TIME_SLEEP = 15  # Minutes to wait between trade iterations.
REPEAT_BACKTEST = True  # Whether to run repeated backtests.

# ----------------------------
# Technical Indicator Parameters
# ----------------------------
timeframe = 15  # Candle timeframe in minutes.
ema_fast = 12  # Fast EMA period.
ema_slow = 26  # Slow EMA period.
rsi_period = 14  # RSI period.

# ----------------------------
# Backtesting Parameters
# ----------------------------
backtest_days = 30  # Days of historical data for backtesting.
analysis_period = 100  # Number of candles used in live analysis.
historical_data_path = "data/historical.csv"  # Path to historical CSV data.

# ----------------------------
# Risk Management Settings
# ----------------------------
RISK_PERCENT = 1.0  # Percentage of balance risked per trade.
RISK_REWARD_RATIO = 2.0  # Reward-to-risk ratio.
SPREAD_THRESHOLD = 20  # Maximum acceptable spread threshold.

# ----------------------------
# Market Regime Detection Parameters
# ----------------------------
adx_trending_threshold = 25  # ADX threshold to classify market as trending.
atr_volatility_multiplier = 1.2  # ATR multiplier for volatility detection.
atr_multiplier_trending = 2.0  # ATR multiplier in trending markets.
atr_multiplier_volatile = 1.5  # ATR multiplier in volatile markets.
atr_multiplier_sideways = 2.0  # ATR multiplier in sideways markets.

# ----------------------------
# Trade Execution Settings
# ----------------------------
min_confidence = 50  # Minimum confidence (percentage) required to trigger a trade.

# ----------------------------
# LSTM Strategy Settings
# ----------------------------
lstm_sequence_length = 50  # Time steps for LSTM sequences.
lstm_epochs = 100  # Increased epochs for better convergence.
lstm_batch_size = 64  # Increased batch size for more stable training.
lstm_patience = 10  # Increased patience for early stopping.

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
