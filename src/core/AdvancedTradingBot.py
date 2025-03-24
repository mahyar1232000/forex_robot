"""
AdvancedTradingBot.py

Main trading bot for advanced trading.
Supports live/backtest modes and parameter optimization.
Integrates market data processing, trade execution, dynamic risk management,
a circuit breaker, performance monitoring, and integrated admin monitoring.
Trained model and scaler files are stored in the 'models' folder.
"""

import os
import time
import numpy as np
import pandas as pd
import talib
import MetaTrader5 as mt5
import pytz
from datetime import datetime, timedelta, time as dtime
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from joblib import load, dump
from config import settings
from src.utils.logger import get_logger
from src.risk.AdvancedRiskManager import calculate_dynamic_risk
from src.risk.circuit_breaker import check_circuit_breaker
from src.risk.PerformanceMonitor import PerformanceMonitor
from src.admin.admin import AdminInterface

logger = get_logger(__name__)

class AdvancedTradingBot:
    """
    AdvancedTradingBot handles live trading, backtesting, and parameter optimization.
    It loads or creates the trained model and scaler from the 'models' folder.
    If admin monitoring is enabled, it periodically triggers an admin check to allow adjustments.
    """

    def __init__(self, config_params=None):
        self.config = settings.__dict__.copy()
        if config_params:
            self.config.update(config_params)
        self.account_balance = self.config.get('INITIAL_BALANCE', 10000)
        self.equity_high = self.account_balance
        self.trade_history = []
        self.current_index = 0
        self.historical_data = pd.DataFrame()
        self.timezone = pytz.timezone('Etc/UTC')
        self.session_id = datetime.now().strftime("%Y%m%d%H%M%S")
        self.feature_columns = [
            'EMA_FAST', 'EMA_SLOW', 'RSI', 'ATR', 'MACD', 'ADX',
            'STOCH_K', 'SMA_50', 'SMA_200', 'OBV', 'VWAP', 'ENGULFING',
            'log_return', 'rolling_volatility', 'momentum',
            'price_rate_of_change', 'ema_sma_diff', 'market_regime'
        ]
        self.model = None
        self.scaler = None
        self.performance_monitor = PerformanceMonitor()
        self.admin_interface = AdminInterface(self)
        self.last_admin_check = datetime.now()
        self.initialize_components()

    def initialize_components(self):
        self.cleanup_legacy_files()
        self.load_or_create_model()

    def cleanup_legacy_files(self):
        # Remove legacy files from the root if they exist.
        for fname in ['deepseek_model.joblib', 'scaler.joblib']:
            if os.path.exists(fname):
                try:
                    os.remove(fname)
                    logger.info(f"Removed legacy file: {fname}")
                except Exception as e:
                    logger.warning(f"Cleanup error for {fname}: {str(e)}")

    def load_or_create_model(self):
        model_path = os.path.join("models", "deepseek_model.joblib")
        scaler_path = os.path.join("models", "scaler.joblib")
        try:
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = load(model_path)
                self.scaler = load(scaler_path)
                if not hasattr(self.model, 'classes_'):
                    raise ValueError("Loaded model not properly trained")
                logger.info("Loaded existing model and scaler from models folder")
            else:
                self.create_new_model()
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.create_new_model()

    def create_new_model(self):
        model_path = os.path.join("models", "deepseek_model.joblib")
        scaler_path = os.path.join("models", "scaler.joblib")
        try:
            if not self.initialize_mt5():
                raise ConnectionError("Could not connect to MT5 for initial data")
            start_date = datetime.now() - timedelta(days=self.config.get('backtest_days', 30))
            end_date = datetime.now()
            init_data = self.get_historical_data(start_date, end_date)
            if init_data.empty or len(init_data) < 100:
                raise ValueError(f"Insufficient training data ({len(init_data)} records)")
            features = init_data[self.feature_columns].values[:-1]
            labels = np.where(init_data['close'].shift(-1) > init_data['close'], 1, 0)[:-1]
            self.scaler = StandardScaler()
            scaled_features = self.scaler.fit_transform(features)
            self.model = SGDClassifier(loss='log_loss', random_state=42)
            self.model.fit(scaled_features, labels)
            dump(self.model, model_path)
            dump(self.scaler, scaler_path)
            logger.info("Successfully created and trained initial model; saved to models folder")
        except Exception as e:
            logger.error(f"Model creation failed: {str(e)}")
            self.model = SGDClassifier(loss='log_loss', random_state=42)
            self.scaler = StandardScaler()
            try:
                dump(self.model, model_path)
                dump(self.scaler, scaler_path)
            except Exception as save_error:
                logger.error(f"Failed to save fallback model: {str(save_error)}")

    def initialize_mt5(self, retries=3, delay=5) -> bool:
        from src.utils.config import load_config
        secret_config = load_config('config/secret.yaml')
        if 'mt5' not in self.config:
            self.config.update(secret_config)
        for attempt in range(retries):
            try:
                logger.info(f"Initializing MT5 (attempt {attempt+1}/{retries})")
                if not mt5.initialize():
                    error_info = mt5.last_error()
                    logger.error(f"MT5 initialization failed. Error: {error_info}")
                else:
                    if not mt5.login(
                        login=self.config['mt5']['account'],
                        password=self.config['mt5']['password'],
                        server=self.config['mt5']['server']
                    ):
                        error_info = mt5.last_error()
                        logger.error(f"MT5 login failed. Error: {error_info}")
                    elif not mt5.symbol_select(self.config['SYMBOL'], True):
                        logger.error(f"Failed to select symbol {self.config['SYMBOL']}")
                    else:
                        logger.info(f"MT5 connected and symbol {self.config['SYMBOL']} selected")
                        return True
                    mt5.shutdown()
            except Exception as e:
                logger.error(f"MT5 initialization exception: {repr(e)}")
            time.sleep(delay)
        return False

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['rolling_volatility'] = df['log_return'].rolling(window=14).std() * np.sqrt(14)
        df['momentum'] = df['close'] - df['close'].shift(10)
        df['price_rate_of_change'] = df['close'].pct_change(periods=10)
        df['ema_sma_diff'] = df['EMA_FAST'] - df['SMA_200']
        df['market_regime'] = df.apply(self.detect_market_regime_row, axis=1).map(
            {'trending': 0, 'volatile': 1, 'sideways': 2})
        return df

    def detect_market_regime_row(self, row):
        try:
            if row['ADX'] >= self.config.get('adx_trending_threshold', 25):
                return 'trending'
            elif row['ATR'] >= row['ATR_MA'] * self.config.get('atr_volatility_multiplier', 1.2):
                return 'volatile'
            return 'sideways'
        except Exception as e:
            logger.error(f"Regime detection error for row: {e}")
            return 'sideways'

    def detect_market_regime(self, df: pd.DataFrame) -> str:
        try:
            if len(df) < 50:
                logger.warning("Not enough data for market regime detection; defaulting to 'sideways'.")
                return "sideways"
            atr_ma_series = df['ATR'].rolling(window=50).mean()
            if atr_ma_series.empty or np.isnan(atr_ma_series.iloc[-1]):
                logger.warning("ATR rolling mean not available; defaulting to 'sideways'.")
                return "sideways"
            atr_ma = atr_ma_series.iloc[-1]
            latest = df.iloc[-1]
            if latest['ADX'] >= self.config.get('adx_trending_threshold', 25):
                return 'trending'
            elif latest['ATR'] >= atr_ma * self.config.get('atr_volatility_multiplier', 1.2):
                return 'volatile'
            return 'sideways'
        except Exception as e:
            logger.error(f"Market regime detection error: {str(e)}")
            return "sideways"

    def get_atr_multiplier(self, regime: str) -> float:
        return {
            'trending': self.config.get('atr_multiplier_trending', 2.0),
            'volatile': self.config.get('atr_multiplier_volatile', 1.5),
            'sideways': self.config.get('atr_multiplier_sideways', 2.0)
        }[regime]

    def process_data(self, rates) -> pd.DataFrame:
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df['spread'] = df['high'] - df['low']
        closes = df['close'].values
        df['EMA_FAST'] = talib.EMA(closes, self.config['ema_fast'])
        df['EMA_SLOW'] = talib.EMA(closes, self.config['ema_slow'])
        df['SMA_50'] = talib.SMA(closes, 50)
        df['SMA_200'] = talib.SMA(closes, 200)
        df['RSI'] = talib.RSI(closes, self.config['rsi_period'])
        df['MACD'], df['MACD_SIGNAL'], _ = talib.MACD(closes)
        df['ADX'] = talib.ADX(df['high'], df['low'], closes, 14)
        df['STOCH_K'], df['STOCH_D'] = talib.STOCH(df['high'], df['low'], df['close'])
        df['ATR'] = talib.ATR(df['high'], df['low'], closes, 14)
        df['OBV'] = talib.OBV(df['close'], df['tick_volume'])
        df['VWAP'] = (df['close'] * df['tick_volume']).cumsum() / df['tick_volume'].cumsum()
        df['ENGULFING'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
        df['ATR_MA'] = df['ATR'].rolling(window=50).mean()
        df = self.feature_engineering(df)
        # Instead of dropping all NaNs, drop only rows where ATR is missing.
        df = df[df['ATR'].notna()]
        logger.info(f"Data after processing: {len(df)} rows")
        return df

    def get_market_data(self) -> pd.DataFrame:
        try:
            if not mt5.terminal_info() and not self.initialize_mt5():
                return pd.DataFrame()
            rates = mt5.copy_rates_from_pos(self.config['SYMBOL'],
                                            self.config.get('timeframe', 15),
                                            0, self.config.get('analysis_period', 100))
            if rates is None or len(rates) == 0:
                logger.error(f"Market data error: {mt5.last_error()}")
                return pd.DataFrame()
            df_live = self.process_data(rates)
            # If live data is insufficient for regime detection, fetch extra historical data.
            if len(df_live) < 50:
                from datetime import datetime, timedelta
                extra_start = datetime.now() - timedelta(days=30)
                extra_rates = mt5.copy_rates_range(self.config['SYMBOL'],
                                                   self.config.get('timeframe', 15),
                                                   extra_start,
                                                   datetime.now())
                if extra_rates is not None and len(extra_rates) >= 50:
                    df_extra = self.process_data(extra_rates)
                    df_combined = pd.concat([df_extra, df_live]).drop_duplicates(subset='time').sort_values('time')
                    return df_combined
                else:
                    logger.warning("Not enough extra data available; using live data only.")
            return df_live
        except Exception as e:
            logger.error(f"Market data processing error: {str(e)}")
            return pd.DataFrame()

    def get_historical_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        try:
            if not mt5.terminal_info():
                logger.info("Reconnecting MT5 for historical data...")
                if not self.initialize_mt5(retries=5, delay=10):
                    logger.error("Historical data aborted – MT5 connection failed")
                    return pd.DataFrame()
            start_date = start_date.astimezone(self.timezone)
            end_date = end_date.astimezone(self.timezone)
            rates = mt5.copy_rates_range(
                self.config['SYMBOL'],
                self.config.get('timeframe', 15),
                start_date,
                end_date
            )
            if rates is None or len(rates) == 0:
                logger.error(f"Failed to fetch historical data. MT5 error: {mt5.last_error()}")
                return pd.DataFrame()
            df = self.process_data(rates)
            if df.empty:
                logger.error("Processed historical data is empty after cleaning")
            elif len(df) < 100:
                logger.warning(f"Limited historical data: Only {len(df)} records available")
            return df
        except Exception as e:
            logger.error(f"Historical data error: {str(e)}")
            return pd.DataFrame()

    def analyze_market(self, df: pd.DataFrame) -> dict:
        if df.empty or len(df) < 100:
            return {'signal': 'HOLD', 'confidence': 0}
        try:
            latest_features = df[self.feature_columns].values[-1].reshape(1, -1)
        except Exception as e:
            logger.error(f"Error extracting latest features: {str(e)}")
            return {'signal': 'HOLD', 'confidence': 0}
        atr_value = df.iloc[-1].get('ATR', None)
        if atr_value is None or np.isnan(atr_value):
            logger.warning("ATR value is not available; returning HOLD signal.")
            return {'signal': 'HOLD', 'confidence': 0}
        try:
            probabilities = np.asarray(self.model.predict_proba(self.scaler.transform(latest_features)))
            confidence = float(probabilities.max())
            logger.info(f"Predicted confidence: {confidence}")
            prediction = self.model.predict(self.scaler.transform(latest_features))
            stop_loss = atr_value * self.get_atr_multiplier(self.detect_market_regime(df))
            return {
                'signal': 'BUY' if prediction[0] == 1 else 'SELL',
                'confidence': confidence,
                'price': df.iloc[-1]['close'],
                'time': df.iloc[-1]['time'],
                'stop_loss': stop_loss,
                'market_regime': self.detect_market_regime(df),
                'RR_ratio': self.config.get('RISK_REWARD_RATIO', 2.0)
            }
        except Exception as e:
            logger.error(f"Error in market analysis: {str(e)}")
            try:
                fallback_prediction = self.model.predict(self.scaler.transform(latest_features))
            except Exception as ex:
                fallback_prediction = [0]
            return {
                'signal': 'BUY' if fallback_prediction[0] == 1 else 'SELL',
                'confidence': 0,
                'price': df.iloc[-1]['close'],
                'time': df.iloc[-1]['time'],
                'stop_loss': 0,
                'market_regime': self.detect_market_regime(df),
                'RR_ratio': self.config.get('RISK_REWARD_RATIO', 2.0)
            }

    def admin_monitor(self):
        """
        Checks periodically if admin intervention is needed.
        If admin_enabled is True and the specified interval has passed, trigger a one-time admin check.
        """
        if self.config.get("admin_enabled", False):
            now = datetime.now()
            interval = self.config.get("admin_check_interval", 1)  # in minutes
            if (now - self.last_admin_check).total_seconds() >= interval * 60:
                logger.info("Admin monitor triggered.")
                self.admin_interface.run_once()
                self.last_admin_check = now

    def execute_trade(self, signal: dict) -> bool:
        if signal.get("signal", "HOLD") == "HOLD":
            logger.info("Signal is HOLD; no trade executed.")
            return False
        if not self.safety_checks():
            return False
        if not check_circuit_breaker(self.trade_history, self.account_balance):
            logger.warning("Circuit breaker activated; trading halted.")
            return False
        dynamic_risk = calculate_dynamic_risk(self.account_balance, self.config.get('RISK_PERCENT', 1.0))
        self.config['RISK_PERCENT'] = dynamic_risk
        self.admin_monitor()  # Check for admin intervention.
        if self.config.get('backtest_mode', False):
            return self.backtest_trade(signal)
        return self.execute_live_trade(signal)

    def backtest_trade(self, signal: dict) -> bool:
        current_data = self.historical_data.iloc[self.current_index]
        entry_price = current_data['close']
        risk_amount = self.account_balance * (self.config.get('RISK_PERCENT', 1.0) / 100)
        stop_loss = signal['stop_loss']
        if stop_loss <= 0:
            logger.error("Non-positive stop_loss in backtest calculation.")
            return False
        pip_value = 0.0001
        position_size = round(risk_amount / (stop_loss * pip_value), 2)
        position_size = min(position_size, 1e6)
        future_data = self.historical_data[self.current_index + 1:]
        profit = 0
        for _, row in future_data.iterrows():
            if signal['signal'] == 'BUY':
                if row['low'] <= (entry_price - stop_loss):
                    profit = -risk_amount
                    break
                if row['high'] >= (entry_price + stop_loss * self.config.get('RISK_REWARD_RATIO', 2.0)):
                    profit = risk_amount * self.config.get('RISK_REWARD_RATIO', 2.0)
                    break
            elif signal['signal'] == 'SELL':
                if row['high'] >= (entry_price + stop_loss):
                    profit = -risk_amount
                    break
                if row['low'] <= (entry_price - stop_loss * self.config.get('RISK_REWARD_RATIO', 2.0)):
                    profit = risk_amount * self.config.get('RISK_REWARD_RATIO', 2.0)
                    break
        if profit == 0:
            close_price = future_data.iloc[-1]['close'] if not future_data.empty else entry_price
            profit = (close_price - entry_price) * position_size if signal['signal'] == 'BUY' else (entry_price - close_price) * position_size
        self.account_balance += profit
        self.equity_high = max(self.equity_high, self.account_balance)
        self.trade_history.append({
            'entry_time': signal['time'],
            'exit_time': datetime.now(),
            'signal': signal['signal'],
            'profit': profit,
            'balance': self.account_balance
        })
        self.performance_monitor.update(self.account_balance)
        return True

    def execute_live_trade(self, signal: dict) -> bool:
        try:
            if not mt5.terminal_info() and not self.initialize_mt5():
                logger.error("MT5 terminal unavailable or reinitialization failed.")
                return False
            symbol = self.config['SYMBOL']
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                logger.error(f"Symbol {symbol} not found.")
                return False
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to select symbol {symbol}.")
                return False
            price = symbol_info.ask if signal['signal'] == 'BUY' else symbol_info.bid
            if price is None or price <= 0:
                logger.error(f"Invalid price for symbol {symbol}: {price}")
                return False
            current_spread = symbol_info.spread
            regime = self.detect_market_regime(self.historical_data)
            spread_limit = self.config.get('SPREAD_THRESHOLD', 20) * {
                'volatile': 2.0,
                'trending': 1.5,
                'sideways': 1.0
            }.get(regime, 1.0)
            if current_spread > spread_limit:
                logger.warning(f"Spread {current_spread:.1f} exceeds limit {spread_limit:.1f} in {regime} market")
                return False
            lot_size = self.calculate_lot_size(signal)
            if lot_size <= 0:
                logger.error(f"Invalid calculated lot size: {lot_size}")
                return False
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lot_size,
                "type": mt5.ORDER_TYPE_BUY if signal['signal'] == 'BUY' else mt5.ORDER_TYPE_SELL,
                "price": price,
                "sl": self.calculate_stop_loss(signal),
                "tp": self.calculate_take_profit(signal),
                "deviation": 20,
                "magic": self.config.get('MAGIC_NUMBER'),
                "comment": f"AI-Trade|{self.session_id}"
            }
            logger.debug(f"Order request: {request}")
            result = mt5.order_send(request)
            if result is None or not hasattr(result, 'retcode'):
                logger.error(f"Order send failed. Request: {request}. Error: {mt5.last_error()}")
                return False
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Trade failed: {result.comment}. Request: {request}")
                return False
            logger.info(f"Executed {signal['signal']} trade at price {price} with volume {lot_size}")
            return True
        except Exception as e:
            logger.error(f"Live trade execution error: {str(e)}")
            return False

    def calculate_lot_size(self, signal: dict) -> float:
        risk_amount = self.account_balance * (self.config.get('RISK_PERCENT', 1.0) / 100)
        pip_value = 0.01 if "XAUUSD" in self.config['SYMBOL'] else 0.0001
        stop_loss_pips = signal['stop_loss'] / pip_value
        lot_size = risk_amount / (stop_loss_pips * 10)
        symbol_info = mt5.symbol_info(self.config['SYMBOL'])
        if symbol_info:
            lot_size = max(symbol_info.volume_min, min(lot_size, symbol_info.volume_max))
            lot_size = round(lot_size / symbol_info.volume_step) * symbol_info.volume_step
        return round(lot_size, 2)

    def calculate_stop_loss(self, signal: dict) -> float:
        tick = mt5.symbol_info_tick(self.config['SYMBOL'])
        entry_price = tick.ask if signal['signal'] == 'BUY' else tick.bid
        return entry_price - signal['stop_loss'] if signal['signal'] == 'BUY' else entry_price + signal['stop_loss']

    def calculate_take_profit(self, signal: dict) -> float:
        tick = mt5.symbol_info_tick(self.config['SYMBOL'])
        entry_price = tick.ask if signal['signal'] == 'BUY' else tick.bid
        rr = self.config.get('RISK_REWARD_RATIO', 2.0)
        return entry_price + (signal['stop_loss'] * rr) if signal['signal'] == 'BUY' else entry_price - (signal['stop_loss'] * rr)

    def safety_checks(self) -> bool:
        if self.config.get('backtest_mode', False):
            return True
        now = datetime.now().time()
        if (dtime(21, 0) <= now <= dtime(23, 59)) or (dtime(0, 0) <= now <= dtime(2, 0)):
            logger.warning("Avoiding trading during market close hours")
            return False
        if self.account_balance < self.config.get('INITIAL_BALANCE', 200.0) * 0.98:
            logger.warning("Account protection: 2% daily drawdown limit reached")
            return False
        return True

    def run_backtest(self):
        if not self.initialize_mt5():
            return
        start_date = datetime.now() - timedelta(days=self.config.get('backtest_days', 30))
        end_date = datetime.now()
        self.historical_data = self.get_historical_data(start_date, end_date)
        if len(self.historical_data) < 200:
            logger.error("Insufficient data for backtest")
            return
        logger.info(f"Starting backtest with {len(self.historical_data)} bars")
        trades_executed = 0
        for self.current_index in range(200, len(self.historical_data)):
            df = self.historical_data.iloc[:self.current_index]
            signal = self.analyze_market(df)
            logger.info(f"At bar {self.current_index}, signal: {signal['signal']}, confidence: {signal['confidence']}")
            if signal['confidence'] >= self.config.get('min_confidence', 0.5):
                if self.execute_trade(signal):
                    trades_executed += 1
            self.admin_monitor()  # Check for admin intervention during backtest.
            if self.current_index % 100 == 0:
                self.report_backtest_progress()
        logger.info(f"Total trades executed: {trades_executed}")
        self.generate_backtest_report()

    def report_backtest_progress(self):
        completed = (self.current_index / len(self.historical_data)) * 100
        logger.info(f"Backtest progress: {completed:.2f}% completed")

    def generate_backtest_report(self):
        logger.info(f"Backtest completed. Final balance: {self.account_balance:.2f}")
        self.performance_monitor.report()

    def optimize_parameters(self, param_grid: dict) -> dict:
        """
        Optimize parameters by running backtests over the parameter grid.
        Returns the best parameter set based on final balance.
        """
        from sklearn.model_selection import ParameterGrid
        best_params = None
        best_balance = -float('inf')
        original_config = self.config.copy()
        for params in ParameterGrid(param_grid):
            logger.info(f"Testing parameters: {params}")
            self.config.update(params)
            self.account_balance = original_config.get('INITIAL_BALANCE', 200.0)
            self.trade_history = []
            if self.historical_data.empty:
                from datetime import datetime, timedelta
                self.historical_data = self.get_historical_data(
                    datetime.now() - timedelta(days=self.config.get('backtest_days', 30)),
                    datetime.now()
                )
            test_data = self.historical_data.copy().tail(300)
            for i in range(200, len(test_data)):
                df = test_data.iloc[:i]
                signal = self.analyze_market(df)
                if signal['confidence'] >= self.config.get('min_confidence', 0.5):
                    self.backtest_trade(signal)
            final_balance = self.account_balance
            logger.info(f"Parameters {params} resulted in final balance: {final_balance}")
            if final_balance > best_balance:
                best_balance = final_balance
                best_params = params
            self.config = original_config.copy()
            self.account_balance = original_config.get('INITIAL_BALANCE', 200.0)
            self.trade_history = []
        logger.info(f"Optimization complete. Best parameters: {best_params} with balance: {best_balance}")
        self.config.update(best_params)
        return best_params

# End of AdvancedTradingBot.py
