import numpy as np
from tqdm import tqdm
from bayes_opt import BayesianOptimization
from ..core import LSTMTradingStrategy
from ..utils.config import ConfigManager


class BayesianOptimizer:
    def optimize(self, data):
        def black_box_function(lookback_window, neurons, dropout):
            params = {
                'lookback_window': int(lookback_window),
                'neurons': int(neurons),
                'dropout': max(0.0, min(dropout, 0.5))  # Ensure dropout is within bounds
            }
            strategy = LSTMTradingStrategy(params)
            performance = self.evaluate(data, strategy)
            return performance

        optimizer = BayesianOptimization(
            f=black_box_function,
            pbounds={
                'lookback_window': (12, 48),
                'neurons': (30, 100),
                'dropout': (0.1, 0.5)
            },
            random_state=42,
            verbose=2
        )

        optimizer.maximize(init_points=5, n_iter=15)
        return optimizer.max['params']

    def evaluate(self, data, strategy):
        portfolio = self._simulate_trading(data, strategy)
        return self._calculate_performance(portfolio)

    def _simulate_trading(self, data, strategy):
        portfolio = {'balance': ConfigManager().load_config('backtest')['initial_balance'], 'positions': []}
        lookback = strategy.params['lookback_window']

        for i in range(lookback, len(data)):
            prediction = strategy.predict(data['close'].iloc[i - lookback:i].values.reshape(-1, 1))
            self._execute_trade(portfolio, data.iloc[i], prediction)

        return portfolio

    def _calculate_performance(self, portfolio):
        returns = np.diff([p['value'] for p in portfolio['positions']])
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        return sharpe


class WalkForwardOptimizer:
    def __init__(self, train_size=180, test_size=30):
        self.train_size = train_size  # Training window size (days)
        self.test_size = test_size  # Testing window size (days)

    def optimize(self, data):
        results = []
        total_days = len(data)

        for i in range(0, total_days - self.train_size - self.test_size, self.test_size):
            train_data = data.iloc[i:i + self.train_size]
            test_data = data.iloc[i + self.train_size:i + self.train_size + self.test_size]

            # Optimize parameters on training data
            optimizer = BayesianOptimizer()
            best_params = optimizer.optimize(train_data)

            # Test on out-of-sample data
            strategy = LSTMTradingStrategy(best_params)
            performance = self.evaluate(test_data, strategy)

            results.append((best_params, performance))

        # Return the best parameters based on average performance
        return max(results, key=lambda x: x[1])[0]

    def evaluate(self, data, strategy):
        portfolio = self._simulate_trading(data, strategy)
        return self._calculate_performance(portfolio)

    def _simulate_trading(self, data, strategy):
        portfolio = {'balance': ConfigManager().load_config('backtest')['initial_balance'], 'positions': []}
        lookback = strategy.params['lookback_window']

        for i in range(lookback, len(data)):
            prediction = strategy.predict(data['close'].iloc[i - lookback:i].values.reshape(-1, 1))
            self._execute_trade(portfolio, data.iloc[i], prediction)

        return portfolio

    def _calculate_performance(self, portfolio):
        returns = np.diff([p['value'] for p in portfolio['positions']])
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        return sharpe
