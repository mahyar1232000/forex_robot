import numpy as np
from tqdm import tqdm
from ..core import DataManager, LSTMTradingStrategy, AdvancedRiskManager
from ..utils.config import ConfigManager


class Backtester:
    def __init__(self):
        self.config = ConfigManager().load_config('backtest')
        self.data_manager = DataManager(mode='backtest')
        self.risk_manager = AdvancedRiskManager()
        self.results = {
            'optimization_history': [],
            'equity_curve': [],
            'final_balance': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0
        }

    def optimize_parameters(self):
        best_performance = -np.inf
        best_params = {}
        param_combinations = self._generate_parameter_combinations()

        for params in tqdm(param_combinations, desc="Optimizing parameters"):
            performance = self.run_single_backtest(params)
            self.results['optimization_history'].append((params, performance))

            if performance > best_performance:
                best_performance = performance
                best_params = params

        # Save best parameters to live config
        ConfigManager().update_live_config(best_params)

        # Run final backtest with best parameters
        final_results = self.run_single_backtest(best_params, is_final=True)
        self.results.update(final_results)

        return self.results

    def run_single_backtest(self, params, is_final=False):
        data = self.data_manager.get_historical_data()
        strategy = LSTMTradingStrategy(params)
        strategy.train(data['close'].values)

        portfolio = self._simulate_trading(data, strategy)
        performance = self._calculate_performance(portfolio)

        if is_final:
            self.results['equity_curve'] = portfolio['equity_curve']
            self.results['final_balance'] = portfolio['balance']
            self.results['sharpe_ratio'] = performance
            self.results['max_drawdown'] = self._calculate_max_drawdown(portfolio['equity_curve'])
            self.results['win_rate'] = self._calculate_win_rate(portfolio['trades'])

        return performance

    def _simulate_trading(self, data, strategy):
        portfolio = {
            'balance': self.config['initial_balance'],
            'positions': [],
            'equity_curve': [],
            'trades': []
        }
        lookback = strategy.params['lookback_window']

        for i in range(lookback, len(data)):
            prediction = strategy.predict(data['close'].iloc[i - lookback:i].values.reshape(-1, 1))
            self._execute_trade(portfolio, data.iloc[i], prediction)
            portfolio['equity_curve'].append(portfolio['balance'])

        return portfolio

    def _execute_trade(self, portfolio, market_data, prediction):
        # Extract scalar value from prediction array
        prediction_value = prediction[0][0]  # Assuming prediction is a 2D array

        # Simplified trade execution logic
        if prediction_value > market_data['close']:
            trade = {'type': 'buy', 'profit': market_data['close'] - market_data['open']}
        else:
            trade = {'type': 'sell', 'profit': market_data['open'] - market_data['close']}

        portfolio['balance'] += trade['profit']
        portfolio['trades'].append(trade)

    def _calculate_performance(self, portfolio):
        returns = np.diff([p['value'] for p in portfolio['positions']])
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        return sharpe

    def _calculate_max_drawdown(self, equity_curve):
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        return np.max(drawdown)

    def _calculate_win_rate(self, trades):
        wins = sum(1 for trade in trades if trade['profit'] > 0)
        return wins / len(trades) if trades else 0.0

    def _generate_parameter_combinations(self):
        params = self.config['optimization']['parameters']
        from itertools import product
        return [dict(zip(params.keys(), v)) for v in product(*params.values())]
