README.md

# Forex Robot

This project implements an AI Forex Trading Robot with advanced trading strategies, dynamic risk management, and parameter optimization. The system includes:

- **Live Trading:** Executes trades in real time.
- **Backtesting:** Tests strategy performance on historical data.
- **Model Training:** Trains an LSTM model with validation monitoring.
- **Parameter Optimization:** Optimizes trading parameters via grid search.
- **DeepSeek Integration:** Uses Ollama-hosted DeepSeek-R1-8B for comprehensive code reviews and technical analysis of the project.

## Project Structure

See the provided project structure for details.

## Setup

1. **Activate the Conda Environment:**
   ```bash
   conda activate tf_env

## Features

- **Live Trading**: Real-time trading with optimized parameters.
- **Backtesting**: Historical data simulation with parameter optimization.
- **Risk Management**: Advanced position sizing and circuit breaker.
- **GUI**: Real-time dashboard for live trading and backtest results.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/mahyar1232000/forex_robot.git
   cd forex_robot
2. Install dependencies:
   pip install -r requirements.txt
3. Configure the
   config/live.yaml
   ,
   config/backtest.yaml
   , and
   config/secrets.yaml
   files.

## Usage

Live Mode:
python main.py --mode live

Backtest Mode:
python main.py --mode backtest

# Forex Trading Robot

This project implements a Forex trading robot with live, backtesting, optimization, and training modes.

## Project Structure

forex_robot/ ├── main.py ├── config/ ├── data/ ├── docs/ ├── logs/ ├── models/ ├── src/ └── tests/

## Usage

Run the robot in your desired mode:

```bash
python main.py --mode live
python main.py --mode backtest
python main.py --mode optimization
python main.py --mode train
shell
Copy
Edit

### forex_robot/requirements.txt
pandas numpy matplotlib PyYAML tensorflow keras scikit-learn MetaTrader5 talib joblib pytz

Usage
-------

Live Trading:
python main.py --mode live


Backtesting:
python main.py --mode backtest



Model Training:
python main.py --mode train



Parameter Optimization:
python main.py --mode optimize



Code Review with DeepSeek: Run:
python src/ollama_reviewer.py

to review all Python files in the project.