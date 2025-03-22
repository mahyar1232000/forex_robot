README.md

# AI Forex Trading Robot

This project implements an AI-powered Forex trading robot that connects to the Lite Finance broker via MetaTrader 5. It
supports both live trading and backtesting modes.

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

less
Copy
Edit

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