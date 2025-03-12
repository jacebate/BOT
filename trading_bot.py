import os  # Added for file existence checks
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import requests
import telegram

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Connect to MT5
if not mt5.initialize():
    logging.error("Failed to initialize MT5")
    quit()

# Global variables for risk management
MAX_TRADES = 3  # Maximum number of open trades at a time
RISK_PER_TRADE_PERCENT = 2  # Risk per trade as a percentage of account balance
MIN_EQUITY_PERCENT = 10  # Minimum equity as a percentage of account balance

# Store historical trade data
historical_trades = []

# LSTM Model for price prediction
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

# Load pre-trained LSTM model
def load_model(input_size=12, hidden_size=64, num_layers=2, output_size=1, filename="lstm_model.pth"):
    """Load pre-trained LSTM model if available."""
    model = LSTM(input_size, hidden_size, num_layers, output_size)
    
    if os.path.exists(filename):
        model.load_state_dict(torch.load(filename))
        logging.info(f"Model loaded from {filename}")
    else:
        logging.warning(f"Model file '{filename}' not found. Using untrained model.")
    
    return model

# ✅ Get Account Info
def get_account_info():
    """Retrieve account balance and equity."""
    account_info = mt5.account_info()
    if account_info is None:
        logging.error("Failed to get account info")
        return None, None
    return account_info.balance, account_info.equity

# ✅ Fetch Market Data
def get_market_data(symbol, timeframe, bars=100):
    """Fetch historical market data from MT5."""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None:
        logging.error(f"Failed to get market data for {symbol}")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')  # Convert timestamp to readable format
    return df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]

# ✅ Calculate Position Size
def calculate_position_size(balance, risk_percent, sl_pips, pip_value, symbol):
    """Calculate trade volume based on account balance and risk percentage."""
    if sl_pips == 0 or pip_value == 0:
        logging.error("SL Pips or Pip Value is zero. Cannot calculate position size.")
        return 0.0

    risk_amount = balance * (risk_percent / 100)
    lot_size = risk_amount / (sl_pips * pip_value)

    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logging.error(f"Failed to get symbol info for {symbol}")
        return 0.0

    min_lot = symbol_info.volume_min
    max_lot = symbol_info.volume_max
    lot_step = symbol_info.volume_step

    lot_size = round(lot_size / lot_step) * lot_step
    return max(min_lot, min(lot_size, max_lot))

# ✅ Add Technical Indicators
def add_technical_indicators(df):
    """Add technical indicators to the DataFrame."""
    df['moving_avg_5'] = df['close'].rolling(window=5).mean()
    df['moving_avg_10'] = df['close'].rolling(window=10).mean()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()

    # Bollinger Bands
    df['bb_middle'] = df['close'].rolling(window=20).mean()
    std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_middle'] + 2 * std
    df['bb_lower'] = df['bb_middle'] - 2 * std

    df.dropna(inplace=True)
    return df

# Prepare data for LSTM prediction
def prepare_data(df):
    """Prepare data for LSTM prediction."""
    X, y = [], []
    for i in range(20, len(df)):  # Use 20 periods for lookback
        X.append(df.iloc[i-20:i, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]].values)  # All features
        y.append(df.iloc[i, 3])  # Predicting 'close' price
    return np.array(X), np.array(y)

# Auto-Adaptive Risk Management
def calculate_volatility(df):
    """Calculate market volatility using ATR."""
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1)))
    )
    atr = df['tr'].rolling(window=14).mean().iloc[-1]
    return atr

def adjust_risk_based_on_volatility(atr, base_risk=RISK_PER_TRADE_PERCENT):
    """Adjust risk percentage based on market volatility."""
    if atr > 0.01:  # High volatility
        return base_risk * 0.5  # Reduce risk
    elif atr < 0.005:  # Low volatility
        return base_risk * 1.5  # Increase risk
    return base_risk

# ✅ Retrieve Trade Details for Alerts
def get_trade_details():
    """Retrieve the last trade details for Telegram alerts."""
    if not historical_trades:
        return "No trade history available."

    last_trade = historical_trades[-1]
    return f"Symbol: {last_trade['symbol']}, Type: {last_trade['type']}, Volume: {last_trade['volume']}, SL: {last_trade['sl']}, TP: {last_trade['tp']}, Profit: {last_trade['profit']}"

# ✅ Send Telegram Alert
def send_telegram_alert(message):
    """Send a message via Telegram bot."""
    bot_token = "YOUR_BOT_TOKEN"
    chat_id = "YOUR_CHAT_ID"
    bot = telegram.Bot(token=bot_token)
    
    try:
        bot.send_message(chat_id=chat_id, text=message)
    except Exception as e:
        logging.error(f"Telegram alert failed: {e}")

# ✅ RL Agent Class
class RLAgent:
    def __init__(self, state_size, action_size, filename="rl_model.pth"):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self.build_model()
        
        if os.path.exists(filename):
            self.model.load_state_dict(torch.load(filename))
            logging.info(f"RL model loaded from {filename}")
        else:
            logging.warning("No RL model found. Training a new one.")

    def build_model(self):
        """Build a simple neural network for RL."""
        model = nn.Sequential(
            nn.Linear(self.state_size, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size),
            nn.Softmax(dim=1)
        )
        return model

    def choose_action(self, state):
        """Choose an action based on the current state."""
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        probs = self.model(state)
        action = torch.argmax(probs).item()
        return action

# Advanced News Filtering
def is_high_impact_news(symbol):
    """Check if there is high-impact news for the symbol using Forex Factory API."""
    url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    response = requests.get(url).json()
    for event in response:
        if symbol[:3] in event['currency'] or symbol[3:] in event['currency']:
            if event['impact'] == 'High':
                return True
    return False

# Check equity
def check_equity(balance, equity):
    """Check if equity is below the minimum allowed threshold."""
    min_equity_threshold = balance * (MIN_EQUITY_PERCENT / 100)
    if equity < min_equity_threshold:
        logging.warning(f"Equity ({equity}) is below the minimum allowed ({min_equity_threshold}). Stopping trading.")
        return True
    return False

# Analyze historical trades
def analyze_historical_trades():
    """Analyze historical trades to refine future trading decisions."""
    if not historical_trades:
        return RISK_PER_TRADE_PERCENT  # Default risk percentage

    winning_trades = [trade for trade in historical_trades if trade["profit"] > 0]
    win_rate = len(winning_trades) / len(historical_trades) if historical_trades else 0

    # Adjust risk percentage based on win rate
    if win_rate > 0.6:
        return RISK_PER_TRADE_PERCENT * 1.2  # Increase risk if win rate is high
    elif win_rate < 0.4:
        return RISK_PER_TRADE_PERCENT * 0.8  # Decrease risk if win rate is low
    return RISK_PER_TRADE_PERCENT

# Save trade history
def save_trade_history(historical_trades):
    """Save trade history to a CSV file."""
    trade_df = pd.DataFrame(historical_trades)
    trade_df.to_csv("trade_history.csv", index=False)
    logging.info("Trade history saved to trade_history.csv.")

def place_trade(symbol, volume, trade_type, sl, tp):
    """Place a trade on MT5."""
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logging.error(f"Failed to get symbol info for {symbol}")
        return

    # Get the latest bid and ask prices
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        logging.error(f"Failed to get tick data for {symbol}")
        return

    price = tick.ask if trade_type == mt5.ORDER_TYPE_BUY else tick.bid

    # Validate SL and TP
    sl, tp = validate_stops(symbol, price, sl, tp, trade_type)
    if sl is None or tp is None:
        logging.error(f"Invalid stop levels for {symbol}. Skipping trade.")
        return

    # Create order request
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": trade_type,
        "price": price,
        "sl": sl,  # Stop Loss
        "tp": tp,  # Take Profit
        "deviation": 10,
        "magic": 123456,
        "comment": "SMC trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)

    # Check if trade was placed successfully
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.error(f"Trade placement failed: {result}")
    else:
        logging.info(f"Trade placed successfully: {result}")

        historical_trades.append({
            "symbol": symbol,
            "type": "BUY" if trade_type == mt5.ORDER_TYPE_BUY else "SELL",
            "volume": volume,
            "sl": sl,
            "tp": tp,
            "profit": 0,  # Will be updated when the trade is closed
            "timestamp": datetime.now()
        })

# Validate stop levels
def validate_stops(symbol, price, sl, tp, trade_type):
    """Validate SL and TP levels to ensure they meet broker requirements."""
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logging.error(f"Failed to get symbol info for {symbol}")
        return None, None

    # Get minimum stop distance (in points)
    min_stop_distance = symbol_info.trade_stops_level * symbol_info.point if symbol_info.trade_stops_level else symbol_info.point * 10

    # Ensure SL and TP respect minimum stop distance
    if trade_type == mt5.ORDER_TYPE_BUY:
        if sl >= price - min_stop_distance:
            sl = price - min_stop_distance  # Adjust SL downwards
        if tp <= price + min_stop_distance:
            tp = price + min_stop_distance  # Adjust TP upwards
    elif trade_type == mt5.ORDER_TYPE_SELL:
        if sl <= price + min_stop_distance:
            sl = price + min_stop_distance  # Adjust SL upwards
        if tp >= price - min_stop_distance:
            tp = price - min_stop_distance  # Adjust TP downwards

    return sl, tp

# ✅ Main Trading Bot
def smc_trading_bot(symbols, timeframe=mt5.TIMEFRAME_M15):
    """Main loop for the SMC trading bot."""
    model = RLAgent(state_size=3, action_size=3)  # RL Agent for decision-making

    while True:
        balance, equity = get_account_info()
        if balance is None:
            continue

        # Fetch market data
        for symbol in symbols:
            df = get_market_data(symbol, timeframe, 100)
            if df is None:
                continue

            # Add indicators before using RL agent
            df = add_technical_indicators(df)

            # Ensure indicators are calculated
            if not {'rsi', 'macd', 'bb_middle'}.issubset(df.columns):
                logging.error("Missing indicators in DataFrame. Skipping trade.")
                continue

            # Define RL state variables
            state = df.iloc[-1][['rsi', 'macd', 'bb_middle']].values
            action = model.choose_action(state)

            # Define trade parameters
            latest_price = df['close'].iloc[-1]
            sl_pips = 50  # Stop-loss pips
            pip_value = 0.1  # Pip value placeholder
            volume = calculate_position_size(balance, RISK_PER_TRADE_PERCENT, sl_pips, pip_value, symbol)

            sl = latest_price - (sl_pips * 0.0001) if action == 0 else latest_price + (sl_pips * 0.0001)
            tp = latest_price + (100 * 0.0001) if action == 0 else latest_price - (100 * 0.0001)

            # Trade execution based on RL decision
            if action == 0:  # Buy
                trade_type = mt5.ORDER_TYPE_BUY
            elif action == 1:  # Sell
                trade_type = mt5.ORDER_TYPE_SELL
            else:
                logging.info("RL agent decided to stay out of the market.")
                continue

            # Place trade
            place_trade(symbol, volume, trade_type, sl, tp)

            # Send trade execution alert
            send_telegram_alert(f"Trade executed: {get_trade_details()}")

        time.sleep(60)

# ✅ Run the Trading Bot
if __name__ == "__main__":
    symbols = ["BTCUSD", "ETHUSD"]
    smc_trading_bot(symbols)