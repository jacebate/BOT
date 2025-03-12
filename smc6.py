import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta

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

# Store historical trade data for backtesting
historical_trades = []

def get_account_info():
    """Retrieve account balance and equity."""
    account_info = mt5.account_info()
    if account_info is None:
        logging.error("Failed to get account info")
        return None
    return account_info.balance, account_info.equity

def get_market_data(symbol, timeframe, bars=100):
    """Fetch historical market data."""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None:
        logging.error(f"Failed to get market data for {symbol}")
        return None
    return pd.DataFrame(rates)[['time', 'open', 'high', 'low', 'close']]

def detect_break_of_structure(df):
    """Identify Break of Structure (BOS)."""
    df['prev_high'] = df['high'].shift(1)
    df['prev_low'] = df['low'].shift(1)
    df['BOS_up'] = df['high'] > df['prev_high']
    df['BOS_down'] = df['low'] < df['prev_low']
    return df

def find_order_blocks(df):
    """Identify potential order blocks."""
    bullish_ob = df[(df['BOS_up']) & (df['low'] < df['low'].shift(1))]
    bearish_ob = df[(df['BOS_down']) & (df['high'] > df['high'].shift(1))]
    return bullish_ob, bearish_ob

def get_pip_value(symbol):
    """Retrieve the pip value for a given symbol."""
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info:
        return symbol_info.point * 10  # Assuming 1 pip = 10 points
    return 0.1  # Default fallback

def calculate_position_size(balance, risk_percent, sl_pips, pip_value, symbol):
    """Calculate trade volume based on account balance and risk percentage."""
    risk_amount = balance * (risk_percent / 100)
    lot_size = risk_amount / (sl_pips * pip_value)

    # Get the minimum and maximum allowed lot size for the symbol
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logging.error(f"Failed to get symbol info for {symbol}")
        return 0.0

    min_lot = symbol_info.volume_min
    max_lot = symbol_info.volume_max
    lot_step = symbol_info.volume_step

    # Round the lot size to the nearest step
    lot_size = round(lot_size / lot_step) * lot_step

    # Ensure the lot size is within the allowed range
    return max(min_lot, min(lot_size, max_lot))

def validate_stops(symbol, price, sl, tp, trade_type):
    """Validate SL and TP levels to ensure they meet broker requirements."""
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        logging.error(f"Failed to get symbol info for {symbol}")
        return None, None  # Ensure valid return values

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

def place_trade(symbol, volume, trade_type, sl, tp):
    """Place a trade on MT5 with a spread check."""
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
    spread = abs(tick.ask - tick.bid)  # Spread in points

    # Check if spread is too high (e.g., greater than 20 points)
    # Set max spread as a percentage of the average spread over recent ticks
    recent_ticks = mt5.copy_ticks_from(symbol, time.time() - 300, 100, mt5.COPY_TICKS_INFO)
    if recent_ticks is not None and len(recent_ticks) > 0:
        avg_spread = np.mean([tick['ask'] - tick['bid'] for tick in recent_ticks])
        max_allowed_spread = avg_spread * 1.5  # Allow up to 1.5x the average spread
    else:
        max_allowed_spread = symbol_info.point * 30  # Default fallback

    if spread > max_allowed_spread:
        logging.warning(f"Spread too high for {symbol} ({spread} points). Skipping trade.")
        return  # Avoid entering trades with high spreads

    # Validate SL and TP
    sl, tp = validate_stops(symbol, price, sl, tp, trade_type)
    if not sl or not tp:
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

def monitor_trades():
    """Monitor open trades and close if SL or TP is hit."""
    positions = mt5.positions_get()
    if positions is None or len(positions) == 0:
        return
    for position in positions:
        tick = mt5.symbol_info_tick(position.symbol)
        if tick is None:
            continue
        current_price = tick.ask if position.type == mt5.ORDER_TYPE_BUY else tick.bid
        if (position.type == mt5.ORDER_TYPE_BUY and current_price <= position.sl) or \
           (position.type == mt5.ORDER_TYPE_SELL and current_price >= position.sl):
            close_trade(position)

def close_trade(position):
    """Close a trade when SL or TP is hit."""
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
        "position": position.ticket,
        "price": mt5.symbol_info_tick(position.symbol).bid if position.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(position.symbol).ask,
        "deviation": 10,
        "magic": 123456,
        "comment": "SL/TP Hit",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logging.error(f"Failed to close trade: {result}")
    else:
        logging.info(f"Trade closed successfully: {result}")
        # Update historical trade data with profit/loss
        for trade in historical_trades:
            if trade["symbol"] == position.symbol and trade["type"] == ("BUY" if position.type == mt5.ORDER_TYPE_BUY else "SELL"):
                trade["profit"] = position.profit
                break

def check_equity(balance, equity):
    """Check if equity is less than 10% of the account balance."""
    if equity < balance * (MIN_EQUITY_PERCENT / 100):
        logging.warning(f"Equity ({equity}) is less than {MIN_EQUITY_PERCENT}% of the account balance ({balance}). Stopping trading.")
        return True
    return False

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

def check_equity(balance, equity):
    """Restrict trading if equity falls below 89.5% of balance."""
    min_equity_threshold = balance * 0.895  # 89.5% of balance

    if equity < min_equity_threshold:
        logging.warning(f"Equity ({equity}) is below the minimum allowed ({min_equity_threshold}). Stopping trading.")
        return True  # Stop trading

    return False  # Continue trading


def smc_trading_bot(symbols, timeframe=mt5.TIMEFRAME_M15):
    """Main loop for the SMC trading bot."""
    while True:
        balance, equity = get_account_info()
        if balance is None:
            continue

        # Check if equity is below the minimum threshold
        if check_equity(balance, equity):
            break

        # Analyze historical trades to adjust risk percentage
        risk_percent = analyze_historical_trades()

        positions = mt5.positions_get()
        if positions is None:
            positions = []

        # Only proceed if the number of open trades is less than MAX_TRADES
        if len(positions) >= MAX_TRADES:
            logging.info(f"Maximum number of trades ({MAX_TRADES}) reached. Skipping new trades.")
            time.sleep(60)
            continue

        for symbol in symbols:
            df = get_market_data(symbol, timeframe, 100)
            if df is None:
                continue
            df = detect_break_of_structure(df)
            bullish_ob, bearish_ob = find_order_blocks(df)
            latest_price = df['close'].iloc[-1]

            if not bullish_ob.empty:
                ob_price = bullish_ob['low'].iloc[-1]
                sl_pips = abs(latest_price - ob_price) * 10000
                pip_value = 0.1  # Placeholder for actual pip value
                volume = calculate_position_size(balance, risk_percent, sl_pips, pip_value, symbol)
                sl = ob_price - (0.002 * ob_price)
                tp = latest_price + (0.004 * latest_price)
                place_trade(symbol, volume, mt5.ORDER_TYPE_BUY, sl, tp)
            elif not bearish_ob.empty:
                ob_price = bearish_ob['high'].iloc[-1]
                sl_pips = abs(latest_price - ob_price) * 10000
                pip_value = 0.1  # Placeholder for actual pip value
                volume = calculate_position_size(balance, risk_percent, sl_pips, pip_value, symbol)
                sl = ob_price + (0.002 * ob_price)
                tp = latest_price - (0.004 * latest_price)
                place_trade(symbol, volume, mt5.ORDER_TYPE_SELL, sl, tp)
        
        monitor_trades()
        time.sleep(60)  # Wait before checking again

if __name__ == "__main__":
    symbols = ["BTCUSD", "ETHUSD"]  # Add more symbols here if needed
    smc_trading_bot(symbols)
