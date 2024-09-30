import streamlit as st
import os
import json
import sys
import pandas as pd
import mplfinance as mpf
from binance.client import Client
from binance import ThreadedWebsocketManager
import requests
import logging
from textblob import TextBlob
from datetime import datetime
import os
import json
import sys
import threading

# Load API credentials from a separate file
CREDENTIALS_FILE = 'api_credentials.json'
if not os.path.exists(CREDENTIALS_FILE):
    logging.error(f"{CREDENTIALS_FILE} not found. Please provide the API credentials.")
    sys.exit(1)

with open(CREDENTIALS_FILE, 'r') as file:
    credentials = json.load(file)
    api_key = credentials.get('API_KEY')
    api_secret = credentials.get('API_SECRET')

if not api_key or not api_secret:
    logging.error("API credentials not found in the credentials file.")
    sys.exit(1)

# Set up Binance client
client = Client(api_key, api_secret)

# Telegram bot details (consider loading these from a secure file or environment variables)
TOKEN = 'your_telegram_token'
CHAT_ID = 'your_telegram_chat_id'

# Fetch BTC/USDT data from Binance (e.g., 1 day candles for the last 100 days)
symbol = 'BTCUSDT'
candles = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1DAY, limit=100)

# Process data into a pandas DataFrame
data = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                      'close_time', 'quote_asset_volume', 'number_of_trades', 
                                      'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])

# Convert timestamps to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
data.set_index('timestamp', inplace=True)

# Convert columns to numeric values
data['open'] = pd.to_numeric(data['open'])
data['high'] = pd.to_numeric(data['high'])
data['low'] = pd.to_numeric(data['low'])
data['close'] = pd.to_numeric(data['close'])
data['volume'] = pd.to_numeric(data['volume'])

# Improved logging function to track trades and patterns
def log_trade_info(trade_side, symbol, entry_price, stop_loss, take_profit, result=None):
    logging.info(f"Trade: {trade_side} {symbol} at {entry_price}, SL: {stop_loss}, TP: {take_profit}. Result: {result if result else 'Open'}")

# Streamlit Dashboard Initialization
st.title("Trading Dashboard")
st.sidebar.title("Trading Bot Settings")
selected_symbol = st.sidebar.selectbox("Select Symbol", ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'])
risk_pct = st.sidebar.slider("Risk Percentage per Trade", 1, 10, 2)
stop_loss_pct = st.sidebar.slider("Stop Loss Percentage", 1, 10, 3)
take_profit_pct = st.sidebar.slider("Take Profit Percentage", 1, 20, 5)

# Function to send Telegram alert
def send_telegram_alert(pattern, timestamp, symbol):
    message = f"A candlestick pattern '{pattern}' was detected on {timestamp} for {symbol}."
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        'chat_id': CHAT_ID,
        'text': message
    }
    try:
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            logging.info(f"Telegram alert sent: {pattern} detected on {timestamp}")
        else:
            logging.error(f"Failed to send Telegram alert: {response.text}")
    except Exception as e:
        logging.error(f"Error sending Telegram alert: {str(e)}")

# Sentiment analysis on headlines
def analyze_sentiment(news_headlines):
    polarity = 0
    for headline in news_headlines:
        analysis = TextBlob(headline)
        polarity += analysis.sentiment.polarity
    return polarity / len(news_headlines) if news_headlines else 0

# Get latest news headlines for sentiment analysis
def get_news_headlines():
    try:
        response = requests.get('https://api.currentsapi.services/v1/latest-news?apiKey=your_api_key')
        headlines = [news['title'] for news in response.json()['news']]
        return headlines
    except Exception as e:
        logging.error(f"Error fetching news headlines: {str(e)}")
        return []

# Volume-based pattern confirmation
def confirm_pattern_with_volume(data, pattern_timestamp):
    avg_volume = data['volume'].rolling(window=20).mean()
    volume_at_pattern = data.loc[pattern_timestamp]['volume']
    return volume_at_pattern > avg_volume.loc[pattern_timestamp] * 1.5

# Calculate position size based on risk tolerance
def calculate_position_size(balance, risk_pct, entry_price, stop_loss_price):
    risk_amount = balance * (risk_pct / 100)
    position_size = risk_amount / abs(entry_price - stop_loss_price)
    return position_size

# Place a trade with stop-loss and take-profit
def place_trade(symbol, trade_side, quantity, stop_loss_pct, take_profit_pct):
    try:
        # Get current price				   
        ticker = client.get_symbol_ticker(symbol=symbol)
        current_price = float(ticker['price'])
        # Get current price										  
										  
        order = client.order_market(symbol=symbol, side=trade_side, quantity=quantity)
		
        # Calculate stop-loss and take-profit prices												
        stop_loss_price = current_price * (1 - stop_loss_pct / 100) if trade_side == 'BUY' else current_price * (1 + stop_loss_pct / 100)
        take_profit_price = current_price * (1 + take_profit_pct / 100) if trade_side == 'BUY' else current_price * (1 - take_profit_pct / 100)
        
        # Set stop-loss and take-profit orders
        client.create_order(symbol=symbol, side='SELL' if trade_side == 'BUY' else 'BUY',
                            type='STOP_LOSS_LIMIT', quantity=quantity, stopPrice=stop_loss_price, price=stop_loss_price)
        client.create_order(symbol=symbol, side='SELL' if trade_side == 'BUY' else 'BUY',
                            type='TAKE_PROFIT_LIMIT', quantity=quantity, stopPrice=take_profit_price, price=take_profit_price)
        
        log_trade_info(trade_side, symbol, current_price, stop_loss_price, take_profit_price)
    except Exception as e:
        logging.error(f"Error placing trade: {e}")

# Real-time data fetching via WebSocket
def handle_socket_message(msg):
    global data
    if msg['e'] == 'kline':
        kline = msg['k']
        timestamp = pd.to_datetime(kline['t'], unit='ms')
        new_row = {
            'timestamp': timestamp,
            'open': float(kline['o']),
            'high': float(kline['h']),
            'low': float(kline['l']),
            'close': float(kline['c']),
            'volume': float(kline['v'])
        }
        data = data.append(pd.DataFrame([new_row]).set_index('timestamp'))
        logging.info(f"New candle added for {timestamp}")

# Run WebSocket for real-time data in a separate thread
def run_websocket(symbol):
    def websocket_thread():
        twm = ThreadedWebsocketManager(api_key=api_key, api_secret=api_secret)
        twm.start()
        twm.start_kline_socket(callback=handle_socket_message, symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE)
        twm.join()
    
    thread = threading.Thread(target=websocket_thread)
    thread.daemon = True  # So it closes when the app is closed
    thread.start()

# Detect and alert patterns with sentiment analysis and volume confirmation
def detect_patterns_with_alerts(data, symbol):
    hammer_patterns = detect_hammer(data)
    three_line_strike_patterns = detect_three_line_strike(data)
    sentiment = analyze_sentiment(get_news_headlines())

    for pattern in patterns:
        pattern_timestamp = pattern[0]
        pattern_name = pattern[1]
        if confirm_pattern_with_volume(data, pattern_timestamp) and sentiment > 0:
            logging.info(f"Confirmed {pattern_name} on {pattern_timestamp} for {symbol}")
            send_telegram_alert(pattern_name, pattern_timestamp, symbol)

# Example: Detecting various patterns
def detect_hammer(data):
    patterns = []
    for i in range(1, len(data)):
        body = abs(data['close'].iloc[i] - data['open'].iloc[i])
        lower_shadow = data['low'].iloc[i] - min(data['open'].iloc[i], data['close'].iloc[i])
        upper_shadow = data['high'].iloc[i] - max(data['open'].iloc[i], data['close'].iloc[i])
        if lower_shadow > body * 2 and upper_shadow < body:
            patterns.append((data.index[i], 'Hammer'))
			
    return patterns

def detect_three_line_strike(data):
    patterns = []
    for i in range(3, len(data)):
        if data['close'].iloc[i-3] < data['open'].iloc[i-3] and \
           data['close'].iloc[i-2] < data['open'].iloc[i-2] and \
           data['close'].iloc[i-1] < data['open'].iloc[i-1] and \
           data['close'].iloc[i] > data['open'].iloc[i] and \
           data['close'].iloc[i] > data['open'].iloc[i-3]:
            patterns.append((data.index[i], 'Bullish Three Line Strike'))
            
        if data['close'].iloc[i-3] > data['open'].iloc[i-3] and \
           data['close'].iloc[i-2] > data['open'].iloc[i-2] and \
           data['close'].iloc[i-1] > data['open'].iloc[i-1] and \
           data['close'].iloc[i] < data['open'].iloc[i] and \
           data['close'].iloc[i] < data['open'].iloc[i-3]:
            patterns.append((data.index[i], 'Bearish Three Line Strike'))
    
    return patterns
	
def detect_morning_star(data):
    patterns = []
    for i in range(2, len(data)):
        if data['close'].iloc[i-2] < data['open'].iloc[i-2] and \
           abs(data['close'].iloc[i-1] - data['open'].iloc[i-1]) < (data['high'].iloc[i-1] - data['low'].iloc[i-1]) * 0.1 and \
           data['close'].iloc[i] > data['open'].iloc[i] and \
           data['close'].iloc[i] > (data['close'].iloc[i-2] + data['open'].iloc[i-2]) / 2:
            patterns.append((data.index[i], 'Morning Star'))
    
    return patterns
	
def detect_evening_star(data):
    patterns = []
    for i in range(2, len(data)):
        if data['close'].iloc[i-2] > data['open'].iloc[i-2] and \
           abs(data['close'].iloc[i-1] - data['open'].iloc[i-1]) < (data['high'].iloc[i-1] - data['low'].iloc[i-1]) * 0.1 and \
           data['close'].iloc[i] < data['open'].iloc[i] and \
           data['close'].iloc[i] < (data['close'].iloc[i-2] + data['open'].iloc[i-2]) / 2:
            patterns.append((data.index[i], 'Evening Star'))
    
    return patterns
	
def detect_abandoned_baby(data):
    patterns = []
    for i in range(2, len(data)):
        if abs(data['close'].iloc[i-1] - data['open'].iloc[i-1]) < (data['high'].iloc[i-1] - data['low'].iloc[i-1]) * 0.1:
            if data['close'].iloc[i-2] > data['open'].iloc[i-2] and data['close'].iloc[i] < data['open'].iloc[i] and \
               data['low'].iloc[i-1] > data['high'].iloc[i-2] and data['high'].iloc[i-1] < data['low'].iloc[i]:
                patterns.append((data.index[i], 'Bearish Abandoned Baby'))
            elif data['close'].iloc[i-2] < data['open'].iloc[i-2] and data['close'].iloc[i] > data['open'].iloc[i] and \
               data['low'].iloc[i-1] > data['high'].iloc[i-2] and data['high'].iloc[i-1] < data['low'].iloc[i]:
                patterns.append((data.index[i], 'Bullish Abandoned Baby'))
    
    return patterns

# Example: Detecting various patterns
hammer_patterns = detect_hammer(data)
three_line_strike_patterns = detect_three_line_strike(data)
morning_star_patterns = detect_morning_star(data)
evening_star_patterns = detect_evening_star(data)
abandoned_baby_patterns = detect_abandoned_baby(data)

# Combining all patterns detected
all_patterns = hammer_patterns + three_line_strike_patterns + morning_star_patterns + evening_star_patterns + abandoned_baby_patterns

# Mark the patterns on the chart
apdict = [mpf.make_addplot(data['close'], color='g')]

# Highlighting detected patterns on the chart
for pattern in all_patterns:
    print(f"Detected {pattern[1]} on {pattern[0].strftime('%Y-%m-%d')}")

# Visualizing the chart with the detected patterns
mpf.plot(data[['open', 'high', 'low', 'close', 'volume']], type='candle', volume=True, 
         title='BTC/USDT Daily with Detected Patterns', style='charles', mav=(3,6,9), addplot=apdict)



# Main function to handle multiple assets and run the analysis
def analyze_all_assets(symbols):
    for symbol in symbols:
        candles = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1DAY, limit=100)
        data = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data.set_index('timestamp', inplace=True)
        detect_patterns_with_alerts(data, symbol)
        # Display candlestick chart in the dashboard
        st.subheader(f"{symbol} Candlestick Chart")
        mpf.plot(data[['open', 'high', 'low', 'close']], type='candle', style='charles', title=f"{symbol} Daily Chart", volume=True)

def backtest_pattern_performance(data, pattern_timestamps):
    results = []
    for timestamp in pattern_timestamps:
        pattern_index = data.index.get_loc(timestamp)
        try:
            # Check the price movement after the pattern (e.g., 3 candles later)
            price_after_3 = data['close'].iloc[pattern_index + 3]
            price_at_pattern = data['close'].iloc[pattern_index]
            pct_change = (price_after_3 - price_at_pattern) / price_at_pattern * 100
            results.append(pct_change)
        except IndexError:
            # If not enough candles, ignore
            continue
    
    # Calculate statistics
    avg_move = sum(results) / len(results) if results else 0
    win_rate = len([r for r in results if r > 0]) / len(results) if results else 0
    print(f"Average price movement after pattern: {avg_move:.2f}%")
    print(f"Win rate: {win_rate * 100:.2f}%")

# Sidebar Settings
st.sidebar.header("Settings")
symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
selected_symbols = st.sidebar.multiselect("Select Symbols", symbols, default=symbols)

# Run analysis
if st.button("Run Bot"):
    analyze_all_assets(selected_symbols)

# Start WebSocket for real-time data
if st.button("Start WebSocket for BTCUSDT"):
    run_websocket('BTCUSDT')

# Main Loop: Detect Patterns and Run WebSocket
if __name__ == "__main__":
    # Detect initial patterns in the historical data
    detect_patterns_with_alerts(data, symbol)

    # Start the WebSocket for real-time data and pattern detection
    run_websocket(symbol)

# Binance API URL
BASE_URL = 'https://api.binance.com'

# Load API credentials from a separate file
CREDENTIALS_FILE = 'api_credentials.json'
if not os.path.exists(CREDENTIALS_FILE):
    logging.error(f"{CREDENTIALS_FILE} not found. Please provide the API credentials.")
    sys.exit(1)

with open(CREDENTIALS_FILE, 'r') as file:
    credentials = json.load(file)
    api_key = credentials.get('API_KEY')
    api_secret = credentials.get('API_SECRET')

if not api_key or not api_secret:
    logging.error("API credentials not found in the credentials file.")
    sys.exit(1)

# Set up Binance client
client = Client(api_key, api_secret)

# Telegram bot details (consider loading these from a secure file or environment variables)
TOKEN = 'your_telegram_token'
CHAT_ID = 'your_telegram_chat_id'

# Fetch BTC/USDT data from Binance (e.g., 1 day candles for the last 100 days)
symbol = 'BTCUSDT'
candles = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1DAY, limit=100)

# Process data into a pandas DataFrame
data = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                      'close_time', 'quote_asset_volume', 'number_of_trades', 
                                      'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])

# Convert timestamps to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
data.set_index('timestamp', inplace=True)

# Convert columns to numeric values
data['open'] = pd.to_numeric(data['open'])
data['high'] = pd.to_numeric(data['high'])
data['low'] = pd.to_numeric(data['low'])
data['close'] = pd.to_numeric(data['close'])
data['volume'] = pd.to_numeric(data['volume'])

# Plot candlestick chart using mplfinance
mpf.plot(data[['open', 'high', 'low', 'close', 'volume']], type='candle', volume=True, 
         title='BTC/USDT Daily Candlestick Chart', style='charles', mav=(3,6,9))

# Streamlit Dashboard Initialization
st.title("Trading Dashboard")
st.sidebar.title("Trading Bot Settings")
selected_symbol = st.sidebar.selectbox("Select Symbol", ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'])
risk_pct = st.sidebar.slider("Risk Percentage per Trade", 1, 10, 2)
stop_loss_pct = st.sidebar.slider("Stop Loss Percentage", 1, 10, 3)
take_profit_pct = st.sidebar.slider("Take Profit Percentage", 1, 20, 5)

# Function to send Telegram alert
def send_telegram_alert(pattern, timestamp, symbol):
    message = f"A candlestick pattern '{pattern}' was detected on {timestamp} for {symbol}."
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        'chat_id': CHAT_ID,
        'text': message
    }
    try:
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            logging.info(f"Telegram alert sent: {pattern} detected on {timestamp}")
        else:
            logging.error(f"Failed to send Telegram alert: {response.text}")
    except Exception as e:
        logging.error(f"Error sending Telegram alert: {str(e)}")

# Sentiment analysis on headlines
def analyze_sentiment(news_headlines):
    polarity = 0
    for headline in news_headlines:
        analysis = TextBlob(headline)
        polarity += analysis.sentiment.polarity
    return polarity / len(news_headlines) if news_headlines else 0

# Get latest news headlines for sentiment analysis
def get_news_headlines():
    try:
        response = requests.get('https://api.currentsapi.services/v1/latest-news?apiKey=your_api_key')
        headlines = [news['title'] for news in response.json()['news']]
        return headlines
    except Exception as e:
        logging.error(f"Error fetching news headlines: {str(e)}")
        return []

# Volume-based pattern confirmation
def confirm_pattern_with_volume(data, pattern_timestamp):
    avg_volume = data['volume'].rolling(window=20).mean()
    volume_at_pattern = data.loc[pattern_timestamp]['volume']
    return volume_at_pattern > avg_volume.loc[pattern_timestamp] * 1.5

# Calculate position size based on risk tolerance
def calculate_position_size(balance, risk_pct, entry_price, stop_loss_price):
    risk_amount = balance * (risk_pct / 100)
    position_size = risk_amount / abs(entry_price - stop_loss_price)
    return position_size

# Place a trade with stop-loss and take-profit
def place_trade(symbol, trade_side, quantity, stop_loss_pct, take_profit_pct):
    try:
        # Get current price
        ticker = client.get_symbol_ticker(symbol=symbol)
        current_price = float(ticker['price'])
        
        # Place market order (buy or sell)
        order = client.order_market(symbol=symbol, side=trade_side, quantity=quantity)
        
        # Calculate stop-loss and take-profit prices
        stop_loss_price = current_price * (1 - stop_loss_pct / 100) if trade_side == 'BUY' else current_price * (1 + stop_loss_pct / 100)
        take_profit_price = current_price * (1 + take_profit_pct / 100) if trade_side == 'BUY' else current_price * (1 - take_profit_pct / 100)
        
        # Set stop-loss and take-profit orders
        client.create_order(symbol=symbol, side='SELL' if trade_side == 'BUY' else 'BUY',
                            type='STOP_LOSS_LIMIT', quantity=quantity, stopPrice=stop_loss_price, price=stop_loss_price)
        client.create_order(symbol=symbol, side='SELL' if trade_side == 'BUY' else 'BUY',
                            type='TAKE_PROFIT_LIMIT', quantity=quantity, stopPrice=take_profit_price, price=take_profit_price)
        
        print(f"Placed {trade_side} trade for {symbol} at {current_price}. Stop-loss: {stop_loss_price}, Take-profit: {take_profit_price}")
    except Exception as e:
        print(f"Error placing trade: {e}")

# Real-time data fetching via WebSocket
def handle_socket_message(msg):
    global data
    if msg['e'] == 'kline':
        kline = msg['k']
        timestamp = pd.to_datetime(kline['t'], unit='ms')
        new_row = {
            'timestamp': timestamp,
            'open': float(kline['o']),
            'high': float(kline['h']),
            'low': float(kline['l']),
            'close': float(kline['c']),
            'volume': float(kline['v'])
        }
        data = data.append(pd.DataFrame([new_row]).set_index('timestamp'))
        logging.info(f"New candle added for {timestamp}")

# Run WebSocket for real-time data in a separate thread
def run_websocket(symbol):
    def websocket_thread():
        twm = ThreadedWebsocketManager(api_key=api_key, api_secret=api_secret)
        twm.start()
        twm.start_kline_socket(callback=handle_socket_message, symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE)
        twm.join()
    
    thread = threading.Thread(target=websocket_thread)
    thread.daemon = True  # So it closes when the app is closed
    thread.start()

# Detect and alert patterns with sentiment analysis and volume confirmation
def detect_patterns_with_alerts(data, symbol):
    hammer_patterns = detect_hammer(data)
    sentiment = analyze_sentiment(get_news_headlines())

    for pattern in hammer_patterns:
        pattern_timestamp = pattern[0]
        pattern_name = pattern[1]
        if confirm_pattern_with_volume(data, pattern_timestamp) and sentiment > 0:
            logging.info(f"Confirmed {pattern_name} on {pattern_timestamp} for {symbol}")
            send_telegram_alert(pattern_name, pattern_timestamp, symbol)

# Pattern detection functions (simplified example: Hammer pattern)
def detect_three_line_strike(data):
    patterns = []
    for i in range(3, len(data)):
        if data['close'].iloc[i-3] < data['open'].iloc[i-3] and \
           data['close'].iloc[i-2] < data['open'].iloc[i-2] and \
           data['close'].iloc[i-1] < data['open'].iloc[i-1] and \
           data['close'].iloc[i] > data['open'].iloc[i] and \
           data['close'].iloc[i] > data['open'].iloc[i-3]:
            patterns.append((data.index[i], 'Bullish Three Line Strike'))
            
        if data['close'].iloc[i-3] > data['open'].iloc[i-3] and \
           data['close'].iloc[i-2] > data['open'].iloc[i-2] and \
           data['close'].iloc[i-1] > data['open'].iloc[i-1] and \
           data['close'].iloc[i] < data['open'].iloc[i] and \
           data['close'].iloc[i] < data['open'].iloc[i-3]:
            patterns.append((data.index[i], 'Bearish Three Line Strike'))
    
    return patterns

def detect_morning_star(data):
    patterns = []
    for i in range(2, len(data)):
        if data['close'].iloc[i-2] < data['open'].iloc[i-2] and \
           abs(data['close'].iloc[i-1] - data['open'].iloc[i-1]) < (data['high'].iloc[i-1] - data['low'].iloc[i-1]) * 0.1 and \
           data['close'].iloc[i] > data['open'].iloc[i] and \
           data['close'].iloc[i] > (data['close'].iloc[i-2] + data['open'].iloc[i-2]) / 2:
            patterns.append((data.index[i], 'Morning Star'))
    
    return patterns

def detect_evening_star(data):
    patterns = []
    for i in range(2, len(data)):
        if data['close'].iloc[i-2] > data['open'].iloc[i-2] and \
           abs(data['close'].iloc[i-1] - data['open'].iloc[i-1]) < (data['high'].iloc[i-1] - data['low'].iloc[i-1]) * 0.1 and \
           data['close'].iloc[i] < data['open'].iloc[i] and \
           data['close'].iloc[i] < (data['close'].iloc[i-2] + data['open'].iloc[i-2]) / 2:
            patterns.append((data.index[i], 'Evening Star'))
    
    return patterns

def detect_abandoned_baby(data):
    patterns = []
    for i in range(2, len(data)):
        if abs(data['close'].iloc[i-1] - data['open'].iloc[i-1]) < (data['high'].iloc[i-1] - data['low'].iloc[i-1]) * 0.1:
            if data['close'].iloc[i-2] > data['open'].iloc[i-2] and data['close'].iloc[i] < data['open'].iloc[i] and \
               data['low'].iloc[i-1] > data['high'].iloc[i-2] and data['high'].iloc[i-1] < data['low'].iloc[i]:
                patterns.append((data.index[i], 'Bearish Abandoned Baby'))
            elif data['close'].iloc[i-2] < data['open'].iloc[i-2] and data['close'].iloc[i] > data['open'].iloc[i] and \
               data['low'].iloc[i-1] > data['high'].iloc[i-2] and data['high'].iloc[i-1] < data['low'].iloc[i]:
                patterns.append((data.index[i], 'Bullish Abandoned Baby'))
    
    return patterns

# Advanced Pattern Detection Functions
def detect_hammer(data):
    patterns = []
    for i in range(1, len(data)):
        body = abs(data['close'].iloc[i] - data['open'].iloc[i])
        lower_shadow = data['low'].iloc[i] - min(data['open'].iloc[i], data['close'].iloc[i])
        upper_shadow = data['high'].iloc[i] - max(data['open'].iloc[i], data['close'].iloc[i])
        
        if lower_shadow > body * 2 and upper_shadow < body:
            patterns.append((data.index[i], 'Hammer'))
    
    return patterns

# Example: Detecting various patterns
hammer_patterns = detect_hammer(data)
three_line_strike_patterns = detect_three_line_strike(data)
morning_star_patterns = detect_morning_star(data)
evening_star_patterns = detect_evening_star(data)
abandoned_baby_patterns = detect_abandoned_baby(data)

# Combining all patterns detected
all_patterns = hammer_patterns + three_line_strike_patterns + morning_star_patterns + evening_star_patterns + abandoned_baby_patterns

# Mark the patterns on the chart
apdict = [mpf.make_addplot(data['close'], color='g')]

# Highlighting detected patterns on the chart
for pattern in all_patterns:
    print(f"Detected {pattern[1]} on {pattern[0].strftime('%Y-%m-%d')}")

# Visualizing the chart with the detected patterns
mpf.plot(data[['open', 'high', 'low', 'close', 'volume']], type='candle', volume=True, 
         title='BTC/USDT Daily with Detected Patterns', style='charles', mav=(3,6,9), addplot=apdict)

# Main function to handle multiple assets and run the analysis
def analyze_all_assets(symbols):
    for symbol in symbols:
        candles = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1DAY, limit=100)
        data = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
        data.set_index('timestamp', inplace=True)
        detect_patterns_with_alerts(data, symbol)
        # Display candlestick chart in the dashboard
        st.subheader(f"{symbol} Candlestick Chart")
        mpf.plot(data[['open', 'high', 'low', 'close']], type='candle', style='charles', title=f"{symbol} Daily Chart", volume=True)

def backtest_pattern_performance(data, pattern_timestamps):
    results = []
    for timestamp in pattern_timestamps:
        pattern_index = data.index.get_loc(timestamp)
        try:
            # Check the price movement after the pattern (e.g., 3 candles later)
            price_after_3 = data['close'].iloc[pattern_index + 3]
            price_at_pattern = data['close'].iloc[pattern_index]
            pct_change = (price_after_3 - price_at_pattern) / price_at_pattern * 100
            results.append(pct_change)
        except IndexError:
            # If not enough candles, ignore
            continue
    
    # Calculate statistics
    avg_move = sum(results) / len(results) if results else 0
    win_rate = len([r for r in results if r > 0]) / len(results) if results else 0
    print(f"Average price movement after pattern: {avg_move:.2f}%")
    print(f"Win rate: {win_rate * 100:.2f}%")

# Sidebar Settings
st.sidebar.header("Settings")
symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
selected_symbols = st.sidebar.multiselect("Select Symbols", symbols, default=symbols)

# Run analysis
if st.button("Run Bot"):
    analyze_all_assets(selected_symbols)

# Start WebSocket for real-time data
if st.button("Start WebSocket for BTCUSDT"):
    run_websocket('BTCUSDT')
    
    # Main Loop: Detect Patterns and Run WebSocket
if __name__ == "__main__":
    # Detect initial patterns in the historical data
    detect_patterns_with_alerts(data)

    # Start the WebSocket for real-time data and pattern detection
    run_websocket()
