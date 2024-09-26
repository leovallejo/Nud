import requests
import logging

logger = logging.getLogger(__name__)

BINANCE_BASE_URL = "https://api.binance.com/api/v3"

def get_binance_klines(symbol="ETHUSDT", interval="1h", limit=5000):
    """
    Fetch kline/candlestick data from Binance API.
    """
    endpoint = f"{BINANCE_BASE_URL}/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    
    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Error fetching klines data: {str(e)}")
        raise

def get_binance_ticker(symbol="ETHUSDT"):
    """
    Fetch current price ticker from Binance API.
    """
    endpoint = f"{BINANCE_BASE_URL}/ticker/price"
    params = {"symbol": symbol}
    
    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Error fetching ticker data: {str(e)}")
        raise
