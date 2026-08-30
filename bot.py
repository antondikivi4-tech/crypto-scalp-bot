import os
import time
import asyncio
import ccxt
from telegram import Bot

# --- НАСТРОЙКИ ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")  
CHAT_ID = os.getenv("CHAT_ID", "673791974")  

SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
TIMEFRAME = "15"

THRESHOLD_PERCENT = 0.2 

bot = Bot(token=TELEGRAM_TOKEN)
exchange = ccxt.bybit()

def get_support_resistance(candles):
    highs = [c[2] for c in candles]
    lows = [c[3] for c in candles]
    
    resistance = max(highs[:-1])
    support = min(lows[:-1])
    
    return support, resistance

async def send_alert(message):
    await bot.send_message(chat_id=CHAT_ID, text=message, parse_mode="Markdown")

def check_markets():
    print("Проверка рынков через Bybit...")
    for symbol in SYMBOLS:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=50)
            current_price = ohlcv[-1][4]
            support, resistance = get_support_resistance(ohlcv)
            
            support_diff = abs(current_price - support) / support * 100
            if support_diff <= THRESHOLD_PERCENT:
                message = (
                    f"🟢 **СИГНАЛ: ПОДДЕРЖКА (Bybit)**\n"
                    f"Монета: `{symbol}`\n"
                    f"Таймфрейм: `{TIMEFRAME}m`\n"
                    f"Цена: `{current_price}`\n"
                    f"Уровень поддержки: `{support:.4f}`"
                )
                asyncio.run(send_alert(message))
            
            resistance_diff = abs(current_price - resistance) / resistance * 100
            if resistance_diff <= THRESHOLD_PERCENT:
                message = (
                    f"🔴 **СИГНАЛ: СОПРОТИВЛЕНИЕ (Bybit)**\n"
                    f"Монета: `{symbol}`\n"
                    f"Таймфрейм: `{TIMEFRAME}m`\n"
                    f"Цена: `{current_price}`\n"
                    f"Уровень сопротивления: `{resistance:.4f}`"
                )
                asyncio.run(send_alert(message))
                
        except Exception as e:
            print(f"Ошибка при обработке {symbol}: {e}")

if __name__ == "__main__":
    print("Бот запущен и следит за уровнями через Bybit...")
    while True:
        check_markets()
        time.sleep(180)
