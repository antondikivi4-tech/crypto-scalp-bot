import os
import time
import requests
import ccxt

# --- НАСТРОЙКИ ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")  
CHAT_ID = os.getenv("CHAT_ID", "673791974")  

TIMEFRAME = "15"
THRESHOLD_PERCENT = 0.15 
IMPULSE_PERCENT = 1.5
TOP_COINS_LIMIT = 6  блок динамического отбора топ монет

exchange = ccxt.bybit()

def get_top_symbols(limit=6):
    try:
        tickers = exchange.fetch_tickers()
        usdt_tickers = {s: t for s, t in tickers.items() if s.endswith('/USDT')}
        sorted_tickers = sorted(
            usdt_tickers.items(), 
            key=lambda x: x[1].get('quoteVolume', 0) or 0, 
            reverse=True
        )
        top_symbols = [item[0] for item in sorted_tickers[:limit]]
        print(f"Динамический топ монет по объему: {top_symbols}")
        return top_symbols
    except Exception as e:
        print(f"Ошибка получения топ монет, используем резерв: {e}")
        return ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]

def send_telegram_message(message, symbol):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    base, quote = symbol.split('/')
    bybit_url = f"https://www.bybit.com/trade/spot/{base}/{quote}"
    
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
        "reply_markup": {
            "inline_keyboard": [
                [{"text": f"📈 Открыть {symbol} на Bybit", "url": bybit_url}]
            ]
        }
    }
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Ошибка отправки в Telegram: {e}")

def calculate_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50.0
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def get_support_resistance(candles):
    highs = [c[2] for c in candles]
    lows = [c[3] for c in candles]
    resistance = max(highs[:-1])
    support = min(lows[:-1])
    return support, resistance

def check_markets():
    print("Проверка рынков (Топ объем + RSI + Объемы + Уровни)...")
    symbols = get_top_symbols(TOP_COINS_LIMIT)
    
    for symbol in symbols:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=50)
            if not ohlcv or len(ohlcv) < 30:
                continue
                
            closes = [c[4] for c in ohlcv]
            volumes = [c[5] for c in ohlcv]
            
            last_candle = ohlcv[-1]
            prev_candle = ohlcv[-2]
            
            open_price = last_candle[1]
            current_price = last_candle[4]
            last_volume = last_candle[5]
            
            # Средний объем за последние 20 свечей для поиска всплеска
            avg_volume = sum(volumes[-21:-1]) / 20 if len(volumes) >= 21 else last_volume
            is_volume_spike = last_volume >= (avg_volume * 1.8)
            
            rsi = calculate_rsi(closes)
            support, resistance = get_support_resistance(ohlcv)
            
            # 1. Анализ импульса с подтверждением объема
            candle_change = ((current_price - open_price) / open_price) * 100
            if abs(candle_change) >= IMPULSE_PERCENT and is_volume_spike:
                direction_text = "🚀 МОЩНЫЙ ИМПУЛЬС РОСТА" if candle_change > 0 else "⚡️ МОЩНЫЙ ИМПУЛЬС ПАДЕНИЯ"
                impulse_msg = (
                    f"📊 *{direction_text}*\n"
                    f"Монета: `{symbol}`\n"
                    f"Изменение свечи: `{candle_change:.2f}%`\n"
                    f"Всплеск объема: `x{(last_volume/avg_volume):.1f} от нормы` ✅\n"
                    f"RSI: `{rsi:.1f}`\n"
                    f"Цена сейчас: `{current_price}`"
                )
                send_telegram_message(impulse_msg, symbol)

            # 2. Подход к поддержке с фильтрами RSI и безопасности
            support_diff = abs(current_price - support) / support * 100
            if support_diff <= THRESHOLD_PERCENT:
                is_safe_bounce = (prev_candle[4] < prev_candle[1]) and (current_price > open_price)
                rsi_filter = rsi < 40  # Зона перепроданности для лонга
                
                status_desc = "🛡 *Отличная точка (RSI внизу + отскок)*" if (is_safe_bounce and rsi_filter) else "⚠️ *Подход к уровню*"
                
                message = (
                    f"🟢 *СИГНАЛ: ПОДДЕРЖКА (Bybit)*\n"
                    f"Статус: {status_desc}\n"
                    f"Монета: `{symbol}`\n"
                    f"Цена: `{current_price}`\n"
                    f"Уровень: `{support:.4f}`\n"
                    f"RSI (14): `{rsi:.1f}` {'🟢 Перепроданность' if rsi_filter else ''}\n"
                    f"Объем выше среднего: {'Да ✅' if is_volume_spike else 'Нет ❌'}"
                )
                send_telegram_message(message, symbol)
            
            # 3. Подход к сопротивлению с фильтрами RSI и безопасности
            resistance_diff = abs(current_price - resistance) / resistance * 100
            if resistance_diff <= THRESHOLD_PERCENT:
                is_safe_rejection = (prev_candle[4] > prev_candle[1]) and (current_price < open_price)
                rsi_filter = rsi > 60  # Зона перекупленности для шорта
                
                status_desc = "🛡 *Отличная точка (RSI вверху + отскок)*" if (is_safe_rejection and rsi_filter) else "⚠️ *Подход к уровню*"
                
                message = (
                    f"🔴 *СИГНАЛ: СОПРОТИВЛЕНИЕ (Bybit)*\n"
                    f"Статус: {status_desc}\n"
                    f"Монета: `{symbol}`\n"
                    f"Цена: `{current_price}`\n"
                    f"Уровень: `{resistance:.4f}`\n"
                    f"RSI (14): `{rsi:.1f}` {'🔴 Перекупленность' if rsi_filter else ''}\n"
                    f"Объем выше среднего: {'Да ✅' if is_volume_spike else 'Нет ❌'}"
                )
                send_telegram_message(message, symbol)
                
        except Exception as e:
            print(f"Ошибка при обработке {symbol}: {e}")

if __name__ == "__main__":
    print("Супер-бот запущен с динамическим отбором, RSI и инлайн-кнопками...")
    while True:
        check_markets()
        time.sleep(180)
