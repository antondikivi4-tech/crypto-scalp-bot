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
TOP_COINS_LIMIT = 6

# Словарь для защиты от спама (кулдаун 1 час)
sent_signals = {}
COOLDOWN_TIME = 3600  

exchange = ccxt.bybit({
    'options': {
        'defaultType': 'linear',  # Переключаемся на фьючерсы (ликвидации есть только там)
    }
})

def get_top_symbols(limit=6):
    try:
        # Получаем тикеры для бессрочных фьючерсов (linear)
        tickers = exchange.fetch_tickers()
        usdt_tickers = {s: t for s, t in tickers.items() if s.endswith('/USDT:USDT') or ('/USDT' in s and ':' in s)}
        if not usdt_tickers:
            usdt_tickers = {s: t for s, t in tickers.items() if s.endswith('/USDT')}
            
        sorted_tickers = sorted(
            usdt_tickers.items(), 
            key=lambda x: x[1].get('quoteVolume', 0) or 0, 
            reverse=True
        )
        top_symbols = [item[0] for item in sorted_tickers[:limit]]
        return top_symbols
    except Exception as e:
        print(f"Ошибка получения топ монет, используем резерв: {e}")
        return ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT"]

def send_telegram_message(message, symbol, signal_key):
    current_time = time.time()
    if current_time - sent_signals.get(signal_key, 0) < COOLDOWN_TIME:
        return  
        
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    
    # Очищаем символ для ссылки на Bybit (например, BTC/USDT:USDT -> BTCUSDT)
    clean_symbol = symbol.split(':')[0].replace('/', '')
    bybit_url = f"https://www.bybit.com/trade/usdt/{clean_symbol}"
    
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
        "reply_markup": {
            "inline_keyboard": [
                [{"text": f"📈 Открыть {symbol.split(':')[0]} на Bybit", "url": bybit_url}]
            ]
        }
    }
    try:
        response = requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            sent_signals[signal_key] = current_time  
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

def check_liquidations(symbol):
    try:
        # Получаем последние ликвидации по инструменту
        liquidations = exchange.fetch_liquidations(symbol, limit=20)
        if not liquidations:
            return None
        
        # Считаем суммарный объем ликвидаций за последнее время
        recent_liq_volume = 0
        longs_liquidated = 0
        shorts_liquidated = 0
        
        current_time = time.time() * 1000  копируем в мс
        for liq in liquidations:
            # Берем ликвидации за последние 15 минут (900000 мс)
            if current_time - liq.get('timestamp', 0) <= 900000:
                amount = liq.get('usdValue', 0) or (liq.get('amount', 0) * liq.get('price', 0))
                recent_liq_volume += amount
                if liq.get('side') == 'buy':  # Ликвидация шорта (бай)
                    shorts_liquidated += amount
                elif liq.get('side') == 'sell':  # Ликвидация лонга (селл)
                    longs_liquidated += amount
                    
        # порог крупного каскада ликвидаций (например, от $100,000 за 15 минут)
        if recent_liq_volume >= 100000:
            side_text = "🟢 Сбриты Шорты (Каскад роста)" if shorts_liquidated > longs_liquidated else "🔴 Сбриты Лонги (Каскад падения)"
            return {
                "text": side_text,
                "volume": recent_liq_volume
            }
    except Exception as e:
        # Некоторые пары могут не отдавать ликвидации через публичный API, пропускаем тихо
        pass
    return None

def check_markets():
    print("Проверка рынков (Топ объем + RSI + Объемы + Уровни + Ликвидации)...")
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
            
            avg_volume = sum(volumes[-21:-1]) / 20 if len(volumes) >= 21 else last_volume
            is_volume_spike = last_volume >= (avg_volume * 1.8)
            
            rsi = calculate_rsi(closes)
            support, resistance = get_support_resistance(ohlcv)
            
            # 0. Проверка ликвидаций (Каскад сбрития стопов)
            liq_data = check_liquidations(symbol)
            if liq_data:
                liq_msg = (
                    f"💥 *КАСКАД ЛИКВИДАЦИЙ*\n"
                    f"Монета: `{symbol.split(':')[0]}`\n"
                    f"Тип: *{liq_data['text']}*\n"
                    f"Объем ликвидаций: `${liq_data['volume']:,.0f}`\n"
                    f"Цена сейчас: `{current_price}`"
                )
                signal_key = f"{symbol}_liquidation"
                send_telegram_message(liq_msg, symbol, signal_key)

            # 1. Импульс
            candle_change = ((current_price - open_price) / open_price) * 100
            if abs(candle_change) >= IMPULSE_PERCENT and is_volume_spike:
                direction_text = "🚀 МОЩНЫЙ ИМПУЛЬС РОСТА" if candle_change > 0 else "⚡️ МОЩНЫЙ ИМПУЛЬС ПАДЕНИЯ"
                impulse_msg = (
                    f"📊 *{direction_text}*\n"
                    f"Монета: `{symbol.split(':')[0]}`\n"
                    f"Изменение свечи: `{candle_change:.2f}%`\n"
                    f"Всплеск объема: `x{(last_volume/avg_volume):.1f} от нормы` ✅\n"
                    f"RSI: `{rsi:.1f}`\n"
                    f"Цена сейчас: `{current_price}`"
                )
                signal_key = f"{symbol}_impulse"
                send_telegram_message(impulse_msg, symbol, signal_key)

            # 2. Поддержка
            support_diff = abs(current_price - support) / support * 100
            if support_diff <= THRESHOLD_PERCENT:
                is_safe_bounce = (prev_candle[4] < prev_candle[1]) and (current_price > open_price)
                rsi_filter = rsi < 40
                
                status_desc = "🛡 *Отличная точка (RSI внизу + отскок)*" if (is_safe_bounce and rsi_filter) else "⚠️ *Подход к уровню*"
                
                message = (
                    f"🟢 *СИГНАЛ: ПОДДЕРЖКА (Bybit)*\n"
                    f"Статус: {status_desc}\n"
                    f"Монета: `{symbol.split(':')[0]}`\n"
                    f"Цена: `{current_price}`\n"
                    f"Уровень: `{support:.4f}`\n"
                    f"RSI (14): `{rsi:.1f}` {'🟢 Перепроданность' if rsi_filter else ''}\n"
                    f"Объем выше среднего: {'Да ✅' if is_volume_spike else 'Нет ❌'}"
                )
                signal_key = f"{symbol}_support"
                send_telegram_message(message, symbol, signal_key)
            
            # 3. Сопротивление
            resistance_diff = abs(current_price - resistance) / resistance * 100
            if resistance_diff <= THRESHOLD_PERCENT:
                is_safe_rejection = (prev_candle[4] > prev_candle[1]) and (current_price < open_price)
                rsi_filter = rsi > 60
                
                status_desc = "🛡 *Отличная точка (RSI вверху + отскок)*" if (is_safe_rejection and rsi_filter) else "⚠️ *Подход к уровню*"
                
                message = (
                    f"🔴 *СИГНАЛ: СОПРОТИВЛЕНИЕ (Bybit)*\n"
                    f"Статус: {status_desc}\n"
                    f"Монета: `{symbol.split(':')[0]}`\n"
                    f"Цена: `{current_price}`\n"
                    f"Уровень: `{resistance:.4f}`\n"
                    f"RSI (14): `{rsi:.1f}` {'🔴 Перекупленность' if rsi_filter else ''}\n"
                    f"Объем выше среднего: {'Да ✅' if is_volume_spike else 'Нет ❌'}"
                )
                signal_key = f"{symbol}_resistance"
                send_telegram_message(message, symbol, signal_key)
                
        except Exception as e:
            print(f"Ошибка при обработке {symbol}: {e}")

if __name__ == "__main__":
    print("Супер-бот запущен с мониторингом ликвидаций на фьючерсах...")
    while True:
        check_markets()
        time.sleep(180)
