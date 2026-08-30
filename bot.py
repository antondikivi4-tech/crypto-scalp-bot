import os
import time
import requests
import ccxt
from datetime import datetime, timezone
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

# ====================== НАСТРОЙКИ ======================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID", "673791974")

TIMEFRAME = "15m"
HIGHER_TF_1H = "1h"
HIGHER_TF_4H = "4h"

THRESHOLD_PERCENT = 0.15
IMPULSE_PERCENT = 1.7
TOP_COINS_LIMIT = 8
COOLDOWN_TIME = 3200
CACHE_TOP_SECONDS = 300
CACHE_HIGHER_TF_SECONDS = 240          # кэш старшего ТФ 4 минуты
MAX_RETRIES = 3
BACKOFF_BASE = 1.6

ATR_SL_MULT = 1.35
ATR_TP1_MULT = 2.1
ATR_TP2_MULT = 3.4

MIN_VOLUME_SPIKE = 2.1
MIN_RISING_VOLUME_BARS = 2
RSI_OVERSOLD = 33
RSI_OVERBOUGHT = 67
STOCH_OVERSOLD = 20
STOCH_OVERBOUGHT = 80
MAX_BTC_VOLATILITY_ATR = 2.6
LOW_LIQUIDITY_HOURS_UTC = {0, 1, 2, 3, 4, 5}

REQUIRE_HIGHER_TF_ALIGN = True
REQUIRE_VOLUME_SPIKE = True
MIN_RR_RATIO = 1.6
STRICT_BTC_FILTER = True
MAX_SIGNALS_PER_CYCLE = 2

# Параллельность (важно для скорости)
MAX_WORKERS = 6                    # сколько монет проверять одновременно

sent_signals = {}
top_symbols_cache = {"symbols": [], "timestamp": 0}
fear_greed_cache = {"value": 50, "timestamp": 0}
higher_tf_cache = {}               # {symbol: {"bias": ..., "rsi": ..., "ts": ...}}

exchange = ccxt.bybit({
    'options': {'defaultType': 'linear'},
    'enableRateLimit': True,
    'timeout': 15000,
})

# ====================== УТИЛИТЫ ======================
def retry_on_error(max_retries=MAX_RETRIES, base_delay=BACKOFF_BASE):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (ccxt.NetworkError, ccxt.ExchangeError, requests.RequestException) as e:
                    last_exc = e
                    delay = base_delay ** attempt + np.random.uniform(0, 0.4)
                    time.sleep(delay)
                except Exception as e:
                    print(f"[Error] {func.__name__}: {e}")
                    return None
            return None
        return wrapper
    return decorator

def is_low_liquidity_session():
    return datetime.now(timezone.utc).hour in LOW_LIQUIDITY_HOURS_UTC

# ====================== ИНДИКАТОРЫ (оптимизированные) ======================
def calculate_atr(candles, period=14):
    if len(candles) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(candles)):
        high, low, prev_close = candles[i][2], candles[i][3], candles[i-1][4]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    return float(np.mean(trs[-period:])) if len(trs) >= period else float(np.mean(trs) if trs else 0)

def calculate_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_rsi_series(closes, period=14):
    """Считает RSI один раз для всего ряда (для дивергенции)"""
    if len(closes) < period + 1:
        return [50.0] * len(closes)
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    
    rsi_values = [50.0] * period
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi_values.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100 - (100 / (1 + rs)))
    return rsi_values

def calculate_macd(closes, fast=12, slow=26, signal=9):
    if len(closes) < slow + signal:
        return 0.0, 0.0, 0.0
    closes = np.asarray(closes, dtype=float)
    ema_fast = np.zeros_like(closes)
    ema_slow = np.zeros_like(closes)
    ema_fast[fast-1] = np.mean(closes[:fast])
    ema_slow[slow-1] = np.mean(closes[:slow])
    alpha_f, alpha_s = 2/(fast+1), 2/(slow+1)
    for i in range(fast, len(closes)):
        ema_fast[i] = closes[i]*alpha_f + ema_fast[i-1]*(1-alpha_f)
    for i in range(slow, len(closes)):
        ema_slow[i] = closes[i]*alpha_s + ema_slow[i-1]*(1-alpha_s)
    macd_line = ema_fast - ema_slow
    signal_line = np.zeros_like(macd_line)
    signal_line[slow+signal-2] = np.mean(macd_line[slow-1:slow+signal-1])
    alpha_sig = 2/(signal+1)
    for i in range(slow+signal-1, len(closes)):
        signal_line[i] = macd_line[i]*alpha_sig + signal_line[i-1]*(1-alpha_sig)
    return float(macd_line[-1]), float(signal_line[-1]), float(macd_line[-1] - signal_line[-1])

def calculate_stochastic(candles, k_period=14, d_period=3):
    if len(candles) < k_period + d_period:
        return 50.0, 50.0
    highs = np.array([c[2] for c in candles])
    lows = np.array([c[3] for c in candles])
    closes = np.array([c[4] for c in candles])
    k_values = []
    for i in range(k_period-1, len(candles)):
        highest = np.max(highs[i-k_period+1:i+1])
        lowest = np.min(lows[i-k_period+1:i+1])
        k = 50.0 if highest == lowest else 100 * (closes[i] - lowest) / (highest - lowest)
        k_values.append(k)
    d = np.mean(k_values[-d_period:]) if len(k_values) >= d_period else k_values[-1]
    return float(k_values[-1]), float(d)

def detect_rsi_divergence(closes, rsi_values, lookback=18):
    if len(closes) < lookback or len(rsi_values) < lookback:
        return None
    price_slice = closes[-lookback:]
    rsi_slice = rsi_values[-lookback:]
    price_low_idx = int(np.argmin(price_slice))
    rsi_low_idx = int(np.argmin(rsi_slice))
    if price_low_idx > lookback//2 and rsi_low_idx < price_low_idx - 3:
        if price_slice[price_low_idx] < min(price_slice[:price_low_idx]) and rsi_slice[rsi_low_idx] > rsi_slice[price_low_idx]:
            return "bullish"
    price_high_idx = int(np.argmax(price_slice))
    rsi_high_idx = int(np.argmax(rsi_slice))
    if price_high_idx > lookback//2 and rsi_high_idx < price_high_idx - 3:
        if price_slice[price_high_idx] > max(price_slice[:price_high_idx]) and rsi_slice[rsi_high_idx] < rsi_slice[price_high_idx]:
            return "bearish"
    return None

def find_fractal_swings(candles, left=2, right=2, min_strength=0.15):
    highs = [c[2] for c in candles]
    lows = [c[3] for c in candles]
    volumes = [c[5] for c in candles]
    swing_highs, swing_lows = [], []
    for i in range(left, len(candles) - right):
        if all(highs[i] > highs[i-j] for j in range(1, left+1)) and all(highs[i] > highs[i+j] for j in range(1, right+1)):
            strength = (highs[i] - min(lows[i-left:i+right+1])) / highs[i] * 100
            vol_strength = volumes[i] / (np.mean(volumes[max(0,i-10):i+1]) + 1e-9)
            if strength >= min_strength and vol_strength > 0.8:
                swing_highs.append({"price": highs[i]})
        if all(lows[i] < lows[i-j] for j in range(1, left+1)) and all(lows[i] < lows[i+j] for j in range(1, right+1)):
            strength = (max(highs[i-left:i+right+1]) - lows[i]) / lows[i] * 100
            vol_strength = volumes[i] / (np.mean(volumes[max(0,i-10):i+1]) + 1e-9)
            if strength >= min_strength and vol_strength > 0.8:
                swing_lows.append({"price": lows[i]})
    return swing_highs, swing_lows

def get_dynamic_levels(candles, atr, current_price):
    swing_highs, swing_lows = find_fractal_swings(candles)
    resistance_candidates = [s["price"] for s in swing_highs[-5:]] if swing_highs else []
    support_candidates = [s["price"] for s in swing_lows[-5:]] if swing_lows else []

    if swing_highs:
        last_high = swing_highs[-1]["price"]
        resistance_candidates += [last_high + atr*0.3, last_high - atr*0.2]
    if swing_lows:
        last_low = swing_lows[-1]["price"]
        support_candidates += [last_low - atr*0.3, last_low + atr*0.2]

    highs = [c[2] for c in candles[:-1]]
    lows = [c[3] for c in candles[:-1]]
    if highs: resistance_candidates.append(max(highs))
    if lows: support_candidates.append(min(lows))

    resistance = min([r for r in resistance_candidates if r > current_price], default=current_price*1.02)
    support = max([s for s in support_candidates if s < current_price], default=current_price*0.98)

    if abs(resistance - current_price)/current_price > 0.04:
        resistance = current_price + atr * 1.8
    if abs(support - current_price)/current_price > 0.04:
        support = current_price - atr * 1.8
    return support, resistance

# ====================== ДАННЫЕ С КЭШЕМ ======================
@retry_on_error()
def get_top_symbols(limit=TOP_COINS_LIMIT):
    global top_symbols_cache
    now = time.time()
    if now - top_symbols_cache["timestamp"] < CACHE_TOP_SECONDS and top_symbols_cache["symbols"]:
        return top_symbols_cache["symbols"]

    tickers = exchange.fetch_tickers()
    usdt = {s: t for s, t in tickers.items() if s.endswith('/USDT:USDT') or (s.endswith('/USDT') and ':' in s)}
    if not usdt:
        usdt = {s: t for s, t in tickers.items() if s.endswith('/USDT')}

    sorted_t = sorted(usdt.items(), key=lambda x: x[1].get('quoteVolume', 0) or 0, reverse=True)
    symbols = [item[0] for item in sorted_t[:limit]]
    for must in ["BTC/USDT:USDT", "ETH/USDT:USDT"]:
        if must not in symbols:
            symbols.insert(0, must)
    symbols = symbols[:limit]
    top_symbols_cache = {"symbols": symbols, "timestamp": now}
    return symbols

@retry_on_error()
def get_btc_context():
    try:
        ohlcv = exchange.fetch_ohlcv("BTC/USDT:USDT", timeframe="1h", limit=40)
        closes = [c[4] for c in ohlcv]
        atr = calculate_atr(ohlcv, 14)
        atr_pct = (atr / closes[-1]) * 100 if closes[-1] else 0
        rsi = calculate_rsi(closes)
        change_24h = ((closes[-1] - closes[-24]) / closes[-24] * 100) if len(closes) >= 24 else 0
        extreme_vol = atr_pct > MAX_BTC_VOLATILITY_ATR
        trend = "бычий" if rsi > 55 and change_24h > 0.8 else ("медвежий" if rsi < 45 and change_24h < -0.8 else "нейтральный")
        return {"atr_pct": atr_pct, "rsi": rsi, "change_24h": change_24h, "extreme_vol": extreme_vol, "trend": trend}
    except:
        return {"atr_pct": 1.0, "rsi": 50, "change_24h": 0, "extreme_vol": False, "trend": "нейтральный"}

@retry_on_error()
def get_fear_greed():
    global fear_greed_cache
    now = time.time()
    if now - fear_greed_cache["timestamp"] < 1800:
        return fear_greed_cache["value"]
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=6)
        value = int(r.json()["data"][0]["value"])
        fear_greed_cache = {"value": value, "timestamp": now}
        return value
    except:
        return fear_greed_cache["value"]

def get_higher_tf_bias_cached(symbol):
    """Кэшированный старший таймфрейм"""
    now = time.time()
    cached = higher_tf_cache.get(symbol)
    if cached and now - cached["ts"] < CACHE_HIGHER_TF_SECONDS:
        return cached["bias"], cached["rsi"]

    try:
        ohlcv_1h = exchange.fetch_ohlcv(symbol, timeframe=HIGHER_TF_1H, limit=50)
        ohlcv_4h = exchange.fetch_ohlcv(symbol, timeframe=HIGHER_TF_4H, limit=30)
        if not ohlcv_1h or not ohlcv_4h:
            return "нейтральное", 50.0

        closes_1h = [c[4] for c in ohlcv_1h]
        closes_4h = [c[4] for c in ohlcv_4h]
        rsi_1h = calculate_rsi(closes_1h)

        ema20_1h = np.mean(closes_1h[-20:])
        ema50_1h = np.mean(closes_1h[-50:]) if len(closes_1h) >= 50 else ema20_1h
        ema20_4h = np.mean(closes_4h[-20:])
        ema50_4h = np.mean(closes_4h[-30:]) if len(closes_4h) >= 30 else ema20_4h

        score = 0
        if closes_1h[-1] > ema20_1h > ema50_1h: score += 2
        elif closes_1h[-1] < ema20_1h < ema50_1h: score -= 2
        if closes_4h[-1] > ema20_4h > ema50_4h: score += 2
        elif closes_4h[-1] < ema20_4h < ema50_4h: score -= 2
        if rsi_1h > 55: score += 1
        elif rsi_1h < 45: score -= 1

        bias = "бычье (растёт)" if score >= 3 else ("медвежье (падает)" if score <= -3 else "нейтральное")
        higher_tf_cache[symbol] = {"bias": bias, "rsi": rsi_1h, "ts": now}
        return bias, rsi_1h
    except:
        return "нейтральное", 50.0

def get_funding(symbol):
    try:
        fr = exchange.fetch_funding_rate(symbol)
        return float(fr.get('fundingRate', 0) or 0) * 100
    except:
        return 0.0

def check_liquidations(symbol, support, resistance, current_price):
    try:
        liquidations = exchange.fetch_liquidations(symbol, limit=40)
        if not liquidations:
            return None
        now_ms = time.time() * 1000
        recent = [l for l in liquidations if now_ms - l.get('timestamp', 0) <= 45*60*1000]
        if not recent:
            return None

        total_vol = longs_liq = shorts_liq = 0.0
        for liq in recent:
            amount = liq.get('usdValue') or (liq.get('amount', 0) * liq.get('price', 0))
            total_vol += amount
            side = liq.get('side', '').lower()
            if side == 'buy':
                shorts_liq += amount
            elif side == 'sell':
                longs_liq += amount

        if total_vol < 120000:
            return None

        near_level = abs(current_price - support)/support*100 < 0.45 or abs(current_price - resistance)/resistance*100 < 0.45
        if shorts_liq > longs_liq * 1.4:
            return {"text": "много людей выбило из коротких позиций — цена может продолжить рост",
                    "volume": total_vol, "near_level": near_level, "preferred": "long"}
        else:
            return {"text": "много людей выбило из длинных позиций — цена может продолжить падение",
                    "volume": total_vol, "near_level": near_level, "preferred": "short"}
    except:
        return None

# ====================== TELEGRAM ======================
def send_telegram_message(message, symbol, signal_key):
    now = time.time()
    if now - sent_signals.get(signal_key, 0) < COOLDOWN_TIME:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    clean = symbol.split(':')[0].replace('/', '')
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
        "reply_markup": {"inline_keyboard": [[{"text": f"📈 Открыть {symbol.split(':')[0]} на Bybit", "url": f"https://www.bybit.com/trade/usdt/{clean}"}]]}
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code == 200:
            sent_signals[signal_key] = now
            print(f"✅ {signal_key}")
            return True
    except Exception as e:
        print(f"Telegram error: {e}")
    return False

def build_sl_tp(current_price, atr, direction):
    if direction == "long":
        sl = current_price - atr * ATR_SL_MULT
        tp1 = current_price + atr * ATR_TP1_MULT
        tp2 = current_price + atr * ATR_TP2_MULT
        rr = (tp1 - current_price) / (current_price - sl) if current_price > sl else 0
    else:
        sl = current_price + atr * ATR_SL_MULT
        tp1 = current_price - atr * ATR_TP1_MULT
        tp2 = current_price - atr * ATR_TP2_MULT
        rr = (current_price - tp1) / (sl - current_price) if sl > current_price else 0
    return sl, tp1, tp2, rr

def explain_rsi(rsi):
    if rsi < 30: return f"{rsi:.0f} — сильно перепродан (хорошо для покупки)"
    if rsi < 40: return f"{rsi:.0f} — перепродан"
    if rsi > 70: return f"{rsi:.0f} — сильно перекуплен (хорошо для продажи)"
    if rsi > 60: return f"{rsi:.0f} — перекуплен"
    return f"{rsi:.0f} — нормальная зона"

def explain_stoch(k):
    if k < 20: return f"{k:.0f} — сильная перепроданность"
    if k < 30: return f"{k:.0f} — перепроданность"
    if k > 80: return f"{k:.0f} — сильная перекупленность"
    if k > 70: return f"{k:.0f} — перекупленность"
    return f"{k:.0f} — нейтрально"

def explain_macd(hist):
    if hist > 0.001: return "сила у покупателей"
    if hist > 0: return "покупатели чуть сильнее"
    if hist < -0.001: return "сила у продавцов"
    if hist < 0: return "продавцы чуть сильнее"
    return "нет преимущества"

def explain_divergence(div):
    if div == "bullish": return "бычья (признак разворота вверх)"
    if div == "bearish": return "медвежья (признак разворота вниз)"
    return "нет"

def explain_funding(fund):
    if abs(fund) < 0.01: return f"{fund:.4f}% — почти ноль"
    if fund > 0.03: return f"{fund:.4f}% — высокая плата за лонги"
    if fund < -0.03: return f"{fund:.4f}% — высокая плата за шорты"
    return f"{fund:.4f}%"

def explain_fng(fng):
    if fng >= 75: return f"{fng} — очень жадный"
    if fng >= 55: return f"{fng} — жадность"
    if fng <= 25: return f"{fng} — страх"
    if fng <= 45: return f"{fng} — скорее страх"
    return f"{fng} — спокойный"

def explain_volume(is_spike, rising):
    if is_spike and rising: return "очень высокий и растущий"
    if is_spike: return "повышенный"
    if rising: return "растёт"
    return "обычный"

# ====================== ОБРАБОТКА ОДНОЙ МОНЕТЫ ======================
def process_symbol(symbol, btc_ctx, fng):
    """Обрабатывает одну монету. Возвращает список кандидатов."""
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=70)
        if not ohlcv or len(ohlcv) < 40:
            return []

        closes = [c[4] for c in ohlcv]
        volumes = [c[5] for c in ohlcv]
        last = ohlcv[-1]
        prev = ohlcv[-2]
        open_p = last[1]
        current_price = last[4]
        last_vol = last[5]

        atr = calculate_atr(ohlcv)
        rsi = calculate_rsi(closes)
        macd_line, _, macd_hist = calculate_macd(closes)
        stoch_k, _ = calculate_stochastic(ohlcv)
        rsi_series = calculate_rsi_series(closes)
        divergence = detect_rsi_divergence(closes, rsi_series)

        support, resistance = get_dynamic_levels(ohlcv, atr, current_price)
        bias_1h, _ = get_higher_tf_bias_cached(symbol)

        avg_vol = np.mean(volumes[-21:-1]) if len(volumes) >= 21 else last_vol
        is_volume_spike = last_vol >= avg_vol * MIN_VOLUME_SPIKE
        rising_volume = len(volumes) > 3 and volumes[-1] > volumes[-2] > volumes[-3]

        clean_name = symbol.split(':')[0]
        candidates = []

        # --- Ликвидации (делаем только если близко к уровню) ---
        near_support = abs(current_price - support)/support*100 < 0.6
        near_resist = abs(current_price - resistance)/resistance*100 < 0.6
        if near_support or near_resist:
            liq = check_liquidations(symbol, support, resistance, current_price)
            if liq and (liq["near_level"] or liq["volume"] > 300000):
                direction = liq["preferred"]
                if STRICT_BTC_FILTER and ((direction == "long" and btc_ctx["trend"] == "медвежий") or (direction == "short" and btc_ctx["trend"] == "бычий")):
                    pass
                elif REQUIRE_HIGHER_TF_ALIGN and ((direction == "long" and "медвежье" in bias_1h) or (direction == "short" and "бычье" in bias_1h)):
                    pass
                else:
                    sl, tp1, tp2, rr = build_sl_tp(current_price, atr, direction)
                    if rr >= MIN_RR_RATIO:
                        funding = get_funding(symbol)
                        action = "ПОКУПАТЬ (ЛОНГ)" if direction == "long" else "ПРОДАВАТЬ (ШОРТ)"
                        msg = (
                            f"💥 *РЕЗКОЕ ДВИЖЕНИЕ ИЗ-ЗА ЛИКВИДАЦИЙ*\n\n"
                            f"Монета: `{clean_name}`\nТекущая цена: `{current_price}`\n\n"
                            f"*Что делать:* {action}\n\n*Почему:*\n{liq['text']}\n\n"
                            f"*Куда ставить:*\n• Вход: `{current_price}`\n• Стоп: `{sl:.4f}`\n• Цель 1: `{tp1:.4f}`\n• Цель 2: `{tp2:.4f}`\n\n"
                            f"Риск/Прибыль: 1 к {rr:.1f}\n\n——————————————\n"
                            f"• RSI: {explain_rsi(rsi)}\n• Stochastic: {explain_stoch(stoch_k)}\n"
                            f"• MACD: {explain_macd(macd_hist)}\n• Дивергенция: {explain_divergence(divergence)}\n"
                            f"• 1 час: {bias_1h}\n• Funding: {explain_funding(funding)}\n"
                            f"• Настроение: {explain_fng(fng)}\n• Объём: {explain_volume(is_volume_spike, rising_volume)}"
                        )
                        score = 82 + (8 if liq["near_level"] else 0)
                        candidates.append({"score": score, "msg": msg, "symbol": symbol, "key": f"{symbol}_liq"})

        # --- Импульс ---
        candle_change = ((current_price - open_p) / open_p) * 100
        strong_body = abs(current_price - open_p) / (last[2] - last[3] + 1e-9) > 0.68
        if abs(candle_change) >= IMPULSE_PERCENT and is_volume_spike and rising_volume and strong_body:
            direction = "long" if candle_change > 0 else "short"
            if not (btc_ctx["extreme_vol"] and abs(candle_change) < 2.3):
                if not (REQUIRE_HIGHER_TF_ALIGN and ((direction == "long" and "медвежье" in bias_1h) or (direction == "short" and "бычье" in bias_1h))):
                    if not (STRICT_BTC_FILTER and ((direction == "long" and btc_ctx["trend"] == "медвежий") or (direction == "short" and btc_ctx["trend"] == "бычий"))):
                        macd_ok = (macd_hist > 0 and direction == "long") or (macd_hist < 0 and direction == "short")
                        stoch_ok = (stoch_k < 78 and direction == "long") or (stoch_k > 22 and direction == "short")
                        if macd_ok or stoch_ok:
                            sl, tp1, tp2, rr = build_sl_tp(current_price, atr, direction)
                            if rr >= MIN_RR_RATIO:
                                funding = get_funding(symbol)
                                title = "🚀 СИЛЬНЫЙ РОСТ" if direction == "long" else "⚡️ СИЛЬНОЕ ПАДЕНИЕ"
                                action = "ПОКУПАТЬ (ЛОНГ)" if direction == "long" else "ПРОДАВАТЬ (ШОРТ)"
                                reason = "цена резко выросла на большом объёме" if direction == "long" else "цена резко упала на большом объёме"
                                msg = (
                                    f"*{title}*\n\nМонета: `{clean_name}`\nТекущая цена: `{current_price}`\n\n"
                                    f"*Что делать:* {action}\n\n*Почему:*\n{reason}\n\n"
                                    f"*Куда ставить:*\n• Вход: `{current_price}`\n• Стоп: `{sl:.4f}`\n• Цель 1: `{tp1:.4f}`\n• Цель 2: `{tp2:.4f}`\n\n"
                                    f"Риск/Прибыль: 1 к {rr:.1f}\n\n——————————————\n"
                                    f"• RSI: {explain_rsi(rsi)}\n• Stochastic: {explain_stoch(stoch_k)}\n"
                                    f"• MACD: {explain_macd(macd_hist)}\n• Дивергенция: {explain_divergence(divergence)}\n"
                                    f"• 1 час: {bias_1h}\n• Funding: {explain_funding(funding)}\n"
                                    f"• Настроение: {explain_fng(fng)}\n• Объём: {explain_volume(is_volume_spike, rising_volume)}"
                                )
                                score = 72 + min(abs(candle_change)*2.5, 18)
                                candidates.append({"score": score, "msg": msg, "symbol": symbol, "key": f"{symbol}_impulse"})

        # --- Поддержка ---
        support_diff = abs(current_price - support) / support * 100
        if support_diff <= THRESHOLD_PERCENT:
            is_bounce = (prev[4] < prev[1]) and (current_price > open_p)
            rsi_ok = rsi < RSI_OVERSOLD or divergence == "bullish"
            stoch_ok = stoch_k < STOCH_OVERSOLD
            volume_ok = is_volume_spike if REQUIRE_VOLUME_SPIKE else True

            if not (REQUIRE_HIGHER_TF_ALIGN and "медвежье" in bias_1h) and not (STRICT_BTC_FILTER and btc_ctx["trend"] == "медвежий"):
                strong = is_bounce and (rsi_ok or stoch_ok) and volume_ok
                if strong or support_diff < 0.09:
                    sl, tp1, tp2, rr = build_sl_tp(current_price, atr, "long")
                    if rr >= MIN_RR_RATIO:
                        funding = get_funding(symbol)
                        status = "цена подошла к поддержке и отскочила на объёме" if strong else "цена очень близко к сильной поддержке"
                        msg = (
                            f"🟢 *ХОРОШАЯ ТОЧКА ДЛЯ ПОКУПКИ*\n\nМонета: `{clean_name}`\nТекущая цена: `{current_price}`\n\n"
                            f"*Что делать:* ПОКУПАТЬ (ЛОНГ)\n\n*Почему:*\n{status}\n\n"
                            f"*Куда ставить:*\n• Вход: `{current_price}`\n• Стоп: `{sl:.4f}`\n• Цель 1: `{tp1:.4f}`\n• Цель 2: `{tp2:.4f}`\n\n"
                            f"Риск/Прибыль: 1 к {rr:.1f}\n\n——————————————\n"
                            f"• RSI: {explain_rsi(rsi)}\n• Stochastic: {explain_stoch(stoch_k)}\n"
                            f"• MACD: {explain_macd(macd_hist)}\n• Дивергенция: {explain_divergence(divergence)}\n"
                            f"• 1 час: {bias_1h}\n• Funding: {explain_funding(funding)}\n"
                            f"• Настроение: {explain_fng(fng)}\n• Объём: {explain_volume(is_volume_spike, rising_volume)}"
                        )
                        score = 68 + (14 if strong else 0) + (8 if support_diff < 0.08 else 0)
                        candidates.append({"score": score, "msg": msg, "symbol": symbol, "key": f"{symbol}_support"})

        # --- Сопротивление ---
        resist_diff = abs(current_price - resistance) / resistance * 100
        if resist_diff <= THRESHOLD_PERCENT:
            is_reject = (prev[4] > prev[1]) and (current_price < open_p)
            rsi_ok = rsi > RSI_OVERBOUGHT or divergence == "bearish"
            stoch_ok = stoch_k > STOCH_OVERBOUGHT
            volume_ok = is_volume_spike if REQUIRE_VOLUME_SPIKE else True

            if not (REQUIRE_HIGHER_TF_ALIGN and "бычье" in bias_1h) and not (STRICT_BTC_FILTER and btc_ctx["trend"] == "бычий"):
                strong = is_reject and (rsi_ok or stoch_ok) and volume_ok
                if strong or resist_diff < 0.09:
                    sl, tp1, tp2, rr = build_sl_tp(current_price, atr, "short")
                    if rr >= MIN_RR_RATIO:
                        funding = get_funding(symbol)
                        status = "цена подошла к сопротивлению и отбилась на объёме" if strong else "цена очень близко к сильному сопротивлению"
                        msg = (
                            f"🔴 *ХОРОШАЯ ТОЧКА ДЛЯ ПРОДАЖИ*\n\nМонета: `{clean_name}`\nТекущая цена: `{current_price}`\n\n"
                            f"*Что делать:* ПРОДАВАТЬ (ШОРТ)\n\n*Почему:*\n{status}\n\n"
                            f"*Куда ставить:*\n• Вход: `{current_price}`\n• Стоп: `{sl:.4f}`\n• Цель 1: `{tp1:.4f}`\n• Цель 2: `{tp2:.4f}`\n\n"
                            f"Риск/Прибыль: 1 к {rr:.1f}\n\n——————————————\n"
                            f"• RSI: {explain_rsi(rsi)}\n• Stochastic: {explain_stoch(stoch_k)}\n"
                            f"• MACD: {explain_macd(macd_hist)}\n• Дивергенция: {explain_divergence(divergence)}\n"
                            f"• 1 час: {bias_1h}\n• Funding: {explain_funding(funding)}\n"
                            f"• Настроение: {explain_fng(fng)}\n• Объём: {explain_volume(is_volume_spike, rising_volume)}"
                        )
                        score = 68 + (14 if strong else 0) + (8 if resist_diff < 0.08 else 0)
                        candidates.append({"score": score, "msg": msg, "symbol": symbol, "key": f"{symbol}_resistance"})

        return candidates
    except Exception as e:
        print(f"Ошибка {symbol}: {e}")
        return []

# ====================== ГЛАВНЫЙ ЦИКЛ ======================
def check_markets():
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Проверка...")
    if is_low_liquidity_session():
        print("⏳ Ночь UTC — пропуск")
        return

    symbols = get_top_symbols()
    btc_ctx = get_btc_context()
    fng = get_fear_greed()

    all_candidates = []

    # === Параллельная обработка монет ===
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_symbol, sym, btc_ctx, fng): sym for sym in symbols}
        for future in as_completed(futures):
            result = future.result()
            if result:
                all_candidates.extend(result)

    # Отправляем лучшие
    if all_candidates:
        all_candidates.sort(key=lambda x: x["score"], reverse=True)
        sent = 0
        for cand in all_candidates:
            if sent >= MAX_SIGNALS_PER_CYCLE:
                break
            if send_telegram_message(cand["msg"], cand["symbol"], cand["key"]):
                sent += 1
        print(f"Отправлено: {sent}")
    else:
        print("Сигналов нет")

# ====================== ЗАПУСК ======================
if __name__ == "__main__":
    print("🚀 Бот запущен (оптимизированная версия + параллельность)")
    while True:
        try:
            check_markets()
        except Exception as e:
            print(f"Ошибка цикла: {e}")
        time.sleep(180)
