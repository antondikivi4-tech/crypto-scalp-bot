import os
import time
import requests
import ccxt
from datetime import datetime, timezone
from functools import wraps
import numpy as np

# ====================== НАСТРОЙКИ (СРЕДНЕЕ УЖЕСТОЧЕНИЕ) ======================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID", "673791974")

TIMEFRAME = "15m"
HIGHER_TF_1H = "1h"
HIGHER_TF_4H = "4h"

# --- Средние ужесточённые параметры ---
THRESHOLD_PERCENT = 0.15          # было 0.25
IMPULSE_PERCENT = 1.7             # было 1.3
TOP_COINS_LIMIT = 8
COOLDOWN_TIME = 3200              # ~53 минуты
CACHE_TOP_SECONDS = 300
MAX_RETRIES = 4
BACKOFF_BASE = 1.8

ATR_SL_MULT = 1.35
ATR_TP1_MULT = 2.1
ATR_TP2_MULT = 3.4

MIN_VOLUME_SPIKE = 2.1            # было 1.7
MIN_RISING_VOLUME_BARS = 2
RSI_OVERSOLD = 33                 # было 38
RSI_OVERBOUGHT = 67               # было 62
STOCH_OVERSOLD = 20
STOCH_OVERBOUGHT = 80
MAX_BTC_VOLATILITY_ATR = 2.6
LOW_LIQUIDITY_HOURS_UTC = {0, 1, 2, 3, 4, 5}

# Новые жёсткие фильтры
REQUIRE_HIGHER_TF_ALIGN = True    # обязательно совпадение со старшим ТФ
REQUIRE_VOLUME_SPIKE = True       # без всплеска объёма — не отправляем
MIN_RR_RATIO = 1.6                # минимальный R:R
STRICT_BTC_FILTER = True
MAX_SIGNALS_PER_CYCLE = 2         # максимум сигналов за одну проверку

sent_signals = {}
top_symbols_cache = {"symbols": [], "timestamp": 0}
fear_greed_cache = {"value": 50, "timestamp": 0}

exchange = ccxt.bybit({
    'options': {'defaultType': 'linear'},
    'enableRateLimit': True,
    'timeout': 20000,
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
                    delay = base_delay ** attempt + np.random.uniform(0, 0.5)
                    print(f"[Retry {attempt+1}/{max_retries}] {func.__name__}: {e} → sleep {delay:.1f}s")
                    time.sleep(delay)
                except Exception as e:
                    print(f"[Error] {func.__name__}: {e}")
                    return None
            print(f"[Fail] {func.__name__} после {max_retries} попыток: {last_exc}")
            return None
        return wrapper
    return decorator

def is_low_liquidity_session():
    hour = datetime.now(timezone.utc).hour
    return hour in LOW_LIQUIDITY_HOURS_UTC

# ====================== ИНДИКАТОРЫ ======================
def calculate_atr(candles, period=14):
    if len(candles) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(candles)):
        high = candles[i][2]
        low = candles[i][3]
        prev_close = candles[i-1][4]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
    if len(trs) < period:
        return float(np.mean(trs)) if trs else 0.0
    return float(np.mean(trs[-period:]))

def calculate_rsi(closes, period=14):
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(closes, fast=12, slow=26, signal=9):
    if len(closes) < slow + signal:
        return 0.0, 0.0, 0.0
    closes = np.array(closes, dtype=float)
    ema_fast = np.zeros_like(closes)
    ema_slow = np.zeros_like(closes)
    ema_fast[fast-1] = np.mean(closes[:fast])
    ema_slow[slow-1] = np.mean(closes[:slow])
    alpha_f = 2 / (fast + 1)
    alpha_s = 2 / (slow + 1)
    for i in range(fast, len(closes)):
        ema_fast[i] = closes[i] * alpha_f + ema_fast[i-1] * (1 - alpha_f)
    for i in range(slow, len(closes)):
        ema_slow[i] = closes[i] * alpha_s + ema_slow[i-1] * (1 - alpha_s)
    macd_line = ema_fast - ema_slow
    signal_line = np.zeros_like(macd_line)
    signal_line[slow + signal - 2] = np.mean(macd_line[slow-1:slow+signal-1])
    alpha_sig = 2 / (signal + 1)
    for i in range(slow + signal - 1, len(closes)):
        signal_line[i] = macd_line[i] * alpha_sig + signal_line[i-1] * (1 - alpha_sig)
    hist = macd_line - signal_line
    return float(macd_line[-1]), float(signal_line[-1]), float(hist[-1])

def calculate_stochastic(candles, k_period=14, d_period=3):
    if len(candles) < k_period + d_period:
        return 50.0, 50.0
    highs = [c[2] for c in candles]
    lows = [c[3] for c in candles]
    closes = [c[4] for c in candles]
    k_values = []
    for i in range(k_period - 1, len(candles)):
        highest = max(highs[i - k_period + 1:i + 1])
        lowest = min(lows[i - k_period + 1:i + 1])
        if highest == lowest:
            k = 50.0
        else:
            k = 100 * (closes[i] - lowest) / (highest - lowest)
        k_values.append(k)
    if len(k_values) < d_period:
        return float(k_values[-1]), float(k_values[-1])
    d = np.mean(k_values[-d_period:])
    return float(k_values[-1]), float(d)

def detect_rsi_divergence(closes, rsi_values, lookback=20):
    if len(closes) < lookback or len(rsi_values) < lookback:
        return None
    price_slice = closes[-lookback:]
    rsi_slice = rsi_values[-lookback:]
    price_low_idx = int(np.argmin(price_slice))
    rsi_low_idx = int(np.argmin(rsi_slice))
    if price_low_idx > lookback // 2 and rsi_low_idx < price_low_idx - 3:
        if price_slice[price_low_idx] < min(price_slice[:price_low_idx]) and rsi_slice[rsi_low_idx] > rsi_slice[price_low_idx]:
            return "bullish"
    price_high_idx = int(np.argmax(price_slice))
    rsi_high_idx = int(np.argmax(rsi_slice))
    if price_high_idx > lookback // 2 and rsi_high_idx < price_high_idx - 3:
        if price_slice[price_high_idx] > max(price_slice[:price_high_idx]) and rsi_slice[rsi_high_idx] < rsi_slice[price_high_idx]:
            return "bearish"
    return None

def find_fractal_swings(candles, left=2, right=2, min_strength=0.15):
    highs = [c[2] for c in candles]
    lows = [c[3] for c in candles]
    volumes = [c[5] for c in candles]
    swing_highs = []
    swing_lows = []
    for i in range(left, len(candles) - right):
        if all(highs[i] > highs[i - j] for j in range(1, left + 1)) and all(highs[i] > highs[i + j] for j in range(1, right + 1)):
            strength = (highs[i] - min(lows[i - left:i + right + 1])) / highs[i] * 100
            vol_strength = volumes[i] / (np.mean(volumes[max(0, i - 10):i + 1]) + 1e-9)
            if strength >= min_strength and vol_strength > 0.8:
                swing_highs.append({"price": highs[i], "idx": i, "strength": strength})
        if all(lows[i] < lows[i - j] for j in range(1, left + 1)) and all(lows[i] < lows[i + j] for j in range(1, right + 1)):
            strength = (max(highs[i - left:i + right + 1]) - lows[i]) / lows[i] * 100
            vol_strength = volumes[i] / (np.mean(volumes[max(0, i - 10):i + 1]) + 1e-9)
            if strength >= min_strength and vol_strength > 0.8:
                swing_lows.append({"price": lows[i], "idx": i, "strength": strength})
    return swing_highs, swing_lows

def get_dynamic_levels(candles, atr, current_price):
    swing_highs, swing_lows = find_fractal_swings(candles)
    resistance_candidates = [s["price"] for s in swing_highs[-6:]] if swing_highs else []
    support_candidates = [s["price"] for s in swing_lows[-6:]] if swing_lows else []

    if swing_highs:
        last_high = swing_highs[-1]["price"]
        resistance_candidates += [last_high + atr * 0.3, last_high - atr * 0.2]
    if swing_lows:
        last_low = swing_lows[-1]["price"]
        support_candidates += [last_low - atr * 0.3, last_low + atr * 0.2]

    highs = [c[2] for c in candles[:-1]]
    lows = [c[3] for c in candles[:-1]]
    if highs:
        resistance_candidates.append(max(highs))
    if lows:
        support_candidates.append(min(lows))

    resistance = min([r for r in resistance_candidates if r > current_price], default=current_price * 1.02)
    support = max([s for s in support_candidates if s < current_price], default=current_price * 0.98)

    if abs(resistance - current_price) / current_price > 0.04:
        resistance = current_price + atr * 1.8
    if abs(support - current_price) / current_price > 0.04:
        support = current_price - atr * 1.8

    return support, resistance

def get_higher_tf_bias(symbol):
    try:
        ohlcv_1h = exchange.fetch_ohlcv(symbol, timeframe=HIGHER_TF_1H, limit=60)
        ohlcv_4h = exchange.fetch_ohlcv(symbol, timeframe=HIGHER_TF_4H, limit=40)
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
        if closes_1h[-1] > ema20_1h > ema50_1h:
            score += 2
        elif closes_1h[-1] < ema20_1h < ema50_1h:
            score -= 2
        if closes_4h[-1] > ema20_4h > ema50_4h:
            score += 2
        elif closes_4h[-1] < ema20_4h < ema50_4h:
            score -= 2
        if rsi_1h > 55:
            score += 1
        elif rsi_1h < 45:
            score -= 1

        if score >= 3:
            return "бычье (растёт)", rsi_1h
        elif score <= -3:
            return "медвежье (падает)", rsi_1h
        return "нейтральное", rsi_1h
    except Exception as e:
        print(f"Ошибка higher TF {symbol}: {e}")
        return "нейтральное", 50.0

# ====================== РЫНОЧНЫЕ ДАННЫЕ ======================
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
        ohlcv = exchange.fetch_ohlcv("BTC/USDT:USDT", timeframe="1h", limit=50)
        closes = [c[4] for c in ohlcv]
        atr = calculate_atr(ohlcv, 14)
        atr_pct = (atr / closes[-1]) * 100 if closes[-1] else 0
        rsi = calculate_rsi(closes)
        change_24h = ((closes[-1] - closes[-24]) / closes[-24] * 100) if len(closes) >= 24 else 0

        extreme_vol = atr_pct > MAX_BTC_VOLATILITY_ATR
        trend = "бычий" if rsi > 55 and change_24h > 0.8 else ("медвежий" if rsi < 45 and change_24h < -0.8 else "нейтральный")
        return {
            "atr_pct": atr_pct,
            "rsi": rsi,
            "change_24h": change_24h,
            "extreme_vol": extreme_vol,
            "trend": trend
        }
    except Exception as e:
        print(f"BTC context error: {e}")
        return {"atr_pct": 1.0, "rsi": 50, "change_24h": 0, "extreme_vol": False, "trend": "нейтральный"}

@retry_on_error()
def get_funding_and_oi(symbol):
    funding = 0.0
    try:
        fr = exchange.fetch_funding_rate(symbol)
        funding = float(fr.get('fundingRate', 0) or 0) * 100
    except:
        pass
    return funding, 0.0

@retry_on_error()
def get_fear_greed():
    global fear_greed_cache
    now = time.time()
    if now - fear_greed_cache["timestamp"] < 1800:
        return fear_greed_cache["value"]
    try:
        r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=8)
        data = r.json()
        value = int(data["data"][0]["value"])
        fear_greed_cache = {"value": value, "timestamp": now}
        return value
    except:
        return fear_greed_cache["value"]

def check_liquidations_improved(symbol, support, resistance, current_price):
    try:
        liquidations = exchange.fetch_liquidations(symbol, limit=50)
        if not liquidations:
            return None

        now_ms = time.time() * 1000
        window_ms = 45 * 60 * 1000
        recent = [l for l in liquidations if now_ms - l.get('timestamp', 0) <= window_ms]
        if not recent:
            return None

        total_vol = 0.0
        longs_liq = 0.0
        shorts_liq = 0.0
        for liq in recent:
            amount = liq.get('usdValue') or (liq.get('amount', 0) * liq.get('price', 0))
            total_vol += amount
            side = liq.get('side', '').lower()
            if side == 'buy':
                shorts_liq += amount
            elif side == 'sell':
                longs_liq += amount

        if total_vol < 120000:  # повысили порог
            return None

        near_level = abs(current_price - support) / support * 100 < 0.45 or abs(current_price - resistance) / resistance * 100 < 0.45

        if shorts_liq > longs_liq * 1.4:
            side_text = "много людей выбило из коротких позиций — цена может продолжить рост"
            preferred = "long"
        else:
            side_text = "много людей выбило из длинных позиций — цена может продолжить падение"
            preferred = "short"

        return {
            "text": side_text,
            "volume": total_vol,
            "near_level": near_level,
            "preferred": preferred
        }
    except Exception:
        return None

# ====================== TELEGRAM ======================
def send_telegram_message(message, symbol, signal_key):
    now = time.time()
    if now - sent_signals.get(signal_key, 0) < COOLDOWN_TIME:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    clean = symbol.split(':')[0].replace('/', '')
    bybit_url = f"https://www.bybit.com/trade/usdt/{clean}"
    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True,
        "reply_markup": {
            "inline_keyboard": [[{"text": f"📈 Открыть {symbol.split(':')[0]} на Bybit", "url": bybit_url}]]
        }
    }
    try:
        r = requests.post(url, json=payload, timeout=12)
        if r.status_code == 200:
            sent_signals[signal_key] = now
            print(f"✅ Сигнал отправлен: {signal_key}")
            return True
        else:
            print(f"Telegram error {r.status_code}: {r.text[:200]}")
            return False
    except Exception as e:
        print(f"Ошибка Telegram: {e}")
        return False

def build_sl_tp(current_price, atr, direction):
    if direction == "long":
        sl = current_price - atr * ATR_SL_MULT
        tp1 = current_price + atr * ATR_TP1_MULT
        tp2 = current_price + atr * ATR_TP2_MULT
        risk = current_price - sl
        reward1 = tp1 - current_price
        rr1 = reward1 / risk if risk > 0 else 0
    else:
        sl = current_price + atr * ATR_SL_MULT
        tp1 = current_price - atr * ATR_TP1_MULT
        tp2 = current_price - atr * ATR_TP2_MULT
        risk = sl - current_price
        reward1 = current_price - tp1
        rr1 = reward1 / risk if risk > 0 else 0
    return sl, tp1, tp2, rr1

def explain_rsi(rsi):
    if rsi < 30:
        return f"{rsi:.0f} — рынок сильно перепродан (хороший момент для покупки)"
    if rsi < 40:
        return f"{rsi:.0f} — рынок перепродан (можно рассматривать покупку)"
    if rsi > 70:
        return f"{rsi:.0f} — рынок сильно перекуплен (хороший момент для продажи)"
    if rsi > 60:
        return f"{rsi:.0f} — рынок перекуплен (можно рассматривать продажу)"
    return f"{rsi:.0f} — рынок в нормальной зоне"

def explain_stoch(k):
    if k < 20:
        return f"{k:.0f} — сильная перепроданность (возможен разворот вверх)"
    if k < 30:
        return f"{k:.0f} — перепроданность (возможен разворот вверх)"
    if k > 80:
        return f"{k:.0f} — сильная перекупленность (возможен разворот вниз)"
    if k > 70:
        return f"{k:.0f} — перекупленность (возможен разворот вниз)"
    return f"{k:.0f} — нейтральная зона"

def explain_macd(hist):
    if hist > 0.001:
        return "положительный и растущий — сила у покупателей"
    if hist > 0:
        return "слабо положительный — покупатели немного сильнее"
    if hist < -0.001:
        return "отрицательный и падающий — сила у продавцов"
    if hist < 0:
        return "слабо отрицательный — продавцы немного сильнее"
    return "около нуля — нет явного преимущества"

def explain_divergence(div):
    if div == "bullish":
        return "бычья — цена ещё падала, а сила уже росла (признак разворота вверх)"
    if div == "bearish":
        return "медвежья — цена ещё росла, а сила уже падала (признак разворота вниз)"
    return "нет"

def explain_funding(fund):
    if fund > 0.03:
        return f"{fund:.4f}% — высокая плата за лонги (осторожно с покупками)"
    if fund > 0.01:
        return f"{fund:.4f}% — небольшая плата за лонги"
    if fund < -0.03:
        return f"{fund:.4f}% — высокая плата за шорты (осторожно с продажами)"
    if fund < -0.01:
        return f"{fund:.4f}% — небольшая плата за шорты"
    return f"{fund:.4f}% — почти ноль (нейтрально)"

def explain_fng(fng):
    if fng >= 75:
        return f"{fng} — рынок очень жадный (высокий риск коррекции)"
    if fng >= 55:
        return f"{fng} — рынок в зоне жадности"
    if fng <= 25:
        return f"{fng} — рынок в страхе (возможны хорошие точки для покупки)"
    if fng <= 45:
        return f"{fng} — рынок скорее боится"
    return f"{fng} — рынок спокойный"

def explain_volume(is_spike, rising):
    if is_spike and rising:
        return "очень высокий и растущий — сильный интерес крупных игроков"
    if is_spike:
        return "повышенный — есть интерес"
    if rising:
        return "растёт — интерес увеличивается"
    return "обычный"

# ====================== ОСНОВНАЯ ЛОГИКА ======================
def check_markets():
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Проверка рынков...")
    if is_low_liquidity_session():
        print("⏳ Низкая ликвидность (ночь UTC) — пропускаем цикл")
        return

    symbols = get_top_symbols()
    btc_ctx = get_btc_context()
    fng = get_fear_greed()

    if btc_ctx["extreme_vol"]:
        print(f"⚠️ Экстремальная волатильность BTC — фильтруем слабые сигналы")

    candidates = []  # сюда собираем все потенциальные сигналы

    for symbol in symbols:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=80)
            if not ohlcv or len(ohlcv) < 40:
                continue

            closes = [c[4] for c in ohlcv]
            volumes = [c[5] for c in ohlcv]
            last = ohlcv[-1]
            prev = ohlcv[-2]
            open_p = last[1]
            current_price = last[4]
            last_vol = last[5]

            atr = calculate_atr(ohlcv, 14)
            rsi = calculate_rsi(closes)
            macd, macd_sig, macd_hist = calculate_macd(closes)
            stoch_k, stoch_d = calculate_stochastic(ohlcv)
            rsi_series = [calculate_rsi(closes[:i+1]) for i in range(20, len(closes))]
            divergence = detect_rsi_divergence(closes, rsi_series) if len(rsi_series) > 15 else None

            support, resistance = get_dynamic_levels(ohlcv, atr, current_price)
            bias_1h, rsi_1h = get_higher_tf_bias(symbol)
            funding, _ = get_funding_and_oi(symbol)

            avg_vol = np.mean(volumes[-21:-1]) if len(volumes) >= 21 else last_vol
            is_volume_spike = last_vol >= avg_vol * MIN_VOLUME_SPIKE
            rising_volume = all(volumes[-i] > volumes[-i-1] for i in range(1, MIN_RISING_VOLUME_BARS + 1)) if len(volumes) > MIN_RISING_VOLUME_BARS else False

            clean_name = symbol.split(':')[0]

            # ========== 0. ЛИКВИДАЦИИ ==========
            liq = check_liquidations_improved(symbol, support, resistance, current_price)
            if liq and (liq["near_level"] or liq["volume"] > 300000):
                direction = "long" if liq["preferred"] == "long" else "short"
                
                # Фильтры
                if STRICT_BTC_FILTER:
                    if (direction == "long" and btc_ctx["trend"] == "медвежий") or (direction == "short" and btc_ctx["trend"] == "бычий"):
                        continue
                if REQUIRE_HIGHER_TF_ALIGN:
                    if (direction == "long" and "медвежье" in bias_1h) or (direction == "short" and "бычье" in bias_1h):
                        continue

                sl, tp1, tp2, rr = build_sl_tp(current_price, atr, direction)
                if rr < MIN_RR_RATIO:
                    continue

                action = "ПОКУПАТЬ (ЛОНГ)" if direction == "long" else "ПРОДАВАТЬ (ШОРТ)"
                msg = (
                    f"💥 *РЕЗКОЕ ДВИЖЕНИЕ ИЗ-ЗА ЛИКВИДАЦИЙ*\n\n"
                    f"Монета: `{clean_name}`\n"
                    f"Текущая цена: `{current_price}`\n\n"
                    f"*Что делать:* {action}\n\n"
                    f"*Почему сигнал появился:*\n{liq['text']}\n\n"
                    f"*Куда ставить ордера:*\n"
                    f"• Вход: около `{current_price}`\n"
                    f"• Стоп-лосс (защита): `{sl:.4f}`\n"
                    f"• Цель 1: `{tp1:.4f}`\n"
                    f"• Цель 2: `{tp2:.4f}`\n\n"
                    f"Соотношение риска к прибыли: примерно 1 к {rr:.1f}\n\n"
                    f"——————————————\n"
                    f"*Что показывают индикаторы:*\n\n"
                    f"• Сила тренда (RSI): {explain_rsi(rsi)}\n"
                    f"• Момент разворота (Stochastic): {explain_stoch(stoch_k)}\n"
                    f"• Сила движения (MACD): {explain_macd(macd_hist)}\n"
                    f"• Расхождение цены и силы: {explain_divergence(divergence)}\n"
                    f"• Направление на 1 часе: {bias_1h}\n"
                    f"• Плата за перенос позиции: {explain_funding(funding)}\n"
                    f"• Настроение рынка: {explain_fng(fng)}\n"
                    f"• Объём торгов: {explain_volume(is_volume_spike, rising_volume)}"
                )
                score = 80 + (10 if liq["near_level"] else 0) + (5 if is_volume_spike else 0)
                candidates.append({"score": score, "msg": msg, "symbol": symbol, "key": f"{symbol}_liq"})

            # ========== 1. ИМПУЛЬС ==========
            candle_change = ((current_price - open_p) / open_p) * 100
            strong_body = abs(current_price - open_p) / (last[2] - last[3] + 1e-9) > 0.68
            if abs(candle_change) >= IMPULSE_PERCENT and is_volume_spike and rising_volume and strong_body:
                direction = "long" if candle_change > 0 else "short"

                if btc_ctx["extreme_vol"] and abs(candle_change) < 2.3:
                    continue
                if REQUIRE_HIGHER_TF_ALIGN:
                    if (direction == "long" and "медвежье" in bias_1h) or (direction == "short" and "бычье" in bias_1h):
                        continue
                if STRICT_BTC_FILTER:
                    if (direction == "long" and btc_ctx["trend"] == "медвежий") or (direction == "short" and btc_ctx["trend"] == "бычий"):
                        continue

                macd_ok = (macd_hist > 0 and direction == "long") or (macd_hist < 0 and direction == "short")
                stoch_ok = (stoch_k < 78 and direction == "long") or (stoch_k > 22 and direction == "short")

                if not (macd_ok or stoch_ok):
                    continue

                sl, tp1, tp2, rr = build_sl_tp(current_price, atr, direction)
                if rr < MIN_RR_RATIO:
                    continue

                if direction == "long":
                    title = "🚀 СИЛЬНЫЙ РОСТ"
                    action = "ПОКУПАТЬ (ЛОНГ)"
                    reason = "цена резко выросла на очень большом объёме — это признак сильного интереса покупателей"
                else:
                    title = "⚡️ СИЛЬНОЕ ПАДЕНИЕ"
                    action = "ПРОДАВАТЬ (ШОРТ)"
                    reason = "цена резко упала на очень большом объёме — это признак сильного интереса продавцов"

                msg = (
                    f"*{title}*\n\n"
                    f"Монета: `{clean_name}`\n"
                    f"Текущая цена: `{current_price}`\n\n"
                    f"*Что делать:* {action}\n\n"
                    f"*Почему сигнал появился:*\n{reason}\n\n"
                    f"*Куда ставить ордера:*\n"
                    f"• Вход: около `{current_price}`\n"
                    f"• Стоп-лосс (защита): `{sl:.4f}`\n"
                    f"• Цель 1: `{tp1:.4f}`\n"
                    f"• Цель 2: `{tp2:.4f}`\n\n"
                    f"Соотношение риска к прибыли: примерно 1 к {rr:.1f}\n\n"
                    f"——————————————\n"
                    f"*Что показывают индикаторы:*\n\n"
                    f"• Сила тренда (RSI): {explain_rsi(rsi)}\n"
                    f"• Момент разворота (Stochastic): {explain_stoch(stoch_k)}\n"
                    f"• Сила движения (MACD): {explain_macd(macd_hist)}\n"
                    f"• Расхождение цены и силы: {explain_divergence(divergence)}\n"
                    f"• Направление на 1 часе: {bias_1h}\n"
                    f"• Плата за перенос позиции: {explain_funding(funding)}\n"
                    f"• Настроение рынка: {explain_fng(fng)}\n"
                    f"• Объём торгов: {explain_volume(is_volume_spike, rising_volume)}"
                )
                score = 70 + min(abs(candle_change) * 3, 20) + (10 if is_volume_spike else 0)
                candidates.append({"score": score, "msg": msg, "symbol": symbol, "key": f"{symbol}_impulse"})

            # ========== 2. ПОДДЕРЖКА (ЛОНГ) ==========
            support_diff = abs(current_price - support) / support * 100
            if support_diff <= THRESHOLD_PERCENT:
                is_bounce = (prev[4] < prev[1]) and (current_price > open_p)
                rsi_ok = rsi < RSI_OVERSOLD or divergence == "bullish"
                stoch_ok = stoch_k < STOCH_OVERSOLD
                volume_ok = is_volume_spike if REQUIRE_VOLUME_SPIKE else (is_volume_spike or rising_volume)

                if REQUIRE_HIGHER_TF_ALIGN and "медвежье" in bias_1h:
                    continue
                if STRICT_BTC_FILTER and btc_ctx["trend"] == "медвежий":
                    continue

                strong = is_bounce and (rsi_ok or stoch_ok) and volume_ok

                if strong or support_diff < 0.09:
                    sl, tp1, tp2, rr = build_sl_tp(current_price, atr, "long")
                    if rr < MIN_RR_RATIO:
                        continue

                    status = "цена подошла к сильному уровню поддержки и отскочила вверх на объёме" if strong else \
                             "цена очень близко к сильному уровню поддержки"

                    msg = (
                        f"🟢 *ХОРОШАЯ ТОЧКА ДЛЯ ПОКУПКИ*\n\n"
                        f"Монета: `{clean_name}`\n"
                        f"Текущая цена: `{current_price}`\n\n"
                        f"*Что делать:* ПОКУПАТЬ (ЛОНГ)\n\n"
                        f"*Почему сигнал появился:*\n{status}\n\n"
                        f"*Куда ставить ордера:*\n"
                        f"• Вход: около `{current_price}`\n"
                        f"• Стоп-лосс (защита): `{sl:.4f}`\n"
                        f"• Цель 1: `{tp1:.4f}`\n"
                        f"• Цель 2: `{tp2:.4f}`\n\n"
                        f"Соотношение риска к прибыли: примерно 1 к {rr:.1f}\n\n"
                        f"——————————————\n"
                        f"*Что показывают индикаторы:*\n\n"
                        f"• Сила тренда (RSI): {explain_rsi(rsi)}\n"
                        f"• Момент разворота (Stochastic): {explain_stoch(stoch_k)}\n"
                        f"• Сила движения (MACD): {explain_macd(macd_hist)}\n"
                        f"• Расхождение цены и силы: {explain_divergence(divergence)}\n"
                        f"• Направление на 1 часе: {bias_1h}\n"
                        f"• Плата за перенос позиции: {explain_funding(funding)}\n"
                        f"• Настроение рынка: {explain_fng(fng)}\n"
                        f"• Объём торгов: {explain_volume(is_volume_spike, rising_volume)}"
                    )
                    score = 65 + (15 if strong else 0) + (10 if support_diff < 0.08 else 0)
                    candidates.append({"score": score, "msg": msg, "symbol": symbol, "key": f"{symbol}_support"})

            # ========== 3. СОПРОТИВЛЕНИЕ (ШОРТ) ==========
            resist_diff = abs(current_price - resistance) / resistance * 100
            if resist_diff <= THRESHOLD_PERCENT:
                is_reject = (prev[4] > prev[1]) and (current_price < open_p)
                rsi_ok = rsi > RSI_OVERBOUGHT or divergence == "bearish"
                stoch_ok = stoch_k > STOCH_OVERBOUGHT
                volume_ok = is_volume_spike if REQUIRE_VOLUME_SPIKE else (is_volume_spike or rising_volume)

                if REQUIRE_HIGHER_TF_ALIGN and "бычье" in bias_1h:
                    continue
                if STRICT_BTC_FILTER and btc_ctx["trend"] == "бычий":
                    continue

                strong = is_reject and (rsi_ok or stoch_ok) and volume_ok

                if strong or resist_diff < 0.09:
                    sl, tp1, tp2, rr = build_sl_tp(current_price, atr, "short")
                    if rr < MIN_RR_RATIO:
                        continue

                    status = "цена подошла к сильному уровню сопротивления и отбилась вниз на объёме" if strong else \
                             "цена очень близко к сильному уровню сопротивления"

                    msg = (
                        f"🔴 *ХОРОШАЯ ТОЧКА ДЛЯ ПРОДАЖИ*\n\n"
                        f"Монета: `{clean_name}`\n"
                        f"Текущая цена: `{current_price}`\n\n"
                        f"*Что делать:* ПРОДАВАТЬ (ШОРТ)\n\n"
                        f"*Почему сигнал появился:*\n{status}\n\n"
                        f"*Куда ставить ордера:*\n"
                        f"• Вход: около `{current_price}`\n"
                        f"• Стоп-лосс (защита): `{sl:.4f}`\n"
                        f"• Цель 1: `{tp1:.4f}`\n"
                        f"• Цель 2: `{tp2:.4f}`\n\n"
                        f"Соотношение риска к прибыли: примерно 1 к {rr:.1f}\n\n"
                        f"——————————————\n"
                        f"*Что показывают индикаторы:*\n\n"
                        f"• Сила тренда (RSI): {explain_rsi(rsi)}\n"
                        f"• Момент разворота (Stochastic): {explain_stoch(stoch_k)}\n"
                        f"• Сила движения (MACD): {explain_macd(macd_hist)}\n"
                        f"• Расхождение цены и силы: {explain_divergence(divergence)}\n"
                        f"• Направление на 1 часе: {bias_1h}\n"
                        f"• Плата за перенос позиции: {explain_funding(funding)}\n"
                        f"• Настроение рынка: {explain_fng(fng)}\n"
                        f"• Объём торгов: {explain_volume(is_volume_spike, rising_volume)}"
                    )
                    score = 65 + (15 if strong else 0) + (10 if resist_diff < 0.08 else 0)
                    candidates.append({"score": score, "msg": msg, "symbol": symbol, "key": f"{symbol}_resistance"})

        except Exception as e:
            print(f"Ошибка обработки {symbol}: {e}")

    # ===== Отправляем только лучшие сигналы (максимум 2) =====
    if candidates:
        candidates.sort(key=lambda x: x["score"], reverse=True)
        sent_count = 0
        for cand in candidates:
            if sent_count >= MAX_SIGNALS_PER_CYCLE:
                break
            if send_telegram_message(cand["msg"], cand["symbol"], cand["key"]):
                sent_count += 1
        print(f"Отправлено сигналов в этом цикле: {sent_count}")
    else:
        print("Подходящих сигналов не найдено")

# ====================== ЗАПУСК ======================
if __name__ == "__main__":
    print("🚀 Бот запущен (среднее ужесточение + макс. 2 сигнала за цикл)")
    print("Режим: только сигналы. Реальную торговлю веди вручную!")
    while True:
        try:
            check_markets()
        except Exception as e:
            print(f"Критическая ошибка цикла: {e}")
        time.sleep(180)
