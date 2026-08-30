import os
import time
import requests
import ccxt
from datetime import datetime, timezone
from functools import wraps
import numpy as np

# ====================== НАСТРОЙКИ ======================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID", "673791974")

TIMEFRAME = "15m"               # основной ТФ
HIGHER_TF_1H = "1h"
HIGHER_TF_4H = "4h"

THRESHOLD_PERCENT = 0.25        # допуск к уровню (%)
IMPULSE_PERCENT = 1.3           # мин. импульс свечи
TOP_COINS_LIMIT = 8
COOLDOWN_TIME = 2700            # 45 минут
CACHE_TOP_SECONDS = 300         # кэш топ-монет 5 минут
MAX_RETRIES = 4
BACKOFF_BASE = 1.8

# Риск-параметры (для подсказок в сообщениях)
RISK_PERCENT = 0.8              # % депозита на сделку (подсказка)
ATR_SL_MULT = 1.4
ATR_TP1_MULT = 2.0
ATR_TP2_MULT = 3.5

# Фильтры
MIN_VOLUME_SPIKE = 1.7
MIN_RISING_VOLUME_BARS = 2
RSI_OVERSOLD = 38
RSI_OVERBOUGHT = 62
STOCH_OVERSOLD = 25
STOCH_OVERBOUGHT = 75
MAX_BTC_VOLATILITY_ATR = 2.8    # если ATR BTC слишком высокий — фильтруем
LOW_LIQUIDITY_HOURS_UTC = {0, 1, 2, 3, 4, 5}  # часы низкой ликвидности (UTC)

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
        return np.mean(trs) if trs else 0.0
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
    signal_line[slow+signal-2] = np.mean(macd_line[slow-1:slow+signal-1])
    alpha_sig = 2 / (signal + 1)
    for i in range(slow+signal-1, len(closes)):
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
    for i in range(k_period-1, len(candles)):
        highest = max(highs[i-k_period+1:i+1])
        lowest = min(lows[i-k_period+1:i+1])
        if highest == lowest:
            k = 50.0
        else:
            k = 100 * (closes[i] - lowest) / (highest - lowest)
        k_values.append(k)
    if len(k_values) < d_period:
        return k_values[-1], k_values[-1]
    d = np.mean(k_values[-d_period:])
    return float(k_values[-1]), float(d)

def detect_rsi_divergence(closes, rsi_values, lookback=20):
    """Простая бычья/медвежья дивергенция RSI"""
    if len(closes) < lookback or len(rsi_values) < lookback:
        return None
    price_slice = closes[-lookback:]
    rsi_slice = rsi_values[-lookback:]
    # Бычья: цена делает ниже лоу, RSI — выше
    price_low_idx = int(np.argmin(price_slice))
    rsi_low_idx = int(np.argmin(rsi_slice))
    if price_low_idx > lookback // 2 and rsi_low_idx < price_low_idx - 3:
        if price_slice[price_low_idx] < min(price_slice[:price_low_idx]) and rsi_slice[rsi_low_idx] > rsi_slice[price_low_idx]:
            return "bullish"
    # Медвежья
    price_high_idx = int(np.argmax(price_slice))
    rsi_high_idx = int(np.argmax(rsi_slice))
    if price_high_idx > lookback // 2 and rsi_high_idx < price_high_idx - 3:
        if price_slice[price_high_idx] > max(price_slice[:price_high_idx]) and rsi_slice[rsi_high_idx] < rsi_slice[price_high_idx]:
            return "bearish"
    return None

def find_fractal_swings(candles, left=2, right=2, min_strength=0.15):
    """Фрактальные хаи/лоу + минимальная сила (в % от цены)"""
    highs = [c[2] for c in candles]
    lows = [c[3] for c in candles]
    volumes = [c[5] for c in candles]
    swing_highs = []
    swing_lows = []
    for i in range(left, len(candles) - right):
        # Fractal High
        if all(highs[i] > highs[i-j] for j in range(1, left+1)) and all(highs[i] > highs[i+j] for j in range(1, right+1)):
            strength = (highs[i] - min(lows[i-left:i+right+1])) / highs[i] * 100
            vol_strength = volumes[i] / (np.mean(volumes[max(0, i-10):i+1]) + 1e-9)
            if strength >= min_strength and vol_strength > 0.8:
                swing_highs.append({"price": highs[i], "idx": i, "strength": strength, "vol": vol_strength})
        # Fractal Low
        if all(lows[i] < lows[i-j] for j in range(1, left+1)) and all(lows[i] < lows[i+j] for j in range(1, right+1)):
            strength = (max(highs[i-left:i+right+1]) - lows[i]) / lows[i] * 100
            vol_strength = volumes[i] / (np.mean(volumes[max(0, i-10):i+1]) + 1e-9)
            if strength >= min_strength and vol_strength > 0.8:
                swing_lows.append({"price": lows[i], "idx": i, "strength": strength, "vol": vol_strength})
    return swing_highs, swing_lows

def get_dynamic_levels(candles, atr, current_price):
    """Комбинация фракталов + ATR-зоны"""
    swing_highs, swing_lows = find_fractal_swings(candles, left=2, right=2, min_strength=0.12)
    resistance_candidates = [s["price"] for s in swing_highs[-6:]] if swing_highs else []
    support_candidates = [s["price"] for s in swing_lows[-6:]] if swing_lows else []

    # Добавляем ATR-зоны вокруг последних экстремумов
    if swing_highs:
        last_high = swing_highs[-1]["price"]
        resistance_candidates.append(last_high + atr * 0.3)
        resistance_candidates.append(last_high - atr * 0.2)
    if swing_lows:
        last_low = swing_lows[-1]["price"]
        support_candidates.append(last_low - atr * 0.3)
        support_candidates.append(last_low + atr * 0.2)

    # Классические max/min как запасной вариант
    highs = [c[2] for c in candles[:-1]]
    lows = [c[3] for c in candles[:-1]]
    if highs:
        resistance_candidates.append(max(highs))
    if lows:
        support_candidates.append(min(lows))

    resistance = min([r for r in resistance_candidates if r > current_price], default=current_price * 1.02)
    support = max([s for s in support_candidates if s < current_price], default=current_price * 0.98)

    # Если слишком далеко — берём ближайший
    if abs(resistance - current_price) / current_price > 0.04:
        resistance = current_price + atr * 1.8
    if abs(support - current_price) / current_price > 0.04:
        support = current_price - atr * 1.8

    return support, resistance, swing_highs, swing_lows

def get_higher_tf_bias(symbol):
    """Структура 1h + 4h"""
    try:
        ohlcv_1h = exchange.fetch_ohlcv(symbol, timeframe=HIGHER_TF_1H, limit=60)
        ohlcv_4h = exchange.fetch_ohlcv(symbol, timeframe=HIGHER_TF_4H, limit=40)
        if not ohlcv_1h or not ohlcv_4h:
            return "neutral", 50.0

        closes_1h = [c[4] for c in ohlcv_1h]
        closes_4h = [c[4] for c in ohlcv_4h]
        rsi_1h = calculate_rsi(closes_1h)
        rsi_4h = calculate_rsi(closes_4h)

        # Простая EMA-структура
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
        if rsi_4h > 55:
            score += 1
        elif rsi_4h < 45:
            score -= 1

        if score >= 3:
            return "bullish", rsi_1h
        elif score <= -3:
            return "bearish", rsi_1h
        return "neutral", rsi_1h
    except Exception as e:
        print(f"Ошибка higher TF {symbol}: {e}")
        return "neutral", 50.0

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
    # Всегда держим BTC и ETH
    for must in ["BTC/USDT:USDT", "ETH/USDT:USDT"]:
        if must not in symbols:
            symbols.insert(0, must)
    symbols = symbols[:limit]
    top_symbols_cache = {"symbols": symbols, "timestamp": now}
    return symbols

@retry_on_error()
def get_btc_context():
    """Волатильность + тренд BTC + доминация (прокси)"""
    try:
        ohlcv = exchange.fetch_ohlcv("BTC/USDT:USDT", timeframe="1h", limit=50)
        closes = [c[4] for c in ohlcv]
        atr = calculate_atr(ohlcv, 14)
        atr_pct = (atr / closes[-1]) * 100 if closes[-1] else 0
        rsi = calculate_rsi(closes)
        change_24h = ((closes[-1] - closes[-24]) / closes[-24] * 100) if len(closes) >= 24 else 0

        # Прокси доминации через относительную силу (упрощённо)
        dominance_bias = "neutral"
        if change_24h > 2.5 and rsi > 55:
            dominance_bias = "btc_strong"
        elif change_24h < -2.5 and rsi < 45:
            dominance_bias = "btc_weak"

        extreme_vol = atr_pct > MAX_BTC_VOLATILITY_ATR
        trend = "bullish" if rsi > 55 and change_24h > 0.8 else ("bearish" if rsi < 45 and change_24h < -0.8 else "neutral")
        return {
            "atr_pct": atr_pct,
            "rsi": rsi,
            "change_24h": change_24h,
            "extreme_vol": extreme_vol,
            "trend": trend,
            "dominance_bias": dominance_bias
        }
    except Exception as e:
        print(f"BTC context error: {e}")
        return {"atr_pct": 1.0, "rsi": 50, "change_24h": 0, "extreme_vol": False, "trend": "neutral", "dominance_bias": "neutral"}

@retry_on_error()
def get_funding_and_oi(symbol):
    funding = 0.0
    oi_change = 0.0
    try:
        fr = exchange.fetch_funding_rate(symbol)
        funding = float(fr.get('fundingRate', 0) or 0) * 100  # в %
    except:
        pass
    try:
        # OI change за последние ~1-2 часа (если биржа отдаёт)
        oi = exchange.fetch_open_interest(symbol)
        # У Bybit через ccxt иногда есть только текущий OI, поэтому делаем простую заглушку
        # Для реальной динамики лучше хранить предыдущее значение, но здесь упрощённо
        oi_change = 0.0
    except:
        pass
    return funding, oi_change

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
        window_ms = 45 * 60 * 1000  # 45 минут
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
            if side == 'buy':      # ликвидация шортов
                shorts_liq += amount
            elif side == 'sell':    # ликвидация лонгов
                longs_liq += amount

        if total_vol < 80000:
            return None

        # Предпочитаем каскады возле уровней
        near_level = False
        if abs(current_price - support) / support * 100 < 0.6 or abs(current_price - resistance) / resistance * 100 < 0.6:
            near_level = True

        side_text = ""
        if shorts_liq > longs_liq * 1.3:
            side_text = "🟢 Сбриты Шорты (импульс вверх → ищем ЛОНГ)"
            preferred = "long"
        else:
            side_text = "🔴 Сбриты Лонги (импульс вниз → ищем ШОРТ)"
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
        return
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
        else:
            print(f"Telegram error {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"Ошибка Telegram: {e}")

def build_sl_tp(current_price, atr, direction):
    """Подсказки SL / TP на основе ATR"""
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
        print(f"⚠️ Экстремальная волатильность BTC (ATR {btc_ctx['atr_pct']:.2f}%) — фильтруем слабые сигналы")

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

            support, resistance, swing_highs, swing_lows = get_dynamic_levels(ohlcv, atr, current_price)
            bias_1h, rsi_1h = get_higher_tf_bias(symbol)
            funding, oi_change = get_funding_and_oi(symbol)

            # Объём
            avg_vol = np.mean(volumes[-21:-1]) if len(volumes) >= 21 else last_vol
            is_volume_spike = last_vol >= avg_vol * MIN_VOLUME_SPIKE
            rising_volume = all(volumes[-i] > volumes[-i-1] for i in range(1, MIN_RISING_VOLUME_BARS+1)) if len(volumes) > MIN_RISING_VOLUME_BARS else False

            # ========== 0. ЛИКВИДАЦИИ ==========
            liq = check_liquidations_improved(symbol, support, resistance, current_price)
            if liq and (liq["near_level"] or liq["volume"] > 250000):
                direction = "long" if liq["preferred"] == "long" else "short"
                # Фильтр против тренда BTC
                if (direction == "long" and btc_ctx["trend"] == "bearish" and btc_ctx["extreme_vol"]) or \
                   (direction == "short" and btc_ctx["trend"] == "bullish" and btc_ctx["extreme_vol"]):
                    pass
                else:
                    sl, tp1, tp2, rr = build_sl_tp(current_price, atr, direction)
                    msg = (
                        f"💥 *КАСКАД ЛИКВИДАЦИЙ*\n"
                        f"Монета: `{symbol.split(':')[0]}`\n"
                        f"{liq['text']}\n"
                        f"Объём: `${liq['volume']:,.0f}` {'✅ возле уровня' if liq['near_level'] else ''}\n"
                        f"Цена: `{current_price}`\n"
                        f"Bias 1h: `{bias_1h}` | RSI 1h: `{rsi_1h:.0f}`\n"
                        f"Funding: `{funding:.4f}%` | F&G: `{fng}`\n"
                        f"────────────────\n"
                        f"🎯 *Рекомендация:* {'ЛОНГ 🟢' if direction=='long' else 'ШОРТ 🔴'}\n"
                        f"SL: `{sl:.4f}` | TP1: `{tp1:.4f}` | TP2: `{tp2:.4f}`\n"
                        f"R:R ≈ `{rr:.1f}`"
                    )
                    send_telegram_message(msg, symbol, f"{symbol}_liq")

            # ========== 1. ИМПУЛЬС ==========
            candle_change = ((current_price - open_p) / open_p) * 100
            strong_body = abs(current_price - open_p) / (last[2] - last[3] + 1e-9) > 0.65
            if abs(candle_change) >= IMPULSE_PERCENT and is_volume_spike and rising_volume and strong_body:
                direction = "long" if candle_change > 0 else "short"
                # Фильтры
                if btc_ctx["extreme_vol"] and abs(candle_change) < 2.2:
                    continue
                if (direction == "long" and bias_1h == "bearish") or (direction == "short" and bias_1h == "bullish"):
                    continue
                # OI + цена (если бы был реальный OI — усиливали бы)
                oi_confirm = True  # заглушка, при наличии реального OI можно ужесточить

                macd_ok = (macd_hist > 0 and direction == "long") or (macd_hist < 0 and direction == "short")
                stoch_ok = (stoch_k < 80 and direction == "long") or (stoch_k > 20 and direction == "short")

                if macd_ok or stoch_ok:
                    sl, tp1, tp2, rr = build_sl_tp(current_price, atr, direction)
                    action = "🟢 ИЩЕМ ЛОНГ (импульс + объём)" if direction == "long" else "🔴 ИЩЕМ ШОРТ (импульс + объём)"
                    dir_text = "🚀 МОЩНЫЙ ИМПУЛЬС РОСТА" if direction == "long" else "⚡️ МОЩНЫЙ ИМПУЛЬС ПАДЕНИЯ"
                    msg = (
                        f"📊 *{dir_text}*\n"
                        f"🎯 {action}\n"
                        f"Монета: `{symbol.split(':')[0]}`\n"
                        f"Изменение: `{candle_change:+.2f}%` | Объём x`{(last_vol/avg_vol):.1f}` ✅\n"
                        f"RSI: `{rsi:.1f}` | Stoch: `{stoch_k:.0f}` | MACD hist: `{macd_hist:.5f}`\n"
                        f"Bias 1h: `{bias_1h}` | Funding: `{funding:.4f}%`\n"
                        f"F&G: `{fng}` | BTC trend: `{btc_ctx['trend']}`\n"
                        f"────────────────\n"
                        f"Вход ≈ `{current_price}`\n"
                        f"SL: `{sl:.4f}` | TP1: `{tp1:.4f}` | TP2: `{tp2:.4f}`\n"
                        f"R:R ≈ `{rr:.1f}`"
                    )
                    send_telegram_message(msg, symbol, f"{symbol}_impulse")

            # ========== 2. ПОДДЕРЖКА (ЛОНГ) ==========
            support_diff = abs(current_price - support) / support * 100
            if support_diff <= THRESHOLD_PERCENT:
                is_bounce = (prev[4] < prev[1]) and (current_price > open_p)
                rsi_ok = rsi < RSI_OVERSOLD or (divergence == "bullish")
                stoch_ok = stoch_k < STOCH_OVERSOLD or stoch_d < STOCH_OVERSOLD
                macd_ok = macd_hist > macd_hist  # просто наличие
                volume_ok = is_volume_spike or rising_volume

                # Сильный фильтр против тренда
                if bias_1h == "bearish" and btc_ctx["trend"] == "bearish":
                    status = "⚠️ Подход к поддержке (старший ТФ против — ждём подтверждения)"
                    strong = False
                elif is_bounce and (rsi_ok or stoch_ok) and volume_ok:
                    status = "🛡 *Отличная точка ЛОНГ* (отскок + объём + осциллятор)"
                    strong = True
                else:
                    status = "⚠️ Подход к поддержке (ждём паттерн / объём)"
                    strong = False

                if strong or support_diff < 0.12:
                    sl, tp1, tp2, rr = build_sl_tp(current_price, atr, "long")
                    msg = (
                        f"🟢 *СИГНАЛ: УРОВЕНЬ ПОДДЕРЖКИ*\n"
                        f"🎯 Рекомендация: *ЛОНГ 🟢*\n"
                        f"Статус: {status}\n"
                        f"Монета: `{symbol.split(':')[0]}`\n"
                        f"Цена: `{current_price}` | Уровень: `{support:.4f}`\n"
                        f"RSI: `{rsi:.1f}` {'🟢' if rsi_ok else ''} | Stoch K: `{stoch_k:.0f}`\n"
                        f"MACD hist: `{macd_hist:.5f}` | Дивергенция: `{divergence or 'нет'}`\n"
                        f"Bias 1h: `{bias_1h}` | Funding: `{funding:.4f}%` | F&G: `{fng}`\n"
                        f"Объём: {'Да ✅' if volume_ok else 'Нет'}\n"
                        f"────────────────\n"
                        f"Вход ≈ `{current_price}`\n"
                        f"SL: `{sl:.4f}` | TP1: `{tp1:.4f}` | TP2: `{tp2:.4f}`\n"
                        f"R:R ≈ `{rr:.1f}`"
                    )
                    send_telegram_message(msg, symbol, f"{symbol}_support")

            # ========== 3. СОПРОТИВЛЕНИЕ (ШОРТ) ==========
            resist_diff = abs(current_price - resistance) / resistance * 100
            if resist_diff <= THRESHOLD_PERCENT:
                is_reject = (prev[4] > prev[1]) and (current_price < open_p)
                rsi_ok = rsi > RSI_OVERBOUGHT or (divergence == "bearish")
                stoch_ok = stoch_k > STOCH_OVERBOUGHT or stoch_d > STOCH_OVERBOUGHT
                volume_ok = is_volume_spike or rising_volume

                if bias_1h == "bullish" and btc_ctx["trend"] == "bullish":
                    status = "⚠️ Подход к сопротивлению (старший ТФ против — ждём)"
                    strong = False
                elif is_reject and (rsi_ok or stoch_ok) and volume_ok:
                    status = "🛡 *Отличная точка ШОРТ* (отбой + объём + осциллятор)"
                    strong = True
                else:
                    status = "⚠️ Подход к сопротивлению (ждём паттерн / объём)"
                    strong = False

                if strong or resist_diff < 0.12:
                    sl, tp1, tp2, rr = build_sl_tp(current_price, atr, "short")
                    msg = (
                        f"🔴 *СИГНАЛ: УРОВЕНЬ СОПРОТИВЛЕНИЯ*\n"
                        f"🎯 Рекомендация: *ШОРТ 🔴*\n"
                        f"Статус: {status}\n"
                        f"Монета: `{symbol.split(':')[0]}`\n"
                        f"Цена: `{current_price}` | Уровень: `{resistance:.4f}`\n"
                        f"RSI: `{rsi:.1f}` {'🔴' if rsi_ok else ''} | Stoch K: `{stoch_k:.0f}`\n"
                        f"MACD hist: `{macd_hist:.5f}` | Дивергенция: `{divergence or 'нет'}`\n"
                        f"Bias 1h: `{bias_1h}` | Funding: `{funding:.4f}%` | F&G: `{fng}`\n"
                        f"Объём: {'Да ✅' if volume_ok else 'Нет'}\n"
                        f"────────────────\n"
                        f"Вход ≈ `{current_price}`\n"
                        f"SL: `{sl:.4f}` | TP1: `{tp1:.4f}` | TP2: `{tp2:.4f}`\n"
                        f"R:R ≈ `{rr:.1f}`"
                    )
                    send_telegram_message(msg, symbol, f"{symbol}_resistance")

        except Exception as e:
            print(f"Ошибка обработки {symbol}: {e}")

# ====================== ЗАПУСК ======================
if __name__ == "__main__":
    print("🚀 Супер-бот v2 запущен (фракталы + ATR + мульти-ТФ + фильтры + информативные алерты)")
    print("Режим: только сигналы. Реальную торговлю веди вручную с жёстким риском!")
    while True:
        try:
            check_markets()
        except Exception as e:
            print(f"Критическая ошибка цикла: {e}")
        time.sleep(180)
