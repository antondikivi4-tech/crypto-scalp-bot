import asyncio
import logging
import os
import sqlite3
from datetime import datetime
import httpx
from fastapi import FastAPI
import uvicorn
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
DB_NAME = "bot_database.db"

# Инициализация FastAPI для прохождения проверок Render
web_app = FastAPI()

@web_app.get("/")
def home():
    return {"status": "Bot is running!"}

# ==========================================
# 1. ИНИЦИАЛИЗАЦИЯ БАЗЫ ДАННЫХ
# ==========================================
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS subscribers (
            user_id INTEGER PRIMARY KEY,
            subscribed_at TEXT
        )
    """)
    conn.commit()
    conn.close()

def add_subscriber(user_id: int):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR IGNORE INTO subscribers (user_id, subscribed_at) VALUES (?, ?)",
        (user_id, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

def remove_subscriber(user_id: int):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM subscribers WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()

def get_all_active_subscribers() -> list[int]:
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT user_id FROM subscribers")
    rows = cursor.fetchall()
    conn.close()
    return [row[0] for row in rows]

# ==========================================
# 2. АНАЛИТИКА И РАБОТА С BYBIT API
# ==========================================
def calculate_rsi(prices: list[float], period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    gains, losses = [], []
    for i in range(1, len(prices)):
        change = prices[i] - prices[i - 1]
        if change > 0:
            gains.append(change)
            losses.append(0.0)
        else:
            gains.append(0.0)
            losses.append(abs(change))
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100 - (100 / (1 + rs)), 2)

async def get_bybit_market_data(ticker: str = "BTC") -> dict | None:
    symbol = f"{ticker}USDT"
    url = f"https://api.bybit.com/v5/market/kline?category=linear&symbol={symbol}&interval=15&limit=50"
    async with httpx.AsyncClient(timeout=10) as client:
        try:
            response = await client.get(url)
            if response.status_code == 200:
                result = response.json()
                candles = result.get("result", {}).get("list", [])
                if not candles:
                    return None
                candles.reverse()
                closes = [float(c[4]) for c in candles]
                return {
                    "ticker": ticker,
                    "price": closes[-1],
                    "rsi": calculate_rsi(closes),
                    "candles": candles
                }
        except Exception as e:
            logger.error(f"Ошибка запроса к Bybit для {ticker}: {e}")
            return None
    return None

def calculate_support_resistance(candles: list) -> tuple[float, float]:
    highs = [float(c[2]) for c in candles]
    lows = [float(c[3]) for c in candles]
    return min(lows[-30:]), max(highs[-30:])

async def check_scalp_levels(ticker: str = "BTC") -> dict | None:
    market = await get_bybit_market_data(ticker)
    if not market:
        return None
    current_price = market["price"]
    rsi = market["rsi"]
    candles = market["candles"]
    if len(candles) >= 30:
        support, resistance = calculate_support_resistance(candles)
        if abs(current_price - support) / support <= 0.004 and rsi < 35:
            return {"ticker": ticker, "signal": "LONG", "setup": "Отскок от уровня поддержки", "level": support, "price": current_price, "rsi": rsi}
        elif abs(current_price - resistance) / resistance <= 0.004 and rsi > 65:
            return {"ticker": ticker, "signal": "SHORT", "setup": "Отскок от уровня сопротивления", "level": resistance, "price": current_price, "rsi": rsi}
    return None

# ==========================================
# 3. ФОНОВЫЙ МОНИТОРИНГ РЫНКА
# ==========================================
async def monitor_scalp_levels_loop(app: Application):
    logger.info("Скальперский монитор уровней запущен.")
    tickers = ["BTC", "ETH", "SOL", "TON"]
    await asyncio.sleep(5)
    while True:
        try:
            subscribers = get_all_active_subscribers()
            if subscribers:
                for ticker in tickers:
                    signal_data = await check_scalp_levels(ticker)
                    if signal_data:
                        sig, price, lvl, rsi, setup, tck = (
                            signal_data["signal"], signal_data["price"],
                            signal_data["level"], signal_data["rsi"],
                            signal_data["setup"], signal_data["ticker"]
                        )
                        emoji = "🟢" if sig == "LONG" else "🔴"
                        text = (
                            f"⚡️ <b>СКАЛЬПИНГ СИГНАЛ • УРОВЕНЬ</b>\n"
                            f"━━━━━━━━━━━━━━━━━━━\n"
                            f"{emoji} Направление: <code>{sig} ({tck})</code>\n"
                            f"📌 Формация: <b>{setup}</b>\n"
                            f"🎯 Уровень: <code>${lvl:,.2f}</code>\n"
                            f"💰 Текущая цена: <code>${price:,.2f}</code>\n"
                            f"📊 RSI(14): <code>{rsi}</code>\n"
                            f"━━━━━━━━━━━━━━━━━━━"
                        )
                        for user_id in subscribers:
                            try:
                                await app.bot.send_message(chat_id=user_id, text=text, parse_mode="HTML")
                                await asyncio.sleep(0.1)
                            except Exception as ex:
                                logger.warning(f"Ошибка отправки {user_id}: {ex}")
                        await asyncio.sleep(300)
            await asyncio.sleep(60)
        except Exception as e:
            logger.error(f"Ошибка в цикле мониторинга: {e}")
            await asyncio.sleep(60)

# ==========================================
# 4. ТЕЛЕГРАМ ХЭНДЛЕРЫ И ЗАПУСК
# ==========================================
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    add_subscriber(user.id)
    await update.message.reply_text(f"Привет, <b>{user.first_name}</b>! Вы подписаны на уведомления.", parse_mode="HTML")

async def cmd_stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    remove_subscriber(update.effective_user.id)
    await update.message.reply_text("🔕 Вы отписались от уведомлений.")

async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("🔄 Анализирую рынок...")
    tickers = ["BTC", "ETH", "SOL", "TON"]
    report = ["📊 <b>Текущий срез рынка (Bybit):</b>\n"]
    for ticker in tickers:
        market = await get_bybit_market_data(ticker)
        if market:
            support, resistance = calculate_support_resistance(market["candles"])
            report.append(f"▪️ <b>{ticker}</b>: ${market['price']:,.2f} | RSI: <code>{market['rsi']}</code>\n   └ Поддержка: ${support:,.2f} | Сопротивление: ${resistance:,.2f}")
    await msg.edit_text("\n".join(report), parse_mode="HTML")

async def run_telegram_bot():
    init_db()
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("stop", cmd_stop))
    app.add_handler(CommandHandler("status", cmd_status))
    
    async def post_init(application: Application):
        asyncio.create_task(monitor_scalp_levels_loop(application))
    app.post_init = post_init
    
    await app.initialize()
    await app.start()
    await app.updater.start_polling()

@web_app.on_event("startup")
async def startup_event():
    asyncio.create_task(run_telegram_bot())

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(web_app, host="0.0.0.0", port=port)
