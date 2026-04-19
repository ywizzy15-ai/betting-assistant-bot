"""
main.py - Central entry point for the Telegram Betting Assistant Bot.
Integrates Aiogram webhook handler with aiohttp web server for Render deployment.
"""
import os
import logging
from pathlib import Path
from aiohttp import web
from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from dotenv import load_dotenv

# Load environment variables early
load_dotenv()

# ==================== CONFIGURATION ====================
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
RENDER_URL = os.getenv("RENDER_EXTERNAL_URL", "http://localhost:8080").rstrip("/")
PORT = int(os.getenv("PORT", "8080"))
WEBHOOK_PATH = f"/webhook/{BOT_TOKEN.split(':')[0]}"  # Obscured path for security
WEBHOOK_URL = f"{RENDER_URL}{WEBHOOK_PATH}"

# # ==================== PROJECT IMPORTS ====================
# Core
from database import db

# Telegram handlers (in telegram/ folder)
from telegram.bot import bot_router
from telegram.miniapp_api import api_bets_route

# Web routes (in web/ folder)
from web.health import health_route, ready_route


# ==================== LOGGING SETUP ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ==================== AI & WEB SERVER INIT ====================
# Aiogram Bot & Dispatcher
bot = Bot(token=BOT_TOKEN, parse_mode=ParseMode.HTML)
dp = Dispatcher()
dp.include_router(bot_router)

# Aiohttp Web Application
app = web.Application()

# Register Telegram Webhook Handler
SimpleRequestHandler(dispatcher=dp, bot=bot).register(app, path=WEBHOOK_PATH)
setup_application(app, dp, bot=bot)

# Register REST & Health Routes
app.router.add_get("/health", health_route)
app.router.add_get("/ready", ready_route)
app.router.add_get("/api/bets", api_bets_route)

# MiniApp Frontend Route
MINIAPP_PATH = Path(__file__).parent / "web" / "miniapp.html"

async def serve_miniapp(request: web.Request) -> web.Response:
    """Serves the standalone MiniApp HTML file."""
    if not MINIAPP_PATH.exists():
        logger.error("❌ miniapp.html not found in web/ directory")
        return web.Response(text="MiniApp frontend missing. Check deployment.", status=404)
    return web.FileResponse(MINIAPP_PATH)

app.router.add_get("/miniapp", serve_miniapp)

# ==================== LIFECYCLE HOOKS ====================
async def on_startup(app: web.Application) -> None:
    """Initialize DB, set webhook, and log startup."""
    await db.init_db()
    await bot.set_webhook(
        url=WEBHOOK_URL,
        allowed_updates=dp.resolve_used_update_types(),
        drop_pending_updates=True,
    )
    logger.info("✅ Database initialized successfully")
    logger.info(f"🔗 Telegram webhook set to: {WEBHOOK_URL}")
    logger.info("🤖 Bot is live and ready for commands!")

async def on_shutdown(app: web.Application) -> None:
    """Clean up webhook on redeploy/shutdown."""
    await bot.delete_webhook(drop_pending_updates=True)
    logger.info("👋 Webhook deleted. Graceful shutdown complete.")

app.on_startup.append(on_startup)
app.on_shutdown.append(on_shutdown)

# ==================== LOCAL DEV FALLBACK ====================
if __name__ == "__main__":
    logger.info(f"🚀 Starting local dev server on http://0.0.0.0:{PORT}")
    web.run_app(app, host="0.0.0.0", port=PORT)

