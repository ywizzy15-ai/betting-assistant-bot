"""
bot.py - Telegram Bot handlers using aiogram 3.x
Implements HITL workflow: parses commands -> calls prediction engine -> 
displays triple-layer probabilities + divergence grading -> manual execution.
"""
import logging
from aiogram import Router, types, F
from aiogram.filters import Command, CommandStart
from aiogram.types import WebAppInfo
from aiogram.utils.keyboard import InlineKeyboardBuilder

from config import ADMIN_IDS, RENDER_EXTERNAL_URL
from database import db
from parser import parse_betting_command
from engine import predict_matches

logger = logging.getLogger(__name__)
bot_router = Router()

# ==================== COMMAND HANDLERS ====================
@bot_router.message(CommandStart())
async def cmd_start(msg: types.Message):
    """Handle /start command, register user, show welcome + MiniApp."""
    await db.ensure_user(msg.from_user.id, msg.from_user.username or "anon")
    role = await db.get_role(msg.from_user.id)
    
    kb = InlineKeyboardBuilder()
    kb.button(text="📊 Open MiniApp", web_app=WebAppInfo(url=f"{RENDER_EXTERNAL_URL}/miniapp"))
    kb.button(text="⚙️ Settings", callback_data="settings")
    
    await msg.answer(
        f"🤖 <b>Welcome to Betting Assistant!</b>\n"
        f"👤 Role: <code>{role}</code>\n\n"
        f"💡 <b>Examples:</b>\n"
        f"• <code>book me 20 odd football</code>\n"
        f"• <code>5 draws with 65% probability</code>\n"
        f"• <code>/predict over 2.5 basketball</code>",
        reply_markup=kb.as_markup(),
        parse_mode="HTML"
    )

@bot_router.message(Command("predict") | F.text.lower().contains("book"))
async def cmd_predict(msg: types.Message):
    """Parse natural language command -> fetch predictions -> display results."""
    role = await db.get_role(msg.from_user.id)
    if role == "guest":
        await msg.answer("⚠️ Access restricted. Contact an admin to register.")
        return

    query = msg.text.replace("/predict", "").strip()
    if not query:
        query = "football match winner"

    await msg.answer("🔍 Analyzing markets, Polymarket divergence & ML probabilities...")

    try:
        params = parse_betting_command(msg.text)
        opportunities = await predict_matches(params)

        if not opportunities:
            await msg.answer("❌ No high-confidence opportunities found matching your criteria.\n💡 Try adjusting odds range or probability threshold.")
            return

        for opp in opportunities:
            # Format triple-layer data
            poly_str = f"{opp['poly_prob']*100:.1f}%" if opp['poly_prob'] else "N/A"
            text = (
                f"📈 <b>{opp['sport']} | {opp['market']}</b>\n"
                f"⚽ {opp['teams']}\n"
                f"🏆 League: {opp['league']}\n"
                f"💰 Odds: <b>{opp['odds']}</b> | 📉 ML: <b>{opp['ml_prob']*100:.1f}%</b> | Poly: <b>{poly_str}</b>\n"
                f"🔍 {opp['divergence_status']} | 🔐 Conf: <b>{opp['confidence']:.1f}% ({opp['grade']})</b>\n"
                f"💡 {opp['insight']}\n"
                f"📱 Available on: {', '.join(opp['platforms'])}"
            )

            kb = InlineKeyboardBuilder()
            kb.button(text="📱 View in MiniApp", web_app=WebAppInfo(url=f"{RENDER_EXTERNAL_URL}/miniapp"))
            kb.button(text="✅ Copy Details", callback_data=f"copy_{opp['id']}")

            await msg.answer(text, reply_markup=kb.as_markup(), parse_mode="HTML")
            # Log bet request for analytics
            await db.log_bot_bet(
                msg.from_user.id, 
                opp["platforms"][0], 
                opp["market"], 
                opp["teams"], 
                opp["odds"], 
                opp["confidence"]
            )

    except Exception as e:
        logger.error(f"Prediction pipeline failed: {e}", exc_info=True)
        await msg.answer("⚠️ An error occurred while analyzing matches. Please try again later.")

# ==================== CALLBACK HANDLERS ====================
@bot_router.callback_query(F.data.startswith("copy_"))
async def cb_copy(cb: types.CallbackQuery):
    """Acknowledge copy action. Actual clipboard copy happens client-side in MiniApp."""
    await cb.answer("✅ Bet details copied! Place manually on your preferred platform.", show_alert=False)

@bot_router.callback_query(F.data == "settings")
async def cb_settings(cb: types.CallbackQuery):
    """Placeholder for user preferences panel."""
    await cb.answer("⚙️ Settings panel coming in v1.2", show_alert=True)

