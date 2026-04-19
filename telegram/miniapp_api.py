"""
miniapp_api.py - REST API endpoint for the Telegram MiniApp.
Fetches, filters, and serves prediction data from the database.
Designed for async aiohttp routing and seamless MiniApp integration.
"""
import json
import logging
from aiohttp import web
from database import db

logger = logging.getLogger(__name__)

async def api_bets_route(request: web.Request) -> web.Response:
    """
    GET /api/bets endpoint.
    Query params: sport, market, platform, grade, min_confidence, limit
    Returns: JSON array of recent predictions filtered by MiniApp UI selections.
    """
    try:
        # Parse query parameters from MiniApp
        limit = int(request.query.get("limit", 50))
        sport_filter = request.query.get("sport", "").strip().lower()
        market_filter = request.query.get("market", "").strip().lower()
        platform_filter = request.query.get("platform", "").strip().lower()
        grade_filter = request.query.get("grade", "").strip().upper()
        min_confidence = float(request.query.get("min_confidence", 0.0))

        # Fetch recent predictions from async SQLite
        predictions = await db.get_recent_predictions(limit)

        # Apply in-memory filtering (efficient for <50 records)
        filtered = []
        for p in predictions:
            # Confidence threshold
            if p.get("confidence", 0) < min_confidence:
                continue
            
            # Grade filter
            if grade_filter and p.get("grade", "").upper() != grade_filter:
                continue
                
            # Sport & Market substring matching
            if sport_filter and sport_filter not in p.get("sport", "").lower():
                continue
            if market_filter and market_filter not in p.get("market", "").lower():
                continue
                
            # Platform array check
            if platform_filter:
                platforms = p.get("platforms", [])
                if not any(platform_filter in str(pl).lower() for pl in platforms):
                    continue
                    
            filtered.append(p)

        logger.info(f"📊 API /api/bets: served {len(filtered)}/{limit} predictions")
        return web.json_response(filtered)

    except ValueError as e:
        logger.warning(f"⚠️ Invalid query parameters: {e}")
        return web.json_response({"error": "Invalid parameters. Use: limit, sport, market, platform, grade, min_confidence"}, status=400)
    except Exception as e:
        logger.error(f"❌ API /api/bets failed: {e}", exc_info=True)
        return web.json_response({"error": "Internal server error"}, status=500)

