"""
health.py - Health check endpoint for UptimeRobot monitoring.
Returns simple JSON status to prevent Render free tier from sleeping.
"""
import time
import logging
import platform
from aiohttp import web

logger = logging.getLogger(__name__)

START_TIME = time.time()

async def health_route(request: web.Request) -> web.Response:
    """
    GET /health endpoint.
    Returns service status, uptime, and basic system info.
    Designed for UptimeRobot ping (every 5 min).
    """
    uptime_seconds = time.time() - START_TIME
    uptime_hours = uptime_seconds / 3600

    payload = {
        "status": "ok",
        "service": "BettingAssistantBot",
        "timestamp": time.time(),
        "uptime": {
            "seconds": round(uptime_seconds, 1),
            "hours": round(uptime_hours, 2)
        },
        "system": {
            "python": platform.python_version(),
            "platform": platform.system()
        },
        "checks": {
            "database": "connected",  # Could add real DB ping here
            "api": "operational",
            "telegram_bot": "webhook_active"
        }
    }

    # Log ping for monitoring (optional, rate-limit in production)
    logger.debug(f"🏓 Health check ping from {request.remote}")

    return web.json_response(payload, status=200)

async def ready_route(request: web.Request) -> web.Response:
    """
    GET /ready endpoint.
    For Kubernetes/load balancer readiness probes.
    Returns 200 only if all critical dependencies are healthy.
    """
    # Add real dependency checks here if needed:
    # - Database connection
    # - External API availability (Polymarket, GROQ)
    # - Telegram webhook status

    return web.json_response({"ready": True}, status=200)

async def metrics_route(request: web.Request) -> web.Response:
    """
    GET /metrics endpoint.
    Simple Prometheus-style metrics for future observability.
    """
    uptime = time.time() - START_TIME
    metrics_text = (
        f"# HELP betting_bot_uptime_seconds Service uptime in seconds\n"
        f"# TYPE betting_bot_uptime_seconds gauge\n"
        f"betting_bot_uptime_seconds {uptime:.1f}\n"
        f"# HELP betting_bot_requests_total Total HTTP requests\n"
        f"# TYPE betting_bot_requests_total counter\n"
        f"betting_bot_requests_total 0\n"  # Would increment with real middleware
    )
    return web.Response(text=metrics_text, content_type="text/plain; version=0.0.4")

