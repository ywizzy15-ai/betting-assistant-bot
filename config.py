"""
config.py - Central configuration & constants for the Betting Assistant Bot
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ==================== CORE ENVIRONMENT ====================
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
RENDER_EXTERNAL_URL = os.getenv("RENDER_EXTERNAL_URL", "http://localhost:8080").rstrip("/")
PORT = int(os.getenv("PORT", "8080"))
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
FOOTBALL_DATA_KEY = os.getenv("FOOTBALL_DATA_KEY", "").strip()

# Access Control
ADMIN_IDS = [int(x) for x in os.getenv("ADMIN_IDS", "").split(",") if x.strip()]

# ==================== API ENDPOINTS ====================
POLYMARKET_GAMMA_API = "https://gamma-api.polymarket.com"
POLYMARKET_CLOB_API = "https://clob.polymarket.com"
FOOTBALL_DATA_BASE = "https://api.football-data.org/v4"

# ==================== FEATURE ENGINEERING CONSTANTS ====================
# ELO Rating System (FIFA/FiveThirtyEight hybrid)
ELO_K_FACTOR = 32
ELO_HOME_ADVANTAGE = 65

# xG Proxy Approximation
XG_SOT_CONVERSION = 0.30   # ~30% of shots on target become goals
XG_OFF_TARGET_CONVERSION = 0.03

# Fatigue & Fixture Congestion
FATIGUE_THRESHOLD_DAYS = 3  # <3 rest days = fatigued
DEFAULT_INITIAL_REST = 14   # Default for first match in dataset

# ==================== TRIPLE-LAYER PROBABILITY WEIGHTS ====================
# Blending: 40% ML Model + 35% Polymarket + 25% Bookmaker
TRIPLE_BLEND_WEIGHTS = {
    "ml": 0.40,
    "polymarket": 0.35,
    "bookmaker": 0.25
}

# ==================== DIVERGENCE & GRADING THRESHOLDS ====================
# Absolute probability gaps between sources
DIVERGENCE_THRESHOLDS = {
    "high_agreement": 0.05,   # <5% gap
    "moderate_divergence": 0.08, # 5-8% gap
    "high_divergence": 0.08   # >8% gap (potential edge)
}

# Confidence grading scale (edge percentage + liquidity bonus)
CONFIDENCE_GRADES = {
    "HIGH": 15.0,
    "MEDIUM": 5.0
    # <5.0 defaults to LOW
}

# ==================== PATHS & DIRECTORIES ====================
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "betting.db"
WEB_DIR = BASE_DIR / "web"
LOG_DIR = BASE_DIR / "logs"

# Ensure required directories exist
WEB_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ==================== LLM / GROQ SETTINGS ====================
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_TEMPERATURE = 0.3
GROQ_MAX_TOKENS = 60

