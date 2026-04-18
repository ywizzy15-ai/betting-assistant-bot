"""
database.py - Async SQLite management for users, roles, and prediction tracking.
Aligned with triple-layer probability storage and MiniApp data retrieval.
"""
import json
import aiosqlite
import logging
from config import DB_PATH

logger = logging.getLogger(__name__)

class Database:
    def __init__(self):
        self.db_path = DB_PATH

    async def init_db(self) -> None:
        """Initialize database tables with optimized schema."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    role TEXT DEFAULT 'user',
                    username TEXT,
                    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    sport TEXT,
                    market TEXT,
                    teams TEXT,
                    odds REAL,
                    ml_prob REAL,
                    poly_prob REAL,
                    bk_prob REAL,
                    blended_prob REAL,
                    confidence REAL,
                    grade TEXT,
                    divergence_status TEXT,
                    edge_percentage REAL,
                    insight TEXT,
                    platforms TEXT, -- Stored as JSON array string
                    status TEXT DEFAULT 'pending', -- pending/won/lost (manual update later)
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_predictions_user ON predictions(user_id);
                CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at DESC);
            """)
            await db.commit()
        logger.info("✅ Database schema initialized successfully.")

    async def get_role(self, user_id: int) -> str:
        """Retrieve user role. Returns 'guest' if not found."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute("SELECT role FROM users WHERE user_id = ?", (user_id,)) as cursor:
                row = await cursor.fetchone()
                return row[0] if row else 'guest'

    async def ensure_user(self, user_id: int, username: str = "anon") -> None:
        """Register user if not exists, update username/last_active."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT OR IGNORE INTO users (user_id, role, username) VALUES (?, 'user', ?)",
                (user_id, username)
            )
            await db.execute(
                "UPDATE users SET username = ?, last_active = CURRENT_TIMESTAMP WHERE user_id = ?",
                (username, user_id)
            )
            await db.commit()

    async def log_prediction(self, user_id: int, prediction: dict) -> None:
        """Log a generated prediction opportunity with full triple-layer metadata."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT INTO predictions (
                    user_id, sport, market, teams, odds, ml_prob, poly_prob, bk_prob,
                    blended_prob, confidence, grade, divergence_status, edge_percentage,
                    insight, platforms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    user_id,
                    prediction.get("sport", "Football"),
                    prediction.get("market", "Match Winner"),
                    prediction.get("teams", "Unknown"),
                    prediction.get("odds", 0.0),
                    prediction.get("ml_prob", 0.0),
                    prediction.get("poly_prob", 0.0),
                    prediction.get("bk_prob", 0.0),
                    prediction.get("blended_prob", 0.0),
                    prediction.get("confidence", 0.0),
                    prediction.get("grade", "LOW"),
                    prediction.get("divergence_status", "N/A"),
                    prediction.get("edge_percentage", 0.0),
                    prediction.get("insight", ""),
                    json.dumps(prediction.get("platforms", []))
                )
            )
            await db.commit()

    async def get_recent_predictions(self, limit: int = 50) -> list[dict]:
        """Fetch recent predictions for the MiniApp API endpoint."""
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """SELECT id, sport, market, teams, odds, ml_prob, poly_prob, bk_prob,
                        blended_prob, confidence, grade, divergence_status, edge_percentage,
                        insight, platforms, created_at
                   FROM predictions ORDER BY created_at DESC LIMIT ?""",
                (limit,)
            ) as cursor:
                rows = await cursor.fetchall()
                return [
                    {
                        "id": r[0], "sport": r[1], "market": r[2], "teams": r[3], "odds": r[4],
                        "ml_prob": r[5], "poly_prob": r[6], "bk_prob": r[7], "blended_prob": r[8],
                        "confidence": r[9], "grade": r[10], "divergence_status": r[11],
                        "edge_percentage": r[12], "insight": r[13],
                        "platforms": json.loads(r[14]), "created_at": r[15]
                    } for r in rows
                ]

# Singleton instance
db = Database()

