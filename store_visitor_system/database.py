"""
Database management module for VisiTrack.

Supports PostgreSQL with seamless SQLite fallback.
Manages visitor records, daily visitor counts, visit histories, and timestamps.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from .config import Config

logger = logging.getLogger(__name__)

# SQL schema definitions compatible with both PostgreSQL and SQLite
_CREATE_VISITORS_PG = """
CREATE TABLE IF NOT EXISTS visitors (
    visitor_id VARCHAR(64) PRIMARY KEY,
    first_seen TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    total_visits INT DEFAULT 1,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
"""

_CREATE_VISIT_EVENTS_PG = """
CREATE TABLE IF NOT EXISTS visit_events (
    id SERIAL PRIMARY KEY,
    visitor_id VARCHAR(64) REFERENCES visitors(visitor_id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    confidence REAL DEFAULT 0.0
);
"""

_CREATE_VISITORS_SQLITE = """
CREATE TABLE IF NOT EXISTS visitors (
    visitor_id TEXT PRIMARY KEY,
    first_seen TEXT DEFAULT (datetime('now')),
    last_seen TEXT DEFAULT (datetime('now')),
    total_visits INTEGER DEFAULT 1,
    created_at TEXT DEFAULT (datetime('now'))
);
"""

_CREATE_VISIT_EVENTS_SQLITE = """
CREATE TABLE IF NOT EXISTS visit_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    visitor_id TEXT REFERENCES visitors(visitor_id) ON DELETE CASCADE,
    timestamp TEXT DEFAULT (datetime('now')),
    confidence REAL DEFAULT 0.0
);
"""


class DatabaseManager:
    """Handles persistence of store visitor logs to PostgreSQL or SQLite.

    Automatically handles connection pooling/reconnection, table creation,
    visit deduplication via cooldown periods, and query reporting.
    """

    def __init__(self, config: "Config") -> None:
        self._config = config
        self._is_postgres = False
        self._pg_conn_str: Optional[str] = None
        self._sqlite_path = Path("visitrack.db")
        self._lock = threading.Lock()

        self._init_connection()
        self._create_tables()

    # ── Initialization & Connection ─────────────────────────────────

    def _init_connection(self) -> None:
        """Initialize database connection. Try PostgreSQL first, fall back to SQLite."""
        if self._config.use_postgres:
            try:
                import psycopg2

                conn_str = (
                    f"host={self._config.postgres_host} "
                    f"port={self._config.postgres_port} "
                    f"dbname={self._config.postgres_db} "
                    f"user={self._config.postgres_user} "
                    f"password={self._config.postgres_password} "
                    f"connect_timeout=3"
                )
                conn = psycopg2.connect(conn_str)
                conn.close()
                self._is_postgres = True
                self._pg_conn_str = conn_str
                logger.info(
                    "✅ Connected to PostgreSQL at %s:%d/%s",
                    self._config.postgres_host,
                    self._config.postgres_port,
                    self._config.postgres_db,
                )
                return
            except Exception as exc:
                logger.warning(
                    "PostgreSQL connection failed (%s). "
                    "Falling back to local SQLite database (%s).",
                    exc,
                    self._sqlite_path.resolve(),
                )

        self._is_postgres = False
        logger.info("Using SQLite database: %s", self._sqlite_path.resolve())

    def _get_connection(self):
        """Return a database connection instance for the current thread."""
        if self._is_postgres:
            import psycopg2
            return psycopg2.connect(self._pg_conn_str)
        else:
            conn = sqlite3.connect(str(self._sqlite_path))
            conn.row_factory = sqlite3.Row
            return conn

    def _create_tables(self) -> None:
        """Create database tables if they do not exist."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                if self._is_postgres:
                    cur.execute(_CREATE_VISITORS_PG)
                    cur.execute(_CREATE_VISIT_EVENTS_PG)
                else:
                    cur.execute(_CREATE_VISITORS_SQLITE)
                    cur.execute(_CREATE_VISIT_EVENTS_SQLITE)
                conn.commit()
                logger.info("Database tables verified.")
            finally:
                conn.close()

    # ── Core Operations ─────────────────────────────────────────────

    def record_visit(
        self,
        visitor_id: str,
        confidence: float = 0.0,
        timestamp: Optional[datetime] = None,
    ) -> bool:
        """Record a visitor visit in the database.

        Applies a cooldown period (default 10 minutes). If the visitor was last
        seen within the cooldown window, updates `last_seen` without incrementing
        the visit counter or creating a duplicate event.

        Args:
            visitor_id: Unique visitor identifier (e.g. "VISITOR-001")
            confidence: Detection confidence score
            timestamp: Visit timestamp (defaults to current time)

        Returns:
            True if a new visit event was registered, False if updated within cooldown.
        """
        now = timestamp or datetime.now()
        now_str = now.isoformat()
        cooldown_delta = timedelta(minutes=self._config.visit_cooldown_minutes)

        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()

                # Check if visitor already exists
                if self._is_postgres:
                    cur.execute(
                        "SELECT last_seen, total_visits FROM visitors WHERE visitor_id = %s",
                        (visitor_id,),
                    )
                else:
                    cur.execute(
                        "SELECT last_seen, total_visits FROM visitors WHERE visitor_id = ?",
                        (visitor_id,),
                    )

                row = cur.fetchone()

                if row is None:
                    # New visitor -> Insert visitor & insert visit event
                    if self._is_postgres:
                        cur.execute(
                            "INSERT INTO visitors (visitor_id, first_seen, last_seen, total_visits, created_at) "
                            "VALUES (%s, %s, %s, 1, %s)",
                            (visitor_id, now, now, now),
                        )
                        cur.execute(
                            "INSERT INTO visit_events (visitor_id, timestamp, confidence) VALUES (%s, %s, %s)",
                            (visitor_id, now, confidence),
                        )
                    else:
                        cur.execute(
                            "INSERT INTO visitors (visitor_id, first_seen, last_seen, total_visits, created_at) "
                            "VALUES (?, ?, ?, 1, ?)",
                            (visitor_id, now_str, now_str, now_str),
                        )
                        cur.execute(
                            "INSERT INTO visit_events (visitor_id, timestamp, confidence) VALUES (?, ?, ?)",
                            (visitor_id, now_str, confidence),
                        )
                    conn.commit()
                    logger.info("🆕 New visitor registered in DB: %s", visitor_id)
                    return True
                else:
                    # Existing visitor -> check cooldown
                    last_seen_val = row[0] if self._is_postgres else row["last_seen"]
                    if isinstance(last_seen_val, str):
                        try:
                            last_seen_dt = datetime.fromisoformat(last_seen_val)
                        except ValueError:
                            last_seen_dt = now - cooldown_delta - timedelta(seconds=1)
                    elif isinstance(last_seen_val, datetime):
                        last_seen_dt = last_seen_val.replace(tzinfo=None) if last_seen_val.tzinfo else last_seen_val
                    else:
                        last_seen_dt = now

                    # Cooldown check
                    if (now - last_seen_dt) < cooldown_delta:
                        # Re-appeared within cooldown -> update last_seen only
                        if self._is_postgres:
                            cur.execute(
                                "UPDATE visitors SET last_seen = %s WHERE visitor_id = %s",
                                (now, visitor_id),
                            )
                        else:
                            cur.execute(
                                "UPDATE visitors SET last_seen = ? WHERE visitor_id = ?",
                                (now_str, visitor_id),
                            )
                        conn.commit()
                        return False
                    else:
                        # Re-visiting after cooldown -> increment visit count & add new visit_event
                        if self._is_postgres:
                            cur.execute(
                                "UPDATE visitors SET last_seen = %s, total_visits = total_visits + 1 WHERE visitor_id = %s",
                                (now, visitor_id),
                            )
                            cur.execute(
                                "INSERT INTO visit_events (visitor_id, timestamp, confidence) VALUES (%s, %s, %s)",
                                (visitor_id, now, confidence),
                            )
                        else:
                            cur.execute(
                                "UPDATE visitors SET last_seen = ?, total_visits = total_visits + 1 WHERE visitor_id = ?",
                                (now_str, visitor_id),
                            )
                            cur.execute(
                                "INSERT INTO visit_events (visitor_id, timestamp, confidence) VALUES (?, ?, ?)",
                                (visitor_id, now_str, confidence),
                            )
                        conn.commit()
                        logger.info("🔁 Visitor %s returned (re-visit event logged)", visitor_id)
                        return True
            except Exception as exc:
                logger.error("Failed to record visit for %s: %s", visitor_id, exc)
                return False
            finally:
                conn.close()

    # ── Reporting & Analytics Queries ────────────────────────────────

    def get_daily_stats(self) -> Dict[str, Any]:
        """Return daily visitor summary statistics."""
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_start_str = today_start.isoformat()

        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                if self._is_postgres:
                    cur.execute(
                        "SELECT COUNT(DISTINCT visitor_id) FROM visit_events WHERE timestamp >= %s",
                        (today_start,),
                    )
                    unique_today = cur.fetchone()[0] or 0

                    cur.execute(
                        "SELECT COUNT(*) FROM visit_events WHERE timestamp >= %s",
                        (today_start,),
                    )
                    total_today = cur.fetchone()[0] or 0

                    cur.execute("SELECT COUNT(*) FROM visitors")
                    total_registered = cur.fetchone()[0] or 0
                else:
                    cur.execute(
                        "SELECT COUNT(DISTINCT visitor_id) FROM visit_events WHERE timestamp >= ?",
                        (today_start_str,),
                    )
                    unique_today = cur.fetchone()[0] or 0

                    cur.execute(
                        "SELECT COUNT(*) FROM visit_events WHERE timestamp >= ?",
                        (today_start_str,),
                    )
                    total_today = cur.fetchone()[0] or 0

                    cur.execute("SELECT COUNT(*) FROM visitors")
                    total_registered = cur.fetchone()[0] or 0

                return {
                    "unique_visitors_today": unique_today,
                    "total_visits_today": total_today,
                    "total_registered_visitors": total_registered,
                    "date": today_start.strftime("%Y-%m-%d"),
                }
            except Exception as exc:
                logger.error("Error fetching daily stats: %s", exc)
                return {
                    "unique_visitors_today": 0,
                    "total_visits_today": 0,
                    "total_registered_visitors": 0,
                    "date": today_start.strftime("%Y-%m-%d"),
                }
            finally:
                conn.close()

    def get_visitors(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Return list of visitors sorted by most recent visit."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                if self._is_postgres:
                    cur.execute(
                        "SELECT visitor_id, first_seen, last_seen, total_visits FROM visitors "
                        "ORDER BY last_seen DESC LIMIT %s",
                        (limit,),
                    )
                    rows = cur.fetchall()
                    return [
                        {
                            "visitor_id": r[0],
                            "first_seen": str(r[1]),
                            "last_seen": str(r[2]),
                            "total_visits": r[3],
                        }
                        for r in rows
                    ]
                else:
                    cur.execute(
                        "SELECT visitor_id, first_seen, last_seen, total_visits FROM visitors "
                        "ORDER BY last_seen DESC LIMIT ?",
                        (limit,),
                    )
                    rows = cur.fetchall()
                    return [
                        {
                            "visitor_id": r["visitor_id"],
                            "first_seen": str(r["first_seen"]),
                            "last_seen": str(r["last_seen"]),
                            "total_visits": r["total_visits"],
                        }
                        for r in rows
                    ]
            except Exception as exc:
                logger.error("Error fetching visitors: %s", exc)
                return []
            finally:
                conn.close()

    def get_recent_visits(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent visit events sorted by timestamp descending."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                if self._is_postgres:
                    cur.execute(
                        "SELECT id, visitor_id, timestamp, confidence FROM visit_events "
                        "ORDER BY timestamp DESC LIMIT %s",
                        (limit,),
                    )
                    rows = cur.fetchall()
                    return [
                        {
                            "id": r[0],
                            "visitor_id": r[1],
                            "timestamp": str(r[2]),
                            "confidence": round(r[3], 2),
                        }
                        for r in rows
                    ]
                else:
                    cur.execute(
                        "SELECT id, visitor_id, timestamp, confidence FROM visit_events "
                        "ORDER BY timestamp DESC LIMIT ?",
                        (limit,),
                    )
                    rows = cur.fetchall()
                    return [
                        {
                            "id": r["id"],
                            "visitor_id": r["visitor_id"],
                            "timestamp": str(r["timestamp"]),
                            "confidence": round(r["confidence"], 2),
                        }
                        for r in rows
                    ]
            except Exception as exc:
                logger.error("Error fetching visit events: %s", exc)
                return []
            finally:
                conn.close()
