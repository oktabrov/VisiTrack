"""
Lightweight FastAPI Web Dashboard for VisiTrack.

Provides a user-friendly single-page dashboard at http://localhost:8000
to view daily store visitor counts, visitor IDs, visit frequencies, and timestamps.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Optional

import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

if TYPE_CHECKING:
    from .config import Config
    from .database import DatabaseManager

logger = logging.getLogger(__name__)

# FastAPI App
app = FastAPI(title="VisiTrack Dashboard", version="1.0.0")

# Global variables for WebSockets and DB
_db: Optional["DatabaseManager"] = None
active_connections: List[WebSocket] = []
_loop: Optional[asyncio.AbstractEventLoop] = None


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Render the main single-page web dashboard."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VisiTrack — Visitor Dashboard</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0c10;
            --bg-card: rgba(22, 27, 34, 0.75);
            --border-color: rgba(255, 255, 255, 0.1);
            --accent-green: #10b981;
            --accent-blue: #3b82f6;
            --accent-purple: #8b5cf6;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-primary);
            background-image: 
                radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.12) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(139, 92, 246, 0.12) 0px, transparent 50%);
            color: var(--text-main);
            min-height: 100vh;
            padding: 2rem;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border-color);
        }

        .logo {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .logo-icon {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 1.25rem;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        }

        h1 {
            font-size: 1.75rem;
            font-weight: 700;
            letter-spacing: -0.02em;
        }

        .live-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: var(--accent-green);
            padding: 0.4rem 0.8rem;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
        }

        .pulse-dot {
            width: 8px;
            height: 8px;
            background-color: var(--accent-green);
            border-radius: 50%;
            animation: pulse 1.8s infinite;
        }

        @keyframes pulse {
            0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
            70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
            100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }

        /* Stat Cards Grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2.5rem;
        }

        .stat-card {
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }

        .stat-card:hover {
            transform: translateY(-2px);
            border-color: rgba(255, 255, 255, 0.2);
        }

        .stat-title {
            color: var(--text-muted);
            font-size: 0.9rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }

        .stat-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--text-main);
            line-height: 1;
        }

        .stat-sub {
            margin-top: 0.75rem;
            font-size: 0.85rem;
            color: var(--text-muted);
        }

        /* Data Section */
        .section-title {
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .table-card {
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            margin-bottom: 2rem;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            text-align: left;
        }

        th {
            background: rgba(255, 255, 255, 0.03);
            color: var(--text-muted);
            font-weight: 600;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 1rem 1.5rem;
            border-bottom: 1px solid var(--border-color);
        }

        td {
            padding: 1rem 1.5rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            font-size: 0.95rem;
        }

        tr:last-child td {
            border-bottom: none;
        }

        tr:hover td {
            background: rgba(255, 255, 255, 0.02);
        }

        .badge-id {
            font-family: monospace;
            background: rgba(59, 130, 246, 0.15);
            color: #60a5fa;
            padding: 0.25rem 0.6rem;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 600;
        }

        .badge-count {
            display: inline-block;
            background: rgba(139, 92, 246, 0.2);
            color: #c084fc;
            font-weight: 600;
            padding: 0.2rem 0.6rem;
            border-radius: 12px;
            font-size: 0.85rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo">
                <div class="logo-icon">V</div>
                <div>
                    <h1>VisiTrack Analytics</h1>
                    <p style="font-size: 0.85rem; color: var(--text-muted);">Real-Time Store Visitor Counter</p>
                </div>
            </div>
            <div class="live-badge">
                <span class="pulse-dot"></span>
                LIVE STREAMING
            </div>
        </header>

        <!-- Stat Cards -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-title">Unique Visitors Today</div>
                <div class="stat-value" id="stat-unique-today">0</div>
                <div class="stat-sub">Distinct people seen today</div>
            </div>
            <div class="stat-card">
                <div class="stat-title">This Week</div>
                <div class="stat-value" id="stat-unique-week">0</div>
                <div class="stat-sub">Distinct people seen this week</div>
            </div>
            <div class="stat-card">
                <div class="stat-title">This Month</div>
                <div class="stat-value" id="stat-unique-month">0</div>
                <div class="stat-sub">Distinct people seen this month</div>
            </div>
            <div class="stat-card">
                <div class="stat-title">This Year</div>
                <div class="stat-value" id="stat-unique-year">0</div>
                <div class="stat-sub">Distinct people seen this year</div>
            </div>
            <div class="stat-card">
                <div class="stat-title">Total Visit Events Today</div>
                <div class="stat-value" id="stat-visits-today">0</div>
                <div class="stat-sub">Includes repeat visits after cooldown</div>
            </div>
            <div class="stat-card">
                <div class="stat-title">Total Registered Visitors</div>
                <div class="stat-value" id="stat-total-reg">0</div>
                <div class="stat-sub">All-time unique customer database</div>
            </div>
        </div>

        <!-- Visitor Directory -->
        <div class="section-title">
            👥 Store Visitor Log & Frequency
        </div>
        <div class="table-card">
            <table>
                <thead>
                    <tr>
                        <th>Visitor ID</th>
                        <th>Visit Count</th>
                        <th>First Seen</th>
                        <th>Last Seen</th>
                    </tr>
                </thead>
                <tbody id="visitors-table-body">
                    <tr><td colspan="4" style="text-align: center; color: var(--text-muted);">Loading visitors...</td></tr>
                </tbody>
            </table>
        </div>

        <!-- Recent Visit Events -->
        <div class="section-title">
            🕒 Recent Visit Timestamps
        </div>
        <div class="table-card">
            <table>
                <thead>
                    <tr>
                        <th>Event ID</th>
                        <th>Visitor ID</th>
                        <th>Timestamp</th>
                        <th>Confidence</th>
                    </tr>
                </thead>
                <tbody id="events-table-body">
                    <tr><td colspan="4" style="text-align: center; color: var(--text-muted);">Loading visit history...</td></tr>
                </tbody>
            </table>
        </div>
    </div>

    <script>
        async function updateDashboard() {
            try {
                // Fetch daily stats
                const statsRes = await fetch('/api/stats');
                const stats = await statsRes.json();
                document.getElementById('stat-unique-today').innerText = stats.unique_visitors_today;
                document.getElementById('stat-unique-week').innerText = stats.unique_visitors_week;
                document.getElementById('stat-unique-month').innerText = stats.unique_visitors_month;
                document.getElementById('stat-unique-year').innerText = stats.unique_visitors_year;
                document.getElementById('stat-visits-today').innerText = stats.total_visits_today;
                document.getElementById('stat-total-reg').innerText = stats.total_registered_visitors;

                // Fetch visitors list
                const visitorsRes = await fetch('/api/visitors');
                const visitors = await visitorsRes.json();
                const vBody = document.getElementById('visitors-table-body');
                if (visitors.length === 0) {
                    vBody.innerHTML = '<tr><td colspan="4" style="text-align: center; color: var(--text-muted);">No visitors recorded yet today.</td></tr>';
                } else {
                    vBody.innerHTML = visitors.map(v => `
                        <tr>
                            <td><span class="badge-id">${v.visitor_id}</span></td>
                            <td><span class="badge-count">${v.total_visits} ${v.total_visits === 1 ? 'visit' : 'visits'}</span></td>
                            <td style="color: var(--text-muted);">${formatDate(v.first_seen)}</td>
                            <td><strong>${formatDate(v.last_seen)}</strong></td>
                        </tr>
                    `).join('');
                }

                // Fetch recent events
                const eventsRes = await fetch('/api/events');
                const events = await eventsRes.json();
                const eBody = document.getElementById('events-table-body');
                if (events.length === 0) {
                    eBody.innerHTML = '<tr><td colspan="4" style="text-align: center; color: var(--text-muted);">No visit events recorded yet.</td></tr>';
                } else {
                    eBody.innerHTML = events.map(e => `
                        <tr>
                            <td style="color: var(--text-muted);">#${e.id}</td>
                            <td><span class="badge-id">${e.visitor_id}</span></td>
                            <td>${formatDate(e.timestamp)}</td>
                            <td>${(e.confidence * 100).toFixed(0)}%</td>
                        </tr>
                    `).join('');
                }
            } catch (err) {
                console.error("Dashboard refresh error:", err);
            }
        }

        function formatDate(isoStr) {
            if (!isoStr) return '-';
            try {
                const d = new Date(isoStr);
                return d.toLocaleString();
            } catch (e) {
                return isoStr;
            }
        }

        // Initial load & backup periodic 10s refresh
        updateDashboard();
        setInterval(updateDashboard, 10000);

        // WebSockets for instant, real-time stats updates
        let wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        let wsUrl = `${wsProtocol}//${window.location.host}/ws`;
        let socket = new WebSocket(wsUrl);

        socket.onopen = function(e) {
            console.log("✅ WebSocket connection established.");
        };

        socket.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);
                console.log("⚡ Real-time update received:", data);
                
                // If it is a new visit, trigger dashboard refresh immediately
                if (data.type === 'new_visit') {
                    updateDashboard();
                }
            } catch (err) {
                console.error("Error processing real-time message:", err);
            }
        };

        socket.onclose = function(event) {
            console.log("❌ WebSocket closed. Reconnecting in 5 seconds...");
            setTimeout(() => {
                window.location.reload();
            }, 5000);
        };
    </script>
</body>
</html>"""
    return HTMLResponse(content=html_content)


@app.get("/api/stats")
async def get_stats():
    """API Endpoint: Daily visitor statistics."""
    if _db is None:
        return {"error": "Database not initialized"}
    return _db.get_daily_stats()


@app.get("/api/visitors")
async def get_visitors(limit: int = 50):
    """API Endpoint: Visitor directory and frequencies."""
    if _db is None:
        return []
    return _db.get_visitors(limit=limit)


@app.get("/api/events")
async def get_events(limit: int = 50):
    """API Endpoint: Recent visit event history."""
    if _db is None:
        return []
    return _db.get_recent_visits(limit=limit)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint to establish persistent real-time connections."""
    global _loop
    _loop = asyncio.get_running_loop()
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            # Keep connection alive, listen for ping/close
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        active_connections.remove(websocket)


async def broadcast_ws(message: dict):
    """Broadcast JSON message to all active WebSocket clients."""
    for connection in active_connections:
        try:
            await connection.send_json(message)
        except Exception:
            pass


def notify_visit_event(visitor_id: str, confidence: float, timestamp: str) -> None:
    """Thread-safe function to broadcast visitor event to all open dashboards instantly."""
    global _loop, _db
    if _loop and active_connections:
        stats = _db.get_daily_stats() if _db else {}
        event_data = {
            "type": "new_visit",
            "visitor_id": visitor_id,
            "confidence": confidence,
            "timestamp": timestamp,
            "stats": stats,
        }
        asyncio.run_coroutine_threadsafe(broadcast_ws(event_data), _loop)


class WebServerThread(threading.Thread):
    """Runs Uvicorn Web Server in a background thread."""

    def __init__(self, db_manager: "DatabaseManager", port: int = 8000) -> None:
        super().__init__(name="VisiTrack-WebServerThread", daemon=True)
        global _db
        _db = db_manager
        self._port = port
        self._server: Optional[uvicorn.Server] = None

    def run(self) -> None:
        logger.info("🌐 Starting Web Dashboard on http://localhost:%d …", self._port)
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=self._port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._server.run()

    def stop(self) -> None:
        if self._server:
            self._server.should_exit = True
