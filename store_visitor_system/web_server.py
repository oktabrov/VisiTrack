"""
Lightweight FastAPI Web Dashboard for VisiTrack.

Provides a user-friendly single-page dashboard at http://localhost:8000
to view daily store visitor counts, visitor IDs, visit frequencies, and timestamps.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Optional

import os
from pathlib import Path
from pydantic import BaseModel
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

# Pipeline & Camera status
pipeline_status = "OFFLINE"
camera_info = "Not Connected"
stream_resolution = "N/A"
decoder_backend = "N/A"


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
            font-family: inherit;
            cursor: pointer;
            outline: none;
            transition: filter 0.2s ease, transform 0.1s ease;
        }

        .live-badge:hover {
            filter: brightness(1.2);
        }

        .live-badge:active {
            transform: scale(0.97);
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

        /* Primary/Today's Stats Section */
        .primary-stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }

        .primary-card {
            background: rgba(22, 27, 34, 0.85);
            backdrop-filter: blur(16px);
            border-radius: 20px;
            padding: 2rem;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .primary-card:hover {
            transform: translateY(-3px);
        }

        .primary-card-unique {
            border: 1px solid rgba(16, 185, 129, 0.3);
            box-shadow: 0 10px 40px rgba(16, 185, 129, 0.08), 0 0 1px 1px rgba(16, 185, 129, 0.2);
        }
        .primary-card-unique:hover {
            box-shadow: 0 12px 48px rgba(16, 185, 129, 0.15), 0 0 2px 2px rgba(16, 185, 129, 0.4);
        }

        .primary-card-visits {
            border: 1px solid rgba(59, 130, 246, 0.3);
            box-shadow: 0 10px 40px rgba(59, 130, 246, 0.08), 0 0 1px 1px rgba(59, 130, 246, 0.2);
        }
        .primary-card-visits:hover {
            box-shadow: 0 12px 48px rgba(59, 130, 246, 0.15), 0 0 2px 2px rgba(59, 130, 246, 0.4);
        }

        .primary-title {
            color: var(--text-muted);
            font-size: 0.95rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.75rem;
        }

        .primary-value {
            font-size: 3.5rem;
            font-weight: 800;
            line-height: 1;
            margin-bottom: 0.75rem;
        }
        
        .primary-card-unique .primary-value {
            background: linear-gradient(135deg, #10b981, #34d399);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .primary-card-visits .primary-value {
            background: linear-gradient(135deg, #3b82f6, #60a5fa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .primary-sub {
            font-size: 0.9rem;
            color: var(--text-muted);
        }

        /* Secondary Stats Grid */
        .secondary-stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.25rem;
            margin-bottom: 2.5rem;
        }

        .secondary-card {
            background: rgba(22, 27, 34, 0.45);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 14px;
            padding: 1.25rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }

        .secondary-card:hover {
            transform: translateY(-2px);
            border-color: rgba(255, 255, 255, 0.15);
            background: rgba(22, 27, 34, 0.6);
        }

        .secondary-title {
            color: var(--text-muted);
            font-size: 0.8rem;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }

        .secondary-value {
            font-size: 1.85rem;
            font-weight: 700;
            color: #d1d5db;
            line-height: 1;
            margin-bottom: 0.4rem;
        }

        .secondary-sub {
            font-size: 0.8rem;
            color: rgba(156, 163, 175, 0.7);
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
            <button class="live-badge" id="status-badge" style="background: rgba(156, 163, 175, 0.15); border: 1px solid rgba(156, 163, 175, 0.3); color: #9ca3af;" onclick="handleStatusClick()">
                <span class="pulse-dot" id="status-dot" style="background-color: #9ca3af;"></span>
                <span id="status-text">PIPELINE OFFLINE</span>
            </button>
        </header>

        <!-- Camera Information Info Bar -->
        <div class="camera-info-bar" style="display: flex; flex-wrap: wrap; gap: 2rem; font-size: 0.85rem; color: var(--text-muted); margin-bottom: 2rem; border: 1px solid var(--border-color); background: rgba(255, 255, 255, 0.02); padding: 0.6rem 1.2rem; border-radius: 12px; backdrop-filter: blur(8px);">
            <div>📹 Camera Stream: <span id="info-camera" style="color: var(--text-main); font-weight: 500;">Not Connected</span></div>
            <div>📐 Resolution: <span id="info-resolution" style="color: var(--text-main); font-weight: 500;">N/A</span></div>
            <div>⚙️ Decoder Backend: <span id="info-backend" style="color: var(--text-main); font-weight: 500;">N/A</span></div>
        </div>

        <!-- Primary/Today's Stats (Main Indicators) -->
        <div class="section-title" style="margin-top: 1rem;">
            📊 Today's Metrics (Real-Time)
        </div>
        <div class="primary-stats-grid">
            <div class="primary-card primary-card-unique">
                <div class="primary-title">Unique Visitors Today</div>
                <div class="primary-value" id="stat-unique-today">0</div>
                <div class="primary-sub">Distinct people identified in the store today</div>
            </div>
            <div class="primary-card primary-card-visits">
                <div class="primary-title">Total Visit Events Today</div>
                <div class="primary-value" id="stat-visits-today">0</div>
                <div class="primary-sub">Includes repeat customers entering after 10m cooldown</div>
            </div>
        </div>

        <!-- Secondary Periodical Stats -->
        <div class="section-title">
            📈 Historical Analytics
        </div>
        <div class="secondary-stats-grid">
            <div class="secondary-card">
                <div class="secondary-title">This Week</div>
                <div class="secondary-value" id="stat-unique-week">0</div>
                <div class="secondary-sub">Distinct visitors this week</div>
            </div>
            <div class="secondary-card">
                <div class="secondary-title">This Month</div>
                <div class="secondary-value" id="stat-unique-month">0</div>
                <div class="secondary-sub">Distinct visitors this month</div>
            </div>
            <div class="secondary-card">
                <div class="secondary-title">This Year</div>
                <div class="secondary-value" id="stat-unique-year">0</div>
                <div class="secondary-sub">Distinct visitors this year</div>
            </div>
            <div class="secondary-card">
                <div class="secondary-title">All-Time Database</div>
                <div class="secondary-value" id="stat-total-reg">0</div>
                <div class="secondary-sub">Total registered visitor IDs</div>
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

                // Update camera status details
                updateStatusUI(
                    stats.pipeline_status,
                    stats.camera_info,
                    stats.resolution,
                    stats.backend
                );

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

        function updateStatusUI(status, camera, resolution, backend) {
            const badge = document.getElementById('status-badge');
            const dot = document.getElementById('status-dot');
            const text = document.getElementById('status-text');

            document.getElementById('info-camera').innerText = camera || 'Not Connected';
            document.getElementById('info-resolution').innerText = resolution || 'N/A';
            document.getElementById('info-backend').innerText = backend || 'N/A';

            if (status === 'STREAMING') {
                badge.style.background = 'rgba(16, 185, 129, 0.15)';
                badge.style.border = '1px solid rgba(16, 185, 129, 0.3)';
                badge.style.color = '#10b981';
                dot.style.backgroundColor = '#10b981';
                text.innerText = 'LIVE STREAMING';
            } else if (status === 'CONNECTING') {
                badge.style.background = 'rgba(245, 158, 11, 0.15)';
                badge.style.border = '1px solid rgba(245, 158, 11, 0.3)';
                badge.style.color = '#f59e0b';
                dot.style.backgroundColor = '#f59e0b';
                text.innerText = 'CONNECTING CAMERA...';
            } else if (status === 'ERROR') {
                badge.style.background = 'rgba(239, 68, 68, 0.15)';
                badge.style.border = '1px solid rgba(239, 68, 68, 0.3)';
                badge.style.color = '#ef4444';
                dot.style.backgroundColor = '#ef4444';
                text.innerText = 'CONNECTION ERROR';
            } else {
                badge.style.background = 'rgba(156, 163, 175, 0.15)';
                badge.style.border = '1px solid rgba(156, 163, 175, 0.3)';
                badge.style.color = '#9ca3af';
                dot.style.backgroundColor = '#9ca3af';
                text.innerText = 'PIPELINE OFFLINE';
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

        let isStarting = false;
        async function handleStatusClick() {
            const badge = document.getElementById('status-badge');
            const text = document.getElementById('status-text');

            const currentStatus = text.innerText;
            if (currentStatus === 'LIVE STREAMING' || currentStatus === 'CONNECTING CAMERA...') {
                return;
            }

            if (isStarting) return;
            isStarting = true;

            // Instantly transition to Connecting UI state
            updateStatusUI('CONNECTING', '', '', '');

            try {
                const res = await fetch('/api/start_pipeline', { method: 'POST' });
                const result = await res.json();
                console.log("Pipeline start response:", result);

                if (result.status === 'missing_rtsp_url') {
                    // Prompt user to enter RTSP URL
                    const url = prompt(
                        "RTSP Camera Stream URL is missing or set to default placeholder.\n\nPlease enter your RTSP URL (e.g. rtsp://admin:pass@192.168.1.50:554/stream):"
                    );
                    if (url && url.trim() !== '') {
                        const saveRes = await fetch('/api/save_rtsp_url', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({ rtsp_url: url.trim() })
                        });
                        const saveResult = await saveRes.json();
                        console.log("Save RTSP response:", saveResult);
                    } else {
                        // User cancelled
                        updateStatusUI('OFFLINE', '', '', '');
                    }
                }
            } catch (err) {
                console.error("Error launching pipeline:", err);
                updateStatusUI('ERROR', '', '', '');
            } finally {
                isStarting = false;
            }
        }

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
                
                if (data.type === 'new_visit') {
                    updateDashboard();
                } else if (data.type === 'status_update') {
                    updateStatusUI(data.status, data.camera_info, data.resolution, data.backend);
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
    """API Endpoint: Daily visitor statistics and pipeline/camera status."""
    stats = {}
    if _db is not None:
        stats = _db.get_daily_stats()
    else:
        stats = {
            "unique_visitors_today": 0,
            "total_visits_today": 0,
            "unique_visitors_week": 0,
            "unique_visitors_month": 0,
            "unique_visitors_year": 0,
            "total_registered_visitors": 0,
            "date": datetime.now().strftime("%Y-%m-%d"),
        }
    stats.update({
        "pipeline_status": pipeline_status,
        "camera_info": camera_info,
        "resolution": stream_resolution,
        "backend": decoder_backend,
    })
    return stats


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


def update_pipeline_status(status: str, camera: str = "", resolution: str = "", backend: str = "") -> None:
    """Thread-safe function to update and broadcast camera/pipeline status changes instantly."""
    global pipeline_status, camera_info, stream_resolution, decoder_backend, _loop
    pipeline_status = status
    if camera:
        camera_info = camera
    if resolution:
        stream_resolution = resolution
    if backend:
        decoder_backend = backend

    if _loop and active_connections:
        event_data = {
            "type": "status_update",
            "status": pipeline_status,
            "camera_info": camera_info,
            "resolution": stream_resolution,
            "backend": decoder_backend,
        }
        asyncio.run_coroutine_threadsafe(broadcast_ws(event_data), _loop)


class RTSPConfig(BaseModel):
    rtsp_url: str


@app.post("/api/save_rtsp_url")
async def save_rtsp_url(data: RTSPConfig):
    """Save the RTSP URL entered by the user to .env file and trigger pipeline start."""
    global pipeline_status, _pipeline_thread
    url = data.rtsp_url.strip()
    if not url:
        return {"status": "error", "message": "RTSP URL cannot be empty"}

    # Update environment variables so Config instantiations see the update
    os.environ["RTSP_URL"] = url

    # Write to local .env file
    env_path = Path(".env")
    lines = []
    updated = False
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip().startswith("RTSP_URL="):
                    lines.append(f"RTSP_URL={url}\n")
                    updated = True
                else:
                    lines.append(line)
    
    if not updated:
        lines.append(f"\nRTSP_URL={url}\n")

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    logger.info("Saved new RTSP URL to .env file: %s", url)

    # Automatically start pipeline
    if pipeline_status not in ("STREAMING", "CONNECTING"):
        _pipeline_thread = PipelineRunnerThread(_db)
        _pipeline_thread.start()
        return {"status": "started"}
    
    return {"status": "already_running"}


_pipeline_thread: Optional[PipelineRunnerThread] = None


@app.post("/api/start_pipeline")
async def start_pipeline():
    """Trigger the VisiTrack AI video processing pipeline execution."""
    global pipeline_status, _pipeline_thread
    if pipeline_status in ("STREAMING", "CONNECTING"):
        return {"status": "already_running"}

    # Check if RTSP URL is missing or set to placeholder
    try:
        from .config import Config
        config = Config()
        url = config.rtsp_url
        if not url or url.strip() == "" or "192.168.1.100" in url:
            return {"status": "missing_rtsp_url"}
    except Exception as exc:
        logger.error("Failed to validate RTSP URL configuration: %s", exc)

    _pipeline_thread = PipelineRunnerThread(_db)
    _pipeline_thread.start()
    return {"status": "started"}


class PipelineRunnerThread(threading.Thread):
    """Runs VisiTrack AI/video inference pipeline in a background thread."""

    def __init__(self, db_manager: "DatabaseManager") -> None:
        super().__init__(name="VisiTrack-PipelineRunnerThread", daemon=True)
        self._db = db_manager

    def run(self) -> None:
        logger.info("Initializing VisiTrack AI Pipeline runner thread …")
        try:
            from .config import Config
            from .gpu import GPUManager
            from .pipeline import InferencePipeline

            config = Config()
            gpu = GPUManager(config)
            
            gpu.select_device()
            pipeline = InferencePipeline(gpu, config, db_manager=self._db)
            pipeline.run()
        except Exception as exc:
            logger.error("Failed to run AI pipeline thread: %s", exc)
            update_pipeline_status("ERROR")
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
