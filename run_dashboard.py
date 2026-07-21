"""
Standalone launcher for VisiTrack Web Dashboard.

Run this script to start the web server at http://localhost:8000
without requiring an active video stream.
"""

import sys
import os
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, os.getcwd())

from store_visitor_system.config import Config
from store_visitor_system.database import DatabaseManager
import uvicorn
from store_visitor_system.web_server import app, _db
import store_visitor_system.web_server as ws

def main():
    print("=" * 60)
    print("  VisiTrack — Web Dashboard Server")
    print("=" * 60)
    
    config = Config()
    db = DatabaseManager(config)
    ws._db = db
    
    print(f"\n🌐 Web Dashboard is live at: http://localhost:{config.web_server_port}")
    print("   Press Ctrl+C to stop.\n")
    
    uvicorn.run(app, host="0.0.0.0", port=config.web_server_port, log_level="info")

if __name__ == "__main__":
    main()
