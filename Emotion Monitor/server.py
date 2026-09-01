"""
Server Application Entrypoint
Wraps around the locked Video Analyzer in app.py without modifying its internal code.
Mounts Extended APIs for Sessions, Analytics, Reports, and ML Intelligence.
"""

import os
import sys
from app import app, monitor, landmarker, cap
from extended_api import extended_bp, init_adapter

# Initialize Adapter with Video Analyzer engine
init_adapter(monitor)

# Register Extended Endpoints
app.register_blueprint(extended_bp)

@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 65)
    print("  AI EMOTION & ATTENTION MONITOR — INTEGRATED PLATFORM")
    print("=" * 65)
    print(f"  • Video Analyzer Engine : ACTIVE (DirectShow)")
    print(f"  • Session Management    : READY (SQLite WAL)")
    print(f"  • Analytics & Reports   : READY")
    print(f"  • ML Intelligence       : READY")
    print(f"  • Server URL            : http://localhost:{port}")
    print("=" * 65)
    try:
        app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
    finally:
        monitor.running = False
        if cap is not None:
            cap.release()
        landmarker.close()
